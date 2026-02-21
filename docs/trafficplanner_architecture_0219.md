# TrafficPlanner Model 아키텍처 (2026-02-19 최신)

## 개요

`trafficplanner_model.py`는 STRIVE의 `fit_traffic_model.py`를 기반으로 확장한 모델입니다:
- **이중 잠재변수**: `z_global` (32dim, pretrain 후 frozen) + `z_local` (64dim, 매 스텝 ego 전용)
- **수정된 GCN**: `IndividualSceneInteractionNet` — ego/other 분리 출력
- **2-layer Cross-Attention + Self-Attention**: z_local의 interaction 모델링
- **Teacher Forcing**: cosine annealing으로 segment 길이 점진 확대
- **가변/고정 윈도우**: `growing_window` 플래그로 전환

---

## 1. 학습 전략

### Phase 1: 사전학습 (Normal 데이터)
- 전체 모델 학습 (z_global + z_local + decoder 전부)
- `z_global`이 과거 궤적에서 주행 의도를 학습
- Loss: `L_recon + λ_kl * L_kl(z_global)`

### Phase 2: 파인튜닝 (Adversarial 데이터)
- **z_global 인코더 고정** (prior_net, posterior_net, GCN, GRU, Transformer, map)
- **z_local + ego decoder만 학습**
- 목표: z_local이 위험 상황에 대한 반응 학습
- Loss: `L_recon` (z_global frozen이므로 KL loss 없음)

```
┌──────────────────────────────────────────────────────────────┐
│  freeze_for_finetuning()                                      │
│                                                                │
│  FROZEN (requires_grad=False):                                 │
│  ├── latent_prior_net          (MLP)                          │
│  ├── latent_posterior_net      (MLP)                          │
│  ├── step_feature_extractor    (MLP)                          │
│  ├── temporal_gcn_encoder      (IndividualSceneInteractionNet)│
│  ├── prior_temporal_gru        (GRU, 2-layer)                 │
│  ├── positional_encoding       (Sinusoidal PE)                │
│  ├── transformer_encoder       (3-layer, 8-head)              │
│  ├── map_conv                  (6-layer CNN)                  │
│  ├── map_feature               (Linear)                       │
│  ├── decoder_net               (IndividualSceneInteractionNet)│
│  └── decoder_memory            (GRU, 3-layer)                 │
│                                                                │
│  TRAINABLE (requires_grad=True):                               │
│  ├── z_local_gcn               (IndividualSceneInteractionNet)│
│  ├── z_local_temporal_gru      (GRU, 1-layer)                 │
│  ├── z_local_ego_temporal_gru  (GRU, 1-layer, ego_full_query) │
│  ├── cross_attn_1 / norm_1     (MultiheadAttention + LN)      │
│  ├── cross_attn_2 / norm_2     (MultiheadAttention + LN)      │
│  ├── interaction_self_attn / norm (MultiheadAttention + LN)   │
│  ├── ego_warmup_gru            (GRU, 3-layer)                 │
│  ├── z_combine_mlp             (MLP)                          │
│  ├── ego_decoder_gru           (GRU, 3-layer)                 │
│  └── ego_output_mlp            (MLP)                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 모델 전체 구조

### 2.1 하이퍼파라미터 요약

```
┌─────────────────────────────────────────────┐
│  하이퍼파라미터                                │
├─────────────────────────────────────────────┤
│  PT (past_len)           = 4                │
│  FT (future_len)         = 12               │
│  NC (nclasses)           = 2 (car, truck)   │
│  state_size              = 6 (x,y,hx,hy,s,hdot) │
│  att_feat_size           = 2 (l, w)         │
│  gcn_hidden_dim          = 64               │
│  map_feat_size           = 64               │
│  past_feat_size          = 64               │
│  z_size (z_global)       = 32               │
│  z_local_size            = 32 (unused, kept for compat) │
│  z_local_out_size        = 64 (actual)      │
│  z_local_gcn_feat_size   = 64               │
│  z_local_window          = 4~8 (configurable) │
│  growing_window          = True/False       │
│  ego_full_query          = True/False       │
│  cross_attn_heads        = 4                │
│  transformer_nhead       = 8                │
│  transformer_nlayer      = 3                │
│  num_memory_layers       = 3                │
│  output_bicycle          = True → traj_out=2 (a, hdot) │
│  dt                      = 0.5              │
│  tf_max_annealing_epoch  = 1600             │
│  tf_init_segment_len     = 1                │
└─────────────────────────────────────────────┘
```

### 2.2 전체 데이터 흐름

```
입력: scene_graph (past, future, edge_index, lw, sem, batch), map_idx, map_env

Step 1: Map Feature
├── map_conv (6-layer CNN: 4→16→32→64→64→128→128, kernel=[7,5,5,3,3,3], stride=2)
├── flatten → map_feature (Linear: 128*H'*W' → 64)
└── map_feat: (NA, 64)

Step 2: z_global Prior (past only)
├── _run_temporal_encoder(past, PT=4, use_transformer=False)
│   ├── Per-step: state(6)+lw(2)+vis(1)+sem(2) → step_feature_extractor(MLP: 11→128→64)
│   ├── Per-step: temporal_gcn_encoder(IndivGCN: 64→128→64) → ego/other feat merge
│   ├── Stack: (NA, 4, 64)
│   └── prior_temporal_gru(GRU: 64→64, 2-layer) → context = last step (NA, 64)
├── prior_in = concat(context, map_feat, sem) → (NA, 64+64+2 = 130)
├── latent_prior_net(MLP: 130→128→64) → split → z_prior_mean(32), z_prior_var(32)
└── past_seq_out: (NA, 4, 64)  ← decoder에서 past_feat 추출용

Step 3: z_global Posterior (past + future, 학습 시만)
├── _run_temporal_encoder(future, FT=12, use_transformer=True)
│   ├── Per-step GCN: 동일
│   ├── Stack: (NA, 12, 64)
│   ├── positional_encoding(Sinusoidal PE, max_len=12)
│   └── transformer_encoder(3-layer, 8-head, d_model=64) → future_context (NA, 64)
├── posterior_in = concat(past_context, future_context, map_feat, sem) → (NA, 194)
└── latent_posterior_net(MLP: 194→128→64) → z_post_mean(32), z_post_var(32)

Step 4: z_global 샘플링
└── z = rsample(z_post_mean, z_post_var) → (NA, 32)

Step 5: Decoder (autoregressive, FT=12 steps)
└── (아래 섹션 2.4~2.6에서 상세 설명)
```

---

### 2.3 GCN: IndividualSceneInteractionNet

모든 GCN은 동일한 클래스를 사용하며, 용도별로 다른 차원의 입출력을 가집니다.

```
┌─────────────────────────────────────────────────────────────┐
│  IndividualSceneInteractionNet                                │
│                                                               │
│  1. Input MLP (shared):                                       │
│     mlp_in: in_channels → msg_node_channels                  │
│                                                               │
│  2. Message Passing (shared):                                 │
│     for layer in self.msg:  # AgentInteractionConv            │
│         x = layer(x, edge_index, pos, sem, h=hidden)         │
│                                                               │
│  3. Output MLP (ego/other separate):                          │
│     ego_feat   = mlp_out_ego(x[ego_idx])                     │
│     other_feat = mlp_out_other(x[other_idx])                 │
└─────────────────────────────────────────────────────────────┘
```

| 용도 | 인스턴스명 | in → msg → out | 입력 구성 |
|------|-----------|----------------|-----------|
| Temporal Encoder | `temporal_gcn_encoder` | 64→128→64 | step_feature_extractor 출력 |
| Sur Decoder | `decoder_net` | 164→64→2 | z(32)+past_feat(64)+map(64)+sem(2)+lw(2) |
| z_local Per-step | `z_local_gcn` | 10→64→64 | state(6)+lw(2)+sem(2) |

---

### 2.4 z_local 인코더 (Ego 전용, 매 스텝)

```
┌─────────────────────────────────────────────────────────────────────┐
│  z_local 계산 파이프라인 (_compute_z_local)                           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  1. 입력 준비                                                    │ │
│  │                                                                   │ │
│  │  ego_feat_window:   list of (num_ego, 64)   — W 스텝            │ │
│  │  other_feat_window: list of (num_other, 64) — W 스텝            │ │
│  │                                                                   │ │
│  │  W = z_local_window (고정) 또는 4+t (growing)                    │ │
│  │                                                                   │ │
│  │  growing_window=False, z_local_window=6:                         │ │
│  │    t=0: W=4, t=1: W=5, t=2~11: W=6 (고정)                      │ │
│  │                                                                   │ │
│  │  growing_window=True:                                            │ │
│  │    t=0: W=4, t=5: W=9, t=11: W=15                               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  2. Other 시간 인코딩                                            │ │
│  │                                                                   │ │
│  │  other_feat_window: (W, num_other, 64)                          │ │
│  │       ↓ permute (1,0,2)                                          │ │
│  │  (num_other, W, 64)                                              │ │
│  │       ↓ z_local_temporal_gru (GRU: 64→64, 1-layer)              │ │
│  │  (num_other, W, 64)                                              │ │
│  │       ↓ reshape                                                  │ │
│  │  KV = (1, num_other×W, 64)                                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  3. Ego Query 구성                                               │ │
│  │                                                                   │ │
│  │  ego_full_query=False:                                           │ │
│  │    Q = ego_feat_window[-1]               → (1, 1, 64)           │ │
│  │                                                                   │ │
│  │  ego_full_query=True:                                            │ │
│  │    ego_feat_window: (W, num_ego, 64)                            │ │
│  │         ↓ z_local_ego_temporal_gru (GRU: 64→64, 1-layer)        │ │
│  │    Q = (1, W, 64)                                                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  4. 2-Layer Cross-Attention (UniAD/GameFormer style)             │ │
│  │                                                                   │ │
│  │  Q: (1, Q_len, 64)    KV: (1, num_other×W, 64)                 │ │
│  │                                                                   │ │
│  │  Layer 1:                                                        │ │
│  │    attn1 = cross_attn_1(Q, KV, KV)    [4 heads, head_dim=16]   │ │
│  │    attn1 = LayerNorm(attn1 + Q)       [residual + norm]        │ │
│  │                                                                   │ │
│  │  Layer 2:                                                        │ │
│  │    attn2 = cross_attn_2(attn1, KV, KV) [4 heads, head_dim=16] │ │
│  │    attn2 = LayerNorm(attn2 + attn1)    [residual + norm]       │ │
│  │                                                                   │ │
│  │  if ego_full_query: mean pooling → (1, 1, 64)                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  5. Self-Attention (ego ↔ sur mutual interaction)                │ │
│  │                                                                   │ │
│  │  ego_token:  attn2 output                → (1, 1, 64)          │ │
│  │  sur_tokens: other_feat_seq.mean(dim=1)  → (1, num_other, 64)  │ │
│  │  all_tokens: concat                      → (1, 1+num_other, 64)│ │
│  │                                                                   │ │
│  │  sa_out = interaction_self_attn(all, all, all)                   │ │
│  │                          [4 heads, head_dim=16]                  │ │
│  │  sa_out = LayerNorm(sa_out + all_tokens)  [residual + norm]     │ │
│  │                                                                   │ │
│  │  ego_out = sa_out[:, 0, :]               → (64,)               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  출력: z_local = ego_out → (num_ego, 64)   [deterministic, 투영 없음] │
│                                                                       │
│  ※ sur agent가 0인 경우: z_local = zeros(64)                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.5 디코더: Autoregressive

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Autoregressive Decoder                             │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  WARM-UP PHASE                                                 ║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  1. z_local window 초기화 (past 4스텝)                          ║   │
│  ║     for pt in [0,1,2,3]:                                       ║   │
│  ║       state_t = scene_graph.past[:, pt, :]        (NA, 6)     ║   │
│  ║       gcn_in = concat(state_t, lw, sem)           (NA, 10)    ║   │
│  ║       ego_f, other_f = z_local_gcn(gcn_in)        (ego:64, other:64) ║
│  ║       ego_feat_window.append(ego_f)                            ║   │
│  ║       other_feat_window.append(other_f)                        ║   │
│  ║     → window = [feat_p0, feat_p1, feat_p2, feat_p3]  len=4   ║   │
│  ║                                                                 ║   │
│  ║  2. Ego GRU hidden 초기화                                      ║   │
│  ║     ego_warmup_gru(past GCN features)                          ║   │
│  ║       input:  (num_ego, PT, 64)                                ║   │
│  ║       output: hidden (3, num_ego, 64)                          ║   │
│  ║                                                                 ║   │
│  ║  3. 초기 상태                                                   ║   │
│  ║     prev_state = scene_graph.past[:, -1, :]       (NA, 6)     ║   │
│  ║     past_feat  = past_seq_out[:, -1, :]           (NA, 64)    ║   │
│  ║     other_mem_state = past_feat[~ego]  → GRU hidden init      ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  DECODING LOOP: for t = 0 to 11                                ║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  STEP 1: z_local 계산                                          ║   │
│  ║  ┌───────────────────────────────────────────────────────────┐ ║   │
│  ║  │  ego_current = ego_feat_window[-1]         (num_ego, 64) │ ║   │
│  ║  │  z_local = _compute_z_local(                             │ ║   │
│  ║  │      Q=ego_current, KV=other_feat_window                 │ ║   │
│  ║  │  )                                         (num_ego, 64) │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  if use_z_local=False: z_local = zeros(64)  [ablation]   │ ║   │
│  ║  └───────────────────────────────────────────────────────────┘ ║   │
│  ║                                                                 ║   │
│  ║  STEP 2: Sur 차량 디코딩 (STRIVE style)                        ║   │
│  ║  ┌───────────────────────────────────────────────────────────┐ ║   │
│  ║  │  decoder_in = concat(z_global, past_feat, map, sem, lw)  │ ║   │
│  ║  │                           (NA, 32+64+64+2+2 = 164)       │ ║   │
│  ║  │  _, other_out = decoder_net(scene_graph)                 │ ║   │
│  ║  │                           other_out: (num_other, 2)      │ ║   │
│  ║  │  next_other = bicycle_model(prev, a, hdot)               │ ║   │
│  ║  └───────────────────────────────────────────────────────────┘ ║   │
│  ║                                                                 ║   │
│  ║  STEP 3: Ego 차량 디코딩                                       ║   │
│  ║  ┌───────────────────────────────────────────────────────────┐ ║   │
│  ║  │  z_combined = z_combine_mlp(                             │ ║   │
│  ║  │      concat(z_global[ego], z_local)                      │ ║   │
│  ║  │  )               (32+64=96) → MLP(96→64→64) → (64)      │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  gru_in = concat(z_combined, map, sem, lw)               │ ║   │
│  ║  │                   (64+64+2+2 = 132)                      │ ║   │
│  ║  │  gru_out, hidden = ego_decoder_gru(gru_in, hidden)       │ ║   │
│  ║  │                   GRU(132→64, 3-layer) → (num_ego, 64)   │ ║   │
│  ║  │  ego_out = ego_output_mlp(gru_out)                       │ ║   │
│  ║  │                   MLP(64→64→2) → (num_ego, 2) = (a, hdot)│ ║   │
│  ║  │  next_ego = bicycle_model(prev, a, hdot)                 │ ║   │
│  ║  └───────────────────────────────────────────────────────────┘ ║   │
│  ║                                                                 ║   │
│  ║  STEP 4: 상태 업데이트                                          ║   │
│  ║  ┌───────────────────────────────────────────────────────────┐ ║   │
│  ║  │  prev_state ← merge(next_ego, next_other)                │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  # Sur memory 업데이트                                    │ ║   │
│  ║  │  past_feat[other] ← decoder_memory(                      │ ║   │
│  ║  │      state_local, other_mem_state                         │ ║   │
│  ║  │  )                  GRU(4→64, 3-layer)                   │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  # Map 업데이트                                            │ ║   │
│  ║  │  map_feat ← encode_map(next_pos)                         │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  # z_local window 업데이트                                 │ ║   │
│  ║  │  new_ego_f, new_other_f = z_local_gcn(next_state)        │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  if growing_window:                                       │ ║   │
│  ║  │    ego_feat_window.append(new_ego_f)      # 누적         │ ║   │
│  ║  │    other_feat_window.append(new_other_f)                 │ ║   │
│  ║  │  else:                                                    │ ║   │
│  ║  │    if len >= z_local_window:                               │ ║   │
│  ║  │      ego_feat_window.pop(0)               # 고정 슬라이드│ ║   │
│  ║  │      other_feat_window.pop(0)                             │ ║   │
│  ║  │    ego_feat_window.append(new_ego_f)                     │ ║   │
│  ║  │    other_feat_window.append(new_other_f)                 │ ║   │
│  ║  └───────────────────────────────────────────────────────────┘ ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  출력: traj_out (NA, 12, 4) or (NA, NS, 12, 4)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2.6 디코더: Teacher Forcing

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Teacher Forcing Decoder                            │
│                                                                       │
│  Cosine Annealing:                                                   │
│    segment_len = tf_init_segment_len(1) → 12 over tf_max_epoch(1600)│
│    초기: 매 스텝 GT 리셋 → 후기: 12스텝 전체 autoregressive          │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  초기 WARM-UP (Autoregressive와 동일)                          ║   │
│  ║  window = [feat_p0, feat_p1, feat_p2, feat_p3]   len=4       ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  SEGMENT LOOP: for target_t = 0, segment_len, 2*segment_len..║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  1. Segment 시작 시 리셋                                        ║   │
│  ║  ┌───────────────────────────────────────────────────────────┐ ║   │
│  ║  │  ego_prev_state ← GT(target_t - 1)                       │ ║   │
│  ║  │  ego_gru_hidden ← _warmup_ego_gru_hidden(target_t)      │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │  if target_t > 0:                                         │ ║   │
│  ║  │    z_local window 리셋 (GT 기반):                          │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │    growing_window=True:                                    │ ║   │
│  ║  │      past 4스텝 + GT future 0..target_t-1 전부             │ ║   │
│  ║  │      예: target_t=6 → [p0,p1,p2,p3,f0,f1,f2,f3,f4,f5]   │ ║   │
│  ║  │                                            len=10         │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │    growing_window=False (z_local_window=6):                │ ║   │
│  ║  │      최근 z_local_window 스텝만 (clamp로 범위 보호)        │ ║   │
│  ║  │      예: target_t=6 → [f0,f1,f2,f3,f4,f5]    len=6       │ ║   │
│  ║  │      예: target_t=2 → [p2,p3,f0,f1]          len=4       │ ║   │
│  ║  │                       (6개 요청했지만 4개만 가용)           │ ║   │
│  ║  └───────────────────────────────────────────────────────────┘ ║   │
│  ║                                                                 ║   │
│  ║  2. Segment 내부 디코딩 (segment_len 스텝)                      ║   │
│  ║  ┌───────────────────────────────────────────────────────────┐ ║   │
│  ║  │  for step in range(segment_len):                          │ ║   │
│  ║  │    actual_t = target_t + step                             │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │    # ego=predicted/GT-reset, sur=GT                       │ ║   │
│  ║  │    z_local 계산 (Autoregressive와 동일 로직)              │ ║   │
│  ║  │    ego 디코딩 (Autoregressive와 동일 로직)                │ ║   │
│  ║  │                                                           │ ║   │
│  ║  │    # Window 업데이트 (마지막 step 제외)                    │ ║   │
│  ║  │    if step < segment_len - 1:                              │ ║   │
│  ║  │      combined = merge(ego_predicted, sur_GT)              │ ║   │
│  ║  │      new_feats = z_local_gcn(combined)                   │ ║   │
│  ║  │      growing → append / fixed → pop+append               │ ║   │
│  ║  └───────────────────────────────────────────────────────────┘ ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  출력: ego_segments (list of segments), z_local_outputs              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 모듈 상세

### 3.1 z_global 인코더

```
┌─────────────────────────────────────────────────────────────────────┐
│                     _run_temporal_encoder                             │
│                                                                       │
│  입력: traj_data (NA, T, 6), vis_data (NA, T)                       │
│                                                                       │
│  1. Per-step GCN:                                                    │
│     for t in range(T):                                               │
│       step_in = concat(state, lw, vis, sem)           (NA, 11)      │
│       gcn_in  = step_feature_extractor(step_in)       (NA, 64)      │
│                 MLP: 11 → 128 → 64                                  │
│       ego_f, other_f = temporal_gcn_encoder(gcn_in)   (NA, 64)      │
│                 IndivGCN: 64 → 128 → 64                             │
│       gcn_feat = merge(ego_f, other_f)                (NA, 64)      │
│     sequence = stack → (NA, T, 64)                                   │
│                                                                       │
│  2a. Prior (use_transformer=False):                                  │
│     ┌─────────────────────────────────────────────────────────────┐ │
│     │  prior_temporal_gru                                          │ │
│     │    GRU(input=64, hidden=64, layers=2, batch_first=True)     │ │
│     │    input:  (NA, 4, 64)                                       │ │
│     │    output: (NA, 4, 64) → context = [:, -1, :] → (NA, 64)   │ │
│     └─────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  2b. Posterior (use_transformer=True):                                │
│     ┌─────────────────────────────────────────────────────────────┐ │
│     │  positional_encoding (Sinusoidal, max_len=12)               │ │
│     │    (NA, 12, 64) → (NA, 12, 64)                              │ │
│     │                                                               │ │
│     │  transformer_encoder                                         │ │
│     │    TransformerEncoder(                                        │ │
│     │      TransformerEncoderLayer(d_model=64, nhead=8)            │ │
│     │      num_layers=3                                            │ │
│     │    )                                                          │ │
│     │    head_dim = 64/8 = 8                                       │ │
│     │    input:  (NA, 12, 64) + padding_mask                       │ │
│     │    output: (NA, 12, 64) → context = [:, -1, :] → (NA, 64)  │ │
│     └─────────────────────────────────────────────────────────────┘ │
│                                                                       │
│             ┌──────────┴──────────┐                                  │
│             ▼                      ▼                                  │
│        PRIOR Net               POSTERIOR Net                         │
│   MLP: 130 → 128 → 64      MLP: 194 → 128 → 64                    │
│   (64+64+2)                 (64+64+64+2)                             │
│        ↓                          ↓                                  │
│   z_prior (μ:32, σ²:32)    z_post (μ:32, σ²:32)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Ego Decoder 컴포넌트

```
┌─────────────────────────────────────────────────────────────┐
│  Ego Decoder 데이터 흐름                                      │
│                                                               │
│  z_global[ego]  ─── (32) ──┐                                 │
│                             ├── concat ── (96) ──┐            │
│  z_local        ─── (64) ──┘                     │            │
│                                                   ▼            │
│                                         z_combine_mlp          │
│                                         MLP: 96 → 64 → 64    │
│                                                   │            │
│                                                (64)            │
│  map_feat[ego]  ─── (64) ──┐                     │            │
│  sem[ego]       ─── (2)  ──┤── concat ──── (132) │            │
│  lw[ego]        ─── (2)  ──┘                     │            │
│                                                   ▼            │
│                                         ego_decoder_gru        │
│                                         GRU(132→64, 3-layer)  │
│                                                   │            │
│                                                (64)            │
│                                                   ▼            │
│                                         ego_output_mlp         │
│                                         MLP: 64 → 64 → 2     │
│                                                   │            │
│                                             (a, hdot)          │
│                                                   ▼            │
│                                         bicycle_model          │
│                                                   │            │
│                                         next_ego_state (6)     │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Sur Decoder 컴포넌트

```
┌─────────────────────────────────────────────────────────────┐
│  Sur Decoder 데이터 흐름 (STRIVE style)                       │
│                                                               │
│  z_global       ─── (32) ──┐                                 │
│  past_feat      ─── (64) ──┤                                 │
│  map_feat       ─── (64) ──┤── concat ── (164)               │
│  sem            ─── (2)  ──┤                                 │
│  lw             ─── (2)  ──┘                                 │
│                               ▼                               │
│                      decoder_net (IndivGCN)                   │
│                      164 → 64 → 2  (a, hdot)                 │
│                               ▼                               │
│                      bicycle_model                            │
│                               │                               │
│                      next_sur_state (6)                        │
│                                                               │
│  State memory:                                                │
│  decoder_memory: GRU(4→64, 3-layer)                          │
│  → past_feat 업데이트 (매 스텝)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 윈도우 동작 비교

### 4.1 Fixed Window (`growing_window=False`, `z_local_window=6`)

```
AR Decoder:
t=0:  [p0, p1, p2, p3]                          len=4
t=1:  [p0, p1, p2, p3, pred0]                   len=5
t=2:  [p0, p1, p2, p3, pred0, pred1]            len=6
t=3:  [p1, p2, p3, pred0, pred1, pred2]         len=6 (pop+append)
...
t=11: [pred5, pred6, pred7, pred8, pred9, pred10] len=6

TF Decoder (segment_len=3):
target_t=0: [p0, p1, p2, p3]                    len=4 (초기)
  step 0,1,2 → append, window grows to 6
target_t=3: 리셋 → [p3, f0, f1, f2]             len=4 (가용<6)
             → step 내부에서 5, 6으로 성장
target_t=6: 리셋 → [f0, f1, f2, f3, f4, f5]     len=6 (정상)
target_t=9: 리셋 → [f3, f4, f5, f6, f7, f8]     len=6 (정상)
```

### 4.2 Growing Window (`growing_window=True`)

```
AR Decoder:
t=0:  [p0, p1, p2, p3]                          len=4
t=1:  [p0, p1, p2, p3, pred0]                   len=5
t=5:  [p0, p1, p2, p3, pred0..4]                len=9
t=11: [p0, p1, p2, p3, pred0..10]               len=15

TF Decoder (segment_len=3):
target_t=0: [p0, p1, p2, p3]                    len=4
target_t=3: 리셋 → [p0,p1,p2,p3,f0,f1,f2]      len=7
target_t=6: 리셋 → [p0..p3,f0..f5]              len=10
target_t=9: 리셋 → [p0..p3,f0..f8]              len=13
```

---

## 5. 모듈 전체 요약

| 모듈 | 타입 | 차원 | 용도 | Fine-tune |
|------|------|------|------|-----------|
| `step_feature_extractor` | MLP | 11→128→64 | raw state → GCN input | Frozen |
| `temporal_gcn_encoder` | IndivGCN | 64→128→64 | per-step agent interaction | Frozen |
| `prior_temporal_gru` | GRU | 64→64, 2-layer | past temporal encoding | Frozen |
| `positional_encoding` | SinPE | 64, max=12 | future PE | Frozen |
| `transformer_encoder` | TransEnc | 64, 8-head, 3-layer | future temporal encoding | Frozen |
| `latent_prior_net` | MLP | 130→128→64 | z_global prior | Frozen |
| `latent_posterior_net` | MLP | 194→128→64 | z_global posterior | Frozen |
| `map_conv` | CNN | 4→128, 6-layer | map encoding | Frozen |
| `map_feature` | Linear | ?→64 | map feat projection | Frozen |
| `decoder_net` | IndivGCN | 164→64→2 | sur trajectory | Frozen |
| `decoder_memory` | GRU | 4→64, 3-layer | sur state memory | Frozen |
| `z_local_gcn` | IndivGCN | 10→64→64 | per-step z_local GCN | **Trainable** |
| `z_local_temporal_gru` | GRU | 64→64, 1-layer | other temporal in window | **Trainable** |
| `z_local_ego_temporal_gru` | GRU | 64→64, 1-layer | ego temporal (full_query) | **Trainable** |
| `cross_attn_1` + `norm_1` | MHA+LN | 64, 4-head | cross-attn layer 1 | **Trainable** |
| `cross_attn_2` + `norm_2` | MHA+LN | 64, 4-head | cross-attn layer 2 | **Trainable** |
| `interaction_self_attn` + `norm` | MHA+LN | 64, 4-head | ego↔sur self-attn | **Trainable** |
| `ego_warmup_gru` | GRU | 64→64, 3-layer | ego hidden init | **Trainable** |
| `z_combine_mlp` | MLP | 96→64→64 | z_global+z_local merge | **Trainable** |
| `ego_decoder_gru` | GRU | 132→64, 3-layer | ego trajectory decode | **Trainable** |
| `ego_output_mlp` | MLP | 64→64→2 | ego output (a, hdot) | **Trainable** |

---

## 6. 변경 이력 (2026-02-19)

| # | 변경 | 내용 |
|---|------|------|
| 1 | Multi-layer Cross-Attention | 1-layer → 2-layer + residual + LayerNorm |
| 2 | VAE 제거 | z_local: VAE sampling → deterministic projection |
| 3 | GPU 최적화 | `.item()` → `int()`, `ptr` 일괄 CPU 전환 |
| 4 | Ablation 플래그 | `use_z_local=False`로 z_local=0 실험 가능 |
| 5 | Getter 통합 | `_z_local_mean/var` → `_z_local_outputs` |
| 6 | 압축 제거 | z_local: 64→32 projection 제거 → 64dim 직접 전달 |
| 7 | Self-Attention | ego↔sur mutual interaction layer 추가 |
| 8 | Prior GRU 강화 | 1-layer → 2-layer GRU |
| 9 | Posterior 확인 | 이미 8-head, 3-layer (UniAD 동일) — 수정 불필요 |
| 10 | 윈도우 확대 | `growing_window` 플래그로 고정/누적 선택 가능 |
