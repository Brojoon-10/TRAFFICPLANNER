# TrafficPlanner Model Redesign Plan (v6)

> 작성일: 2026-02-22
> 최종 수정: 2026-02-23
> 상태: Phase 1 구현 완료 (test_forward_pass 11개 전체 통과)

---

## 0. 왜 다시 설계해야 하는가

### 근본 문제
현재 모델은 trajectory 좌표를 MSE로 외울 뿐, **"왜 이렇게 움직이는가"의 인과관계를 학습하지 못한다.**

원하는 것:
- z_global: "어디로 갈 것인가" (직진, 좌회전, 우회전) — **글로벌 의도**
- z_local: "어떻게 반응할 것인가" (등속, 감속, 가속, 회피) — **순간 의도**
- z_local은 **수동 카테고리 아님** — 데이터에서 자율적으로 의도 클러스터 발견
- 맵을 "배경 정보"가 아닌 **"선택적으로 참조하는 도로 구조"**로 취급

### 현재 모델의 구체적 실패 원인

**1. Ego decoder가 맥락을 못 봄**
- Old model: 매 스텝 cross-attention으로 past encoder 참조 → 258+NC dim 입력
- New model: ego_gru_out 64dim만으로 예측 → 정보 병목
- **past_seq_out은 초기 4스텝만 포함** → 디코딩 중 지나온 경로 참조 불가

**2. z_local이 Phase 1에서 noise**
- Phase 1에서 z_local은 학습 신호 없이 random 값
- z_combine_mlp(z_global + z_local_noise) → z_global 정보 희석
- Old model 대비 val loss 0.08 악화의 직접 원인

**3. Train/Val gap (TF mismatch)**
- Train: TF로 GT reset, Val: full autoregressive → 0.10 gap
- TF 꺼진 epoch 1600+ 이후에도 수렴 안 됨 (이미 TF에 overfit)

**4. Map을 무시함**
- map_feat 64dim이 concat으로 들어가지만, MSE loss는 좌표만 맞추면 됨
- 모델이 map_feat을 무시해도 loss가 줄어듦 → 도로 구조 학습 안 됨

**5. 구조적 병목**
- 매 스텝: z_local용 GCN + 2-layer cross-attn + self-attn + temporal GRU
- Loss에서 .item()/.cpu() 수십 회
- list append → torch.stack 패턴 만연
- 학습 3일 소요 (3000 epochs)

---

## 1. 설계 원칙

### 1a. 유지 (변경 불가)
- CVAE 구조: prior/posterior net → z_global → KL loss
- GCN: SceneInteractionNet (상호작용 인코딩용으로 유지)
- Map encoding: CNN 기반 (encoder용 64dim 유지)
- Bicycle model output: (acceleration, yaw_rate)
- 12-step prediction at 0.5s interval
- Visualization 코드
- Autoregressive decoding (실시간 ego-sur 상호작용 필수)
- Ego/Sur decoder 모듈 완전 분리 (Phase 2에서 ego 학습이 sur에 영향 zero)

### 1b. 핵심 설계 방향
1. **Ego/Sur 대칭 decoder 구조**: 둘 다 History Attention + Map Cross-Attention + GRU
2. **GCN은 상호작용 인코딩 전용**: decoder prediction이 아닌 history buffer 채우기용
3. **History Attention KV = GCN feature 누적**: 풍부한 상호작용 정보 + PE로 시간 순서
4. **Map Cross-Attention**: CNN spatial features (29×29=841 tokens @256pix, 27×27=729 @240pix), flatten 안 함
5. **Intent Codebook (z_local)**: discrete K개 의도, z_global과 독립, ego만 적용
6. **z_local Phase 1 zero-masking**: z_global 학습 방해 방지
7. **TF annealing 유지**: 기존 segment 방식 유지 (Scheduled Sampling은 불연속 문제)
8. **Loop body 경량화**: pre-allocate buffer, in-place update, Python overhead 최소화

---

## 2. 새 아키텍처

### 2a. 계층 구조 개요

```
Level 0: Scene Understanding (1회, encoder)
  Past 4 steps → GCN + Transformer → past_seq_out (NA, PT, D)
  Map → CNN → 64dim (encoder용, 기존대로)
  Past+Future → Posterior → z_global
  Past only → Prior → z_global

Level 1: z_global (scene-level, 1회)
  CVAE prior/posterior → "이 씬에서 어디로 가는가"
  12스텝 전체에 동일하게 적용

Level 2: z_local Intent Codebook (step-level, 매 스텝, ego만)
  K개 learnable intent 벡터 (K=8, 조절 가능)
  situation → intent_predictor → K개 중 1개 선택
  z_global과 독립 — decoder에서 합류

Level 3: Action Generation (step-level, 매 스텝)
  GCN → history buffer (상호작용 인코딩)
  History Attention (GCN feature 누적 + PE)
  Map Cross-Attention (29×29=841 spatial tokens @256pix, 2.66m 해상도)
  GRU + output_head (Linear) → (acc, yaw_rate) → bicycle model
```

### 2b. Decoder 전체 흐름

```
[Decoder 초기화]
ego_history_buffer = zeros(FT, gcn_feat_dim)   # pre-allocate
sur_history_buffer = zeros(FT, gcn_feat_dim)   # pre-allocate
map_tokens = map_CNN_partial(map_crop)          # (NA, 841, ch) — conv3에서 분기 (29×29 @256pix)

[매 스텝 t = 0..11]

1. 상호작용 인코딩 (GCN):
   scene_graph.x = [state, lw, sem]  ← map 안 넣음, 상호작용 전용
   ego_gcn_feat, sur_gcn_feat = GCN(scene_graph)
   ego_history_buffer[t] = ego_gcn_feat   # in-place 저장
   sur_history_buffer[t] = sur_gcn_feat   # in-place 저장

2. History Attention:
   ego_hist_ctx = Ego HistAttn(Q=ego_gru_hidden, KV=ego_history_buffer[:t])
   predicted_sur_delta = Linear(ego_hist_ctx)  → (B) Sur Pred Loss
   sur_hist_ctx = Sur HistAttn(Q=sur_gru_hidden, KV=sur_history_buffer[:t])
   predicted_ego_delta = Linear(sur_hist_ctx)  → (C) Ego Pred Loss

3. Map Attention:
   ego_map_ctx, ego_attn_w = Ego MapAttn(Q=ego_gru_hidden, KV=map_tokens)
   sur_map_ctx, sur_attn_w = Sur MapAttn(Q=sur_gru_hidden, KV=map_tokens)
   → (D) Map Attn Guidance Loss

4. Intent Selection (ego만):
   intent_input = [ego_state, predicted_sur_delta, ego_map_ctx.detach()]
   z_local, weights = IntentCodebook(intent_input)
   → (A) Intent CE Loss
   Phase 1: z_local = zero (구조만 유지, 출력 zero)

5. GRU + Output:
   ego_gru_input = [ego_hist_ctx, ego_map_ctx, z_global, z_local, lw, sem]
   Ego GRU → output_head → (acc, yaw_rate) → bicycle model
   → Recon MSE Loss

   sur_gru_input = [sur_hist_ctx, sur_map_ctx, z_global, lw, sem]
   Sur GRU → output_head → (acc, yaw_rate) → bicycle model

6. Scene Graph Update:
   scene_graph.pos = new_states  # in-place, 다음 스텝 GCN용
```

### 2c. 모듈 분리 (Phase 2 독립성 보장)

```
완전 분리 (ego/sur 별도 weight):
  - Ego History Attention / Sur History Attention
  - Ego Map Cross-Attention / Sur Map Cross-Attention
  - Ego GRU / Sur GRU
  - Ego output_head (Linear) / Sur output_head (Linear)
  - Intent Codebook (ego만)

공유:
  - 상호작용 GCN (history buffer 채우기용)
  - Encoder (past encoder, prior/posterior)
  - Map CNN

Phase 2 학습 시:
  Freeze: encoder 전체, z_global, sur 모듈 전체, 상호작용 GCN, map CNN
  Train:  ego history attn, ego map attn, ego GRU, ego output_head (Linear), intent codebook
  → sur에 영향 zero
```

### 2d. GCN 역할 변경

**기존:**
```
Sur decoder용 GCN: scene_graph → sur prediction (prediction 담당)
z_local용 GCN: scene_graph → ego/sur feature → z_local attention (feature 추출)
→ GCN이 2종류, prediction + feature 추출 둘 다 담당
```

**제안:**
```
상호작용 GCN 1개: scene_graph → ego_gcn_feat, sur_gcn_feat
→ prediction은 안 함, history buffer 채우기 전용
→ 입력: [state, lw, sem] (map 미포함, 상호작용 인코딩 전용)
→ ego/sur decoder가 이 feature를 history attention으로 참조해서 각자 prediction
```

**장점:**
- GCN이 순수 상호작용 인코딩에만 집중
- map 정보는 별도 Map Cross-Attention에서 처리 (역할 분리)
- 기존 z_local GCN의 무거운 attention 스택 제거 → 가벼움
- 매 스텝 GCN 1회만 (기존: sur GCN + z_local GCN = 2회)

### 2e. Intent Codebook 상세 (z_local)

```python
class IntentCodebook(nn.Module):
    def __init__(self, num_intents=8, intent_dim=32, input_dim=D):
        self.codebook = nn.Embedding(num_intents, intent_dim)  # K개 의도 벡터
        self.intent_predictor = MLP([input_dim, 64, num_intents])  # 상황→의도 확률

    def forward(self, situation, temperature=1.0):
        # 상황만 보고 의도 선택 (z_global과 독립)
        logits = self.intent_predictor(situation)  # (num_ego, K)

        if self.training:
            intent_weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            intent_weights = F.one_hot(logits.argmax(-1), K).float()

        z_local = intent_weights @ self.codebook.weight  # (num_ego, intent_dim)
        return z_local, intent_weights
```

**특성:**
- K개 의도가 뭔지는 모델이 학습 중 자율 결정 (수동 라벨 없음)
- z_global과 독립: decoder에서 z_global + z_local concat으로 합류
- Phase 1: z_local = zero (codebook 비활성화)
- Phase 2: codebook 활성화, temperature annealing (높음→낮음)
- 학습 후: codebook 각 slot 시각화로 의도 해석 가능

### 2f. Decoder History Attention 상세

```python
class DecoderHistoryAttention(nn.Module):
    def __init__(self, d_model, gcn_feat_dim, nhead=4, max_len=12):
        self.feat_proj = nn.Linear(gcn_feat_dim, d_model)  # GCN feat → d_model
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.pe = nn.Embedding(max_len, d_model)  # learnable positional encoding

    def forward(self, query, history_buffer, t):
        """
        query: (N, D) — 현재 GRU hidden
        history_buffer: (N, FT, gcn_feat_dim) — pre-allocated, GCN feature 누적
        t: current timestep
        """
        if t == 0:
            return torch.zeros_like(query)  # 첫 스텝: 히스토리 없음

        # GCN feature → d_model 변환 + positional encoding
        history = self.feat_proj(history_buffer[:, :t])  # (N, t, D)
        positions = self.pe(torch.arange(t, device=query.device))  # (t, D)
        history = history + positions  # 시간 순서 부여

        query = query.unsqueeze(1)  # (N, 1, D)
        attn_out, _ = self.cross_attn(query, history, history)
        return attn_out.squeeze(1)  # (N, D)
```

**history_buffer 내용:**
- GCN feature (상호작용 인코딩 결과, gcn_feat_dim)
- ego_history_buffer: ego 관점 GCN feature 누적
- sur_history_buffer: sur 관점 GCN feature 누적
- Pre-allocate: `zeros(N, 12, gcn_feat_dim)` → list append 없음

**Positional Encoding 역할 (causal confusion 방지):**
- t=3과 t=4의 구분 가능 → "sur가 t=4에서 급접근" 시간적 인과 학습
- 없으면 "sur 접근"과 "ego 감속"의 시간 순서 혼동 가능

### 2g. Map Cross-Attention 상세

```python
class MapCrossAttention(nn.Module):
    def __init__(self, d_model, map_feat_dim, nhead=4):
        self.map_proj = nn.Linear(map_feat_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, query, map_tokens):
        """
        query: (N, D) — agent의 현재 GRU hidden
        map_tokens: (N, num_tokens, map_feat_dim) — conv3 spatial features (29×29=841 @256pix)
        """
        map_proj = self.map_proj(map_tokens)  # (N, num_tokens, D)
        query = query.unsqueeze(1)  # (N, 1, D)
        attn_out, attn_weights = self.cross_attn(query, map_proj, map_proj)
        return attn_out.squeeze(1), attn_weights  # (N, D), (N, 1, num_tokens)
```

**Map token 생성 (CNN conv3에서 분기):**
```python
# Encoder용 (z_global prior/posterior): 기존대로 끝까지 → 64dim
map_feat_encoder = full_map_cnn(map_obs)  # 64dim

# Decoder용 (cross-attention): conv3에서 멈춤
x = map_conv_1(map_obs)   # conv1: 256→125
x = map_conv_2(x)          # conv2: 125→61
x = map_conv_3(x)          # conv3: 61→30
map_tokens = x.flatten(2).permute(0, 2, 1)  # (NA, 841, ch)
```

**해상도:**
- 29×29 = 841 tokens (@256pix), 77m/29 ≈ 2.66m per token
- 도로 폭 3.5m 내에 token ~1.3개 → 도로/경계 분리 가능
- 필요시 conv2 분기 (61×61=3,721 tokens, 1.26m) 가능

**적용: ego + sur 모두** (각자 별도 weight)

### 2h. GRU 입력 정리

**Ego:**
```
ego_gru_input = concat([
    history_context,   # (D) — Ego History Attention 결과 (GCN feature 기반)
    map_context,       # (D) — Ego Map Cross-Attention 결과
    z_global,          # (z_size) — scene-level 의도
    z_local_t,         # (intent_dim) — 순간 의도 (Phase 1: zero)
    lw,                # (2) — 차량 크기
    sem                # (NC) — semantic 정보
])
```

**Sur:**
```
sur_gru_input = concat([
    history_context,   # (D) — Sur History Attention 결과 (GCN feature 기반)
    map_context,       # (D) — Sur Map Cross-Attention 결과
    z_global,          # (z_size) — scene-level 의도
    lw,                # (2) — 차량 크기
    sem                # (NC) — semantic 정보
])
```

**차이: z_local 유무만.** Phase 1에서는 z_local=zero이므로 사실상 동일 구조.

**기존 대비:**
| 항목 | 기존 ego | 기존 sur | 제안 ego | 제안 sur |
|------|---------|---------|---------|---------|
| 과거 참조 | 없음 | 없음 | History Attn (GCN feat+PE) | History Attn (GCN feat+PE) |
| 맵 참조 | 64dim concat | GCN 입력에 포함 | 841 token cross-attn | 841 token cross-attn |
| z_global | z_combine_mlp(+z_local) | GCN 입력에 포함 | concat (분리) | concat (분리) |
| z_local | noise (Phase 1) | 없음 | codebook (Phase 1: zero) | 없음 |
| decoder | GRU | GCN | GRU | GRU |

---

## 2i. 상세 아키텍처 다이어그램

### 하이퍼파라미터 요약

```
┌─────────────────────────────────────────────────────┐
│  하이퍼파라미터 (Redesign)                              │
├─────────────────────────────────────────────────────┤
│  PT (past_len)              = 4                     │
│  FT (future_len)            = 12                    │
│  NC (nclasses)              = 2 (car, truck)        │
│  state_size                 = 6 (x,y,hx,hy,s,hdot) │
│  att_feat_size              = 2 (l, w)              │
│  D (d_model)                = 64                    │
│  gcn_feat_dim               = 64                    │
│  map_feat_size              = 64 (encoder용)        │
│  map_token_ch               = 64 (conv3 output ch)  │
│  map_grid                   = 29×29 = 841 tokens (@256pix) │
│  map_resolution             ≈ 2.66m per token       │
│  z_size (z_global)          = 32                    │
│  intent_dim (z_local)       = 32                    │
│  num_intents (K)            = 8                     │
│  hist_attn_nhead            = 4                     │
│  hist_attn_head_dim         = 16                    │
│  map_attn_nhead             = 4                     │
│  map_attn_head_dim          = 16                    │
│  transformer_nhead          = 8  (encoder용)        │
│  transformer_nlayer         = 3  (encoder용)        │
│  num_gru_layers (ego/sur)   = 3                     │
│  output_bicycle             = True → traj_out=2     │
│  dt                         = 0.5                   │
│  tf_max_annealing_epoch     = 1500                  │
│  tf_init_segment_len        = 3                     │
│  gumbel_temperature_init    = 1.0                   │
│  gumbel_temperature_min     = 0.1                   │
│  sur_pred_dim               = 2 (dx, dy)            │
│  intent_ce_nclass           = 9 (3acc × 3yaw)      │
│  map_gt_steps               = 6 (전방 예측 토큰)    │
│  map_gt_decay_lambda        = 0.3 (exponential decay)│
└─────────────────────────────────────────────────────┘
```

### 전체 모델 데이터 흐름

```
입력: scene_graph (past, future, edge_index, lw, sem, batch), map_idx, map_env

═══════════════════════════════════════════════════════════════════
  ENCODER (1회, 기존과 동일)
═══════════════════════════════════════════════════════════════════

Step 1: Map Feature (encoder용 — 기존 동일)
├── map_conv (6-layer CNN: 4→16→32→64→64→128→128)
├── flatten → map_feature (Linear: 128*H'*W' → 64)
└── map_feat_enc: (NA, 64)

Step 1b: Map Tokens (decoder용 — 신규)
├── map_conv_1 (conv1: 4→16, k=7, s=2)    256→125
├── map_conv_2 (conv2: 16→32, k=5, s=2)   125→61
├── map_conv_3 (conv3: 32→64, k=5, s=2)   61→29
├── flatten(2) + permute(0,2,1)
└── map_tokens: (NA, 841, 64)             ← 29×29 spatial tokens (@256pix)

Step 2: z_global Prior (past only — 기존 동일)
├── Per-step: state→step_feat_extractor(MLP: 11→128→64)
├── Per-step: temporal_gcn_encoder(IndivGCN: 64→128→64)
├── Stack: (NA, 4, 64)
├── prior_temporal_gru(GRU: 64→64, 2-layer) → context (NA, 64)
├── prior_in = concat(context, map_feat_enc, sem) → (NA, 130)
├── latent_prior_net(MLP: 130→128→64) → z_prior(μ:32, σ²:32)
└── past_seq_out: (NA, 4, 64)

Step 3: z_global Posterior (past+future — 기존 동일, 학습 시만)
├── Per-step GCN: 동일
├── positional_encoding(Sinusoidal PE, max_len=12)
├── transformer_encoder(3-layer, 8-head, d_model=64)
├── posterior_in = concat(past_ctx, future_ctx, map_feat_enc, sem) → (NA, 194)
└── latent_posterior_net(MLP: 194→128→64) → z_post(μ:32, σ²:32)

Step 4: z_global 샘플링
└── z_global = rsample(z_post_mean, z_post_var) → (NA, 32)

═══════════════════════════════════════════════════════════════════
  DECODER (autoregressive, 12 steps — 재설계)
═══════════════════════════════════════════════════════════════════

Step 5: Decoder (아래 상세 다이어그램 참조)
```

### Decoder 초기화 상세

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Decoder Initialization                             │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  1. History Buffer Pre-allocation                              ║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  ego_history_buffer = zeros(N_ego, FT, D)    # (N_ego, 12, 64)║   │
│  ║  sur_history_buffer = zeros(N_sur, FT, D)    # (N_sur, 12, 64)║   │
│  ║                                                                 ║   │
│  ║  → list append / torch.stack 완전 제거                          ║   │
│  ║  → 매 스텝 in-place: buffer[:, t, :] = gcn_feat               ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  2. Map Token 준비 (conv3 분기)                                ║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  map_crop: (NA, 4, 256, 256)    ← 4ch (drivable/solid/dash/..)║   │
│  ║       ↓ conv1 (4→16, k=7, s=2, pad=3)                         ║   │
│  ║  (NA, 16, 125, 125)                                             ║   │
│  ║       ↓ conv2 (16→32, k=5, s=2, pad=2)                        ║   │
│  ║  (NA, 32, 61, 61)                                               ║   │
│  ║       ↓ conv3 (32→64, k=5, s=2, pad=2)                        ║   │
│  ║  (NA, 64, 29, 29)                                               ║   │
│  ║       ↓ flatten(2) + permute(0,2,1)                            ║   │
│  ║  map_tokens: (NA, 841, 64)      ← 29×29 spatial tokens        ║   │
│  ║                                                                 ║   │
│  ║  ※ encoder용은 기존대로 conv4~6까지 → flatten → Linear → 64   ║   │
│  ║  ※ decoder용만 conv3에서 분기 (spatial 유지)                    ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  3. GRU Hidden 초기화                                          ║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  ego_gru_hidden = warmup_gru(past_seq_out[ego])  # 3-layer GRU ║   │
│  ║  sur_gru_hidden = warmup_gru(past_seq_out[sur])  # 3-layer GRU║   │
│  ║                                                                 ║   │
│  ║  → Warmup GRU: past 4스텝 → hidden 초기화 (구현 완료)         ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                       │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║  4. 초기 상태                                                   ║   │
│  ╠═══════════════════════════════════════════════════════════════╣   │
│  ║                                                                 ║   │
│  ║  prev_state = scene_graph.past[:, -1, :]        (NA, 6)       ║   │
│  ║  prev_ego_state = prev_state[ego_idx]           (N_ego, 6)    ║   │
│  ║  prev_sur_state = prev_state[sur_idx]           (N_sur, 6)    ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
└─────────────────────────────────────────────────────────────────────┘
```

### Decoder Loop 상세 (매 스텝 t = 0..11)

```
┌─────────────────────────────────────────────────────────────────────┐
│                Decoder Loop: for t = 0 to 11                          │
╠═════════════════════════════════════════════════════════════════════╣
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  STEP 1: 상호작용 인코딩 (GCN) → History Buffer                  │ │
│  │                                                                   │ │
│  │  scene_graph.x = concat(state, lw, sem)          (NA, 10)       │ │
│  │       ↓                                                           │ │
│  │  interaction_gcn (IndivGCN: 10→64→D)                             │ │
│  │       ↓                                                           │ │
│  │  ego_gcn_feat: (N_ego, D=64)    sur_gcn_feat: (N_sur, D=64)    │ │
│  │       ↓                              ↓                            │ │
│  │  ego_history_buffer[:, t, :] = ego_gcn_feat     ← in-place      │ │
│  │  sur_history_buffer[:, t, :] = sur_gcn_feat     ← in-place      │ │
│  │                                                                   │ │
│  │  ※ map 미포함 — 순수 agent 상호작용 인코딩 전용                   │ │
│  │  ※ GCN은 ego/sur 공유 (Phase 2에서 frozen)                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  STEP 2: History Attention (ego/sur 병렬)                        │ │
│  │                                                                   │ │
│  │  ┌───────────────────────── Ego ──────────────────────────────┐ │ │
│  │  │                                                             │ │ │
│  │  │  Q = ego_gru_hidden[-1]                   (N_ego, D)      │ │ │
│  │  │       ↓ unsqueeze(1)                                       │ │ │
│  │  │  Q: (N_ego, 1, D)                                          │ │ │
│  │  │                                                             │ │ │
│  │  │  KV = ego_history_buffer[:, :t, :]        (N_ego, t, D)   │ │ │
│  │  │       ↓ feat_proj (Linear: D→D)                            │ │ │
│  │  │       ↓ + PE (Embedding: t positions → D)                  │ │ │
│  │  │  KV: (N_ego, t, D)                                         │ │ │
│  │  │                                                             │ │ │
│  │  │  if t == 0: ego_hist_ctx = zeros(N_ego, D)                 │ │ │
│  │  │  else:                                                      │ │ │
│  │  │    cross_attn(Q, KV, KV)    [4 heads, head_dim=16]        │ │ │
│  │  │       ↓ squeeze(1)                                          │ │ │
│  │  │  ego_hist_ctx: (N_ego, D=64)                               │ │ │
│  │  │                                                             │ │ │
│  │  │  ┌─ Auxiliary (B): Sur Prediction ─────────────────────┐  │ │ │
│  │  │  │  ego_hist_ctx → sur_pred_head (Linear: D→2)           │  │ │ │
│  │  │  │  predicted_sur_delta: (N_ego, 2)                     │  │ │ │
│  │  │  │  Loss = MSE(predicted_sur_delta, gt_sur_delta)       │  │ │ │
│  │  │  │                                                       │  │ │ │
│  │  │  │  ※ Phase 2에서 predicted_sur_delta는                 │  │ │ │
│  │  │  │    intent_input으로 재활용                              │  │ │ │
│  │  │  └──────────────────────────────────────────────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  ┌───────────────────────── Sur ──────────────────────────────┐ │ │
│  │  │                                                             │ │ │
│  │  │  Q = sur_gru_hidden[-1]                   (N_sur, D)      │ │ │
│  │  │  KV = sur_history_buffer[:, :t, :]        (N_sur, t, D)   │ │ │
│  │  │  (ego와 동일 구조, 별도 weight)                             │ │ │
│  │  │  sur_hist_ctx: (N_sur, D=64)                               │ │ │
│  │  │                                                             │ │ │
│  │  │  ┌─ Auxiliary (C): Ego Prediction (Phase 1만) ──────────┐ │ │ │
│  │  │  │  sur_hist_ctx → ego_pred_head (Linear: D→2)            │ │ │ │
│  │  │  │  predicted_ego_delta: (N_sur, 2)                      │ │ │ │
│  │  │  │  Loss = MSE(predicted_ego_delta, gt_ego_delta)        │ │ │ │
│  │  │  └───────────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  STEP 3: Map Cross-Attention (ego/sur 병렬)                      │ │
│  │                                                                   │ │
│  │  ┌───────────────────────── Ego ──────────────────────────────┐ │ │
│  │  │                                                             │ │ │
│  │  │  Q = ego_gru_hidden[-1]                   (N_ego, 1, D)   │ │ │
│  │  │  KV = map_tokens                           (N_ego, 841, D)│ │ │
│  │  │       ↓ map_proj (Linear: map_ch→D)                        │ │ │
│  │  │  KV: (N_ego, 841, D)                                       │ │ │
│  │  │                                                             │ │ │
│  │  │  cross_attn(Q, KV, KV)    [4 heads, head_dim=16]          │ │ │
│  │  │       ↓                                                     │ │ │
│  │  │  ego_map_ctx: (N_ego, D=64)                                │ │ │
│  │  │  ego_attn_w:  (N_ego, 1, 841)  ← attention weight 캡처   │ │ │
│  │  │                                                             │ │ │
│  │  │  ┌─ Auxiliary (D): Map Attn Guidance ──────────────────┐  │ │ │
│  │  │  │  GT: 전방 6스텝 GT 토큰 위치 → soft label (841,)     │  │ │ │
│  │  │  │  가중치: exp(-λt)/Σexp(-λt), λ=0.3                   │  │ │ │
│  │  │  │  Loss = KL(ego_attn_w.squeeze() || soft_label)       │  │ │ │
│  │  │  │  ※ t=11이면 제외 (전방 없음)                           │  │ │ │
│  │  │  └──────────────────────────────────────────────────────┘  │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  ┌───────────────────────── Sur ──────────────────────────────┐ │ │
│  │  │  (ego와 동일 구조, 별도 weight)                             │ │ │
│  │  │  sur_map_ctx: (N_sur, D=64)                                │ │ │
│  │  │  sur_attn_w:  (N_sur, 1, 841)                              │ │ │
│  │  │  → (D) Map Attn Guidance Loss (Phase 1만)                 │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  STEP 4: Intent Selection (ego만, Phase 2만 활성)                │ │
│  │                                                                   │ │
│  │  intent_input = concat([                                         │ │
│  │      ego_state_t,              # (N_ego, 6)                     │ │
│  │      predicted_sur_delta,      # (N_ego, 2)   ← Step 2에서     │ │
│  │      ego_map_ctx.detach()      # (N_ego, D)   ← Step 3에서     │ │
│  │  ])                             # (N_ego, 6+2+D = 72)           │ │
│  │       ↓                                                           │ │
│  │  intent_predictor (MLP: 72→64→K=8)                               │ │
│  │       ↓                                                           │ │
│  │  logits: (N_ego, 8)                                               │ │
│  │       ↓                                                           │ │
│  │  Train: Gumbel-Softmax(logits, τ)  → weights: (N_ego, 8)       │ │
│  │  Eval:  one_hot(argmax(logits))    → weights: (N_ego, 8)       │ │
│  │       ↓                                                           │ │
│  │  z_local = weights @ codebook.weight  → (N_ego, intent_dim=32) │ │
│  │                                                                   │ │
│  │  ┌─ Auxiliary (A): Intent CE (Phase 2만) ─────────────────────┐ │ │
│  │  │  z_local → intent_ce_head (Linear: 32→9)                   │ │ │
│  │  │  Loss = CE(prediction, gt_acc_yaw_class)                    │ │ │
│  │  │  gt_class = acc_bin(3) × yaw_bin(3) = 9 classes            │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  Phase 1: z_local = zeros(N_ego, intent_dim)  ← 비활성화       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  STEP 5: GRU + Output (ego/sur 병렬)                             │ │
│  │                                                                   │ │
│  │  ┌───── Ego Decoder GRU ──────────────────────────────────────┐ │ │
│  │  │                                                             │ │ │
│  │  │  ego_hist_ctx  ─── (D=64) ──┐                             │ │ │
│  │  │  ego_map_ctx   ─── (D=64) ──┤                             │ │ │
│  │  │  z_global[ego] ─── (32)   ──┤── concat ── (D+D+32+32+2+2)│ │ │
│  │  │  z_local       ─── (32)   ──┤              = 196           │ │ │
│  │  │  lw[ego]       ─── (2)    ──┤                             │ │ │
│  │  │  sem[ego]      ─── (2)    ──┘                             │ │ │
│  │  │                                   ↓                        │ │ │
│  │  │                        ego_decoder_gru                     │ │ │
│  │  │                        GRU(196→D, 3-layer)                │ │ │
│  │  │                                   │                        │ │ │
│  │  │                                (D=64)                      │ │ │
│  │  │                                   ↓                        │ │ │
│  │  │                        ego_output_head                     │ │ │
│  │  │                        Linear: D → 2                      │ │ │
│  │  │                                   │                        │ │ │
│  │  │                            (acc, yaw_rate)                 │ │ │
│  │  │                                   ↓                        │ │ │
│  │  │                        bicycle_model(prev_ego, a, hdot)    │ │ │
│  │  │                                   │                        │ │ │
│  │  │                        next_ego_state: (N_ego, 6)          │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  ┌───── Sur Decoder GRU ──────────────────────────────────────┐ │ │
│  │  │                                                             │ │ │
│  │  │  sur_hist_ctx  ─── (D=64) ──┐                             │ │ │
│  │  │  sur_map_ctx   ─── (D=64) ──┤                             │ │ │
│  │  │  z_global[sur] ─── (32)   ──┤── concat ── (D+D+32+2+2)   │ │ │
│  │  │  lw[sur]       ─── (2)    ──┤              = 164           │ │ │
│  │  │  sem[sur]      ─── (2)    ──┘                             │ │ │
│  │  │                                   ↓                        │ │ │
│  │  │                        sur_decoder_gru                     │ │ │
│  │  │                        GRU(164→D, 3-layer)                │ │ │
│  │  │                                   │                        │ │ │
│  │  │                                (D=64)                      │ │ │
│  │  │                                   ↓                        │ │ │
│  │  │                        sur_output_head                     │ │ │
│  │  │                        Linear: D → 2                      │ │ │
│  │  │                                   │                        │ │ │
│  │  │                            (acc, yaw_rate)                 │ │ │
│  │  │                                   ↓                        │ │ │
│  │  │                        bicycle_model(prev_sur, a, hdot)    │ │ │
│  │  │                                   │                        │ │ │
│  │  │                        next_sur_state: (N_sur, 6)          │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                                                                   │ │
│  │  ※ ego GRU input = 196dim (z_local 포함)                        │ │
│  │  ※ sur GRU input = 164dim (z_local 없음)                        │ │
│  │  ※ Phase 1: z_local=zero → ego도 사실상 164+32(zero) = 196     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       ↓                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  STEP 6: Scene Graph Update                                      │ │
│  │                                                                   │ │
│  │  scene_graph.pos = merge(next_ego_state, next_sur_state)        │ │
│  │  prev_ego_state = next_ego_state                                 │ │
│  │  prev_sur_state = next_sur_state                                 │ │
│  │                                                                   │ │
│  │  ※ map_recrop 설정으로 제어 (config: map_recrop: False)          │ │
│  │    False: 초기 crop 재사용 (빠름, 77m 범위 내 ~50m 이동 OK)     │ │
│  │    True: 매 스텝 re-crop + re-encode (정확, TF에서 GT 캐시)      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
╚═════════════════════════════════════════════════════════════════════╝

출력: ego_traj (N_ego, 12, 6), sur_traj (N_sur, 12, 6)
```

### 모듈별 상세 다이어그램

#### DecoderHistoryAttention (ego/sur 각 1개)

```
┌─────────────────────────────────────────────────────────────────────┐
│  DecoderHistoryAttention                                              │
│                                                                       │
│  파라미터:                                                            │
│    d_model = D = 64                                                   │
│    nhead = 4,  head_dim = D/4 = 16                                   │
│    max_len = FT = 12                                                  │
│                                                                       │
│  서브모듈:                                                            │
│    feat_proj: Linear(D, D)           — GCN feat → query space        │
│    pe:        Embedding(12, D)       — learnable positional encoding │
│    cross_attn: MultiheadAttention(D, 4, batch_first=True)           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Forward(query, history_buffer, t)                               │ │
│  │                                                                   │ │
│  │  query:          (N, D)                — GRU hidden state        │ │
│  │  history_buffer: (N, 12, D)            — pre-allocated           │ │
│  │  t:              int                   — current timestep        │ │
│  │                                                                   │ │
│  │  if t == 0: return zeros(N, D)                                   │ │
│  │                                                                   │ │
│  │  history = history_buffer[:, :t, :]    (N, t, D)                 │ │
│  │       ↓ feat_proj                                                 │ │
│  │  history: (N, t, D)                                               │ │
│  │       ↓ + pe(arange(t))                (t, D) broadcast          │ │
│  │  KV: (N, t, D)                                                    │ │
│  │                                                                   │ │
│  │  Q = query.unsqueeze(1)               (N, 1, D)                  │ │
│  │                                                                   │ │
│  │  attn_out, _ = cross_attn(Q, KV, KV)  [4-head MHA]             │ │
│  │       ↓ squeeze(1)                                                │ │
│  │  output: (N, D)                                                   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  데이터 흐름 예시 (t=5):                                              │
│                                                                       │
│  ego_gru_hidden ──(N,D)──→ Q: (N,1,D)                               │
│                                    ↓                                  │
│  ego_hist_buffer[:,:5,:] ─→ KV: (N,5,D) ─→ cross_attn ─→ (N,D)    │
│   [t=0] [t=1] [t=2] [t=3] [t=4]            4-head                   │
│    +PE0  +PE1  +PE2  +PE3  +PE4             head_dim=16              │
│                                                                       │
│  ※ Q_len=1, KV_len=t → attention weight: (N, 1, t)                 │
│  ※ t 증가에 따라 KV 길이 자연 증가 (0→11)                           │
└─────────────────────────────────────────────────────────────────────┘
```

#### MapCrossAttention (ego/sur 각 1개)

```
┌─────────────────────────────────────────────────────────────────────┐
│  MapCrossAttention                                                    │
│                                                                       │
│  파라미터:                                                            │
│    d_model = D = 64                                                   │
│    map_feat_dim = 64 (conv3 output channels)                         │
│    nhead = 4,  head_dim = D/4 = 16                                   │
│                                                                       │
│  서브모듈:                                                            │
│    map_proj:   Linear(map_feat_dim, D)  — map token → query space   │
│    cross_attn: MultiheadAttention(D, 4, batch_first=True)           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Forward(query, map_tokens)                                      │ │
│  │                                                                   │ │
│  │  query:      (N, D)         — GRU hidden state                  │ │
│  │  map_tokens: (N, 841, 64)   — conv3 spatial features (@256pix) │ │
│  │                                                                   │ │
│  │  map_proj_out = map_proj(map_tokens)   (N, 841, D)              │ │
│  │                                                                   │ │
│  │  Q = query.unsqueeze(1)                (N, 1, D)                 │ │
│  │  KV = map_proj_out                     (N, 841, D)              │ │
│  │                                                                   │ │
│  │  attn_out, attn_weights = cross_attn(Q, KV, KV)                 │ │
│  │                                        [4-head MHA]              │ │
│  │                                                                   │ │
│  │  output:  attn_out.squeeze(1)          (N, D)                    │ │
│  │  weights: attn_weights                 (N, 1, 841) ← 캡처       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  29×29 Map Grid 구조 (@256pix):                                      │
│                                                                       │
│  ┌──────────────────────────────────┐                                │
│  │  (0,0)  (0,1) ... (0,28)        │  ← 77m × 77m 영역             │
│  │  (1,0)  (1,1) ...               │     2.66m per token             │
│  │   ...                            │                                 │
│  │  (28,0)       ... (28,28)        │  ego는 (~6,14) 근처            │
│  └──────────────────────────────────┘                                │
│  flatten → 841 tokens → KV로 사용                                    │
│                                                                       │
│  ※ attention weight (N, 1, 841)를 Auxiliary (D) Loss에 사용         │
│  ※ ego/sur 별도 weight → Phase 2에서 ego만 학습                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### IntentCodebook (ego만)

```
┌─────────────────────────────────────────────────────────────────────┐
│  IntentCodebook                                                       │
│                                                                       │
│  파라미터:                                                            │
│    num_intents (K) = 8                                                │
│    intent_dim = 32                                                    │
│    input_dim = 6 + 2 + 64 = 72   (ego_state + sur_delta + map_ctx)  │
│                                                                       │
│  서브모듈:                                                            │
│    codebook:         Embedding(K=8, intent_dim=32) — 의도 벡터       │
│    intent_predictor: MLP(72→64→K=8)                — 상황→확률       │
│    intent_ce_head:   Linear(32→9)                  — Auxiliary (A)   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Forward(ego_state, predicted_sur_delta, map_ctx_detached)       │ │
│  │                                                                   │ │
│  │  intent_input = concat([                                         │ │
│  │      ego_state,            (N_ego, 6)                            │ │
│  │      predicted_sur_delta,  (N_ego, 2)    ← Step 2에서           │ │
│  │      map_ctx.detach()      (N_ego, 64)   ← Step 3에서, no grad  │ │
│  │  ])                         (N_ego, 72)                           │ │
│  │       ↓                                                           │ │
│  │  intent_predictor (MLP: 72→64→8)                                 │ │
│  │       ↓                                                           │ │
│  │  logits: (N_ego, 8)                                               │ │
│  │       ↓                                                           │ │
│  │  ┌─ Train ─────────────────────────────────────┐                 │ │
│  │  │  weights = Gumbel-Softmax(logits, τ)         │                 │ │
│  │  │  → soft weights: (N_ego, 8)                  │                 │ │
│  │  │  모든 codebook에 gradient 흐름                │                 │ │
│  │  └──────────────────────────────────────────────┘                 │ │
│  │  ┌─ Eval ──────────────────────────────────────┐                 │ │
│  │  │  weights = one_hot(argmax(logits))           │                 │ │
│  │  │  → hard selection: (N_ego, 8)                │                 │ │
│  │  └──────────────────────────────────────────────┘                 │ │
│  │       ↓                                                           │ │
│  │  z_local = weights @ codebook.weight            (N_ego, 32)      │ │
│  │                                                                   │ │
│  │  Phase 1: z_local = zeros(N_ego, 32) 반환 (forward 스킵)        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  Codebook Visualization (학습 후):                                   │
│                                                                       │
│  slot 0: [0.2, -0.1, ...] → "등속 직진"                             │
│  slot 1: [0.5,  0.3, ...] → "가속 직진"                             │
│  slot 2: [-0.4, 0.1, ...] → "감속 직진"                             │
│  slot 3: [0.1,  0.8, ...] → "좌회전"                                │
│  ...                                                                  │
│  slot 7: [-0.6, 0.5, ...] → "급감속 회피"                           │
│  (라벨은 학습 후 해석, 모델이 자율 결정)                              │
│                                                                       │
│  Temperature Annealing:                                               │
│  τ = 1.0 (초기, 부드러운 탐색) → 0.1 (후기, 거의 hard selection)    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Interaction GCN (공유, 1개)

```
┌─────────────────────────────────────────────────────────────────────┐
│  IndividualSceneInteractionNet (상호작용 전용)                        │
│                                                                       │
│  파라미터:                                                            │
│    in_channels = 10 (state:6 + lw:2 + sem:2)                        │
│    msg_node_channels = 64                                             │
│    out_channels = D = 64                                              │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Forward(scene_graph)                                            │ │
│  │                                                                   │ │
│  │  x = concat(state, lw, sem)             (NA, 10)                │ │
│  │       ↓                                                           │ │
│  │  mlp_in: Linear(10→64)                  (NA, 64)                │ │
│  │       ↓                                                           │ │
│  │  msg_passing (AgentInteractionConv):                             │ │
│  │    x = layer(x, edge_index, pos, sem)   (NA, 64)                │ │
│  │       ↓                                                           │ │
│  │  mlp_out_ego:   Linear(64→D)            (N_ego, D=64)           │ │
│  │  mlp_out_other: Linear(64→D)            (N_sur, D=64)           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ※ map 미포함 — agent 간 상호작용만 인코딩                           │
│  ※ 기존 2개 GCN (temporal_gcn + z_local_gcn) → 1개로 통합           │
│  ※ prediction 안 함 — history buffer 채우기 전용                    │
│  ※ Phase 2에서 frozen (상호작용 패턴 보존)                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Auxiliary Loss 데이터 흐름 통합도

```
┌─────────────────────────────────────────────────────────────────────┐
│              Auxiliary Loss 흐름 (매 스텝 t)                          │
│                                                                       │
│  ┌──────────────┐    ego_hist_ctx     ┌──────────────────────────┐  │
│  │ Ego History  │ ──────────────────→ │  Linear(D→2)             │  │
│  │ Attention    │                     │  → predicted_sur_delta   │  │
│  └──────────────┘                     └────────┬─────────────────┘  │
│                                                 │                    │
│                                    ┌────────────┼────────┐          │
│                                    │            │        │          │
│                                    ▼            ▼        │          │
│                              (B) MSE Loss   Intent     │          │
│                              vs gt_sur_delta  Input     │          │
│                                                ↓        │          │
│  ┌──────────────┐    ego_map_ctx     ┌─────────┴──────┐ │          │
│  │ Ego Map      │ ─────────────────→ │ .detach()       │ │          │
│  │ Attention    │                    │ → intent_input  │ │          │
│  │              │  ego_attn_w        └─────────┬──────┘ │          │
│  │              │ ──────────────┐              ↓        │          │
│  └──────────────┘               │    ┌─────────────────┐│          │
│                                 │    │ IntentCodebook  ││          │
│                                 ▼    │ → z_local       ││          │
│                          (D) KL Loss │ → weights       ││          │
│                          vs GT 6-step└────────┬────────┘│          │
│                          soft label           │         │          │
│                                               ▼         │          │
│                                        (A) CE Loss      │          │
│                                        Linear(32→9)     │          │
│                                        vs gt_acc_yaw    │          │
│                                                         │          │
│  ┌──────────────┐    sur_hist_ctx                       │          │
│  │ Sur History  │ ──────────────────────────────────┐   │          │
│  │ Attention    │                                    ▼   │          │
│  └──────────────┘                              (C) MSE  │          │
│                                                vs gt_ego│          │
│  ┌──────────────┐    sur_attn_w                         │          │
│  │ Sur Map      │ ─────────────────→ (D) KL Loss       │          │
│  │ Attention    │                    vs GT 6-step       │          │
│  └──────────────┘                                       │          │
│                                                         │          │
│  ────────────────────────────────────────────────────── │          │
│  Recon Loss:                                            │          │
│  ego_traj, sur_traj → MSE vs GT trajectory              │          │
│  KL Loss: z_global prior vs posterior                   │          │
│  EnvPotential: trajectory vs drivable area              │          │
│  VehColl: trajectory vs vehicle distances               │          │
└─────────────────────────────────────────────────────────────────────┘
```

### 모듈 전체 요약표

| 모듈 | 타입 | 차원 | 용도 | Phase 1 | Phase 2 |
|------|------|------|------|---------|---------|
| `step_feature_extractor` | MLP | 11→128→64 | raw state → GCN input | Train | **Frozen** |
| `temporal_gcn_encoder` | IndivGCN | 64→128→64 | past temporal encoding | Train | **Frozen** |
| `prior_temporal_gru` | GRU-2L | 64→64 | past temporal context | Train | **Frozen** |
| `positional_encoding` | SinPE | 64, max=12 | future PE | Train | **Frozen** |
| `transformer_encoder` | TransEnc | 64, 8h, 3L | future temporal encoding | Train | **Frozen** |
| `latent_prior_net` | MLP | 130→128→64 | z_global prior | Train | **Frozen** |
| `latent_posterior_net` | MLP | 194→128→64 | z_global posterior | Train | **Frozen** |
| `map_conv (full)` | CNN-6L | 4→128 | map encoding (encoder) | Train | **Frozen** |
| `map_conv (conv1~3)` | CNN-3L | 4→64 | map tokens (decoder) | Train | **Frozen** |
| `map_feature` | Linear | ?→64 | map feat projection (enc) | Train | **Frozen** |
| `interaction_gcn` | IndivGCN | 10→64→D | agent interaction → buffer | Train | **Frozen** |
| **`ego_history_attn`** | MHA+PE | D, 4h | ego history cross-attn | **Train** | **Train** |
| **`sur_history_attn`** | MHA+PE | D, 4h | sur history cross-attn | **Train** | **Frozen** |
| **`ego_map_attn`** | MHA | D, 4h, KV=841 | ego map cross-attn | **Train** | **Train** |
| **`sur_map_attn`** | MHA | D, 4h, KV=841 | sur map cross-attn | **Train** | **Frozen** |
| **`intent_codebook`** | Embed+MLP | K=8, 32dim | discrete intent (ego) | ❌ 비활성 | **Train** |
| **`ego_decoder_gru`** | GRU-3L | 196→D | ego trajectory decode | **Train** | **Train** |
| **`sur_decoder_gru`** | GRU-3L | 164→D | sur trajectory decode | **Train** | **Frozen** |
| **`ego_output_head`** | Linear | D→2 | ego output (a, hdot) | **Train** | **Train** |
| **`sur_output_head`** | Linear | D→2 | sur output (a, hdot) | **Train** | **Frozen** |
| `sur_pred_head` | Linear | D→2 | Aux (B): ego→sur pred | **Train** | **Train** |
| `ego_pred_head` | Linear | D→2 | Aux (C): sur→ego pred | **Train** | **Frozen** |
| `intent_ce_head` | Linear | 32→9 | Aux (A): intent CE | ❌ 비활성 | **Train** |

### Freeze/Train 시각 요약

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  Phase 1 (Normal Pretrain):                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  TRAIN: 전부                                                   │  │
│  │  ├── Encoder 전체 (GCN, Transformer, Prior, Posterior)        │  │
│  │  ├── Map CNN 전체                                              │  │
│  │  ├── interaction_gcn                                           │  │
│  │  ├── ego_history_attn, ego_map_attn, ego_gru, ego_mlp        │  │
│  │  ├── sur_history_attn, sur_map_attn, sur_gru, sur_mlp        │  │
│  │  ├── sur_pred_head, ego_pred_head                             │  │
│  │  └── (intent_codebook 비활성 — z_local = zero)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Phase 2 (Adversarial Finetune):                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  FROZEN ❄️:                                                    │  │
│  │  ├── Encoder 전체                                              │  │
│  │  ├── Map CNN 전체                                              │  │
│  │  ├── interaction_gcn                                           │  │
│  │  ├── sur_history_attn, sur_map_attn, sur_gru, sur_mlp        │  │
│  │  └── ego_pred_head                                             │  │
│  │                                                                 │  │
│  │  TRAIN 🔥:                                                     │  │
│  │  ├── ego_history_attn                                          │  │
│  │  ├── ego_map_attn                                              │  │
│  │  ├── ego_decoder_gru                                           │  │
│  │  ├── ego_output_head                                           │  │
│  │  ├── intent_codebook (활성화 + temperature annealing)          │  │
│  │  ├── sur_pred_head                                             │  │
│  │  └── intent_ce_head                                            │  │
│  │                                                                 │  │
│  │  → Sur에 영향 ZERO (완전 격리)                                 │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 병목 제거

### 3a. Decoder Loop 경량화

**기존 매 스텝:**
```
z_local GCN (무거움) + 2-layer cross-attn + self-attn + temporal GRU
+ sur decoder GCN
+ ego GRU
= GCN 2회 + attention 3회 + GRU 2회
```

**제안 매 스텝:**
```
상호작용 GCN 1회 (buffer 채우기)
+ History Attention 2회 (ego/sur, 가벼움)
+ Map Attention 2회 (ego/sur, 가벼움)
+ GRU 2회 (ego/sur)
= GCN 1회 + attention 4회(가벼움) + GRU 2회
```

기존 z_local의 무거운 attention 스택(2-layer cross + self + temporal GRU) 제거가 핵심.

**Pre-allocate + In-place:**
```python
ego_history_buffer = torch.zeros(N_ego, FT, gcn_feat_dim, device=device)
sur_history_buffer = torch.zeros(N_sur, FT, gcn_feat_dim, device=device)
```
→ list append / torch.stack 완전 제거

### 3b. Loss Vectorization

- VehPotentialLoss: nested for-loop → batched distance computation
- EnvPotentialLoss: .cpu().numpy() → GPU-only indexing
- VehCollLoss: list comprehension with .item() → torch.linspace batched
- validation .item() → 배치 처리

### 3c. TF Annealing 유지

기존 segment 방식 유지, 3단계 구조로 최적화 (Scheduled Sampling은 불연속 궤적 문제):
```
segment = 3 → cosine annealing → 12 (tf_max_annealing_epoch=1500)
segment 내: 연속 autoregressive (예측 기반 누적)
segment 경계: GT state, GT GCN cache, GT GRU hidden 복원

구현 (ego_loop_decoder):
  Step A: GT GCN 사전계산 (FT=12회, 캐시)
  Step B: GT forward pass (no_grad) → hidden 스냅샷 13개 저장
  Step C: Sliding window — segment 경계에서 GT 스냅샷 복원, 내에서만 예측

  GCN 총 호출: 12(사전) + segment 내 예측용 ≈ 최적화 완료

Auxiliary output 저장: segment별 리스트 구조 (recon과 동일)
  _sur_pred_outputs[seg_idx] = [pred_step0, pred_step1, ...]
  _ego_map_attn_weights_outputs[seg_idx] = [attn_step0, attn_step1, ...]
  _z_local_outputs[seg_idx] = [z_step0, z_step1, ...]
  _intent_weights_outputs[seg_idx] = [w_step0, w_step1, ...]
  → loss에서 segment별 평균 후 전체 segment 평균 (recon loss와 동일 방식)
  → AR 모드에서는 flat list 유지 (loss에서 자동 분기)
```

LR warmup (10 epochs) + gradient clipping (max_norm=1.0) 추가.

---

## 3.5 Auxiliary Loss 설계

### 설계 원칙

각 모듈이 자기 역할을 회피할 수 없게 만드는 구조.
- Auxiliary head는 **Linear-only** (MLP 금지) — 비선형 head는 잘못된 입력을 올바른 출력으로 포장할 수 있음
- Attention weight 직접 supervision — head 없이 attention 분포 자체에 loss
- 각 loss가 하나의 모듈만 타겟팅 — 역할 회피 불가

**원칙의 근거:**
- Linear Probing (Alain & Bengio, 2016): Linear head는 representation에 정보가 이미 선형 분리 가능한 형태로 있어야만 성공 → 모듈이 해당 정보를 직접 인코딩하도록 강제
- Auxiliary Tasks as Regularization (Liebel & Korner, 2018): 보조 과제가 공유 representation을 풍부하게 만들어 주 과제 일반화 성능 향상
- GameFormer (Huang et al., 2023 ICCV): 계층적 decoder의 매 레벨에 supervision → lazy level 방지. 우리도 z_global(KL), z_local(Intent CE), attention(Sur Pred), decoder(Recon) 각각에 loss

### Loss 전체 구조

```
[기존 Loss — 유지]
├── Recon MSE (ego + sur trajectory)        ← decoder 전체
├── KL Divergence (z_global prior/posterior) ← CVAE 구조
├── EnvPotentialLoss (drivable area 경계)    ← 도로 이탈 패널티
└── VehCollLoss (차량 간 충돌)              ← 충돌 패널티

[Auxiliary Loss — 신규]
├── (A) Intent CE Loss                      ← Intent Codebook 타겟
├── (B) Sur Prediction Loss (ego attn)      ← Ego History Attention 타겟
├── (C) Ego Prediction Loss (sur attn)      ← Sur History Attention 타겟
└── (D) Map Attention Guidance Loss         ← Map Cross-Attention 타겟 (ego + sur)
```

### (A) Intent CE Loss — Codebook이 의미있는 행동을 학습하도록 강제

```
codebook_vector (intent_dim) → Linear(intent_dim, 9) → 9-class CE
```

**타겟:** acc × yaw 방향 = 3(감속/유지/가속) × 3(좌/직/우) = 9 class
**활성:** Phase 2만 (Phase 1에서는 codebook 비활성)
**타겟팅 모듈:** Intent Codebook

**근거:**
- VQ-VAE (van den Oord et al., 2017): 학습 가능한 codebook 벡터로 이산 표현 학습
- Gumbel-Softmax (Jang et al., 2017): 미분 가능한 이산 선택, temperature annealing
- MTR (Shi et al., 2022 NeurIPS): 학습 가능한 motion query가 주행 의도/모드를 자율 학습 — 우리 codebook과 직접 대응
- MFP (Tang & Salakhutdinov, 2019 NeurIPS): 이산 latent가 라벨 없이 의미있는 주행 모드를 자율 발견

**왜 Linear-only:** codebook vector가 직접 행동 의미를 담아야 함. MLP면 무의미한 codebook도 올바른 class로 변환 가능

### (B) Sur Prediction Loss (ego attention) — Ego가 Sur를 추적하도록 강제

```
ego_history_context (D) → Linear(D, state_dim) → sur next delta 예측
Loss = MSE(predicted_sur_delta, gt_sur_delta)
```

**타겟:** sur 차량의 다음 스텝 변화량 (dx, dy) 또는 (dx, dy, dyaw)
**활성:** Phase 1 + Phase 2
**타겟팅 모듈:** Ego History Attention

**추가 역할 (Phase 2):**
predicted_sur_delta + map_ctx를 Intent Codebook 입력으로 재활용:
```
intent_input = [ego_state, predicted_sur_delta, map_ctx.detach()]
z_local = IntentCodebook(intent_input)
```
→ "sur가 다가오고 + 옆 차선 상황" 종합 판단으로 의도 선택
→ map_ctx.detach(): 정보는 쓰되 map attention weight에 gradient 안 흐름 (Phase 2 분리 보장)

**근거:**
- Social-LSTM (Alahi et al., 2016 CVPR): 주변 agent를 예측하면 상호작용 특성 학습이 향상됨
- Social Attention (Vemula et al., 2018 ICRA): attention weight로 주변 agent 중요도 학습 — 우리 history attention이 sur를 추적하는 것과 동일 원리
- Trajectron++ (Salzmann et al., 2020 ECCV): 상호작용 그래프에서 모든 agent 공동 예측 → 전체 예측 품질 향상
- Linear Probing 원칙: history_context에 sur 정보가 선형 분리 가능해야 함 → attention이 sur를 직접 추적해야만 loss 감소

**왜 Linear-only:** history_context에 sur 위치 정보가 이미 녹아있어야만 Linear 변환으로 sur delta 추출 가능. MLP면 attention이 sur를 안 봐도 다른 정보에서 간접 추론 가능

### (C) Ego Prediction Loss (sur attention) — Sur도 Ego를 추적하도록 강제

```
sur_history_context (D) → Linear(D, state_dim) → ego next delta 예측
Loss = MSE(predicted_ego_delta, gt_ego_delta)
```

**타겟:** ego 차량의 다음 스텝 변화량 (dx, dy) 또는 (dx, dy, dyaw)
**활성:** Phase 1만 (Phase 2에서는 sur freeze)
**타겟팅 모듈:** Sur History Attention

**근거:** (B)와 동일. ego/sur 대칭 구조이므로 양방향 추적.
Phase 1에서 sur attention도 상대방을 잘 추적하면 sur trajectory 품질도 향상.

### (D) Map Attention Guidance Loss — Map Attention이 도로를 보도록 강제

```
map_attn_weights (N, 841) vs GT soft label (N, 841) → KL divergence
```

**GT soft label 생성 방식:**

가까운 미래일수록 중요하므로 exponential decay 가중치 적용 (6스텝 lookahead, 3초 전방):
```
가중치: [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  (t+1 ~ t+6)
```

```python
# 매 스텝 t에서, 전방 최대 6스텝 GT 위치를 map grid 좌표로 변환
# Exponential decay: w_k = exp(-λk) / Σexp(-λk), λ=0.3 (config 조절 가능)
# λ=0.3 → [0.311, 0.230, 0.170, 0.126, 0.094, 0.069]
weights = exp(-lambda * arange(6)) / sum(exp(-lambda * arange(6)))
soft_label = zeros(29, 29)  # @256pix
remaining = min(6, total_steps - t - 1)
for k in range(remaining):
    gt_pos = gt_trajectory[t + 1 + k]   # map crop 내 상대좌표
    gi, gj = pos_to_grid(gt_pos)         # 29×29 grid index
    soft_label[gi, gj] += weights[k]
soft_label = soft_label / soft_label.sum()  # normalize → 확률 분포
soft_label = soft_label.flatten()            # (841,)

# 남은 스텝이 없으면 (t=11) 이 loss 제외
# 남은 스텝 < 6이면 남은 만큼만 계산 후 re-normalize (가까운 미래 비중 자동 유지)
```

**시각적 예시 (29×29 grid에서 직진 시):**
```
attention weight (841개에 대한 확률 분포):
[.01, .01, .02, ..., .03, .07, .12, .18, .20, .15, ..., .01]
                      ↑    ↑    ↑    ↑    ↑    ↑
                     t+6  t+5  t+4  t+3  t+2  t+1   ← 모델이 보는 곳

GT soft label (exponential decay, λ=0.3):
[ 0,   0,   0,  ..., .07, .09, .13, .17, .23, .31, ...,  0 ]
                      ↑    ↑    ↑    ↑    ↑    ↑
                     t+6  t+5  t+4  t+3  t+2  t+1   ← 봐야 할 곳

Loss = KL(soft_label || attn_weights)
→ attention이 전방 도로 방향을 따라 보되, 가까운 미래에 더 집중하도록 학습
```

**타겟:** 전방 6스텝 (3초) GT trajectory가 지나는 map token 위치
**활성:** Phase 1 + Phase 2 (sur는 Phase 2에서 freeze)
**적용:** ego + sur 각각 (별도 weight, gradient 포함 저장)
**타겟팅 모듈:** Map Cross-Attention (ego/sur 각각)

**근거:**
- LaPred (Kim et al., 2021 CVPR): map/lane attention이 "어떤 차선을 따라갈 것인가"를 직접 분류하게 함 → attention 무시 방지. 우리 적용: 전방 6스텝 GT 토큰으로 "어디를 봐야 하는가" 직접 지도
- Attention Supervision (GAIN, Li et al., 2018 CVPR): attention map에 직접 GT supervision을 걸어 학습 → 우리가 attn_weights에 직접 CE/KL을 거는 것과 동일
- BEVFormer (Li et al., 2022 ECCV): BEV spatial grid token에 대한 cross-attention — 우리 841 map token cross-attention과 같은 패턴
- VectorNet (Gao et al., 2020 CVPR), LaneGCN (Liang et al., 2020 ECCV): map 요소를 개별 token으로 유지하여 attention — flatten하지 않는 우리 접근의 근거

**왜 attention weight 직접 supervision:** 별도 head 없이 attention 분포 자체가 "예측"이므로 packaging 문제 자체가 없음. 가장 직접적인 모듈 타겟팅.

**6스텝 전방을 보는 이유:** map crop이 ego 중심이 아닌 비대칭(bounds [-17, -38.5, 60, 38.5]). 1스텝만 보면 trivial. 6스텝(3초, ~25m @30km/h)이면 진행 방향이 명확히 드러나 도로 구조 학습이 강제되고, 더 먼 전방까지 선제적 판단을 유도.

### Loss 요약표

| Loss | 입력 | Head | 타겟 | 타겟 모듈 | Phase 1 | Phase 2 |
|------|------|------|------|----------|---------|---------|
| Recon MSE | trajectory | - | GT trajectory | 전체 decoder | ✅ | ✅ |
| KL Div | z_global | - | prior/posterior | CVAE encoder | ✅ | ❌ (freeze) |
| EnvPotential | trajectory | - | drivable area | 간접 | ✅ | ✅ |
| VehColl | trajectory | - | 차량 간 거리 | 간접 | ✅ | ✅ |
| **(A) Intent CE** | codebook vec | Linear→9cls | acc×yaw 방향 | Intent Codebook | ❌ | ✅ ego만 |
| **(B) Ego→Sur Pred** | ego hist_ctx | Linear→2 | sur next delta (dx,dy) | Ego History Attn | ✅ | ✅ |
| **(C) Sur→Ego Pred** | sur hist_ctx | Linear→2 | ego next delta (dx,dy) | Sur History Attn | ✅ | ❌ (freeze) |
| **(D) Ego Map Attn** | attn_weights | 없음 (직접) | 전방6스텝 GT토큰 soft label | Ego Map Attn | ✅ | ✅ |
| **(D) Sur Map Attn** | attn_weights | 없음 (직접) | 전방6스텝 GT토큰 soft label | Sur Map Attn | ✅ | ❌ (freeze) |

### 스텝별 edge case 처리

```
t=0~5:  Map Attn → 6스텝 전부, Sur/Ego Pred → 활성
t=6:    Map Attn → 5스텝 (t+1~t+5), Sur/Ego Pred → 활성
t=7:    Map Attn → 4스텝 (t+1~t+4), Sur/Ego Pred → 활성
t=8:    Map Attn → 3스텝 (t+1~t+3), Sur/Ego Pred → 활성
t=9:    Map Attn → 2스텝 (t+1~t+2), Sur/Ego Pred → 활성
t=10:   Map Attn → 1스텝 (t+1만), Sur/Ego Pred → 활성
t=11:   Map Attn → 제외 (전방 없음), Sur/Ego Pred → 활성 (마지막 delta)
남은 스텝 < 6이면 가중치 앞부분만 사용 후 re-normalize → 분포 합=1 유지
```

### Decoder 흐름 업데이트 (auxiliary loss 반영)

```
[매 스텝 t = 0..11]

1. GCN → history buffer 저장

2. History Attention:
   ego_hist_ctx = Ego HistAttn(Q=ego_gru_hidden, KV=ego_history_buffer[:t])
   predicted_sur_delta = Linear(ego_hist_ctx)  → (B) Sur Pred Loss
   sur_hist_ctx = Sur HistAttn(Q=sur_gru_hidden, KV=sur_history_buffer[:t])
   predicted_ego_delta = Linear(sur_hist_ctx)  → (C) Ego Pred Loss

3. Map Attention:
   ego_map_ctx, ego_attn_w = Ego MapAttn(Q=ego_gru_hidden, KV=map_tokens)
   sur_map_ctx, sur_attn_w = Sur MapAttn(Q=sur_gru_hidden, KV=map_tokens)
   → (D) Map Attn Guidance Loss (ego_attn_w, sur_attn_w vs GT 6-step soft label, KL divergence)

4. Intent Selection (ego만):
   intent_input = [ego_state, predicted_sur_delta, ego_map_ctx.detach()]
   z_local, weights = IntentCodebook(intent_input)
   → (A) Intent CE Loss (Linear(z_local) → 9-class)
   Phase 1: z_local = zero (intent 비활성, 구조만 유지)

5. GRU + Output:
   ego_gru_input = [ego_hist_ctx, ego_map_ctx, z_global, z_local, lw, sem]
   ego_gru_out → output_head → (acc, yaw_rate) → bicycle model
   → Recon MSE Loss

   sur_gru_input = [sur_hist_ctx, sur_map_ctx, z_global, lw, sem]
   sur도 동일 (z_local 제외)
```

---

## 4. Phase 분리

### Phase 1: Normal Driving (Pretrain)
- z_local = zero (Intent Codebook 비활성화)
- 전체 모델 학습: encoder, GCN, ego/sur decoder, map CNN
- ego와 sur가 사실상 동일 구조 (z_local=zero)
- 목표: val loss ≤ 3.70, train/val gap ≤ 0.03

### Phase 2: Adversarial Reactive (Finetune)
- Freeze: encoder 전체, z_global, sur 모듈 전체, 상호작용 GCN, map CNN
- Train: ego history attn, ego map attn, ego GRU, ego output_head (Linear), intent codebook
- Intent Codebook 활성화, temperature annealing
- 목표: ego가 sur 접근 시 의도 선택 (감속/유지/회피 등)

---

## 5. 구현 순서

### Step 1: 모델 구조 변경 (최우선)
- 상호작용 GCN 역할 변경 (prediction 제거, buffer 채우기 전용)
- DecoderHistoryAttention 모듈 구현 (GCN feat + PE) — ego/sur 별도
- MapCrossAttention 모듈 구현 (conv3 분기, 29×29 tokens) — ego/sur 별도
- Ego decoder 재설계 (history attn + map attn + GRU)
- Sur decoder 재설계 (history attn + map attn + GRU, 기존 GCN decoder 대체)
- IntentCodebook 모듈 구현 (K=8, Gumbel-Softmax, ego만)
- z_local Phase 1 zero-masking
- z_combine MLP 제거 → concat
- history_buffer pre-allocation + in-place update

### Step 2: 병목 제거
- Loss vectorization
- Pre-allocate tensors, list append 제거
- scene_graph pos in-place update

### Step 3: Phase 1 학습 및 검증
- Phase 1 학습 실행
- Val loss ≤ 3.70 확인
- Train/Val gap ≤ 0.03 확인

### Step 4: Phase 2 (Intent Codebook 활성화)
- ego 모듈만 학습 (sur 영향 zero 확인)
- temperature annealing
- codebook slot 시각화 — 어떤 의도가 학습되었는지 확인
- Ego reactive behavior 검증

### Step 5: 데이터 확장 (필요시)
- CARLA에서 추가 시나리오 생성
- 다양한 상호작용 패턴 (회피 등)

---

## 6. 예상 효과

| 항목 | 현재 | 목표 |
|------|------|------|
| Val loss | 3.78 | ≤ 3.70 |
| Train/Val gap | 0.10 | ≤ 0.03 |
| 학습 시간 (1000 epochs) | ~1일 | 반나절 이하 |
| Map 인식 | 64dim concat (무시됨) | 841 token cross-attn (선택적) |
| z_local 의미 | noise (64dim continuous) | discrete intent (K=8 codebook) |
| 과거 참조 | 없음 (ego) / 없음 (sur) | GCN feat history + PE |
| GCN 역할 | prediction + feature 2종 | 상호작용 인코딩 전용 1개 |
| Ego/Sur 구조 | 비대칭 (GRU vs GCN) | 대칭 (둘 다 GRU + attention) |
| Phase 2 분리 | 불완전 | ego/sur 모듈 완전 분리 |

---

## 7. 리스크 및 대안

**리스크 1: Codebook collapse** — K=8 중 2-3개만 활성화
- 대안: Codebook utilization loss 추가 (균등 사용 유도)
- 또는 K 조절 후 재학습

**리스크 2: History Attention 초반 불안정** — t=0에서 히스토리 없음
- t=0은 zero context 반환
- t=1부터 attention 시작, 스텝 쌓이면서 안정화

**리스크 3: Map cross-attention이 무시될 수 있음**
- Map Attention Guidance Loss (D)로 직접 해결: 전방 4스텝 GT 토큰에 attention 집중 강제
- EnvPotentialLoss가 추가로 간접 유도
- → 리스크 대폭 감소

**리스크 4: Sur decoder 변경 (GCN → GRU) 성능 저하 가능**
- 기존 sur GCN decoder가 잘 작동했으므로, 성능 저하 시 sur만 GCN 복원 가능
- 다만 상호작용 GCN feature를 history attention으로 참조하므로 GCN 정보는 유지됨

---

## 8. 비판적 자기 검토

### History Attention + Codebook이 인과관계를 학습하는가?

**가능성은 높지만 보장은 아님.**
- History Attention + PE: "t=4에서 sur 접근 → t=5에서 감속" 시간적 인과 학습 가능
- Intent Codebook: "이 상황에서 감속 의도 선택" — 상황-의도 매핑 학습
- 진정한 counterfactual reasoning ("sur가 없었다면?")은 못함
- 하지만 trajectory prediction에서 현실적으로 달성 가능한 최선

### GCN을 상호작용 전용으로 바꾸는 것이 맞는가?

**기존 sur GCN은 prediction까지 한 번에 했음.** 이걸 상호작용 인코딩 + GRU prediction으로 나누면:
- 장점: 역할 분리, ego/sur 대칭, Phase 2 분리 깔끔
- 단점: 검증된 sur GCN decoder를 버림
- 판단: 상호작용 정보는 GCN이 인코딩하고 history buffer로 전달되니 정보 손실 없음

### Map Cross-Attention이 도로 인식을 보장하는가?

**구조 + 직접 supervision으로 보장.**
- 841개 spatial token에서 선택적 참조 구조 갖춤
- Map Attention Guidance Loss (D)가 전방 6스텝 GT 토큰에 attention 집중을 직접 강제
- EnvPotentialLoss가 추가로 간접 유도
- LaPred (Kim et al., 2021)에서 검증된 접근

---

## 9. 변경 영향 범위

### 수정 파일
- `src/models/trafficplanner_model.py` — 전체 decoder 재설계, 새 모듈 추가
- `src/losses/trafficplanner_loss.py` — Loss vectorization
- `src/train_trafficplanner.py` — 병목 제거
- `configs/train_trafficplanner.cfg` — 새 하이퍼파라미터

### 건드리지 않는 파일
- `src/models/interaction_net.py` — SceneInteractionNet 유지
- `src/models/individual_interaction_net.py` — 유지
- `src/models/common.py` — car_dynamics 유지
- `src/datasets/` — 데이터 로딩 로직 유지
- `src/viz*/` — 시각화 코드 유지

### 새로 추가할 모듈 (trafficplanner_model.py 내)
- `DecoderHistoryAttention` — GCN feature 기반 히스토리 참조 + PE (ego/sur 별도)
- `MapCrossAttention` — 맵 spatial token 선택적 참조 (ego/sur 별도)
- `IntentCodebook` — discrete 의도 선택, K=8, Gumbel-Softmax (ego만)
