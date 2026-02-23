# Transformer Decoder 도입 계획

> 작성일: 2026-02-23
> 최종 수정: 2026-02-24
> 상태: 구현 완료, TF/AR shifted prediction 버그 수정 완료
> 기반: redesign_plan.md (v6) → GRU decoder를 Transformer decoder로 교체

---

## 0. 동기

### 현재 구조 문제
- GRU loop 12회 순차 실행 → 병렬화 불가, 학습 느림
- History Attention, Map Attention, A2A가 별도 모듈 → 구조 복잡
- GRU hidden state가 query로 쓰이는 등 모듈 간 의존성 높음

### Transformer 도입 이유
1. **학습 속도**: Training 시 12 step 병렬 처리 (causal mask)
2. **구조 통합**: History Attn → A2T, Map Attn → A2S, 신규 A2A를 하나의 Transformer layer로 통합
3. **A2A 자연스러운 포함**: agent 간 상호작용이 구조의 일부가 됨
4. **TF 세그먼트 불필요**: GT 전체 한번에 넣고 causal mask면 끝

### 참고
- Adv-BMT (arxiv 2506.09485): GPT-style decoder + Triple Cross-Attention (A2T, A2A, A2S)
- 단, Relative PE나 Contour-based Relation은 가져오지 않음 (차별화)

---

## 1. 유지하는 것 (변경 없음)

| 모듈 | 설명 |
|------|------|
| CVAE encoder | Prior/Posterior → z_global, KL loss |
| GCN | 매 timestep agent interaction encoding → 토큰 생성용 |
| Map CNN | 도로 이미지 → spatial tokens (29×29=841) |
| Intent Codebook | MLP → Gumbel-Softmax → codebook 가중합 → z_local |
| Intent Loss | z_local → Linear(32→9) → GT soft label과 KL divergence |
| Bicycle model | output (acc, hdot) → state 변환 |
| 12-step prediction | 0.5s interval |
| Visualization 코드 | 기존 유지 |

---

## 2. 전체 아키텍처 (ASCII)

```
╔══════════════════════════════════════════════════════════════════╗
║                    SCENE UNDERSTANDING (1회)                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Past 4 steps ──┬── GCN ──→ past_gcn_feat (NA, PT, 64)         ║
║                 │                                                ║
║                 ├── Temporal Encoder ──→ past_seq_out            ║
║                 │                                                ║
║                 ├── Prior Net ──→ (prior_mu, prior_var)          ║
║                 │                                                ║
║                 └── Posterior Net ──→ (post_mu, post_var)        ║
║                      (uses GT future)    → KL Loss              ║
║                                                                  ║
║  Map Image ──→ CNN ──→ map_feat (64dim, encoder용)              ║
║                └─→ conv3 ──→ map_tokens (NA, 841, ch)           ║
║                               (29×29 spatial grid)              ║
║                                                                  ║
║  z_global = rsample(post_mu, post_var)  ← training              ║
║  z_global = rsample(prior_mu, prior_var) ← inference            ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                     TOKEN CONSTRUCTION                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Training: GT 16 steps (PT=4 + FT=12) → GCN(16) → gcn_feats   ║
║  Inference: past 4 → GCN(4), then 1 step at a time             ║
║                                                                  ║
║  token[t] = Linear(gcn_feat[t](64) ⊕ z_global(32)              ║
║             ⊕ lw(2) ⊕ sem(NC)) → 128dim                        ║
║             + temporal_PE(t)                                     ║
║                                                                  ║
║  tokens shape: (1, T, NA, 128)                                  ║
║  temporal_PE: nn.Embedding(16, 128), learnable                  ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║              TRANSFORMER DECODER (4 layers)                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────── Layer 0 (상황 파악) ──────────────┐           ║
║  │                                                    │           ║
║  │  A2T ──→ Temporal Self-Attn + Causal Mask         │           ║
║  │          (같은 agent, 시간축 attend)               │           ║
║  │                                                    │           ║
║  │  A2A ──→ Agent Self-Attn (같은 timestep 내)       │           ║
║  │          K/V 공유 + ego_Q/O, sur_Q/O 분리         │           ║
║  │     └→ Pred Loss 추출:                             │           ║
║  │        ego_token → sur_pred_head(128→2) → (B)     │           ║
║  │        sur_token → ego_pred_head(128→2) → (C)     │           ║
║  │        ※ shifted: PE 3~14 tokens → step 0~11     │           ║
║  │                                                    │           ║
║  │  A2S ──→ Map Cross-Attn                           │           ║
║  │          Q: agent token, K/V: map_tokens(841)     │           ║
║  │     └→ Map Attn Weight 추출 → (D) Guidance Loss   │           ║
║  │        ※ TF: shifted [PT-1:-1], AR: last-token   │           ║
║  │                                                    │           ║
║  │  FFN + residual                                    │           ║
║  └────────────────────────────────────────────────────┘           ║
║                          │                                        ║
║  ┌──────── Intent Selection (Layer 0~1 사이) ─────────┐          ║
║  │                                                      │          ║
║  │  ego_token ──→ MLP → Gumbel-Softmax → weights(9)   │          ║
║  │            └→ weights @ codebook → z_local(32)      │          ║
║  │            └→ intent_ce_head: z_local → pred(9)     │          ║
║  │               vs GT soft label → (A) Intent KL Loss │          ║
║  │            └→ concat(token, z_local) → Linear → 128 │          ║
║  │                                                      │          ║
║  │  sur_token ──→ (동일 구조, 별도 codebook)            │          ║
║  │                                                      │          ║
║  └──────────────────────────────────────────────────────┘          ║
║                          │                                        ║
║  ┌─────────── Layer 1~3 (심화, 순수 반복) ───────────┐           ║
║  │  A2T → A2A → A2S → FFN + residual  (×3 layers)    │           ║
║  └────────────────────────────────────────────────────┘           ║
║                          │                                        ║
╠══════════════════════════════════════════════════════════════════╣
║                      OUTPUT HEAD                                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Shifted Prediction (GPT-style):                                 ║
║    PE 3 output → future step 0  (PE 3 = past step 3까지 봄)    ║
║    PE 4 output → future step 1  (PE 4 = GT step 0까지 봄)      ║
║    ...                                                           ║
║    PE 14 output → future step 11 (PE 14 = GT step 10까지 봄)   ║
║                                                                  ║
║  TF: future_tokens = x[:, PT-1:-1, :, :]  (PE 3~14)            ║
║  AR: last token output at each step                              ║
║                                                                  ║
║  ego_token → ego_output_head(Linear 128→2) → (acc, hdot)       ║
║  sur_token → sur_output_head(Linear 128→2) → (acc, hdot)       ║
║                                                                  ║
║  (acc, hdot) → Bicycle Model → (x, y, hx, hy) normalized       ║
║  → Recon MSE Loss vs GT                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. Shifted Prediction 상세 (핵심)

### 3a. TF (Training) — 병렬, Causal Mask

```
토큰 구성: [past_0, past_1, past_2, past_3, future_0, future_1, ..., future_11]
PE index:  [  0       1       2       3        4        5              15    ]

Causal Mask (같은 agent 내):
       PE0  PE1  PE2  PE3  PE4  PE5  PE6  ...  PE15
PE0  [  O    X    X    X    X    X    X   ...    X  ]
PE1  [  O    O    X    X    X    X    X   ...    X  ]
PE2  [  O    O    O    X    X    X    X   ...    X  ]
PE3  [  O    O    O    O    X    X    X   ...    X  ]   ← 여기서 step 0 예측
PE4  [  O    O    O    O    O    X    X   ...    X  ]   ← 여기서 step 1 예측
...
PE15 [  O    O    O    O    O    O    O   ...    O  ]   ← 여기서 step 12? NO!

O = attend 가능, X = -inf (못 봄)

※ PE 15에서 step 11까지의 정보를 다 보지만, output으로 사용하지 않음 (마지막 step 11의 예측은 PE 14)

Output 슬라이싱: x[:, PT-1:-1, :, :]  → PE 3~14 = FT=12개

PE → 예측 대응:
  PE  3 출력 → step  0 예측 (입력: past 0~3까지만 봄, GT future 없음)
  PE  4 출력 → step  1 예측 (입력: past 0~3 + GT step 0까지 봄)
  PE  5 출력 → step  2 예측 (입력: past 0~3 + GT step 0~1까지 봄)
  ...
  PE 14 출력 → step 11 예측 (입력: past 0~3 + GT step 0~10까지 봄)
```

### 3b. AR (Inference) — 순차

```
Step 0: tokens = [past_0(PE0), past_1(PE1), past_2(PE2), past_3(PE3)]
        → Transformer → last token(PE3) output → step 0 예측
        → bicycle model → new_state_0

Step 1: tokens = [past_0, past_1, past_2, past_3, pred_0(PE4)]
        → Transformer → last token(PE4) output → step 1 예측
        → bicycle model → new_state_1

...

Step 11: tokens = [past_0, ..., past_3, pred_0, ..., pred_10(PE14)]
         → Transformer → last token(PE14) output → step 11 예측

PE → 예측 대응 (TF와 동일):
  PE  3 → step  0
  PE  4 → step  1
  ...
  PE 14 → step 11
```

### 3c. TF-AR 일관성 원칙

**TF와 AR은 동일한 모델이다. 입출력의 형태와 역할이 완벽히 같아야 한다.**

| 항목 | TF | AR |
|------|-----|-----|
| PE N 토큰이 보는 정보 | PE 0~N (causal mask) | PE 0~N (누적 토큰) |
| PE N 출력의 예측 대상 | step N-PT+1 | step N-PT+1 |
| GCN 입력 state | GT (모든 step) | 예측값 (sequential) |
| Output head | 동일 weight | 동일 weight |
| Bicycle model | 동일 | 동일 |
| 유일한 차이 | GCN에 GT state 사용 | GCN에 predicted state 사용 |

TF의 GT state 사용은 **에러 누적 방지 + 학습 효율**을 위한 것이지, 다른 모델을 키우려는 것이 아님.

---

## 4. Output Head 상세

### 4a. 출력 값

```
output_head: Linear(128 → 2)

출력: (acc, hdot)
  acc  = acceleration (가속도)
  hdot = heading_rate = yaw_rate (방향 변화율)

※ yaw가 아니라 yaw_rate (hdot)
※ 절대 heading이 아니라 heading의 시간 미분
```

### 4b. Bicycle Model

```
(acc, hdot) → Bicycle Model → (x, y, hx, hy) normalized

prev_state: (x, y, hx, hy, speed, hdot) from previous step
1. speed_new = prev_speed + acc * dt
2. heading_new = prev_heading + hdot * dt
3. x_new = prev_x + speed_new * cos(heading_new) * dt
4. y_new = prev_y + speed_new * sin(heading_new) * dt
5. normalize → (x, y, hx, hy)

TF: prev_state = GT (매 step GT에서 가져옴, 에러 누적 없음)
AR: prev_state = 이전 step 예측값 (에러 누적됨)
```

---

## 5. Auxiliary Loss 구조

### 5a. Loss 요약

```
╔═══════════════════╦═════════════════╦═════════════════════════╦═════════════╦══════════╗
║ Loss              ║ 위치            ║ 역할                    ║ Head 타입   ║ Phase    ║
╠═══════════════════╬═════════════════╬═════════════════════════╬═════════════╬══════════╣
║ Recon MSE         ║ Layer N 출력    ║ 궤적 정확도             ║ Linear(→2)  ║ 1, 2    ║
║ KL divergence     ║ Encoder         ║ prior ≈ posterior       ║ —           ║ 1 only  ║
║ Ego Intent KL     ║ L0~L1 사이      ║ ego codebook → GT 분포 ║ Linear(32→9)║ 1, 2    ║
║ Sur Intent KL     ║ L0~L1 사이      ║ sur codebook → GT 분포 ║ Linear(32→9)║ 1, 2    ║
║ Map Attn Guidance ║ Layer 0 A2S     ║ map attn → GT 위치      ║ — (weight)  ║ ego:1,2 ║
║                   ║                 ║                         ║             ║ sur:1   ║
║ Sur Pred MSE      ║ Layer 0 A2A 후  ║ ego→sur delta 예측     ║ Linear(→2)  ║ 1, 2    ║
║ Ego Pred MSE      ║ Layer 0 A2A 후  ║ sur→ego delta 예측     ║ Linear(→2)  ║ 1 only  ║
╚═══════════════════╩═════════════════╩═════════════════════════╩═════════════╩══════════╝

Phase 제한 근거:
  KL: Phase 2에서 z_global frozen → KL loss 불필요
  Ego Pred: Phase 2에서 sur 모듈 frozen → sur→ego 예측 학습 불필요
  Map Attn Sur: Phase 2에서 sur A2S frozen → sur map attn 학습 불필요
```

### 5b. Shifted Slicing이 적용되는 Loss

모든 auxiliary loss는 **shifted prediction에 맞춘 데이터**를 사용:

```
TF mode에서의 slicing:
  Model output:
    a2a_future    = a2a_out[:, PT-1:-1, :, :]     # PE 3~14 → step 0~11
    ego_tokens_l0 = x[:, PT-1:-1, ego_mask, :]    # PE 3~14 → step 0~11
    future_tokens = x[:, PT-1:-1, :, :]           # PE 3~14 → step 0~11

  Loss에서의 slicing:
    ego_attn_future = ego_attn_all[PT-1:-1]       # PE 3~14 → step 0~11
    sur_attn_future = sur_attn_all[PT-1:-1]       # PE 3~14 → step 0~11

AR mode에서의 처리:
  → 각 step의 last token output만 수집 (이미 future-only)
  → 별도 slicing 불필요
  → Loss에서 tensor size로 TF/AR 자동 판별:
    if size(0) == T_total:  # TF → shifted slicing
    else:                    # AR → 그대로 사용
```

### 5c. Sur/Ego Pred Loss 상세

```
Layer 0 A2A output에서 추출 (shifted: PE 3~14):
  ego_a2a[:, :, ego_mask] → sur_pred_head(Linear 128→2) → sur delta 예측
  sur_a2a[:, :, ~ego_mask] → ego_pred_head(Linear 128→2) → ego delta 예측

GT delta:
  step t: gt_delta = gt_future[:, t, :2] - gt_future[:, t-1, :2]  (t>0)
  step 0: gt_delta = gt_future[:, 0, :2] - past[:, -1, :2]

Loss: MSE(pred_delta, gt_delta)
목적: A2A가 상대 agent 정보를 제대로 attend하도록 강제
Linear head → 정보가 token에 직접 담겨있어야만 예측 가능
```

### 5d. Map Attn Guidance Loss 상세

```
Layer 0 A2S의 attention weight (841dim 분포) 추출:

TF mode:
  Layer 0에서 전체 (T_total, N, 841) 반환
  Loss에서 [PT-1:-1] slicing → (FT, N, 841)

AR mode:
  각 step의 마지막 토큰 weight만 append → list
  getter에서 torch.cat → (FT, N, 841)

GT soft label 생성:
  1. GT 미래 6스텝 (x, y) 좌표
  2. agent local frame → pixel → 29×29 grid index
  3. exponential decay 가중: [0.311, 0.230, 0.170, 0.126, 0.094, 0.069]
  4. scatter → 841dim soft label

Loss: KL divergence(GT soft label, model attn weight)
목적: "실제로 차가 갈 도로 위치"에 map attention 집중
```

### 5e. Intent Soft Label Loss 상세

```
Layer 0~1 사이에서 추출 (shifted: PE 3~14 tokens):

ego_token(128) → IntentCodebook:
  MLP(128→64→9) → Gumbel-Softmax → weights(9) → codebook(9, 32) → z_local(32)

intent_ce_head(Linear 32→9): z_local → pred(9)
  vs GT soft label(9)

GT soft label 생성:
  GT (acc, yaw_rate) → 9개 prototype (3×3 grid) 과 Gaussian distance
  → softmax → 분포 (9dim)
  acc: raw speed diff / dt / a_std
  yaw_rate: raw hdot / hdot_std

Loss: KL divergence(log_softmax(pred), gt_soft_label)
목적: codebook 선택이 실제 driving intent와 일치하도록 강제
Linear head → z_local 자체의 품질 강제 (비선형이면 억지로 맞출 수 있음)
```

### 5f. Loss 설계 철학

```
데이터 적은 환경에서 auxiliary loss가 각 모듈을 명시적으로 가이드:
  Adv-BMT: 480K data → CE loss 하나로 충분
  우리: ~5000 data → multi-task loss로 구조적 학습 유도

  recon loss  → "궤적을 정확하게"     → 전체 모델           (Phase 1, 2)
  KL loss     → "latent 분포 맞춰라"  → VAE encoder         (Phase 1 only)
  intent loss → "의도를 올바르게 골라라" → IntentCodebook      (Phase 1, 2)
  map loss    → "도로를 제대로 봐라"   → A2S                 (ego: 1,2 / sur: 1)
  pred loss   → "상대차를 인지해라"    → A2A                 (sur pred: 1,2 / ego pred: 1)

선형 head 원칙:
  Sur Pred, Ego Pred, Intent CE 모두 Linear head
  → 없는 정보를 만들어낼 수 없음
  → 앞단이 좋은 feature를 만들어야만 loss 감소
  → 앞단 품질을 간접적으로 강제하는 bottleneck 역할
```

---

## 6. Transformer Layer 상세

### 6a. A2T (Temporal Self-Attention)

```
A2T: 같은 agent, 시간축 attend
  reshape: (B, T, N, D) → (B*N, T, D)
  self-attention 1개 (ego/sur 구분 없음)
  causal mask 적용 → 미래를 볼 수 없음
  learnable temporal PE (토큰에 이미 포함)
  residual: x = x + A2T(x)
```

### 6b. A2A (Agent Self-Attention)

```
A2A: 같은 timestep 내, agent축 attend
  reshape: (B, T, N, D) → (B*T, N, D)
  K/V = shared_KV_proj(LayerNorm(x))   — 1개, ego/sur 공통
  Q = ego_Q_proj / sur_Q_proj          — 역할별 분리
  O = ego_O_proj / sur_O_proj          — 역할별 분리
  mask: 없음 (GCN이 spatial filtering 담당)
  PE: 없음 (agent 순서 무의미)
  residual: x = x + A2A(x)

K/V 공유 이유:
  - 같은 key space에서 ego↔sur 상호 참조 가능
  - Q/O 분리로 역할별 query/output 차별화
  - Phase 2에서 K/V+sur_Q/O freeze, ego_Q/O만 학습 가능

GCN과 A2A 역할 분리:
  GCN = 각 agent의 명함 만들기 (주변 관계 포함, spatial snapshot)
  A2A = 명함들 보고 "누구한테 주목할지" 고르기 (선택적 집중)
```

### 6c. A2S (Map Cross-Attention)

```
A2S: agent token → map_tokens cross-attention
  Q: agent token (128dim) — ego_Q_proj / sur_Q_proj 분리
  K: map_tokens → map_K_proj(Linear 64→128)  ─┐ ego/sur 간 공유
  V: map_tokens → map_V_proj(Linear 64→128)  ─┘ (별도 projection)
  → attention weight (841dim): map cell에 대한 확률 분포
  → attn_out → ego_O_proj / sur_O_proj 분리

  K, V projection은 각각 별도 Linear이지만 ego/sur가 같은 weight 공유
  Q/O만 ego/sur 분리 → 역할별 query 차별화
  map_tokens는 에이전트별 위치 기준 crop → 입력 자체가 이미 다름
  residual: x = x + A2S(x)

  map_recrop (config: map_recrop: True):
    Training: GT state 16개로 미리 step별 map_tokens crop → (T_total, NA, num_tokens, ch) 4D
              _a2s_attention이 4D map_tokens 직접 사용 (broadcast 없음)
    Inference(AR): 매 step prev_state 기준 _recompute_map_tokens() 호출
              mult_samp도 지원 (mapixes를 NA*NS로 expand)
    False: 초기 crop(last past 위치) 재사용 → 3D (NA, num_tokens, ch) broadcast
```

### 6d. Layer 구조 요약

```
Layer 0:        상황 파악 → pred loss, map attn weight 추출
Layer 0~1 사이: Intent 선택 (ego/sur, 고정 위치)
Layer 1~N-1:    심화 (순수 반복, 깊이 = N-1)
최종 출력:       PE 3~14 tokens → output head → (acc, hdot) → bicycle model
```

---

## 7. Intent Codebook 상세

```python
# Ego/Sur 각각 별도 IntentCodebook 인스턴스

class IntentCodebook(nn.Module):
    def __init__(self, num_intents=9, intent_dim=32, input_dim=128):
        self.codebook = nn.Embedding(num_intents, intent_dim)  # 9개 의도 벡터
        self.intent_predictor = MLP([input_dim, 64, num_intents])  # 상황→의도 logits

    def forward(self, situation, temperature=1.0):
        logits = self.intent_predictor(situation)  # (N, 9)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            weights = F.one_hot(logits.argmax(-1), 9).float()
        z_local = weights @ self.codebook.weight  # (N, 32)
        return z_local, weights
```

**변경점 (GRU → Transformer)**:
- input_dim: 72 → 128 (token dim = d_model). Layer 0이 상황 파악 완료했으므로.
- **sur에도 별도 IntentCodebook 추가** (ego_codebook, sur_codebook)
- **phase 기반 zero 반환 제거** — flag(use_ego_z_local, use_sur_z_local)로만 on/off

z_local 반영 방식:
```
concat(token(128), z_local(32)) → Linear(160→128) → Layer 1 입력
이유: 더하기는 정보 혼재, concat은 역할 분리 유지
```

z_local 제어 (flag 기반, Phase 무관):
```
use_ego_z_local=True  → ego codebook 활성, intent CE loss 작동
use_ego_z_local=False → ego z_local=zeros, intent CE 무의미
use_sur_z_local=True  → sur codebook 활성, sur intent loss 작동 (현재 config)
use_sur_z_local=False → sur z_local 없음, token 그대로 Layer 1로
```

---

## 8. Ego/Sur 분리 및 Phase 2

### 8a. Transformer 내부 분리 구조

```
A2T: self-attention 1개 (ego/sur 구분 없음, 시간축)
A2A: K/V 공유 1개 + ego_Q/O, sur_Q/O 분리
A2S: K/V 공유 1개 + ego_Q/O, sur_Q/O 분리
FFN: ego_FFN, sur_FFN 완전 분리
Intent: ego_codebook, sur_codebook 별도
Output: ego_output_head, sur_output_head 별도
```

### 8b. Phase 2 Fine-tuning

```
Freeze 대상:
  - Encoder 전체 (prior, posterior, temporal GCN, map CNN)
  - GCN (interaction_gcn)
  - A2T (전체, self-attention 1개)
  - A2A K/V (shared), A2A sur_Q/O
  - A2S K/V (shared), A2S sur_Q/O
  - sur_FFN
  - sur_codebook, sur_intent_predictor, sur_intent_ce_head
  - sur_output_head

Train 대상:
  - A2A ego_Q/O (각 layer)
  - A2S ego_Q/O (각 layer)
  - ego_FFN (각 layer)
  - ego_codebook, ego_intent_predictor, ego_intent_ce_head
  - ego_output_head
```

---

## 9. Training vs Inference

### Training (병렬)

```
1. GT state 16 step → GCN(16) → gcn_feats (NA, 16, 64)
2. build_decoder_tokens() → tokens (1, 16, NA, 128)
   gcn_feat(64) ⊕ z_global(32) ⊕ lw(2) ⊕ sem(NC) → Linear → 128 + temporal_PE
3. Layer 0 (causal mask):
   A2T → A2A → [shifted pred loss 추출: PE 3~14] → A2S → [map attn 추출] → FFN
4. Intent 선택 (shifted: PE 3~14 tokens)
   concat(token, z_local) → Linear(160→128) → intent CE loss
5. Layer 1~3: 심화 (A2T → A2A → A2S → FFN ×3)
6. Output: x[:, PT-1:-1, :, :] (PE 3~14)
   → output_head(128→2) → (acc, hdot) → bicycle model → recon loss

※ bicycle model의 prev_state는 GT 사용 (TF 특성)
※ 12 step 한번에 계산
```

### Inference (autoregressive)

```
1. Past 4 step → GCN(4) → past tokens (1, 4, NA, 128)
2. Transformer → last token (PE 3) output → step 0 (acc, hdot)
3. Bicycle model → new_state → GCN(1) → new token (PE 4)
4. Append → Transformer → last token (PE 4) output → step 1
5. 반복 12회

※ bicycle model의 prev_state는 이전 예측값 사용
※ new_token 빌드 시 t_pos = PT + t (PE 4, 5, ..., 14)
※ t < FT-1 까지만 new_token 생성 (마지막 step은 출력만)
```

---

## 10. Hyperparameters

```yaml
# Transformer Decoder
num_decoder_layers: 4        # Layer 수 (기본 4, 최소 2)
trans_d_model: 128           # 모델 차원
trans_nhead: 8               # attention head 수 (128/8=16 per head)
trans_ffn_dim: 512           # FFN 중간 차원 (d_model×4)
trans_dropout: 0.1           # dropout rate
use_ego_z_local: True        # ego z_local on/off
use_sur_z_local: True        # sur z_local on/off (config 기준)
temporal_pe_type: learnable  # positional encoding type
map_recrop: True             # step별 map re-crop (TF: 미리 16step crop, AR: 매 step crop)

# Projection dimensions
gcn_feat_dim: 64             # GCN output → Linear(64→128) → Transformer
intent_dim: 32               # z_local dimension
num_intents: 9               # codebook entries (3acc × 3yaw)
traj_out_size: 2             # (acc, hdot)

# Shifted prediction
PT: 4                        # past timesteps
FT: 12                       # future timesteps
T_total: 16                  # PT + FT
# Output slice: x[:, PT-1:-1]  (PE 3~14)
# temporal_PE: nn.Embedding(16, 128), indices 0~15

# LR scheduler (step-based cosine annealing with warmup)
use_lr_anneal: True
lr_max: 3e-4                 # warmup 완료 후 peak LR
lr_min: 5e-6                 # cosine decay 종료 시 최소 LR
lr_warmup_steps: 2500        # linear warmup (lr_min → lr_max)
# lr_total_steps: auto       # epochs * (data_size / batch_size)
# KL annealing은 기존대로 epoch 기반 (kl_anneal_end: 50)
```

---

## 11. 버그 수정 이력

### 11a. AR PE Offset 버그 (2026-02-24 수정)

```
BEFORE (bug):
  t_pos = PT + t + 1  → new tokens got PE [5,6,...,15]
  AR: PE 5→step 0, PE 6→step 1, ..., PE 15→step 11

AFTER (fix):
  t_pos = PT + t      → new tokens got PE [4,5,...,14]
  AR: PE 4→step 1, PE 5→step 2, ..., PE 14→step 11
  (step 0는 past token PE 3에서 예측, new_token 필요 없음)

문제: TF에서 PE N→step N-3 이었는데, AR에서 PE N→step N-4
→ 같은 모델인데 PE 의미가 달라짐 → 학습/추론 불일치
```

### 11b. TF Information Leak 버그 (2026-02-24 수정)

```
BEFORE (bug):
  output = x[:, PT:, :, :]  → PE 4~15
  PE 4 토큰 = GT step 0의 GCN feature 포함
  PE 4 출력 = step 0 예측
  → "자기 자신의 정답을 보고 예측" = information leak

AFTER (fix):
  output = x[:, PT-1:-1, :, :]  → PE 3~14
  PE 3 토큰 = past step 3까지만 봄 (GT future 없음)
  PE 3 출력 = step 0 예측
  → "과거만 보고 미래 예측" = GPT-style shifted prediction

모든 관련 코드 일괄 수정:
  - a2a_future slicing: PT → PT-1:-1
  - intent codebook: PT → PT-1:-1
  - output head: PT → PT-1:-1
  - map attn loss: PT → PT-1:-1
```

### 11c. AR Map Attn Weight 미수집 버그 (2026-02-24 수정)

```
BEFORE (bug):
  AR decoder: return_a2s_weights=False
  → map attention weight가 저장 안 됨
  → validation에서 map_attn_loss가 tensorboard에 안 나옴
  → 모델 출력에는 영향 없음 (attention 자체는 정상 작동)

AFTER (fix):
  AR decoder: return_a2s_weights=is_first_layer (Layer 0에서만)
  → 각 step의 last token weight를 list에 append
  → getter에서 torch.cat → (FT, N, 841) tensor
  → validation에서도 map_attn_loss 정상 표시
```

---

## 12. 기존 redesign_plan.md와의 관계

이 문서는 redesign_plan.md의 **Level 3 (Action Generation)** 부분만 교체:
- Level 0 (Scene Understanding): 변경 없음
- Level 1 (z_global): 변경 없음
- Level 2 (z_local Intent Codebook): input_dim 변경 (72→128), **sur에도 확장**
- **Level 3: GRU loop → Transformer decoder**

나머지 (encoder, loss weights, Phase 전략, fine-tuning 전략)는 redesign_plan.md 그대로 유지.
