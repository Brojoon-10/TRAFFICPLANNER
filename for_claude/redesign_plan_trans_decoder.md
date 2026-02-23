# Transformer Decoder 도입 계획

> 작성일: 2026-02-23
> 상태: 설계 확정, 구현 준비 완료
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
| Bicycle model | output (acc, yaw_rate) → state 변환 |
| 12-step prediction | 0.5s interval |
| Visualization 코드 | 기존 유지 |

---

## 2. 제거하는 것

| 모듈 | 대체 |
|------|------|
| ego_decoder_gru / sur_decoder_gru | Transformer Layer가 대체 |
| ego_warmup_gru / sur_warmup_gru | Past token이 대체 (warmup 불필요) |
| ego_history_attn / sur_history_attn | A2T (Temporal Self-Attention)가 대체 |
| ego_map_attn / sur_map_attn | A2S (Map Cross-Attention)가 대체 |
| history_buffer | 불필요 (Transformer가 전체 시퀀스 처리) |
| TF 세그먼트 방식 | causal mask로 대체 |

---

## 3. 새 아키텍처

### 3a. 전체 흐름

```
=== 사전 준비 (1회) ===

1. Encoder:
   Past 4 steps → GCN + Encoder → z_global (prior/posterior)
   Map → CNN → map_tokens (N_agents, 841, ch)

2. GCN 전체 처리 (Training: GT 있으니까 한번에):
   Past 4 step + Future GT 12 step = 16 step
   각 step마다 GCN → ego_gcn_feat(64), sur_gcn_feat(64)
   → 16개 토큰 생성 (병렬 처리 가능)

3. 토큰 구성:
   token[t] = GCN_feat[t] + z_global + lw + sem + temporal_PE(t)
   shape: (B, 16, N_agents, D)

=== Transformer Decoder ===

Layer 0: 상황 파악 (feature 추출 + aux loss)
  A2T (Temporal Self-Attention + causal mask)
    - 같은 agent, 시간축 attend
    - past token은 자유롭게, future는 과거만 봄
    - learnable temporal PE (토큰에 이미 포함)
    - residual: x = x + A2T(x)

  A2A (Agent Self-Attention, mask 없음)
    - 같은 timestep, agent축 attend
    - mask 없이 전체 attend (GCN이 spatial filtering 담당)
    - PE 없음 (agent 순서 무의미)
    - residual: x = x + A2A(x)

  --- Sur/Ego Pred Loss 추출 (A2A 직후, A2S 전) ---
    ego_token → sur_pred_head(Linear 128→2) → sur delta 예측
    sur_token → ego_pred_head(Linear 128→2) → ego delta 예측
    vs GT delta → MSE loss
    목적: A2A attention 품질을 간접 감독
    선형 head이므로 token에 상대차 정보가 직접 담겨있어야 예측 가능
    → A2A가 상대차 정보를 잘 attend하도록 강제
    residual 구조이므로 pred loss가 token 특성을 왜곡할 위험 낮음

  A2S (Map Cross-Attention)
    - Q: token, K/V: map_tokens (841개, conv3 spatial features)
    - map attention weight (841dim 분포) 추출 → map guidance loss용
    - residual: x = x + A2S(x)

  --- Map Attention Guidance Loss 추출 (A2S에서) ---
    A2S attention weight (841dim): 모델이 생각한 "중요한 도로 위치" 분포
    vs
    GT soft label (841dim): GT 미래 6스텝 위치를 map grid에 찍은 분포
      - 미래 위치를 agent local frame → pixel → 29×29 grid index 변환
      - exponential decay 가중 (가까운 미래일수록 높음)
        [0.311, 0.230, 0.170, 0.126, 0.094, 0.069]
      - 841개 cell 중 6개 정도만 값 있음, 나머지 0
    → KL divergence
    목적: map attention이 "차가 실제로 갈 도로"에 집중하도록 가이드

  FFN + residual

--- Intent 선택 (Layer 0~1 사이, 고정) ---

  ego_token (128dim)
       │
       ├── intent_predictor(MLP): ego_token → logits(9)
       │     → Gumbel-Softmax → weights(9)
       │     → weights @ codebook.weight → z_local(32dim)
       │
       ├── intent_ce_head(Linear): z_local → pred(9dim)
       │     vs GT soft label (acc/yaw 기반 9개 prototype 거리) → intent KL loss
       │     선형 head이므로 z_local 자체의 품질을 강제
       │
       └── concat(ego_token, z_local) → Linear(160→128) → Layer 2 입력
           (z_local은 concat으로 역할 분리, projection으로 차원 맞춤)

  sur_token (128dim)
       │
       ├── sur_intent_predictor(MLP): sur_token → logits(9)
       │     → Gumbel-Softmax → weights(9)
       │     → weights @ sur_codebook.weight → sur_z_local(32dim)
       │
       ├── sur_intent_ce_head(Linear): sur_z_local → pred(9dim)
       │     vs GT soft label → sur intent KL loss
       │
       └── concat(sur_token, sur_z_local) → Linear(160→128) → Layer 2 입력

  ※ sur에도 intent를 적용하는 이유:
    - sur의 의도를 명시적으로 모델링 → 나중에 적대적 최적화에서 sur intent 조작 가능
    - ego가 A2A에서 "sur가 뭘 하려는지" 정보를 attend할 수 있게 됨
    - ego/sur 동시 학습이므로 구조적으로 자연스러움

  ※ z_local 제어 방식 (flag 기반, Phase 무관):
    - use_ego_z_local=True → ego codebook 활성, intent CE loss 작동
    - use_ego_z_local=False → ego z_local=0 (zeros), intent CE 무의미
    - use_sur_z_local=True → sur codebook 생성+활성, sur intent loss 작동
    - use_sur_z_local=False → sur z_local 없음, token 그대로 Layer 1로
    - IntentCodebook 내부에 phase 체크 없음 — 항상 Gumbel-Softmax 활성

Layer 1 ~ N-1: 심화 (순수 반복, 깊이 = num_layers - 1)
  A2T (Temporal Self-Attention + causal mask)
  A2A (K/V 공유 + Q/O 분리)
  A2S (K/V 공유 + Q/O 분리)
  FFN (ego/sur 분리) + residual
  (intent 반영된 token이 반복적으로 정교화)

  num_layers=4 → Layer 1,2,3 = 깊이 3

--- 최종 출력 ---

  ego_token → ego_output_head(Linear) → trajectory (acc, yaw_rate)
  sur_token → sur_output_head(Linear) → trajectory (acc, yaw_rate)
  → Bicycle model → state
  → Recon loss (예측 궤적 vs GT)
```

### 3b. Causal Mask 구조

```
Training:
  tokens = [past_0, past_1, past_2, past_3, future_0, ..., future_11]

  A2T causal mask (같은 agent 내):
       p0  p1  p2  p3  f0  f1  f2  ...  f11
  p0 [ O   X   X   X   X   X   X   ...   X  ]
  p1 [ O   O   X   X   X   X   X   ...   X  ]
  p2 [ O   O   O   X   X   X   X   ...   X  ]
  p3 [ O   O   O   O   X   X   X   ...   X  ]
  f0 [ O   O   O   O   O   X   X   ...   X  ]
  f1 [ O   O   O   O   O   O   X   ...   X  ]
  ...
  f11[ O   O   O   O   O   O   O   ...   O  ]

  O = attend 가능, X = -inf (못 봄)

  → 12 step 병렬 학습
  → loss는 future 12 step에 대해서만 계산

Inference:
  past 4 step 넣고, future를 하나씩 autoregressive 생성
  매 step: 예측 → bicycle model → 다음 state → GCN → 새 토큰 추가
```

### 3c. A2A 상세

```
A2A (같은 timestep 내, K/V 공유 + Q/O 분리):
  reshape: (B, T, N, D) → (B*T, N, D), D=128
  K/V = shared_KV_proj(LayerNorm(x))  — 1개, ego/sur 공통
  Q = ego_Q_proj(LayerNorm(x)) 또는 sur_Q_proj(LayerNorm(x))
  O = ego_O_proj(attn_out) 또는 sur_O_proj(attn_out)
  mask: 없음 (GCN이 이미 spatial interaction 포함한 풍부한 토큰 생성)
  PE: 없음 (agent 순서 무의미)

  K/V 공유 이유:
    - 같은 key space에서 ego↔sur 상호 참조 가능 (self-attention 효과 유지)
    - Q/O 분리로 역할별 query/output 차별화
    - Phase 2에서 K/V+sur_Q/O freeze, ego_Q/O만 학습 가능

GCN과 A2A의 역할 분리:
  GCN = 각 agent의 명함 만들기 (주변 관계 포함, spatial snapshot)
  A2A = 명함들 보고 "누구한테 주목할지" 고르기 (선택적 집중)
  → 상호보완, 중복 아님

A2A 품질 감독 (Sur/Ego Pred Loss):
  위치: Layer 0 A2A 직후, A2S 전 (고정)

  ego_token → sur_pred_head(Linear 128→2) → sur delta 예측
  sur_token → ego_pred_head(Linear 128→2) → ego delta 예측
  양방향: 서로가 서로를 예측해야 A2A가 양쪽 다 제대로 작동

  Linear(선형) head인 이유:
    - 비선형 MLP면 없는 정보도 억지로 맞출 수 있음
    - Linear는 token에 상대차 정보가 직접 담겨있어야만 예측 가능
    - → A2A attention 품질을 간접적으로 강제
    - intent_ce_head(Linear 32→9)와 같은 논리: 선형 head가 앞단 품질 강제

  residual 구조 (x = x + A2A(x)) 덕분에:
    - pred loss가 A2A output만 영향, 원본 x는 보존
    - 특성 왜곡 위험 낮음
    - pred loss weight를 작게 주면 (0.1) 가이드 수준
```

### 3d. A2S (Map Cross-Attention) 상세

```
A2S Cross-Attention:
  Q: agent token (128dim) — ego_Q_proj / sur_Q_proj 분리
  K/V: map_tokens (841개 = 29×29, conv3 spatial features, 64ch)
  → shared_map_KV_proj(Linear 64→128): K/V 공유 1개
  → attention weight (841dim): 841개 map cell에 대한 확률 분포
  → attn_out → ego_O_proj / sur_O_proj 분리

  K/V 공유 + Q/O 분리: map projection 로직은 동일, 역할별 query 분리
  map_tokens는 에이전트별 위치 기준 crop → 입력 자체가 이미 다름

Map Attention Guidance Loss:
  위치: Layer 0 A2S에서 attention weight 추출

  모델 출력: attn_weight (841dim) — "어디가 중요한지" 분포
  GT 생성:
    1. GT 미래 6스텝 (x, y) 좌표 가져옴
    2. agent local frame → pixel 좌표 → 29×29 grid index 변환
    3. exponential decay 가중 (가까운 미래 = 높은 가중치)
       w = [0.311, 0.230, 0.170, 0.126, 0.094, 0.069]
    4. 해당 grid cell에 가중치 scatter → soft label (841dim)
       841개 중 ~6개만 값 있음, 나머지 0

  Loss: KL divergence(모델 분포, GT 분포)
  목적: "실제로 차가 갈 도로 위치에 attention 집중하라"

  weight에 스텝 개념 없음 — 841개 cell에 대한 순수 확률 분포
  GT soft label의 decay 가중치가 "곧 갈 곳에 더 집중" 효과를 자연스럽게 줌
```

### 3e. Intent Codebook 상세

```python
# Ego/Sur 각각 별도 IntentCodebook 인스턴스

class IntentCodebook(nn.Module):
    def __init__(self, num_intents=9, intent_dim=32, input_dim=64):
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

**변경점**:
- input_dim: 72(ego_state+sur_delta+map_ctx) → 128(token dim, d_model=128). Layer 0이 상황 파악 완료했으므로.
- **sur에도 별도 IntentCodebook 추가** (ego_codebook, sur_codebook)
  - sur intent 학습 → 적대적 최적화에서 sur intent 조작 가능
  - ego가 A2A에서 sur의 의도 정보를 참조 가능
- **phase 기반 zero 반환 제거** — flag(use_ego_z_local, use_sur_z_local)로만 on/off 제어
  - IntentCodebook.forward()는 항상 Gumbel-Softmax 실행
  - flag=False면 모델 레벨에서 z_local을 zeros로 대체

Intent 선택 구조 (MLP → Gumbel-Softmax → codebook → Linear 검증):
  MLP(비선형): token → logits → "어떤 code 고를지" 결정
  codebook: logits → Gumbel-Softmax → 가중합 → z_local (32dim)
  intent_ce_head(Linear 선형): z_local → 9dim → GT 분포와 KL loss
  → Linear가 z_local 품질 강제 (비선형이면 억지로 맞출 수 있으나 선형은 불가)

GT soft label 생성:
  GT acc/yaw_rate → 9개 prototype (3×3 grid, acc×yaw) 과 Gaussian 거리
  → softmax → 분포 (9dim)
  acc: raw speed diff / dt / a_std (0=등속)
  yaw: raw hdot / hdot_std (0=직진)

z_local 반영 방식:
  concat(token(128), z_local(32)) → Linear(160→128) → Layer 1 입력
  이유:
  - 더하기: 정보가 섞여서 z_local 기여 불명확
  - concat: 역할 분리 유지, Linear가 적절히 결합
  - projection(160→128): Layer 1 입력 차원 맞춤
```

---

## 4. Loss 체계

### 4a. Loss 요약

| Loss | 위치 | 역할 | 가이드 대상 | head 타입 |
|------|------|------|-------------|-----------|
| Recon MSE | Layer N 출력 | 궤적 정확도 | 전체 모델 | Linear |
| KL divergence | Encoder | z_global prior ≈ posterior | VAE encoder | — |
| Ego Intent KL | Layer 0~1 사이 | ego codebook 선택 → GT 분포 | Ego Intent Codebook | Linear(32→9) |
| Sur Intent KL | Layer 0~1 사이 | sur codebook 선택 → GT 분포 | Sur Intent Codebook | Linear(32→9) |
| Map Attn Guidance | Layer 0 A2S | map attention → GT 미래 위치 | A2S | — (attn weight) |
| Sur Pred MSE | Layer 0 A2A 직후 | ego가 sur delta 예측 | A2A 품질 감독 | Linear(128→2) |
| Ego Pred MSE | Layer 0 A2A 직후 | sur가 ego delta 예측 | A2A 품질 감독 | Linear(128→2) |

### 4b. Layer 구조 요약

```
Layer 0:       상황 파악 (feature 추출) → pred loss, map loss 추출
Layer 0~1 사이: Intent 선택 (ego/sur, 고정 위치)
Layer 1~N-1:   심화 (순수 반복, 깊이 = N-1)
최종 출력:      future 12 step → output head → recon loss
```

### 4c. Loss 설계 철학

**데이터 적은 환경에서 auxiliary loss가 각 모듈을 명시적으로 가이드:**
- Adv-BMT: 480K data + A6000×8 → CE loss 하나로 충분 (모델이 알아서 배움)
- 우리: 데이터/자원 제한 → multi-task loss로 구조적 학습 유도 (기도메타 방지)

각 loss의 감독 대상:
```
recon loss     → "궤적을 정확하게" → 전체 모델
KL loss        → "latent 분포 맞춰라" → VAE encoder
intent loss    → "의도를 올바르게 골라라" → IntentCodebook (z_local 품질)
map loss       → "도로를 제대로 봐라" → A2S (map attention 품질)
pred loss      → "상대차를 제대로 인지해라" → A2A (agent attention 품질)
```

### 4c. 선형 head 원칙

Sur Pred, Ego Pred, Intent CE 모두 **Linear head** 사용:
- Linear는 없는 정보를 만들어낼 수 없음
- 앞단(A2A, Codebook)이 좋은 feature/z_local을 만들어야만 loss가 줄어듦
- → **앞단의 품질을 간접적으로 강제하는 bottleneck 역할**
- 비선형 MLP head를 쓰면 변환 과정에서 억지로 맞출 수 있어 감독 효과 약해짐

---

## 5. Ego/Sur 분리 및 Sur Intent

### 5a. Transformer 내부 (확정)

```
A2T: self-attention 1개 (ego/sur 구분 없음, 시간축)
A2A: K/V 공유 1개 + ego_Q/O, sur_Q/O 분리 (agent 상호작용, 역할별 query)
A2S: K/V 공유 1개 + ego_Q/O, sur_Q/O 분리 (map cross-attn, map_tokens는 에이전트별 위치 crop)
FFN: ego_FFN, sur_FFN 완전 분리

Phase 2 Freeze:
  A2T: 전체 freeze
  A2A: K/V + sur_Q/O freeze, ego_Q/O만 학습
  A2S: K/V + sur_Q/O freeze, ego_Q/O만 학습
  FFN: sur freeze, ego만 학습
```

### 5b. Intent Codebook

```
ego_codebook: ego 전용 (9개 code, 32dim)
sur_codebook: sur 전용 (9개 code, 32dim, 별도 학습)
→ ego/sur 각각 자기 상황에 맞는 의도 선택
→ z_local도 ego/sur 별도
```

### 5c. Output

```
ego_token → ego_output_head (별도 Linear) → ego 궤적
sur_token → sur_output_head (별도 Linear) → sur 궤적
→ output head가 별도이므로 한쪽 왜곡이 다른 쪽 궤적에 직접 영향 없음
→ A2A에서 정보 교환은 하되, 최종 궤적 생성은 독립
```

### 5d. Phase 2 Fine-tuning (확정)

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

### 5e. Sur Intent의 적대적 활용 (추후)

```
Sur intent를 조작 → sur 행동 변경 → ego 반응 유도
예: sur_codebook에서 "급가속" intent 강제 선택
    → sur가 공격적 행동 → ego가 회피 반응
    → 적대적 시나리오 자동 생성 가능
```

---

## 6. Training vs Inference

### Training (병렬)
```
1. GT state 16 step → GCN 병렬(batch) → gcn_feats (B, 16, N, 64)
2. build_decoder_tokens() → tokens (B, 16, N, 128)
   GCN(64) → proj(128) + z_global(32) + lw(2) + sem(NC) → Linear → 128 + temporal_pe
3. Layer 0 (causal mask):
   - A2T → A2A → [pred loss 추출] → A2S → [map loss 추출] → FFN
4. Intent 선택 (ego/sur 각각, Layer 0~1 사이)
   concat(token, z_local) → Linear(160→128) → intent loss
5. Layer 1~3: 심화 (A2T → A2A → A2S → FFN 반복, 깊이 3)
6. Output: future 12 step → output head(128→2) → (acc, yaw_rate) → bicycle model → recon loss
7. 모든 loss를 12 step 한번에 계산
```

### Inference (autoregressive)
```
1. Past 4 step → GCN → past 토큰
2. Past 토큰 → Transformer → future step 0 예측
3. 예측 → bicycle model → new state → GCN → 새 토큰
4. Past + future[0] → Transformer → future step 1 예측
5. 반복 12회
```

---

## 7. GCN 역할 (유지, 역할 변경)

```
기존 (redesign_plan.md):
  GCN → history buffer 채우기 → History Attention에서 참조

Transformer 버전:
  GCN → 초기 토큰 생성 (agent interaction 녹아있는 풍부한 feature)
  → Transformer 입력으로 사용
  → A2A가 GCN feature 위에서 "누구에 집중할지" 결정

  Training: GT state 16 step 전부 있으니까 GCN도 한번에 처리 가능
  Inference: 매 step GCN 1회 호출 (autoregressive)

GCN과 A2A의 상호보완:
  GCN: "주변 agent 정보를 섞어준 것" (max pooling, 구분 없이)
  A2A: "누가 중요한지 선택적으로 집중" (attention weight)
  → GCN 없이 A2A만 하면 raw state에서 시작해야 하므로 부담 큼
  → GCN이 풍부한 feature 만들어주면 A2A가 더 정확하게 선택 가능
```

---

## 8. 미결 사항

### 8a. Ego/Sur Transformer weight 분리 (확정)

| Module | 구조 | Phase 2 Freeze |
|--------|------|----------------|
| A2T | self-attention 1개 (구분 없음) | 전체 freeze |
| A2A | K/V 공유 + Q/O 분리 | K/V + sur_Q/O freeze, ego_Q/O 학습 |
| A2S | K/V 공유 + Q/O 분리 | K/V + sur_Q/O freeze, ego_Q/O 학습 |
| FFN | 완전 분리 | sur freeze, ego 학습 |
| Intent | 별도 codebook | sur freeze, ego 학습 |

### 8b. Transformer 하이퍼파라미터 (전부 config에서 조절 가능하게 구현)

```yaml
# configs/train_trafficplanner.cfg 에 추가 예정

# Transformer Decoder
num_decoder_layers: 4        # Layer 수 (기본 4, 최소 2)
trans_d_model: 128           # 모델 차원 (Adv-BMT 참고, GCN 64→128 projection)
trans_nhead: 8               # attention head 수 (128/8=16 per head, Adv-BMT 동일)
trans_ffn_dim: 512           # FFN 중간 차원 (d_model×4, Transformer 관례)
trans_dropout: 0.1           # dropout rate (데이터 적으므로 regularization)
use_ego_z_local: True        # ego z_local on/off (flag, Phase 무관)
use_sur_z_local: False       # sur z_local on/off (flag, Phase 무관)
temporal_pe_type: learnable  # positional encoding (learnable / sinusoidal)
```

참고 (Adv-BMT 설정):
- d_model: 128, nhead: 8, num_layers: 6, dropout: 0.0, FFN: 없음 (MessagePassing)
- 480K data + A6000×8 → dropout 불필요, layer 많아도 과적합 안됨
- 우리: 5000 data + sub-seq → dropout 0.1 필요, layer 4개로 축소

차원 변경으로 인한 projection:
- GCN 출력 (64dim) → Linear(64→128) → Transformer 입력
- Intent codebook: input_dim 128, intent_dim 32 유지
- z_local concat: concat(token(128), z_local(32)) → Linear(160→128)
- Pred head: Linear(128→2)
- Intent CE head: Linear(32→9) (변경 없음)
- Output head: Linear(128→2)

Layer 구조:
- Layer 0: 상황 파악 + aux loss 추출 (pred, map)
- Layer 0~1 사이: Intent 선택 (고정 위치)
- Layer 1 ~ N-1: 심화 (순수 반복, 깊이 = N-1)
- 최종 출력: future 12 step → output head → bicycle model
- **num_layers=2면 최소 구성 (Layer 0 + 심화 1)**
- **num_layers=4면 기본 (Layer 0 + 심화 3, 깊이 3)**

### 8c. Positional Encoding
- Learnable temporal PE (absolute), config에서 선택 가능
- Adv-BMT의 Relative PE / FourierEmbedding은 사용하지 않음 (차별화)
- 16 step이라 absolute PE로 충분

### 8d. A2A mask 추가 실험
- 현재: mask 없이 전체 attend
- 필요 시: scene graph adjacency mask 추가 (코드 한 줄)
- 판단 기준: attention weight 시각화 → 먼 agent에 쓸데없이 attend하면 mask 추가

---

## 9. 구현 순서 (계획)

1. TransformerDecoderLayer 구현 (A2T + A2A + A2S + FFN)
2. Pred head 연결 (A2A 직후 위치)
3. Intent Codebook 연결 (ego + sur, Layer 0~1 사이)
4. Training forward (causal mask, 병렬) 구현
5. Loss 연결 (recon, KL, intent, map guidance, pred — 전부 병렬 계산)
6. Inference forward (autoregressive) 구현
7. 기존 테스트 통과 확인
8. Phase 1 학습 실험
9. A2A attention weight 시각화 → mask 결정

---

## 10. 기존 redesign_plan.md와의 관계

이 문서는 redesign_plan.md의 **Level 3 (Action Generation)** 부분만 교체:
- Level 0 (Scene Understanding): 변경 없음
- Level 1 (z_global): 변경 없음
- Level 2 (z_local Intent Codebook): input_dim 변경 (72→128), **sur에도 확장**
- **Level 3: GRU loop → Transformer decoder**

나머지 (encoder, loss weights, Phase 전략, fine-tuning 전략)는 redesign_plan.md 그대로 유지.
