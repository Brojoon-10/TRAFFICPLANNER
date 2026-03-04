# Encoder & Regularization Debug Plan

> 작성일: 2026-03-04 (최종 수정: 2026-03-04)
> 상태: Issue A~G 수정 완료 / z_global 정규화 + loss 재설계 완료 / 학습 대기
> 기반: redesign_plan.md (v6), redesign_plan_trans_decoder.md

---

## 0. 배경 — 왜 이 작업이 필요했는가

### 학습 실패 현황
- **300 시나리오 학습 결과**: train loss=3.93, val loss=19.72 (**5배 gap**)
- pos_err: train 0.67m, val 5.9m — **순수 과적합**
- 모든 auxiliary loss에서도 train/val gap 존재

### 원인 추적 결과
1. **절대좌표가 MLP에 직접 입력** → 모델이 "좌표 암기" (근본 원인)
2. **Action Blending에 순수 TF 구간 존재** → TF→AR 전환 시 성능 급락

---

## 1. 변경 사항 요약

| # | 변경 | 파일 | 영향 범위 |
|---|------|------|----------|
| **A** | 인코더 GCN: global → local 변환 | trafficplanner_model.py | 인코더 전체 |
| **G** | 디코더 GCN: global → local 변환 (3곳) | trafficplanner_model.py | 디코더 전체 |
| **B1** | Blend: alpha_floor=0.05 도입 | train_trafficplanner.py, .cfg | 학습 스케줄 |
| **B2** | Blend: blend_start_step 제거, 완전 선형 ramp | train_trafficplanner.py | 학습 스케줄 |
| **I** | Intent Codebook: hard=False + inference soft 통일 | trafficplanner_model.py | 추론 일관성 + 적대적 최적화 준비 |
| **D** | recon_pos_weight: 100→1 | .cfg | loss gradient 균형 |
| **E** | z_aux weight: 0.5→0.0 | .cfg | 과적합 원인 제거 |
| **Z1** | dec_past_dropout=0.5 | model.py, train.py, .cfg | z_global 의존 강제 |
| **Z2** | enc_dropout=0.0 | model.py, train.py, .cfg | 인코더 품질 유지 |
| **Z3** | kl_free_bits=0.05 | common.py, loss.py, train.py, .cfg | collapse 방지 |
| **Z4** | KL 스케줄 앞당김 (20→40) | .cfg | z 조기 활성화 |
| **L1** | intent_ce: 0.05→0.005 | .cfg | auxiliary loss 균형 |
| **L2** | map_attn: 0.05→0.005 | .cfg | auxiliary loss 균형 |

---

## 2. [A] 인코더 절대좌표 → Local 변환

### 2a. 문제

원본 STRIVE (`fit_traffic_model.py:471`)는 인코더 MLP 입력 전에 `transform2frame`을 적용:
```python
local_past_kin = transform2frame(scene_graph.past[:, -1, :4], scene_graph.past[:, :, :4])
```

GCN 기반으로 전환할 때 (`fit_traffic_model_trans.py`) 이 변환이 **누락**되었고, 이후 `trafficplanner_model.py`(V5)까지 전파됨.

### 2b. 입력 스케일 비교

| 차원 | global (수정 전) | local (수정 후) | 비고 |
|------|----------------|----------------|------|
| x_norm | **-16.1** | -0.23 | 70x 감소 |
| y_norm | **+27.6** | +0.006 | 4600x 감소 |
| hcos | -0.97 | +1.00 | 비슷 |
| hsin | 0.26 | 0.00 | 비슷 |
| speed_norm | -2.24 | -2.24 | **동일** |
| hdot_norm | 0.40 | 0.40 | **동일** |

**핵심**: x,y가 speed/hdot 대비 **20~30배** 큰 스케일 → MLP gradient가 좌표에 집중 → "이 좌표에서는 이렇게 움직인다" 암기 → 과적합

### 2c. 수정 내용

`_run_temporal_encoder()` (line 1153-1158):
```python
# past[-1]을 기준점으로 전체 trajectory를 local frame 변환
ref_frame = scene_graph.past[:, -1, :4]           # (NA, 4)
local_kin = transform2frame(ref_frame, traj_data[:, :, :4])  # (NA, T, 4)
local_traj = torch.cat([local_kin, traj_data[:, :, 4:]], dim=2)  # speed/hdot 유지

# local_traj → step_feature_extractor(MLP) → GCN node feature
# GCN pos는 global 유지 (edge의 transform2frame은 내부 처리)
```

### 2d. 타당성

- **STRIVE 원본과 동일한 패턴**: `encode_past()`, `encode_future()` 모두 `past[-1]` 기준 local
- **local frame의 의미**: "출발점 대비 어디로 이동했는가" → 궤적 형태 (직진/커브) 인코딩
- **speed, hdot은 변환 불필요**: scalar 값이므로 좌표계와 무관

---

## 3. [G] 디코더 GCN 절대좌표 → Local 변환

### 3a. 문제

디코더 `interaction_gcn`에서도 동일하게 절대좌표가 `mlp_in`에 직접 입력:
```python
# 수정 전 (3곳 모두)
gcn_in = torch.cat([cur_state_6d, cur_lw, cur_sem], dim=-1)  # 6d에 절대좌표!
scene_graph.x = gcn_in  # mlp_in([14, 128, 128, 64])에 직접 입력
```

**STRIVE 원본과의 차이**: STRIVE 디코더 GCN의 `scene_graph.x`에는 좌표가 없음 (latent feature만 입력). 우리 모델은 GCN 기반 디코더로 전환하면서 좌표가 직접 들어가게 됨.

### 3b. 수정 내용 (3곳)

**공통 패턴**:
```python
ref_frame = scene_graph.past[:, -1, :4]  # 인코더와 동일 기준점

# node feature: local 좌표 사용 (MLP 스케일 균형)
local_kin = transform2frame(ref_frame, prev_state[:, :4].unsqueeze(1))[:, 0]
local_6d = _get_6d_state(torch.cat([local_kin, prev_state[:, 4:]], dim=1))
scene_graph.x = torch.cat([local_6d, lw, sem], dim=-1)

# edge feature: global 좌표 유지 (transform2frame 내부 처리, 결과 동일)
scene_graph.pos = cur_state_6d_global[:, :4]
```

수정 위치:
1. `_run_interaction_gcn_parallel()` — TF 모드 디코더 (past + future 16 step)
2. `transformer_decoder_training_blended()` — Blended AR 루프
3. `_run_ar_inference()` — AR sampling 루프 (mult_samp 분기 포함)

### 3c. ref_frame = past[-1] 고정의 타당성

| 선택지 | GCN이 보는 것 | 장단점 |
|--------|-------------|--------|
| **past[-1] 고정** | t=0: (0,0), t=5: (1.3, 0.5), t=11: (2.5, 1.8) | **궤적 형태** 정보 보존. 누적 이동 ~0~3 (적절 스케일) |
| 매 스텝 prev_state | 항상 (~0.2, ~0.0) | 궤적 구분 불가. speed/hdot과 **정보 중복** |

- 인코더도 past[-1] 고정 → 일관성
- STRIVE 원본도 인코더에서 past[-1] 고정 (디코더는 GCN에 좌표 미사용이라 해당 없음)

### 3d. 건드리지 않은 것과 그 이유

| 컴포넌트 | 현재 | 변경? | 이유 |
|----------|------|------|------|
| `scene_graph.pos` | global | **유지** | edge의 transform2frame 결과 동일 (수학적 검증 완료, 1e-4 오차) |
| `accum_states` (A2A) | global | **유지** | pairwise 차이만 사용 → 결과 동일 (5e-4 오차) |
| `prev_state` (bicycle) | global | **유지** | state propagation + GT 비교에 필수 |
| Map recrop | global | **유지** | `get_map_crop_pos(pos_unnorm)` → global 위치로 전역 맵에서 crop |

---

## 4. 좌표계 전체 정리

### 4a. 세 가지 좌표 개념

| 좌표계 | 정의 | 예시 |
|--------|------|------|
| **Global (절대)** | 월드 원점 기준 | x=500, y=-200 |
| **Local** | 특정 기준점(자기 자신) 기준 | "내 앞 3m, 왼쪽 1m" |
| **Relative** | 다른 에이전트와의 차이 | "agent_j가 나보다 5m 앞" |

### 4b. 컴포넌트별 좌표계 매핑

```
┌──────────────────────────────────────────────────────────────────┐
│                     COORDINATE FRAME MAP                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [ENCODER]                                                         │
│  step_feature_extractor(MLP)  ← LOCAL (past[-1] 기준)    ✅ 수정  │
│  temporal_gcn_encoder.pos     ← GLOBAL (edge용)          유지     │
│  Positional Encoding          ← TIME INDEX (좌표 무관)    N/A     │
│                                                                    │
│  [DECODER - GCN]                                                   │
│  interaction_gcn.mlp_in       ← LOCAL (past[-1] 기준)    ✅ 수정  │
│  interaction_gcn.pos          ← GLOBAL (edge용)          유지     │
│                                                                    │
│  [DECODER - Transformer]                                           │
│  A2T (Temporal Self-Attn)     ← LEARNED TOKENS (좌표 무관)        │
│  A2A (Agent Self-Attn)        ← LEARNED TOKENS + RelBias          │
│    └ A2A RelBias              ← GLOBAL pairwise diff     유지     │
│  A2S (Map Cross-Attn)         ← Q: token, K/V: CNN feat           │
│    └ Map tokens               ← LOCAL (agent-centered crop)       │
│    └ Map soft label           ← LOCAL (transform2frame 내부)      │
│  A2Z (z Cross-Attn)           ← LATENT (좌표 무관)                │
│  AdaLN                        ← z_global conditioning              │
│  PE (temporal)                ← TIME INDEX (nn.Embedding)          │
│                                                                    │
│  [OUTPUT]                                                          │
│  Bicycle model prev_state     ← GLOBAL (state propagation)  필수  │
│  traj_out (pred)              ← GLOBAL (GT와 비교)          필수  │
│  Map recrop                   ← GLOBAL (전역 맵 crop 위치)  필수  │
│                                                                    │
│  [LOSS]                                                            │
│  Recon MSE                    ← GLOBAL pred vs GLOBAL GT          │
│  Map Attn KL                  ← LOCAL soft label (자체 변환)      │
│  Intent CE                    ← acc/hdot 기반 (좌표 무관)         │
│  z_aux                        ← action 기반 (좌표 무관)           │
│  Sur/Ego Pred                 ← GLOBAL delta (차이만 사용)        │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 4c. GCN 내부 구조와 좌표 흐름

```
scene_graph.x  ──→  mlp_in([14, 128, 128, 64])  ──→  node feature (64dim)
                     ↑ LOCAL 좌표 (수정 후)              ↓ latent (좌표 아님)
                                                          ↓
scene_graph.pos ──→  message():                    ──→  edge feature
                     transform2frame(pos_i, pos_j)        ↓ relative (항상 안전)
                     ↑ GLOBAL 좌표 (유지)                 ↓
                                                          ↓
                                                    update() → aggregation
                                                          ↓
                                                    GCN output (64dim) ← latent
```

- `mlp_in`: 좌표가 직접 들어가는 유일한 곳 → **여기만 local로 변환**
- `message()`: `transform2frame`이 내부에서 상대 변환 → global이든 local이든 결과 동일
- GCN output: 이미 추상화된 latent → 이후 Transformer에서 안전하게 사용

---

## 5. [B] Action Blending 개선

### 5a. 문제

기존 구조:
```
step 0 ──────── blend_start_step ──────── anneal_end
 │                    │                        │
 α=0 (순수 TF)       α=0→target (ramp)        α=target
```

**순수 TF 구간 (α=0)** 에서 모델이 GT에 완전 의존 → AR 전환 시 성능 급락 (cliff)

### 5b. 수정 내용

```
step 0 ──────────────────────────────── anneal_end
 │                                          │
 α=floor(0.05) ──── 선형 ramp ────→ α=target(1.0)
```

- `blend_alpha_floor = 0.05`: step 0부터 5% 예측 action 혼합
- `blend_start_step` 제거: 순수 TF 구간 없음
- 완전 선형 ramp: `α = floor + (target - floor) × min(step / anneal_steps, 1.0)`

### 5c. 타당성

- **Scheduled Sampling 원리**: 학습 초기부터 소량의 자기 예측에 노출 → AR 전환 충격 완화
- floor=0.05는 학습 안정성에 거의 영향 없음 (95% GT) 하면서 AR 준비
- `blend_anneal_steps=65000`: 300 data × batch 2 ≈ 150 step/epoch → ~430 epoch에 걸쳐 점진 전환

---

## 6. [I] Intent Codebook (z_local) 수정 — hard=False + Inference Soft 통일

### 6a. z_local의 역할

z_local은 **step별 행동 의도**(감속/가속/좌회전/우회전 등)를 인코딩하는 latent:
- **Intent predictor**: Transformer Layer 0 출력 token → MLP → logits (9dim = 3acc × 3yaw)
- **Codebook**: 9개 prototype 벡터(32dim), acc/yaw 격자 [-2, 0, 2] 조합
- **z_local**: logits → softmax weights → codebook 가중합 (32dim)
- z_local은 Layer 1 이후 Transformer token에 concat → 이후 output head에서 action 예측

### 6b. 문제 1 — hard=True의 gradient 문제

```python
# 수정 전
intent_weights = F.gumbel_softmax(logits, tau=temperature, hard=True)
# hard=True: forward에서 one-hot, backward에서 soft gradient (straight-through)
```

**hard=True 문제점**:
- Forward pass에서 **argmax → one-hot** 적용 → codebook 벡터 1개만 선택
- Backward에서는 soft gradient 사용 (straight-through estimator)
- **Forward/backward 불일치**: 학습이 soft gradient로 진행되지만 실제 출력은 discrete
- codebook 9개 벡터 사이의 **보간(interpolation) 불가** → 표현력 제한
- 예: "약간 좌회전하면서 감속" = [0.3, 0.7, 0, ...] 같은 soft weight이 유용하지만 hard에서는 불가능

### 6c. 문제 2 — Train/Inference 불일치

```python
# 수정 전
if self.training:
    intent_weights = F.gumbel_softmax(logits, tau=temperature, hard=True)  # hard one-hot
else:
    intent_weights = F.one_hot(logits.argmax(-1), K).float()  # hard one-hot (Gumbel noise 없음)
```

- Train: Gumbel noise + hard one-hot → **확률적으로 다른 intent 탐색** (exploration)
- Inference: 순수 argmax → 항상 **최대 확률 intent만 선택** (exploitation)
- **Gap**: 학습에서는 noise로 다양한 intent 경험, 추론에서는 가장 확률 높은 것만 → 불일치

### 6d. 수정 내용

```python
# 수정 후
if self.training:
    intent_weights = F.gumbel_softmax(logits, tau=temperature, hard=False)  # soft weights
else:
    intent_weights = F.softmax(logits / temperature, dim=-1)  # soft (Gumbel noise만 제거)
```

**핵심 변경 2가지**:
1. **hard=True → hard=False**: codebook 가중합이 연속적 → 보간 가능, gradient 일관성 확보
2. **Inference one_hot → softmax**: train/inference 모두 soft weights → 분포 일관성 확보

### 6e. 기대 효과

- **표현력 향상**: 9개 prototype의 convex combination → 격자 내부 전체 연속 커버
  - acc ∈ [-2, 2], yaw ∈ [-2, 2] 범위에서 연속적 의도 표현
- **Train/val gap 감소**: forward/backward 일관 + train/inference 일관
- **적대적 최적화 준비**: soft weights → δ_logits perturbation이 smooth gradient chain 형성 (Section 7)

---

## 7. 적대적 시나리오 최적화 계획

> **시점**: 모델 학습 완료 후 적용. 학습된 weight는 고정(frozen)하고, latent를 탐색하여 위험 시나리오 생성.

### 7a. 목적

학습된 trajectory prediction 모델을 활용하여:
- **위험 시나리오 자동 생성**: ego-sur 충돌 근접 시나리오 탐색
- **모델 robustness 검증**: 극단적 latent에서의 예측 품질 평가
- **CARLA 테스트 시나리오 생성**: 실제 시뮬레이터에서 재현 가능한 시나리오 도출

### 7b. 최적화 대상

| 변수 | 차원 | 역할 | 최적화 방식 |
|------|------|------|------------|
| **z_global** | (NA, 32) | 전체 trajectory 방향 (직진/좌/우회전) | continuous → gradient 직접 |
| **δ_logits** | (FT, N_ego, 9) | step별 intent perturbation | logits + δ → softmax (미분 가능) |

#### z_global: 메인 최적화 변수
- CVAE posterior에서 샘플링된 latent vector
- continuous 32dim → Adam 등으로 자유롭게 gradient 최적화
- prior 분포(N(0,1))에서 벗어나면 비현실적 → KL penalty로 제약

#### δ_logits: step별 행동 반응 조작
- intent_predictor 출력 logits에 **perturbation δ** 가산
- `logits_perturbed = logits + δ` → softmax → codebook 가중합 → z_local
- **Section 6의 hard=False 수정이 핵심**: soft weight이어야 δ의 gradient가 smooth
- codebook 9개 벡터의 convex combination → 격자 내부 전체 커버 (acc/yaw ∈ [-2, 2])

### 7c. 최적화 흐름

```
입력: 학습된 모델 M (frozen), 시나리오 scene_graph

최적화 변수:
  z_global: (NA, 32)         ← learnable (init: prior sample)
  δ_logits: (FT, N_ego, 9)   ← learnable (init: zeros)

Forward:
  1. z_global → AdaLN conditioning + A2Z cross-attention
  2. Layer 0 → intent_predictor → logits
  3. logits + δ_logits → softmax(hard=False) → codebook weights → z_local
  4. z_local concat → Layer 1 → output head → bicycle model → trajectory

Loss (최소화):
  L_collision  : ego-sur 최소 거리 (min over steps)     — 충돌 유도
  L_plausibility: KL(z_global ‖ prior)                   — 현실성 제약
  L_smooth     : Σ‖δ_logits[t+1] - δ_logits[t]‖²        — 급격한 intent 전환 방지
  L_δ_reg      : ‖δ_logits‖²                              — 극단적 perturbation 방지

  total = L_collision + λ₁·L_plausibility + λ₂·L_smooth + λ₃·L_δ_reg

Gradient chain:
  ∂L/∂z_global: L → traj → bicycle → action → output_head → transformer → AdaLN/A2Z → z_global  ✅
  ∂L/∂δ_logits: L → traj → bicycle → action → output_head → transformer → z_local → softmax → logits+δ  ✅
  (softmax, codebook matmul 모두 미분 가능 → gradient chain 정상)
```

### 7d. hard=False가 필수인 이유

```
hard=True 경로:  δ → logits → argmax(one-hot) → codebook[argmax]
                               ↑ 미분 불가 (straight-through은 근사)

hard=False 경로: δ → logits → softmax(continuous) → Σ w_i × codebook_i
                               ↑ 미분 가능 (smooth gradient)
```

- hard=True에서는 δ의 작은 변화 → argmax 결과 불변 → gradient ≈ 0 (plateau)
- hard=False에서는 δ의 작은 변화 → softmax weights 연속 변화 → smooth gradient
- **적대적 최적화의 전제조건**: intent 경로 전체가 미분 가능해야 함

### 7e. 주의사항

- δ_logits L2 regularization: 극단적 δ → softmax가 one-hot에 가까워짐 → 실질적 hard 복귀
- z_global prior 제약: KL penalty 없으면 비현실적 궤적 (물리적으로 불가능한 이동)
- codebook 다양성: 학습 후 codebook 9개 벡터가 충분히 분산되었는지 확인 필요
- **bicycle model 물리 제약**: 출력이 bicycle model을 통과하므로 비물리적 궤적은 자동 차단

---

## 8. 검증 결과

### 8a. GCN edge / A2A RelBias 동일성 (수학적 + 실험적 검증)

```python
# GCN edge: transform2frame(pos_i, pos_j)
# global frame과 common local frame에서 결과 비교
Edge rel (global): [-0.6700, -11.1600, 0.8660, -0.5000]
Edge rel (local):  [-0.6700, -11.1595, 0.8660, -0.5000]
Max difference: 4.9e-4  ✅ floating point 수준

# A2A RelBias: 8개 pairwise feature
A2A 8-feat (global): [-0.67, -11.16, 11.18, 1.93, -4.00, -3.88, 10.0, -0.52]
A2A 8-feat (local):  [-0.67, -11.16, 11.18, 1.93, -4.00, -3.88, 10.0, -0.52]
Max difference: 4.9e-4  ✅ floating point 수준
```

### 8b. Loss 영향 검증

| Loss | 입력 소스 | 좌표 영향 | 결론 |
|------|----------|----------|------|
| Recon MSE | `traj_out` = bicycle model global 출력 | **없음** | ✅ |
| pos_mse / head_mse | 위와 동일 | **없음** | ✅ |
| KL divergence | prior/posterior z latent | 좌표 무관 | ✅ |
| Map Attn KL | attn weights vs soft label (자체 transform2frame) | **없음** | ✅ |
| Intent CE | z_local → acc/yaw prototypes | 좌표 무관 | ✅ |
| z_aux | z_aux_traj vs GT actions | 좌표 무관 | ✅ |
| Sur/Ego Pred | A2A output → position delta | **없음** | ✅ |
| Potential | `pred_future` (global) | **없음** | ✅ |

**핵심**: 수정은 GCN `mlp_in` 입력만 local로 변경. GCN 출력은 latent feature → Transformer → output head → action → bicycle model → **global `traj_out`**. Loss가 보는 모든 값은 global 유지.

### 8c. Bicycle model global 유지 확인

```python
# Blended AR 루프 (line 1704):
cur_state_global, _, cur_bike_state = self._apply_dynamics(
    blended_action, prev_state, ...)  # prev_state = GLOBAL

# prev_state 업데이트 (line 1710):
prev_state = cur_bike_state  # GLOBAL 유지

# local 변환은 prev_state를 "읽기만" — 원본 수정 없음
local_kin = transform2frame(ref_frame, prev_state[:, :4].unsqueeze(1))[:, 0]
```

---

## 9. Issue C: CARLA_NORM_STATS / BIKE_PARAMS — ✅ 7798 데이터 최종 검증 완료

### 9a. 문제 발견

기존 `verify_norm_final.py`가 **전체 trajectory 연속 프레임 + 가변 dt**로 acc/ddh 통계를 계산했으나,
모델의 `_compute_gt_actions()`는 **학습 subsequence future 구간 + dt=0.5 고정**으로 계산.

두 방식의 분포가 다르므로, 기존 stats로 정규화하면 mean≠0, std≠1이 됨.

### 9b. 왜 모델 방식(subsequence)이 정답인가

정규화 통계의 목적 = **모델이 실제로 보는 값을 mean=0, std=1로 만드는 것**.

- `_compute_gt_actions()`: GT action 생성 → `norm_acc = (raw_acc - a_mean) / a_std`
- `_apply_dynamics()`: 모델 출력 역정규화 → `a_out = pred * a_std + a_mean`

두 함수가 **동일한 stats를 공유**하므로, 이 stats는 `_compute_gt_actions()`가 보는 분포에서 구해야 함.

### 9c. 검증 과정

1. `verify_norm_final.py`를 xlsx 직접 읽기로 재작성 (FITDataset 복잡한 의존성 제거)
2. **300 데이터로 FITDataset 직접 로드 결과와 비교 → 6자리까지 완전 일치** 확인
3. **7798 데이터** (`various_driving_data_20260224`) 전체로 최종 산출

### 9d. 최종값 (7798 데이터, train 85% = 6628 files, 38326 subsequences)

| 항목 | 최종값 (mean, std) |
|------|--------|
| speed | (6.225233, 2.673987) |
| hdot | (0.051012, 0.233765) |
| acc | (0.252031, 1.923014) |
| ddh | (0.003361, 0.371728) |
| l, w | (4.9017, 0.001), (2.1283, 0.001) |
| lscale | (0.0, 15.0) — 변경 불필요 (Section 9e 참조) |
| h | (0.0, 1.0) |

수정 파일: `utils.py` (CARLA_BIKE_PARAMS + CARLA_NORM_STATS), `fit_dataset.py` (scenario_path → 7798 데이터)

### 9e. lscale = (0.0, 15.0) 검증 — 변경 불필요

#### lscale의 역할
- state 6차원 `(x, y, hx, hy, speed, hdot)` 중 **x, y만** 정규화: `norm_x = (x - 0) / 15`
- Dataset에서 global 좌표에 바로 적용 (스케일링만, mean shift 없음)
- 모델 내부에서 `transform2frame`으로 local 변환 시 상대 변위가 `/15`로 스케일링됨

#### mean=0 필수 이유
- NVIDIA 원본 5회 강조: "must have mean 0! Make heavy use of this assumption in model"
- position은 `transform2frame` 뺄셈으로 상대 변위가 됨 → global offset을 빼면 안 됨
- CARLA 주행 편향 (future ly mean=-4.3m)은 모델이 학습할 패턴이지 정규화로 제거할 값이 아님

#### std=15의 유래: transform2frame 후 local 좌표의 전체 std

CARLA 7798 데이터 실측 (6628 train, past[-1] 기준 local 변환, 전체 16 step):

| | mean | std | p5 | p95 |
|---|---|---|---|---|
| local_x (종방향) | 0.46m | 14.40m | -24.7m | 25.8m |
| local_y (횡방향) | -3.04m | 15.67m | -32.9m | 23.0m |
| **x+y 전체** | -1.29m | **15.15m** | — | — |

→ **전체 std = 15.15 ≈ 15.0** — NVIDIA가 nuScenes에서 산출한 값과 거의 동일
→ std=15로 normalize 후: past p95=0.80, future dist p95=3.01 (nuScenes와 유사한 분포)

---

## 10. 미해결 이슈 (별도 추적)

| Issue | 상태 | 설명 |
|-------|------|------|
| B | ✅ 검증 완료 | lscale=(0.0, 15.0) — CARLA에서도 std≈15 확인, 변경 불필요 |
| C | ✅ 검증 완료 | CARLA_NORM_STATS / BIKE_PARAMS — 7798 데이터 최종 검증 완료 |
| D | ✅ 수정 완료 | recon_pos_weight: 100→1 (pos/head gradient 1.3:1 균형) |
| E | ✅ 수정 완료 | z_aux weight: 0.5→0.0 (10.1x 과적합, NVIDIA 원본에도 없음) |
| F | 확인됨 | 300 data 과적합 — 7798 데이터로 전환 완료 |

---

## 11. 파일 변경 목록

### trafficplanner_model.py
- `_run_temporal_encoder()`: local frame transform 추가 (Issue A)
- `_run_interaction_gcn_parallel()`: local frame transform 추가 (Issue G)
- `transformer_decoder_training_blended()`: AR 루프 GCN local transform (Issue G)
- `_run_ar_inference()`: AR 루프 GCN local transform + mult_samp (Issue G)
- `IntentCodebook.forward()`: hard=True→False + inference one_hot→softmax (Issue I)
- `__init__`: `enc_dropout`, `dec_past_dropout` 파라미터 추가 (Section 15)
- `PositionalEncoding`: `dropout=enc_dropout` 전달
- `_build_decoder_tokens()`: `dec_past_dropout` 적용

### losses/common.py
- `kl_normal()`: `free_bits` 파라미터 추가, `torch.clamp(kl - free_bits, min=0)` (Section 15)

### losses/trafficplanner_loss.py
- `__init__`: `kl_free_bits` 파라미터 저장, KL 호출 시 전달 (Section 15)

### datasets/utils.py
- `CARLA_BIKE_PARAMS`: a_stats/ddh_stats — 7798 데이터 최종값 (Issue C)
- `CARLA_NORM_STATS`: speed/hdot/a/ddh — 7798 데이터 최종값 (Issue C)
- `lscale`: (0.0, 15.0) 유지 확인 (Issue B)

### datasets/fit_dataset.py
- `scenario_path`: `various_300_sample` → `various_driving_data_20260224` (7798 데이터)

### verify_norm_final.py
- xlsx 직접 읽기 + `_compute_gt_actions()` 동일 방식으로 전면 재작성 (Issue C)

### train_trafficplanner.py
- `blend_alpha` 계산: floor 기반 선형 ramp (Issue B1, B2)
- `blend_start_step` 비활성화
- argparse: `kl_free_bits`, `enc_dropout`, `dec_past_dropout` 추가 (Section 15)
- model 생성: `enc_dropout`, `dec_past_dropout` 전달
- loss 생성: `kl_free_bits` 전달

### train_trafficplanner.cfg
- `blend_alpha_floor: 0.05` 추가
- `blend_anneal_steps: 65000`
- `blend_target_alpha: 1.0`
- `recon_pos_weight: 100 → 1` (Section 13)
- `loss_z_aux: 0.5 → 0.0` (Section 14)
- `loss_intent_ce: 0.05 → 0.005` (Section 16)
- `loss_map_attn: 0.05 → 0.005` (Section 16)
- `enc_dropout: 0.0` 추가 (Section 15)
- `dec_past_dropout: 0.5` 추가 (Section 15)
- `kl_free_bits: 0.05` 추가 (Section 15)
- `kl_start_epoch: 50 → 20` (Section 15)
- `kl_anneal_end: 65 → 40` (Section 15)

---

## 12. 기대 효과

1. **과적합 감소**: 좌표 암기 → 행동 패턴 학습으로 전환. train/val gap 축소 예상
2. **TF-AR gap 축소**: 학습 초기부터 AR 노출 → 전환 충격 없음
3. **Inference 일관성**: Intent soft weights (hard=False) 통일 → val 성능 안정화
4. **적대적 최적화 준비**: z_global + δ_logits gradient chain 완전 미분 가능
5. **z_global 활용도 향상**: dec_past_dropout으로 z 의존 강제 + free bits로 collapse 방지
6. **Loss gradient 균형**: pos/head 1.3:1, auxiliary loss 지배 해소 → 안정적 학습

---

## 13. [D] recon_loss 구조 분석 & pos_weight 수정

### 13a. log_normal 상수항 문제

```
recon_loss = -log_normal(pred, gt, var=1) + pos_weight × pos_mse
           = Σ_{4dim} [0.5×ln(2π) + 0.5×(pred-gt)²] + pos_weight × pos_mse
```

- 상수항: `4 × 0.5 × ln(2π)` = **3.676** — loss 값의 ~96%가 학습 불가능한 상수
- 수렴 시 recon ≈ 3.83 → 학습 가능한 부분 = 0.15 (4dim MSE/2 + pos_weight×pos_mse)
- **gradient에는 영향 없음** (상수 미분 = 0), 하지만 loss 모니터링 시 "학습이 안 되는 것처럼" 보임

### 13b. pos_weight gradient 분석

정규화 스케일 차이로 position과 heading의 raw MSE 크기가 다름:
- position: `/15`로 정규화 → MSE ~`(Δm/15)²` ≈ 작음
- heading: (cos,sin) 범위 [-1,1] → MSE ~`Δ²` ≈ 큼

| pos_weight | position gradient 비율 | heading gradient 비율 | 비율 |
|------------|----------------------|---------------------|------|
| 100 | 155 | 1 | position 압도 |
| 7 | 8 | 1 | position 우세 |
| 1 | 1.3 | 1 | **근사 균형** |
| 0 | ~0.004 | 1 | heading 압도 + recon gradient 미약 |

### 13c. 수정

`recon_pos_weight: 100 → 1`
- position과 heading gradient가 1.3:1로 거의 균형
- 순수 `var=1 log_normal`이 4dim 전체에 균등한 gradient 제공

---

## 14. [E] z_aux_loss 비활성화

### 14a. 과적합 진단

이전 학습 로그 분석 (`train_trafficplanner_out_tf_z_local_trans_z_global_enhancing_debugged`, 300 epoch):

| 항목 | Train | Val | Gap |
|------|-------|-----|-----|
| z_aux | 0.092 | 0.919 | **10.1x** |
| KL | 9.8→130→2.3 | — | posterior collapse 패턴 |
| recon | 3.83 | 19.15 | 5.0x |

- z_aux MLP `(64→128→128→2)`가 학습 데이터 300개를 **독립적으로 암기**
- z_global을 통해 encoder에 잘못된 gradient 전파 (action 예측 방향으로 편향)

### 14b. NVIDIA STRIVE 원본 비교

NVIDIA 원본: z에 대한 auxiliary loss **없음**. KL divergence (β=0.004) + reconstruction만.
- z는 decoder가 "필요해서" 사용하는 구조 (decoder 능력 제한 → z 의존)
- 별도 MLP로 z를 직접 감독하면 encoder가 "action 예측기"로 치우침

### 14c. 수정

`loss_z_aux: 0.5 → 0.0` (코드 유지, weight만 0)
- z_aux MLP 구조는 향후 모니터링용(detach 후)으로 재활용 가능

---

## 15. z_global 정규화 전략

### 15a. 문제 — Posterior Collapse

KL 패턴: 9.8 → 130 → 2.3
1. **초기(9.8)**: posterior가 prior에 가까움 (아직 정보 미인코딩)
2. **중간(130)**: posterior가 정보를 담으려 시도 (KL 급증)
3. **후기(2.3)**: decoder가 z 없이도 해결 가능 → posterior collapse (KL → 0 방향)

**근본 원인**: Transformer decoder + GCN + map attention이 너무 강력 → z 없이 문제 해결 가능

### 15b. 전략 — 4가지 수단

#### (1) dec_past_dropout = 0.5
- GCN에서 나오는 past context feature에 Dropout(0.5) 적용 (training only)
- decoder가 past context에 의존하지 못하게 만듦 → z_global에 의존 강제
- 구현: `_build_decoder_tokens()`에서 `gcn_feats`에 적용

#### (2) enc_dropout = 0.0 (PE dropout 제거)
- 인코더 PositionalEncoding의 dropout을 0으로 설정
- local 좌표 전환으로 과적합 위험 감소 → PE dropout 불필요
- dropout이 인코더 출력 품질을 저하시킬 수 있으므로 제거

#### (3) KL free bits = 0.05
- per-dim KL에 free nats 허용: `kl_dim = max(kl_dim - 0.05, 0)`
- 각 latent dim이 0.05 nats까지 정보를 저장해도 penalty 없음
- 32-dim × 0.05 = 최대 1.6 nats까지 penalty-free
- **효과**: posterior collapse 초기에 "어느 정도 정보를 담아도 된다" → collapse 방지

```
수치 예시 (per-dim avg KL ≈ 0.07 기준):
  dim_i KL = 0.03 → max(0.03 - 0.05, 0) = 0    → penalty 없음 (보호)
  dim_i KL = 0.12 → max(0.12 - 0.05, 0) = 0.07  → 0.07만 penalty
  dim_i KL = 0.00 → max(0.00 - 0.05, 0) = 0    → 이미 collapse된 dim 보호
```

#### (4) KL 스케줄 앞당김
- `kl_start_epoch: 50 → 20`: KL annealing 시작을 앞당김
- `kl_anneal_end: 65 → 40`: 최대 KL weight 도달 시점 앞당김
- **이유**: reconstruction이 충분히 학습된 후에 KL을 켜면 이미 z를 안 쓰는 경로가 형성됨

### 15c. 수정 파일

| 파일 | 변경 |
|------|------|
| `common.py` | `kl_normal()` — `free_bits` 파라미터 추가 |
| `trafficplanner_loss.py` | `__init__` — `kl_free_bits` 파라미터 저장, KL 호출 시 전달 |
| `trafficplanner_model.py` | `__init__` — `enc_dropout`, `dec_past_dropout` 파라미터 추가 |
| | `PositionalEncoding` — `dropout=enc_dropout` 전달 |
| | `_build_decoder_tokens()` — `dec_past_dropout` 적용 |
| `train_trafficplanner.py` | argparse — 3개 파라미터 추가, model/loss 생성 시 전달 |
| `train_trafficplanner.cfg` | 6개 값 변경 (위 참조) |

---

## 16. Loss Weight 재설계

### 16a. 기존 문제

이전 학습 수렴 시점 gradient 기여도 분석:

| Loss | Weight | 값 | 총 loss 비중 | gradient 기여 |
|------|--------|-----|------------|--------------|
| recon | 1.0 | 3.83 | **98.5%** | 미약 (상수항 96%) |
| pos_mse | ×100 | ~0.003 | 0.3 | position에 집중 |
| KL | 0.004 | ~0.04 | 0.2% | 적절 |
| intent_ce | 0.05 | ~0.01 | — | **65.9%** (수렴 시 지배) |
| map_attn | 0.05 | ~0.01 | — | intent와 유사 |
| z_aux | 0.5 | 0.09 | — | 과적합 원인 |

**핵심 문제**: pos_weight=100이 position gradient를 155x로 증폭 + intent/map이 수렴 후 gradient 지배

### 16b. 수정 후

| Loss | Weight | 비고 |
|------|--------|------|
| recon | 1.0 | 유지 |
| recon_pos_weight | **1** | 100→1, pos/head 1.3:1 균형 |
| loss_kl | 0.004 | 유지 |
| loss_z_aux | **0.0** | 0.5→0, 비활성화 |
| loss_intent_ce | **0.005** | 0.05→0.005, 10x 감소 |
| loss_map_attn | **0.005** | 0.05→0.005, 10x 감소 |
| loss_sur_pred | 0.1 | 유지 |
| loss_ego_pred | 0.1 | 유지 |

---

## 17. 다음 단계

1. **학습 실행**: 7798 데이터로 학습 → train/val gap, KL 패턴, free bits 효과 모니터링
2. **blend 스케줄 검토**: 7798 data × batch 2 = 3899 step/epoch → 65000 step ≈ epoch 17에 α=1.0 도달 (조정 검토)
3. **적대적 최적화 구현** (Section 7): 학습 완료 후, z_global + δ_logits 탐색 스크립트 작성
