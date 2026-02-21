# System Architecture: Hierarchical Conditional Latent Planner

본 시스템은 주행의 전역적 의도(Global Intention)를 유지하면서, 동적 객체와의 상호작용 및 도로 제약에 대한 국소적 반응(Local Reaction)을 생성하는 계층적 구조를 갖는다. 전체 시스템은 **Frozen Block**과 **Trainable Block**으로 이원화되어 있다.

---

## 1. Input Specification (입력 데이터 명세)

Reaction Encoder는 다음 세 가지 핵심 요소를 결합(Concatenation)하여 입력으로 받는다. 단순 관측값이 아닌 **시계열적 맥락과 의도**를 포함한다.

### A. Ego Temporal Context

Ego 차량의 단순 현재 상태가 아닌, **GRU의 Hidden State**를 사용한다.

| 항목 | 설명 |
|------|------|
| **Source** | Ego Encoder (GRU)의 마지막 은닉 상태 |
| **Information** | 현재 위치/속도 + 과거 궤적 패턴 (회피 중인지, 복귀 중인지, 정상 주행 중인지) |
| **Role** | 현재 상태만으로는 알 수 없는 "현재 기동 상태"를 인지 |

### B. Surrounding Temporal Context

주변 차량의 단순 위치가 아닌, **Interaction Network(GRU)의 Hidden State**를 사용한다.

| 항목 | 설명 |
|------|------|
| **Source** | Interaction Encoder (GRU)의 마지막 은닉 상태 |
| **Information** | 위치, 속도, 가속도, 각속도 등 고차원 동역학 정보와 이동 패턴이 잠재되어 있음 |
| **Role** | 순간적인 위치뿐만 아니라 접근 속도(Momentum)와 주행 패턴을 인지하여 동적 충돌 위험을 예측 |

### C. Global Intention Vector

Phase 1(Global Planner)에서 생성된 **Frozen Latent Vector**이다.

| 항목 | 설명 |
|------|------|
| **Role** | Reaction Encoder에게 "원래 의도했던 주행 스타일과 목표"를 기준점(Reference)으로 제공 |
| **Usage** | 회피 후 복귀해야 할 목표 궤적을 정의하는 데 필수적 |

---

## 2. Latent Space Disentanglement (잠재 공간의 분리)

주행 정책은 세 가지 독립적인 잠재 변수(Latent Variable)의 선형 결합으로 표현된다.

### A. Global Intention (z_intent)

| 항목 | 설명 |
|------|------|
| **성격** | 확률적(Probabilistic), CVAE 기반 |
| **역할** | 장애물이 없는 이상적인 상황에서의 명목 경로(Nominal Path) 생성 |
| **상태** | **Frozen** (학습되지 않음). Phase 1에서 사전 학습된 가중치를 고정하여 사용 |

### B. Local Reaction (z_avoid, z_recover, α, β)

| 항목 | 설명 |
|------|------|
| **성격** | 결정론적(Deterministic), Auto-regressive Encoder 기반 |
| **역할** | 충돌 회피 및 경로 복귀를 위한 물리적 제어 벡터 생성 |
| **상태** | **Trainable** (학습됨) |

**구성 요소:**

| 변수 | 역할 |
|------|------|
| `z_avoid` | 충돌 회피를 위한 방향성 벡터 |
| `z_recover` | 경로 이탈 시 복귀를 위한 방향성 벡터 |
| `α, β` | 각 잠재 변수의 개입 시점과 강도를 조절하는 Gating scalar |

---

## 3. Decoder & Action Fusion (행동 합성)

매 시간 스텝 t에서 최종 제어 입력(Action)은 다음과 같은 **Gated Additive Formulation**을 따른다.

```
a_t = Dec(z_intent) + α_t · Dec(z_avoid) + β_t · Dec(z_recover)
```

여기서 `Dec(·)`는 Latent Vector를 물리적 제어량(acceleration, steering 등)으로 변환하는 MLP이다.

| 변수 | 역할 |
|------|------|
| `α` | 회피 모듈 활성화 계수 |
| `β` | 복귀 모듈 활성화 계수 |

---

## 4. Mathematical Formulation: Potential Field-based Loss

학습 시 Ground Truth(GT) 경로가 존재하지 않는 회피 시나리오를 다루기 위해, 물리적 제약 조건을 에너지 함수(Energy Function)로 정의하고 이를 최소화하는 **포텐셜 필드(Potential Field)** 방식을 적용한다.

손실 함수(Loss Function)는 시간 t=1부터 H(Prediction Horizon)까지의 누적합으로 정의된다.

```
L_total = Σ_{t=1}^{H} [ L_safety(t) + L_recovery(t) + L_sparsity(t) ]
```

### 1. Safety Potential (L_safety): 회피 목적

이 항목은 `z_avoid`와 `α`의 학습을 유도하는 **척력(Repulsive Force)**이다.

#### A. Dynamic Obstacle Potential (주변 차량)

Ego와 주변 차량(Sur) 사이의 거리가 안전 거리(d_safe)보다 가까워질수록 손실이 급증한다.

```
L_collision = W_coll · max(0, d_safe - ||p_ego,t - p_sur,t||)
```

#### B. Static Map Potential (차선 제약)

도로의 구조적 특성에 따라 다른 가중치를 부여한다.

| 차선 유형 | 제약 강도 | 설명 |
|----------|----------|------|
| **실선 (Solid Line)** | W_solid (높음) | 절대 넘으면 안 되는 벽 |
| **점선 (Dashed Line)** | W_dashed (낮음) | 필요시 넘을 수 있는 연성 제약 |

```
L_map = W_line · max(0, margin - d_line)
```

### 2. Recovery Potential (L_recovery): 복귀 목적

이 항목은 `z_recover`와 `β`의 학습을 유도하는 **인력(Attractive Force)**이다.

GT 데이터가 없으므로, **Frozen Encoder가 생성한 명목 경로(p_nominal)**를 기준점(Reference)으로 설정한다.

```
L_recovery = W_recon · ||p_predicted,t - p_nominal,t||^2
```

**핵심 논리:** p_nominal은 장애물이 없을 때의 이상적인 주행 경로이므로, 회피 기동이 끝난 후 Ego가 수렴해야 할 목표 지점이 된다.

### 3. Sparsity Regularization (L_sparsity): 개입 최소화

불필요한 상황에서 α, β가 활성화되는 것을 억제하여 주행 안정성을 확보한다.

```
L_sparsity = W_reg · (|α_t| + |β_t|)
```

---

## 5. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FROZEN BLOCK                             │
│  ┌─────────────┐                                                │
│  │   Prior     │──→ z_intent ──→ MLP_base ──→ a_base           │
│  │  Encoder    │     (정적)       (frozen)     (명목 경로)       │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINABLE BLOCK                            │
│                                                                 │
│  Input: [ego_gru_hidden, sur_gru_hidden, z_intent]             │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────┐                                                │
│  │  Reaction   │──┬→ z_avoid ───→ MLP_avoid ──→ Δa_avoid       │
│  │  Encoder    │  │                              (회피 보정)     │
│  │             │  │                                             │
│  │             │  ├→ z_recover ─→ MLP_recover → Δa_recover     │
│  │             │  │                              (복귀 보정)     │
│  │             │  │                                             │
│  └─────────────┘  └→ gating ────→ α, β                         │
│                                    (개입 강도)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ACTION FUSION                              │
│                                                                 │
│   a_final = a_base + α · Δa_avoid + β · Δa_recover             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Strategy: BPTT with Joint Optimization

본 구조는 **Backpropagation Through Time (BPTT)**을 사용하여 시퀀스 전체의 인과관계를 학습한다.

### Forward Rollout (Trajectory Generation)

매 스텝 예측된 행동 a_t를 통해 다음 상태 s_{t+1}을 갱신한다. (Auto-regressive)

### Gradient Routing (Decoupled Update)

Total Loss를 합산하여 역전파하지만, 연쇄 법칙(Chain Rule)에 의해 각 파라미터는 자신에게 해당하는 Loss에 의해서만 업데이트된다.

| 파라미터 | 학습 신호 | 활성화 조건 |
|----------|----------|------------|
| `z_avoid` | L_collision, L_map | 충돌 위험이 높고 α가 활성화된 시점 |
| `z_recover` | L_recovery | 경로 이탈이 크고 β가 활성화된 시점 |

---

## 7. Implementation (Code Snippet)

수정된 입력 명세와 Map Constraint Loss가 모두 반영된 코드이다.

```python
# Hyperparameters
W_COLL, W_RECON, W_REG = 10.0, 2.0, 0.1
W_SOLID, W_DASH = 20.0, 5.0  # 실선/점선 가중치 차등

optimizer.zero_grad()
total_sequence_loss = 0.0

# Initial State
state_t = initial_state
current_pos = state_t[:2]

# -------------------------------------------------
# 1. Prepare Inputs
# -------------------------------------------------
# (1) Global Intent (Frozen)
with torch.no_grad():
    # z_intent: [Batch, Latent_Dim]
    # Encodes high-level style (e.g., target speed, lane preference)
    z_intent = global_encoder(ego_history_global)

# (2) Ego Temporal Context (GRU Hidden State)
# ego_gru_hidden: [Batch, Hidden_Dim]
# Contains current state + trajectory history (avoiding, recovering, normal)
ego_gru_hidden = ego_encoder(ego_history)

# (3) Surrounding Temporal Context (GRU Hidden State)
# sur_gru_hidden: [Batch, Hidden_Dim]
# Contains velocity, acceleration, trajectory curvature of Sur vehicles
sur_gru_hidden = interaction_net(sur_history_relative)

# -------------------------------------------------
# 2. Prediction Loop (BPTT)
# -------------------------------------------------
for t in range(PREDICTION_HORIZON):

    # [A] Construct Reaction Input
    # Combine Ego Context + Sur Context + Global Intent
    reaction_input = torch.cat([ego_gru_hidden, sur_gru_hidden, z_intent], dim=-1)

    z_avoid, z_recover, gating_feats = reaction_encoder(reaction_input)

    # [B] Gating Mechanism
    alpha = torch.sigmoid(head_alpha(gating_feats))
    beta  = torch.sigmoid(head_beta(gating_feats))

    # [C] Action Fusion
    # 1. Base Action (From Frozen Intent) -> Reference
    with torch.no_grad():
        base_action = mlp_base(z_intent)

    # 2. Reactive Actions
    delta_avoid = mlp_avoid(z_avoid)
    delta_recover = mlp_recover(z_recover)

    # 3. Final Action Composition
    final_action = base_action + (alpha * delta_avoid) + (beta * delta_recover)

    # [D] State Update (Differentiable Dynamics)
    next_state = dynamics_model(state_t, final_action)
    next_pos = next_state[:2]

    # -------------------------------------------------
    # 3. Loss Calculation (Potential Field)
    # -------------------------------------------------

    # (1) Safety Potential (Repulsive)
    dist_sur = torch.norm(next_pos - sur_pos[t], dim=-1)
    loss_coll = torch.relu(SAFE_DIST - dist_sur).mean()

    # Map Constraints (Solid vs Dashed)
    dist_line, line_type = get_map_constraints(next_pos, map_info[t])
    w_map = torch.where(line_type == 'SOLID', W_SOLID, W_DASH)
    loss_map = (w_map * torch.relu(SAFE_MARGIN - dist_line)).mean()

    # (2) Recovery Potential (Attractive)
    # Target is NOT GT, but the 'Nominal Path' from Base Action
    nominal_pos = dynamics_model(state_t, base_action)[:2]
    loss_recon = torch.nn.functional.mse_loss(next_pos, nominal_pos)

    # (3) Sparsity Regularization
    loss_reg = torch.mean(torch.abs(alpha) + torch.abs(beta))

    # Accumulate Step Loss
    step_loss = (W_COLL * loss_coll) + \
                (1.0 * loss_map) + \
                (W_RECON * loss_recon) + \
                (W_REG * loss_reg)

    total_sequence_loss += step_loss

    # Update for next step
    state_t = next_state

# -------------------------------------------------
# 4. Backpropagation (BPTT)
# -------------------------------------------------
total_sequence_loss.backward()
optimizer.step()
```

---

## 8. Hyperparameter Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| `W_COLL` | 10.0 | 충돌 회피 가중치 (높은 우선순위) |
| `W_SOLID` | 20.0 | 실선 침범 가중치 (절대 금지) |
| `W_DASH` | 5.0 | 점선 침범 가중치 (연성 제약) |
| `W_RECON` | 2.0 | 복귀 가중치 (낮은 우선순위) |
| `W_REG` | 0.1 | Sparsity 가중치 (미세 조정) |
| `SAFE_DIST` | 3.0m | 차량 간 안전 거리 |
| `SAFE_MARGIN` | 0.5m | 차선까지 여유 거리 |
