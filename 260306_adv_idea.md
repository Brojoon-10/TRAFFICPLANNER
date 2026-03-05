# Adversarial Optimization Ideas — 260306

## 개요

학습 완료된 CVAE 모델에서 위험 시나리오를 생성하기 위한 다양한 최적화 방식.
기존 STRIVE의 z_global 직접 최적화 외에 고차원 접근을 탐색.

---

## 1. z_query Test-Time Optimization

### 아이디어
학습된 z_query를 고정하지 않고, inference 시 "위험한 z를 생성하는 query"를 찾는다.

### 기존 z 최적화와의 차이
| | z 최적화 | z_query 최적화 |
|---|---|---|
| 변수 | z_global (32d) | z_query (2×128d) |
| 범위 | 단일 시나리오 | 모든 시나리오에 적용 가능한 패턴 |
| 해석성 | z 공간 위치 → 직접 해석 어려움 | attention weight → "뭘 읽었는지" 시각화 가능 |
| 결과물 | 위험한 latent code 1개 | 위험한 정보 추출 전략 (범용적) |

### 방법
```python
adv_z_query = model.z_query.data.clone().requires_grad_(True)
optimizer = Adam([adv_z_query], lr=1e-3)

for step in range(100):
    q = adv_z_query.unsqueeze(0).expand(NA, -1, -1)
    z_mu, z_var = cross_attn_pooling(q, context_base)
    z = reparameterize(z_mu, z_var)
    traj = model.decode(z, ...)

    loss = -collision_score(traj)
    loss.backward()
    optimizer.step()
```

### 학문적 가치
- "CVAE의 정보 추출 메커니즘 자체의 취약점 분석"
- 최적화된 z_query의 attention weight를 시각화 → "과거의 어떤 정보를 과대/과소 해석하면 위험한가"
- 예: adv_z_query가 map 토큰을 무시하고 속도만 attend → "환경 무시 = 사고 원인" 해석

### 주의점
- z_query가 바뀌면 z 분포 자체가 변함 → prior 범위 밖 z 생성 가능
- 대응: z에 prior KL 패널티 추가, 또는 z clipping

---

## 2. Latent Space Traversal

### 아이디어
두 시나리오의 posterior z를 보간하며 "안전→위험" 경계를 탐색.

### 방법

#### 2-A. 구면 보간 (Spherical Linear Interpolation)
```python
z_safe = model.posterior(safe_scenario)
z_danger = model.posterior(danger_scenario)

def slerp(z1, z2, alpha):
    z1_n, z2_n = F.normalize(z1), F.normalize(z2)
    omega = torch.acos((z1_n * z2_n).sum(-1, keepdim=True).clamp(-1, 1))
    s1 = torch.sin((1-alpha) * omega) / torch.sin(omega)
    s2 = torch.sin(alpha * omega) / torch.sin(omega)
    norm = (1-alpha) * z1.norm(-1, keepdim=True) + alpha * z2.norm(-1, keepdim=True)
    return (s1 * z1_n + s2 * z2_n) * norm

for alpha in torch.linspace(0, 1, 21):
    z_interp = slerp(z_safe, z_danger, alpha)
    traj = model.decode(z_interp, ...)
    record(alpha, collision_score(traj), road_departure(traj))
```

#### 2-B. 안전 경계 이진 탐색
```python
lo, hi = 0.0, 1.0
for _ in range(20):
    mid = (lo + hi) / 2
    z_mid = slerp(z_safe, z_danger, mid)
    traj = model.decode(z_mid, ...)
    if is_collision(traj):
        hi = mid
    else:
        lo = mid
# lo ≈ 안전→위험 전환점
```

#### 2-C. PCA 주성분 탐색
```python
all_z = torch.stack([model.posterior(s) for s in scenarios])
U, S, V = torch.pca_lowrank(all_z, q=5)

z_mean = all_z.mean(0)
for pc_idx in range(5):
    for scale in torch.linspace(-3, 3, 21):
        z_probe = z_mean + scale * V[:, pc_idx]
        traj = model.decode(z_probe, ...)
        # 주성분별 의미 분석: 좌우? 속도? 공격성?
```

### 학문적 가치
- z 공간의 의미론적 구조 (disentanglement) 분석
- 안전 경계의 기하학적 특성 → 얼마나 좁은 마진인지, 방향에 따라 다른지
- 주성분별 trajectory 변화 시각화 → 해석 가능한 latent space 증명

---

## 3. Adversarial Context Generation

### 아이디어
z가 아니라 모델 입력(map, surrounding agents)을 최적화하여 현실적인 위험 시나리오 자동 생성.

### 3-A. Adversarial Map
```python
delta_map = torch.zeros_like(map_image, requires_grad=True)
optimizer = Adam([delta_map], lr=1e-2)

for step in range(200):
    perturbed_map = (map_image + delta_map).clamp(0, 1)
    map_feat, map_tokens = model.encode_map(perturbed_map)
    context_base = model.context_encoder(past_gcn, map_summary)
    z = model.sample_prior(context_base)
    traj = model.decode(z, context_base, map_tokens, ...)

    loss = -road_departure_score(traj) + 0.1 * delta_map.norm()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        delta_map.clamp_(-0.05, 0.05)  # imperceptible perturbation
```

**의미**: "지도/센서가 약간만 틀려도 사고가 나는" 취약 환경 발견. Robustness 분석.

### 3-B. Adversarial Surrounding Agents
```python
sur_delta = torch.zeros(N_sur, FT, 6, requires_grad=True)
optimizer = Adam([sur_delta], lr=1e-2)

for step in range(200):
    perturbed_sur = sur_gt + sur_delta
    traj_ego = model.forward(past, perturbed_sur, sur_gt_replay=True, ...)

    collision_loss = -vehicle_collision_score(traj_ego, perturbed_sur)
    # 물리적 타당성 제약
    speed_pen = F.relu(-perturbed_sur[:,:,4]).sum() + F.relu(perturbed_sur[:,:,4] - 20).sum()
    accel_pen = torch.diff(perturbed_sur[:,:,4:5], dim=1).abs().sum()

    loss = collision_loss + 0.01 * speed_pen + 0.01 * accel_pen
    loss.backward()
    optimizer.step()
```

**의미**: "주변 차가 어떻게 움직이면 ego가 사고를 내는가". Cut-in, 급정거 등 자연스러운 위험 시나리오 자동 생성.

### 3-C. Joint (map + agents)
map과 surrounding trajectory를 동시 최적화 → 가장 포괄적인 adversarial analysis.

### 학문적 가치
- 실질적인 safety-critical scenario 자동 생성 → safety testing 자동화
- 물리적 타당성 제약 하에서 "최소 변화로 사고 유발" → 현실적 edge case 발견
- 시뮬레이터(CARLA) 연동 가능 → 생성된 시나리오를 실제로 시뮬레이션

---

## 물리적 타당성 보장 방법

적대적 최적화 시 비현실적 결과 (맵 뚫기 등) 방지:

### 이미 있는 방어: Bicycle Model
- decoder output = action (a, ddh) → bicycle_model → state
- action 범위 제한만으로 물리적으로 불가능한 경로 원천 차단
- a ∈ [-5, 5] m/s², ddh ∈ [-0.5, 0.5] rad/s 등

### z 공간 제약
```python
# prior 분포 안에 z를 가두기
prior_mu, prior_var = model._generate_z(context_base)
kl_penalty = 0.5 * ((z - prior_mu)**2 / prior_var).sum()
loss = objective + lambda * kl_penalty
```

### Decoder Fine-tuning with Physics Loss
```python
adv_decoder = copy.deepcopy(model.decoder_layers)
loss = (
    -collision_score(traj)
    + road_boundary_violation(traj, map) * 10.0
    + acceleration_limit(traj, a_max=5.0) * 5.0
    + curvature_limit(traj, kappa_max=0.5) * 5.0
)
```
z 범위 문제 자체가 없음. decoder가 "있을 법한 경로 중 위험한 것"을 직접 생성.

### Constrained z Projection
```python
# gradient step 후 prior 범위로 프로젝션
z_normalized = (z - prior_mu) / prior_std
z_normalized.clamp_(-3, 3)  # 3σ
z = z_normalized * prior_std + prior_mu
```

---

## 추천 실험 순서

1. **기본**: z_global 직접 최적화 + prior KL 제약 → z가 의미있게 작동하는지 확인
2. **해석**: Latent traversal (slerp + PCA) → z 공간 구조 이해
3. **고급**: z_query 최적화 → 정보 추출 취약점 분석, attention 시각화
4. **실용**: Adversarial surrounding agents → 위험 시나리오 자동 생성
5. **종합**: Adversarial map + agents joint → 포괄적 safety analysis

1-2는 모델 학습만 완료되면 즉시 가능.
3은 cross-attention pooling 구현 후 가능 (z_query가 있어야).
4-5는 물리적 타당성 제약 설계 필요.

---

## 논문 기여 가능성

| 방식 | 기여 | 신규성 |
|------|------|--------|
| z_query 최적화 | CVAE 정보 추출 메커니즘의 취약점 분석 | 높음 (기존에 없는 관점) |
| Latent traversal | z 공간의 안전 경계 기하학 | 중간 (방법론은 기존, 적용 분야 신규) |
| Adversarial context | 현실적 위험 시나리오 자동 생성 | 중간 (STRIVE 확장) |
| Decoder fine-tuning | 물리 제약 하 worst-case 분석 | 중간 |
| Joint optimization | 종합적 adversarial safety testing | 높음 (scope가 넓음) |
