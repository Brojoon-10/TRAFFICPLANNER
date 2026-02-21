# KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients

## Overview

KING은 ECCV 2022 논문으로, **미분 가능한 시뮬레이션**을 통해 Adversarial 차량의 행동을 최적화하여 **위험 시나리오**를 생성하고, 이를 활용해 운전 모델을 **robust하게 fine-tuning**하는 파이프라인입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KING Pipeline                                      │
│                                                                             │
│   [1] Scenario Generation: Adversary action 최적화 → 충돌 시나리오 생성      │
│   [2] Fine-tuning Data Gen: Expert가 위험 상황 회피 → 학습 데이터 생성       │
│   [3] Fine-tuning: AIM-BEV가 Expert 행동 모방 학습                          │
│   [4] Evaluation: CARLA에서 robust한 모델 평가                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 패키지 전체 구조

```
king/
├── generate_scenarios.py      ← 메인 진입점 (시나리오 생성)
├── run_generation.sh          ← 실행 스크립트
├── run_fine_tuning.sh         ← Fine-tuning 스크립트
│
├── proxy_simulator/           ← 핵심: 미분 가능 시뮬레이터
├── driving_agents/            ← 운전 에이전트들 (carla vs king 분리)
├── external_code/             ← 외부 코드 (LBC, WoR)
├── leaderboard/               ← CARLA Leaderboard (평가 프레임워크)
├── scenario_runner/           ← CARLA Scenario Runner (시나리오 실행)
└── tools/                     ← 유틸리티 스크립트
```

---

## 2. 핵심 패키지 상세

### 2.1 `proxy_simulator/` - 미분 가능 시뮬레이터

**KING의 핵심**. CARLA 없이 PyTorch 텐서 연산으로 시뮬레이션을 수행하여 gradient 계산 가능.

```
proxy_simulator/
├── simulator.py       ← 전체 시뮬레이션 관리 (상태 업데이트, 충돌 체크)
├── motion_model.py    ← BicycleModel (차량 dynamics)
├── bm_policy.py       ← BMActionSequence (adversary의 learnable action)
├── collision.py       ← 충돌 감지 (bounding box 겹침 계산)
├── driving_costs.py   ← Collision cost, Road deviation cost
├── renderer.py        ← BEV 이미지 렌더링 (STN 기반)
├── carla_wrapper.py   ← CARLA 맵 로딩
└── utils.py           ← 유틸리티
```

#### Proxy vs CARLA 비교

| 구분 | CARLA | Proxy Simulator |
|------|-------|-----------------|
| **본질** | 물리 엔진 + 그래픽 렌더링 | PyTorch 텐서 연산 (수치 시뮬레이션) |
| **차량 동역학** | Unreal Engine 물리 | BicycleModel 수식 |
| **렌더링** | 3D 그래픽 | STN으로 BEV 이미지 변환 |
| **충돌 감지** | 물리 엔진 | Bounding box 겹침 계산 |
| **미분 가능** | ✗ | ✓ |
| **속도** | 느림 (GPU 렌더링) | 빠름 (텐서 연산) |

#### BicycleModel

World on Rails (WoR)에서 system identification으로 추출한 파라미터 사용:

```python
# motion_model.py
self.register_buffer("front_wb", torch.tensor([-0.090769015]))
self.register_buffer("rear_wb", torch.tensor([1.4178275]))
self.register_buffer("steer_gain", torch.tensor([0.36848336]))
self.register_buffer("brake_accel", torch.tensor([-4.952399]))
self.register_buffer("throt_accel", torch.tensor([[0.5633837]]))
```

상태 업데이트 수식:
```
next_pos = current_pos + velocity * dt
next_yaw = current_yaw + (velocity / wheelbase) * tan(steer) * dt
next_vel = current_vel + acceleration * dt
```

---

### 2.2 `driving_agents/` - carla vs king 분리

```
driving_agents/
├── carla/              ← CARLA 환경에서 직접 실행
│   ├── aim_bev/
│   │   └── aim_bev_agent.py
│   └── expert/
│       ├── expert_agent.py
│       └── data_agent.py
│
└── king/               ← Proxy simulator에서 실행
    ├── aim_bev/
    │   ├── aim_bev_agent.py
    │   ├── model.py
    │   ├── data.py
    │   ├── robust_train.py
    │   └── king_initializations/
    ├── expert/
    │   └── expert_agent.py
    ├── transfuser/
    └── common/
```

#### 왜 carla와 king으로 분리?

| 구분 | `carla/` | `king/` |
|------|----------|---------|
| **실행 환경** | 실제 CARLA 시뮬레이터 | Proxy simulator (PyTorch) |
| **입력** | CARLA 센서 데이터 | Proxy에서 렌더링된 BEV 텐서 |
| **용도** | 평가, 일반 데이터 수집 | 시나리오 생성, Fine-tuning |
| **인터페이스** | CARLA Python API | PyTorch 텐서 |
| **상태 변수** | 스칼라 (`self.steer = 0.0`) | 배치 (`self.steer = np.zeros(batch_size)`) |

#### Expert 코드 차이

```python
# carla/expert/expert_agent.py
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
self.vehicle_model = EgoModel(dt=...)
self.steer = 0.0  # 스칼라

# king/expert/expert_agent.py
from proxy_simulator.motion_model import BicycleModel
self.vehicle_model = BicycleModel(...).to(device)
self.steer = np.zeros(shape=(self.args.batch_size))  # 배치
```

**로직(PID 제어, 경로 추종)은 유사하지만 인터페이스가 완전히 다름.**

---

### 2.3 `external_code/` - 외부 의존성

```
external_code/
├── lbc/                ← Learning by Cheating (ICCV 2019)
│   └── bird_view/
│       ├── models/     ← BEV 인코더/디코더 네트워크
│       └── utils/      ← BEV 렌더링 유틸
│
└── wor/                ← World on Rails (ICCV 2021)
    └── ego_model.py    ← BicycleModel 파라미터 원본
```

---

### 2.4 `leaderboard/` - CARLA Leaderboard

```
leaderboard/
├── data/
│   ├── routes/         ← 주행 경로 XML 파일들
│   │   ├── subset_20perTown.xml   ← Ego route
│   │   └── adv_all.xml            ← Adversary route 후보
│   └── scenarios/
├── leaderboard/
│   ├── autoagents/     ← 에이전트 베이스 클래스
│   ├── scenarios/
│   └── utils/
└── scripts/
```

---

### 2.5 `scenario_runner/` - CARLA Scenario Runner

NPC 차량 행동 정의, 시나리오 실행, 평가 기준 정의.

---

## 3. 전체 파이프라인

### Phase 1: 시나리오 생성 (generate_scenarios.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Scenario Generation                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Ego (AIM-BEV)                    Adversary (BMActionSequence)             │
│   ├─ weights 고정                   ├─ action 파라미터 최적화               │
│   ├─ BEV 입력 → action 출력         ├─ steer, throttle learnable            │
│   └─ torch.no_grad() (gradient X)  └─ gradient descent로 업데이트          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Optimization Loop                                 │   │
│   │                                                                     │   │
│   │   for iteration in opt_iters:                                       │   │
│   │       1. Ego: AIM-BEV가 BEV 보고 action 계산 (detached)             │   │
│   │       2. Adv: BMActionSequence에서 action 가져옴 (learnable)        │   │
│   │       3. BicycleModel로 다음 상태 계산                              │   │
│   │       4. Collision cost + Road deviation cost 계산                  │   │
│   │       5. Backprop → Adv action 파라미터 업데이트                    │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   Output: adv_actions JSON (충돌 유발하는 adversary 행동 저장)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 실행 명령어

```bash
# CARLA 서버 실행 (맵 로딩용)
carla_server/CarlaUE4.sh --world-port=2000 -opengl

# 시나리오 생성
bash run_generation.sh
```

#### 주요 파라미터

```bash
python generate_scenarios.py \
    --num_agents 4        # adversary 차량 수
    --opt_iters 100       # 최적화 iteration 수
    --w_adv_col 3.0       # collision cost 가중치
    --w_adv_rd 20.0       # road deviation cost 가중치
```

---

### Phase 1 상세: BM Policy & 최적화 방식

#### BMActionSequence 구조

Adversary의 action을 learnable parameter로 정의:

```python
# bm_policy.py:26-36
class BMActionSequence(torch.nn.Module):
    def __init__(self, batch_size, num_agents, sim_horizon):
        # 최적화 대상 파라미터 (전체 horizon에 대해 미리 정의)
        self.throttle = torch.nn.Parameter(
            torch.zeros(batch_size, num_agents, sim_horizon, 1)
        )
        self.steer = torch.nn.Parameter(
            torch.zeros(batch_size, num_agents, sim_horizon, 1)
        )

    def forward(self, observations):
        t = observations["timestep"]
        return {
            "throttle": torch.tanh(self.throttle[:, :, t, :] + 1e-3),  # [-1, 1]
            "steer": torch.tanh(self.steer[:, :, t, :]),               # [-1, 1]
        }
```

**핵심 특징:**
- `throttle`, `steer`가 `nn.Parameter` → gradient 계산됨
- shape: `[batch, num_agents, sim_horizon, 1]` → 각 timestep마다 별도 action
- `tanh`로 [-1, 1] 범위 제한
- **Non-reactive**: 저장된 action sequence 반환 (입력 무시)

#### Route 사용 여부

**Adversary는 경로 제약 없음. 초기 위치만 사용.**

```python
# simulator.py:437-441 - spawn point만 사용
adv_pos[ix][id] = torch.tensor(
    [[[self.adv_spawn_points[ix][id].location.x,
       self.adv_spawn_points[ix][id].location.y]]],
)

# adv_routes는 로드만 하고 실제로 읽는 코드 없음
```

Ego는 route가 planner 입력으로 들어감:
```python
# simulator.py:224-226
self.ego_policy.set_global_plan(self.gps_route, self.route)
```

#### 최적화 루프 구조

```python
# generate_scenarios.py:113-161

# Outer loop: 최적화 반복 (100~150회)
for i in range(self.args.opt_iters):
    scenario_optim.zero_grad()

    # Inner loop: 시뮬레이션 unroll (sim_horizon 스텝)
    cost_dict = self.unroll_simulation()

    # Cost 집계 후 backward
    total_objective = ...
    total_objective.backward()
    scenario_optim.step()
```

**Unroll**: 시뮬레이션을 시간 순서대로 펼쳐서 실행
```
t=0 → t=1 → t=2 → ... → t=79 (sim_horizon)
```

#### 시뮬레이션 Unroll 상세

```python
# generate_scenarios.py:352-386
for t in range(self.args.sim_horizon):
    # 1. BEV 렌더링 (현재 ego, adv 위치 기반)
    observations, _ = self.simulator.renderer.get_observations(
        semantic_grid,
        self.simulator.get_ego_state(),
        self.simulator.get_adv_state(),
    )

    # 2. Ego action 계산 (매 스텝 반응함!)
    ego_actions = self.simulator.ego_policy.run_step(input_data, self.simulator)

    # 3. Ego gradient만 차단 (반응은 하지만 최적화 대상 아님)
    if self.args.detach_ego_path:
        ego_actions["steer"] = ego_actions["steer"].detach()
        ego_actions["throttle"] = ego_actions["throttle"].detach()

    # 4. Adv action (learnable parameter에서 가져옴)
    adv_actions = self.simulator.adv_policy.run_step(input_data)

    # 5. Cost 계산
    ego_col_cost, adv_col_cost, adv_rd_cost = self.compute_cost()

    # 6. 상태 업데이트
    self.simulator.step(ego_actions, adv_actions)
```

**Ego 반응성**: Ego는 매 스텝 BEV를 보고 action을 계산함. Gradient만 detach되어 최적화 대상에서 제외될 뿐, **Adv 움직임에 반응**함.

#### Cost 함수 및 집계

```python
# generate_scenarios.py:131-154

# ego collision: timestep 평균 → batch 최소
cost_dict["ego_col"] = torch.min(torch.mean(...))

# adv collision: timestep 최소 → batch 최소
cost_dict["adv_col"] = torch.min(torch.min(...))

# road deviation: timestep 평균
cost_dict["adv_rd"] = torch.mean(...)

# 최종 objective
total_objective = sum([
    self.args.w_ego_col * cost_dict["ego_col"].mean(),   # ego 충돌 거리 ↓
    self.args.w_adv_rd * cost_dict["adv_rd"].mean(),     # 도로 이탈 ↓
    -1*self.args.w_adv_col * cost_dict["adv_col"].mean() # adv간 충돌 거리 ↑
])
```

| Cost | 목적 | Timestep 집계 |
|------|------|--------------|
| **ego_col** | Ego-Adv 충돌 유도 | mean |
| **adv_col** | Adv끼리 충돌 방지 | min |
| **adv_rd** | 도로 위 유지 | mean |

#### Road Deviation Cost

Adversary에게 경로 제약은 없지만, **도로 이탈 페널티**가 간접적으로 제약:

```python
# driving_costs.py:102
crop_road_rasterized = 1. - crop_road_rasterized  # 도로 밖 = 1, 도로 위 = 0
# 도로 밖에 있으면 cost 증가 → soft constraint
```

#### 충돌 후 처리

**Stop이 아니라 Freeze:**

```python
# motion_model.py:50-55
update_mask = ~is_terminated.view(-1, 1, 1)  # 종료된 batch는 False

state["pos"] = state["pos"] + ... * update_mask  # 종료되면 업데이트 안 함
state["yaw"] = state["yaw"] + ... * update_mask
state["vel"] = state["vel"] + ... * update_mask
```

```
t=0  t=1  t=2  t=3  t=4  t=5  ...  t=79
 │    │    │    │    │    │         │
 ○────○────○────●────●────●────...──●
                ↑
            충돌 발생 (is_terminated = True)
            이후 상태 freeze, 루프는 끝까지 진행
```

#### Gradient 흐름 (Kinematic Gradient)

모든 연산이 PyTorch 미분 가능 연산으로 구성:

```python
# motion_model.py:39-55
wheel = self.steer_gain * actions["steer"]           # 곱셈
beta = torch.atan(                                    # atan (미분 가능)
    self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel)  # tan
)
speed = torch.norm(state["vel"], dim=-1, keepdim=True)  # norm
motion_components = torch.cat(
    [torch.cos(state["yaw"]+beta), torch.sin(state["yaw"]+beta)],  # cos, sin
    dim=-1,
)
state["pos"] = state["pos"] + speed * motion_components * self.delta_t
state["yaw"] = state["yaw"] + speed / self.rear_wb * torch.sin(beta) * self.delta_t
```

**Gradient 체인:**
```
BMActionSequence.steer (Parameter)
        │
        ▼ torch.tanh
    actions["steer"]
        │
        ▼ * steer_gain, torch.tan, torch.atan
      beta
        │
        ▼ torch.sin, torch.cos
  motion_components
        │
        ▼ state update (+, *)
    next_state["pos"]
        │
        ▼ collision cost (거리 계산)
    ego_col_cost
        │
        ▼ .backward()
    steer.grad ← gradient 역전파
```

#### Kinematic Feasibility

| 항목 | 보장 여부 | 방법 |
|------|----------|------|
| Action 범위 [-1, 1] | ✓ | `tanh` |
| 물리적 회전 반경 | ✓ | BicycleModel 수식 (steer_gain, wheelbase) |
| Action 연속성 (급격한 변화 방지) | △ | 직접 보장 안 함, Adam momentum이 간접적으로 smoothing |
| Kinematic feasibility | △ | 완벽하지 않음, road cost 등으로 간접 제약 |

**한계**: Gradient descent는 cost를 줄이는 방향으로만 움직이므로, 급격한 조향 변화(예: t=5에서 steer=0.9, t=6에서 steer=-0.9)가 발생할 수 있음. 다만 이런 경우 도로 이탈 등으로 페널티를 받아 자연스럽게 걸러지는 구조.

#### 최적화 요약

| 항목 | 내용 |
|------|------|
| **최적화 대상** | `BMActionSequence.throttle`, `BMActionSequence.steer` |
| **Optimizer** | Adam (lr, beta1, beta2 설정) |
| **Outer loop** | `opt_iters` 회 (100~150) - 최적화 반복 |
| **Inner loop** | `sim_horizon` 스텝 (예: 80) - 시뮬레이션 unroll |
| **Backward** | outer loop에서 한 번 (전체 horizon 누적 cost) |
| **Ego** | 매 스텝 반응하지만 gradient는 detach |
| **Adv** | 경로 제약 없음, 초기 위치 + action 최적화가 전부 |

---

### Phase 2: Fine-tuning 데이터 생성 (robust_training_engine.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Fine-tuning Data Generation                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   KING 시나리오 재생 (CARLA 없이 Proxy에서 실행)                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                                                                     │   │
│   │   # robust_training_engine.py:101-110                               │   │
│   │                                                                     │   │
│   │   # 1. 저장된 adv_actions 로드 (고정)                               │   │
│   │   adv_actions = scenario_def['adv_actions'][critical_iteration][t]  │   │
│   │                                                                     │   │
│   │   # 2. Expert가 이 상황에서 회피 action 계산                        │   │
│   │   expert_actions = self.simulator.ego_expert.run_step(...)          │   │
│   │                                                                     │   │
│   │   # 3. 시뮬레이션 스텝                                              │   │
│   │   self.simulator.step(expert_actions, adv_actions)                  │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   수집 데이터:                                                               │
│   ├─ 입력 (X): BEV 이미지 (위험 상황)                                       │
│   └─ 정답 (Y): Expert가 출력한 action (회피 행동)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### BEV 이미지 생성 (CARLA 없이)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BEV Rendering (Proxy)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Global Map (CARLA에서 한 번만 추출, 정적)                                  │
│        │                                                                    │
│        │    Ego state (pos, yaw)                                            │
│        │         │                                                          │
│        ▼         ▼                                                          │
│   ┌─────────────────────────────────┐                                       │
│   │   get_local_birdview()          │                                       │
│   │   - F.affine_grid()             │  ← STN (Spatial Transformer Network)  │
│   │   - F.grid_sample()             │                                       │
│   └──────────────┬──────────────────┘                                       │
│                  │                                                          │
│                  ▼                                                          │
│        Local BEV (192x192)                                                  │
│                  │                                                          │
│                  │    Ego/Adv positions, orientations                       │
│                  │         │                                                │
│                  ▼         ▼                                                │
│   ┌─────────────────────────────────┐                                       │
│   │   render_agent_bv()             │                                       │
│   │   - vehicle_template (22x9)     │  ← 차량 박스 텐서                     │
│   │   - rotation_transform          │  ← 각도에 맞게 회전                   │
│   │   - translation_transform       │  ← 위치에 맞게 이동                   │
│   │   - F.grid_sample()             │                                       │
│   └──────────────┬──────────────────┘                                       │
│                  │                                                          │
│                  ▼                                                          │
│        최종 BEV (도로 + 차량들)                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**핵심**: BicycleModel로 위치/각도 계산 → renderer로 BEV 이미지 생성 (순수 텐서 연산)

---

### Phase 3: Fine-tuning (robust_train.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Fine-tuning                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   학습 데이터 = Regular 데이터 + KING 시나리오                               │
│                                                                             │
│   Regular 데이터:                                                            │
│   ├─ CARLA에서 carla/expert로 수집 (일반 주행)                              │
│   └─ 포맷: topdown/*.png + measurements/*.json                              │
│                                                                             │
│   KING 시나리오:                                                             │
│   ├─ Proxy에서 king/expert로 생성 (위험 상황 회피)                          │
│   └─ 포맷: topdown/*.png + measurements/*.json (동일)                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │   CARLA_Data Dataset                                                │   │
│   │   - 둘 다 같은 포맷 → 같은 DataLoader로 로드                        │   │
│   │   - PNG 이미지 → torch 텐서 변환                                    │   │
│   │   - JSON → action labels                                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   학습 목표: AIM-BEV가 Expert의 회피 행동 모방 (Imitation Learning)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 4: 평가 (run_evaluation.sh)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Evaluation                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   실제 CARLA 시뮬레이터에서 평가                                             │
│                                                                             │
│   사용 모듈:                                                                 │
│   ├─ driving_agents/carla/aim_bev/ (CARLA용 에이전트)                       │
│   ├─ leaderboard/ (평가 프레임워크)                                         │
│   └─ scenario_runner/ (시나리오 실행)                                       │
│                                                                             │
│   평가 벤치마크: Town10 Intersections                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 데이터 흐름 요약

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Data Flow Summary                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   [Regular 데이터 수집]           [KING 시나리오 생성]                        │
│   CARLA 시뮬레이터                Proxy simulator                           │
│        │                              │                                     │
│   carla/expert                   generate_scenarios.py                      │
│        │                              │                                     │
│        ▼                              ▼                                     │
│   topdown/*.png              adv_actions JSON                               │
│   measurements/*.json              │                                        │
│        │                           │                                        │
│        │                    [Fine-tuning 데이터 생성]                        │
│        │                    Proxy simulator                                 │
│        │                           │                                        │
│        │                    king/expert + renderer                          │
│        │                           │                                        │
│        │                           ▼                                        │
│        │                    topdown/*.png                                   │
│        │                    measurements/*.json                             │
│        │                           │                                        │
│        └───────────┬───────────────┘                                        │
│                    │                                                        │
│                    ▼                                                        │
│           같은 포맷 (PNG + JSON)                                            │
│                    │                                                        │
│                    ▼                                                        │
│           CARLA_Data Dataset                                                │
│                    │                                                        │
│                    ▼                                                        │
│           AIM-BEV Fine-tuning                                               │
│                    │                                                        │
│                    ▼                                                        │
│           [평가] CARLA + carla/aim_bev                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. JSON 초기화 파일 구조

경로: `driving_agents/king/aim_bev/king_initializations/`

```json
{
  "meta_data": {...},
  "adv_routes": [[waypoint1, waypoint2, ...], ...],     // 사용 안 됨 (legacy)
  "adv_routes_gps": [...],                              // 사용 안 됨
  "adv_spawn_points": [spawn1, spawn2, ...],           // 초기 위치 설정에 사용
  "action_seq": [[action1, action2, ...], ...]         // BMActionSequence 초기화
}
```

| 필드 | 용도 | 실제 사용 |
|------|------|----------|
| `adv_routes` | Adversary 경로 waypoints | ✗ 로드만 하고 안 씀 |
| `adv_spawn_points` | Adversary 초기 위치 (x, y, yaw) | ✓ simulator.py:437 |
| `action_seq` | Adversary action 초기값 | ✓ adv_policy 초기화 |

---

## 6. 핵심 구성요소 요약

| 구성요소 | 역할 |
|---------|------|
| **Ego (AIM-BEV)** | 피해자 모델, weights 고정, BEV 입력 → action 출력 |
| **Adversary (BMActionSequence)** | 공격자, action 파라미터를 gradient descent로 최적화 |
| **BicycleModel** | 미분 가능한 차량 dynamics, World on Rails 파라미터 |
| **Proxy Simulator** | PyTorch 기반 수치 시뮬레이션 |
| **STN Renderer** | 위치/각도 기반 BEV 이미지 생성 (텐서 연산) |
| **Collision Cost** | Ego-Adversary 충돌 유도 |
| **Road Deviation Cost** | Adversary가 도로 이탈 방지 |
| **Expert (king/)** | Privileged information으로 회피 행동 생성 |

---

## 7. 패키지별 파이프라인 참여

| 패키지 | 시나리오 생성 | Fine-tuning 데이터 | Fine-tuning | 평가 |
|--------|:------------:|:-----------------:|:-----------:|:----:|
| `proxy_simulator/` | ✓ | ✓ | - | - |
| `driving_agents/king/` | ✓ | ✓ | ✓ | - |
| `driving_agents/carla/` | - | - | - | ✓ |
| `external_code/` | ✓ | ✓ | - | - |
| `leaderboard/` | ✓ (route만) | - | - | ✓ |
| `scenario_runner/` | - | - | - | ✓ |

---

## References

- **논문**: KING: Generating Safety-Critical Driving Scenarios for Robust Imitation via Kinematics Gradients (ECCV 2022)
- **GitHub**: https://github.com/autonomousvision/king
- **의존성**:
  - Learning by Cheating (LBC, ICCV 2019)
  - World on Rails (WoR, ICCV 2021)
  - CARLA Leaderboard
  - Scenario Runner
