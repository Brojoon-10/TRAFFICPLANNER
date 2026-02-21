# Reactive Ego 학습 전략: Curriculum Learning과 Fine-tuning

## 1. 연구 배경 및 Contribution

### 1.1 연구 목표

주행 데이터를 기반으로 **블랙박스 모델**을 학습하여, 실제 차량처럼 움직이는 Ego가 적대적으로 공격하는 Sur(Surrounding Vehicle)에 **어느 정도 반응**하면서도, 못 피할 수도 있는 현실적인 시나리오를 생성한다.

### 1.2 핵심 Contribution

| 구분 | 설명 |
|------|------|
| **Fidelity (현실성)** | 주행 데이터를 제대로 모방하여 실제 주행과 유사한 결과물 생성 |
| **Reactivity (반응성)** | Ego가 Sur의 공격에 사람처럼 반응 (급브레이크, 회피 등) |
| **Adversarial Generation** | Sur이 반응하는 Ego를 대상으로 적대적 최적화 수행 |

### 1.3 기존 접근법의 한계

단순히 Ego의 움직임을 최적화하거나 새로운 플래너를 학습시키면:
- 주행 양상이 실제 주행과 비슷해지지 않음
- **"일어날 법한 시나리오"**를 만드는 기여를 하지 못함
- 연구가 단순히 새로운 플래너를 만드는 것에 그침

---

## 2. 모델 구조 이해

### 2.1 Encoder vs Decoder 역할 분담

| 구성요소 | 역할 | Fine-tuning 시 상태 |
|----------|------|---------------------|
| **Encoder (Posterior/Prior)** | 주행 의도 및 성향 정의 → z 생성 | **Freeze (고정)** |
| **Decoder (GRU/MLP/Attention)** | 실시간 반응 및 제어 → 경로 생성 | **Train (학습)** |

### 2.2 Encoder 상세

- **Posterior (Training용)**: `Past` + `Future(GT)`를 보고 정답 z 생성. z의 기준점 형성
- **Prior (Inference용)**: `Past`만 보고 z 추론. Posterior 분포를 모사하도록 학습됨
- **z (Latent)**: 한 시퀀스(Scene) 동안 고정되는 **전역적(Global) 주행 의도** (예: 직진, 좌회전, 과속 성향 등)

### 2.3 Decoder 상세

- **Agent Attention**: 주변 차량(Sur)과의 관계를 계산하여 위험도를 가중치로 변환
- **GRU**: 급격한 회피 기동 시에도 차량 움직임을 물리적으로 매끄럽게 연결 (Recovery)
- **MLP**: z(의도)와 Attention(위험 신호)을 입력받아, 위험 시 z보다 Attention에 가중치를 두어 브레이크/조향 결정

---

## 3. 현재 문제점: Agent Attention이 학습되지 않음

### 3.1 증상

Agent Attention 가중치가 **50:50 균등 분포**:
```
Agent 0: self=0.5000, others_sum=0.5000
Agent 1: self=0.5000, others_sum=0.5000
```

### 3.2 원인 분석

**Attention 학습의 필요조건**: 모델이 **"Attention을 보지 않으면 정답을 맞출 수 없는 상황"**에 처해야 함

필요한 데이터 구조 (쌍으로 준비):
```
상황 A (Attack): Sur가 돌진 → Ego가 피함 (GT: Curve)
상황 B (Safety): Sur가 멈춤 → Ego가 직진 (GT: Straight)
```

두 상황이 **동일한 맵, 동일한 Ego 위치**에서 섞여 있어야 모델이 학습:

1. "내 위치(Q)나 맵 정보는 A랑 B가 똑같은데 뭐가 다르지?"
2. "아! Agent Attention 값(V)을 보니까 A일 땐 Sur가 가까웠고 B일 땐 멀었구나!"
3. **"Sur의 상태(K, V)가 나의 행동 결정을 좌우하는구나"** → Attention 가중치 학습

**현재 문제**: NuScenes 데이터는 대부분 정상 주행이라 **대조적 상황(Contrastive Scenarios)**이 부족

---

## 4. 제안 솔루션: Curriculum Learning

### 4.1 핵심 아이디어

학습 목표를 **2단계로 분리**:

| Phase | 데이터 | 목표 | 결과 |
|-------|--------|------|------|
| **Phase 1** | 일반 주행 (NuScenes) | 주행 양상/스타일 학습 | z가 "정상적인 운전자" 분포 형성 |
| **Phase 2** | 상호작용 데이터 (Contrastive) | Agent Attention 활성화 | 위기 대처 능력 획득 |

### 4.2 Phase 1: Pre-training (기초 교육)

- **데이터**: 일반적인 안전 주행 데이터
- **목표**: 차선 유지, 부드러운 가속, 일반적인 교통 흐름 따르기
- **Loss**: Reconstruction Loss + KL Divergence
- **결과**: 잠재 공간 z가 "정상적인 운전자"의 분포를 학습

### 4.3 Phase 2: Fine-tuning (심화 교육)

- **데이터**: Contrastive 데이터 (기존 주행 + 변형된 공격적 Sur 주행)
- **목표**: z의 스타일 유지 + Agent Attention으로 위험 감지/회피
- **핵심**: **Encoder 고정 (또는 낮은 LR)** → z 분포 보존
- **Contribution 보존**: z가 이미 Phase 1에서 사람의 주행 스타일로 고정되었기 때문에, 갑자기 로봇처럼 움직이지 않음

---

## 5. Contrastive 데이터셋 구성 전략

### 5.1 데이터 구조

```
Contrastive Dataset
├── Set A (Normal): 기존 주행 데이터 (원본)
│   ├── Sur: 원래 궤적 그대로
│   └── Ego: 원래 GT 그대로 (직진, 정상 주행)
│
└── Set B (Adversarial): 변형된 공격적 Sur 주행
    ├── Sur: Adv 최적화로 생성된 공격적 궤적
    └── Ego: 회피/급브레이크 반응 (새로운 GT)
```

### 5.2 Adversarial 데이터 생성 파이프라인

**핵심 아이디어**: Phase 1에서 Pre-trained된 모델을 활용하여 Adv 데이터 생성

```
Step 1: Phase 1 Pre-training 완료
        → 모델은 일반 주행은 잘 하지만, Sur 공격에 반응 못함

Step 2: Pre-trained 모델로 Adv 최적화 실행
        → Sur의 z를 최적화하여 "반응 못하는 Ego"를 공격하는 궤적 생성
        → 이 시점의 Ego는 멍청하므로 쉽게 충돌 시나리오 생성됨

Step 3: 생성된 Adv Sur 궤적 + 원본 데이터를 Contrastive 쌍으로 구성
        → 같은 씬에서 (Normal Sur, Normal Ego) vs (Adv Sur, Reactive Ego)

Step 4: Phase 2 Fine-tuning에 사용
        → 모델이 "Sur가 바뀌면 Ego도 바뀌어야 한다"를 학습
```

### 5.3 Contrastive 쌍의 구체적 구성

| 쌍 구분 | Sur 궤적 | Ego GT | 학습 목표 |
|---------|----------|--------|----------|
| **Normal** | 원본 (안전) | 원본 (직진) | "평소에는 z대로 가라" |
| **Adversarial** | Adv 최적화 (공격적) | 회피/브레이크 | "위험하면 Attention 보고 피해라" |

### 5.4 Reactive Ego GT 생성 방법

Adv Sur에 대응하는 Ego GT를 만드는 방법:

**Option 1: Rule-based Planner 사용**
```
- IDM (Intelligent Driver Model) 등으로 급브레이크 생성
- 단순하지만 "사람다움"이 부족할 수 있음
```

**Option 2: 시뮬레이터 (CARLA) 활용**
```
- Adv Sur 궤적을 CARLA에 재현
- 사람이 직접 조작하거나 내장 플래너로 Ego 반응 수집
- 가장 현실적인 GT 확보 가능
```

**Option 3: Collision Loss 기반 최적화**
```
- Ego의 z도 최적화하여 충돌 회피하는 궤적 생성
- 단, Prior Loss로 "사람 범주" 유지
```

---

## 6. Fine-tuning 상세 전략

### 6.1 왜 Encoder를 고정해야 하는가

#### 이유 1: "언어" 보존

Posterior와 Prior는 "주행 상황"을 "z라는 언어"로 번역하는 **사전(Dictionary)** 역할

```
현재 상태: "직진한다" ↔ z=[0.5, 0.1] (Phase 1에서 학습됨)
목표: 이 번역 체계를 유지한 채로, 문장을 만드는 법(Decoder)만 변경
```

만약 Posterior/Prior를 학습시키면 → 단어의 뜻이 바뀜 → 일반 주행 시 엉뚱한 행동 (Catastrophic Forgetting)

#### 이유 2: "꼼수(Shortcut)" 방지

인코더가 학습되면 디코더가 할 일을 뺏어감:

```
상황: Sur가 끼어듦 → 회피해야 함

[인코더 학습 시]
Posterior: "아, 그냥 내가 z를 '회피용 z'로 바꿔줄게. 디코더 넌 가만히 있어."
결과: 디코더는 여전히 멍청하고, Attention은 무시됨

[우리가 원하는 것]
디코더가 Attention을 보고 반응하는 것 → 인코더를 묶어야 함
```

### 6.2 Encoder 고정 코드

```python
# Fine-tuning 단계 시작 전

# 1. z 생성 담당 (뇌) 얼리기
for param in model.latent_posterior_net.parameters():
    param.requires_grad = False

for param in model.latent_prior_net.parameters():
    param.requires_grad = False

# 2. z 변환기 얼리기 (선택사항이지만 추천)
for param in model.z_projection.parameters():
    param.requires_grad = False

# 3. 디코더와 어텐션만 열어두기
# (decoder_output_mlp, agent_cross_attention 등은 True 유지)
```

### 6.3 학습되는 파라미터

Encoder를 고정하면, **Decoder의 모든 구성요소**가 학습 대상:

| 구성요소 | 역할 | 학습 내용 |
|----------|------|----------|
| `agent_cross_attention` | 주변 차량 주의 할당 | **위협 감지 능력** - 빠르게 다가오는 차에 가중치 증가 |
| `decoder_output_mlp` | 최종 좌표 결정 | **z와 Attention 간 우선순위 조율** - z는 직진하라 하지만 Attention이 위험 신호 보내면 핸들 꺾기 |
| `decoder_memory` (GRU) | 상태 갱신 | **Recovery 능력** - 급기동 후 자세 복구 |

---

## 7. "z 고정인데 디코더만으로 급브레이크 가능한가?"

### 7.1 결론: 가능하다

신경망(MLP)의 **비선형성(Non-linearity)**이 가진 힘

```python
Output = MLP(z, Attention, GRU)
```

MLP는 단순한 더하기(+)가 아님. 학습을 통해 **논리회로(Switching Logic)** 구축 가능:

```python
# 학습 전
Output ≈ 0.8 × z + 0.2 × Attention  # Attention이 약함

# 학습 후 (파인튜닝)
if Attention_Score > 0.8:  # 위험
    return Brake  # z가 뭐든 상관없이 0 출력
else:
    return z  # 원래 가려던 대로
```

**결론**: z는 "기본값(Default)" 제공, 디코더는 Attention이라는 "브레이크 페달"을 밟을 권한 학습

### 7.2 사람다움 유지를 위한 안전장치

#### 안전장치 1: GRU (관성)

- GRU는 "이전 스텝의 내 속도와 상태"를 기억
- 속도를 100→0으로 급격히 줄이려 하면, GRU의 기억과 충돌
- 구조적으로 **변화에 저항하는 성질(Smoothness)** 발생

#### 안전장치 2: 고정된 z (성향)

- z를 고정했기 때문에 "완전히 새로운 주행 스타일" 창조 불가
- 사람의 주행 패턴(z) 위에서 수정하는 것이므로 기괴한 행동 억제

#### 안전장치 3: Regularization Loss (추가 필요)

```python
# Fidelity Loss 추가 예시

# 1. 가속도(Acceleration) 제한
acc = pred_vel_t - pred_vel_t_1
loss_acc = torch.relu(torch.abs(acc) - MAX_HUMAN_ACCEL)

# 2. 저크(Jerk, 가속도 변화율) 제한 → 승차감/부드러움
jerk = acc_t - acc_t_1
loss_jerk = torch.mean(jerk ** 2)

# 총 로스
total_loss = loss_collision * 10.0 + loss_acc * 1.0 + loss_jerk * 0.5
```

**효과**: "최대한 급하게 멈추되(Collision), 사람의 목이 꺾이지 않을 정도(Jerk)로만 멈춰라"

---

## 8. Fine-tuning 데이터 및 Loss 전략

### 8.1 데이터 구성

```
Phase 2 학습 데이터:
├── Contrastive Pair (Adversarial): 80%
│   ├── Normal: 원본 주행 데이터
│   └── Adversarial: Adv Sur + Reactive Ego
│
└── Replay Buffer (일반 주행): 20%
    └── Phase 1에서 사용한 일반 주행 데이터
```

### 8.2 왜 일반 데이터를 섞어야 하는가

**위험성**: 적대적 데이터만 넣으면 Catastrophic Forgetting 발생

```
문제: "차가 튀어 나오는 상황"만 계속 보여줌
결과: 디코더가 "이 세상은 차만 타면 무조건 누가 튀어 나온다"고 착각

부작용:
- 평범한 상황(Sur가 멀리 있음)에서도 지레 겁을 먹고 브레이크
- 어텐션 값에 과민 반응 (Paranoid behavior)
```

**해결책**: 일반 데이터 10~20% 혼합 → "평소에는 z대로 가는 게 맞다"는 사실 유지

### 8.3 Loss 전략

| Loss 종류 | 목적 | 가중치 |
|-----------|------|--------|
| **Collision Loss** | GT(직진)보다 충돌 회피 우선 | 높음 |
| **Fidelity Loss** | 급제동/급조향 시 Jerk 제약 | 중간 |
| **Reconstruction Loss** | 일반 주행 능력 유지 | 낮음 |

### 8.4 학습 스케줄 예시 (600 epochs 기준)

```
Epoch 0-400:   Phase 1 (Normal data)
               - 전체 모델 학습
               - Reconstruction + KL Loss

Epoch 400-600: Phase 2 (Contrastive data)
               - Encoder freeze 또는 LR 1/100
               - Decoder만 집중 학습
               - + Fidelity Loss 추가
```

---

## 9. 적대적 최적화에서 z의 제약 원리

### 9.1 학습 vs 적대적 최적화: 핵심 차이

| 단계 | 건드리는 것 | 고정하는 것 | 목표 |
|------|------------|------------|------|
| **학습 (Training)** | Weight (W) | - | z가 주어졌을 때 경로 생성 능력 |
| **Fine-tuning** | Decoder Weight | Encoder Weight | 반응성 강화 + Fidelity 유지 |
| **적대적 최적화** | z 값 | 전체 Weight | 충돌 유발하는 z 탐색 |

### 9.2 Prior를 통한 z 제약

모델은 학습을 통해 사람의 일반적인 주행 스타일을 **Gaussian 분포(정규분포)** 형태의 Prior로 저장

**제약 방법 (`MotionPriorLoss`)**:
- 최적화 과정에서 z가 Gaussian 분포 중심에서 멀어질수록 페널티(Loss) 부여
- **결과**: z가 물리적으로 불가능하거나 사람이 절대 하지 않는 영역(Out-of-distribution)으로 튀는 것을 방지
- **"사람의 주행 범주(Fidelity) 안에서"** 가장 공격적인 z 값을 탐색

### 9.3 적대적 최적화의 Gradient Flow

```
1. Forward:
   고정된 W와 현재 z를 넣어서 경로 예측
   예측 결과: "충돌 안 함 (거리 5m)"

2. Loss 계산:
   목표: "충돌해야 함 (거리 0m)"
   Loss: "거리가 너무 멀다!"

3. Backward (역전파):
   Loss 신호가 모델을 거꾸로 타고 올라감
   - 디코더(W_dec) 지나감 → 업데이트 안 함
   - 인코더(W_enc) 지나감 → 업데이트 안 함
   - z에 도착! → "z를 [0.1, -0.5] 쪽으로 바꾸면 거리가 줄어들겠구나!"

4. Update:
   z 값만 수정: z_new = z_old - α × ∇z
```

---

## 10. 전체 연구 흐름 요약

### Step 1: Pre-training (일반 주행 학습)

```
목표: 사람처럼 운전하는 기본기

방법:
- 전체 모델 학습 (Encoder + Decoder)
- 일반 주행 데이터 사용
- Reconstruction + KL Loss

결과: z가 "정상적인 운전자" 분포 형성
      (단, 이 시점에서 Ego는 Sur 공격에 반응 못함)
```

### Step 2: Adversarial 데이터 생성

```
목표: Fine-tuning용 Contrastive 데이터 확보

방법:
- Pre-trained 모델 (Step 1 결과) 사용
- Sur의 z를 최적화하여 "반응 못하는 Ego"를 공격
- 생성된 Adv Sur 궤적 저장

결과: (Normal Sur, Normal Ego) vs (Adv Sur, Reactive Ego) 쌍 확보
```

### Step 3: Fine-tuning (반응성 학습)

```
목표: 사람처럼 운전하되, 위험하면 피하는 Ego

방법:
- 인코더 가중치 고정 (Freeze) → "사람다움 유지"
- 디코더 가중치 학습 (Update) → "반응성(Reaction) 장착"
- Contrastive 데이터 + 일반 데이터 20% 혼합

결과: Agent Attention 활성화, 위기 대처 능력 획득
```

### Step 4: Adversarial Optimization (최종 시나리오 생성)

```
목표: 완성된 Reactive Ego를 뚫고 사고를 내는 시나리오 탐색

방법:
- 모델 가중치 전체 고정 (더 이상 학습 안 함)
- Sur의 z 값만 최적화 (Update z)
- 조건: Ego는 Step 3에서 배운 대로 반응, Sur는 Prior Loss 범위 내에서 최악의 z 탐색

결과: 현실적이면서 위험한 시나리오 생성
```

---

## 11. 핵심 결론

```
"z 벡터 값을 바꾸는 것(공격)"과 "인코더 가중치를 바꾸는 것(학습)"은 완전히 다르다

- 학습 때: 인코더 가중치 고정 (기준점이 흔들리면 안 됨)
- 공격 때: z 값을 마음껏 최적화 (단, Prior Loss 안에서)

이 논리로 "데이터 기반의 리얼한 Ego"와 "적대적 공격" 모두 만족

최종 결과: "사람처럼 운전하되, 위기엔 베테랑처럼 반응하는 모델"
```
