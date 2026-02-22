# RULES.md — Claude 작업 규칙 (반드시 준수)

> 이 룰을 어기고 싶으면 반드시 이유를 설명하고 허가를 받아야 함.

---

## Rule 1: 실제 코드 기반으로만 작업
추측하지 말 것. 반드시 실제 import/사용되는 코드를 읽고 확인한 후 작업.
"이거겠지" 싶어서 쓰지 말고, 파일을 열어서 확인.

### 참조해야 할 실제 파일 목록 (train_trafficplanner.py 기준)

**핵심 파일 (학습 로직 직접 관여):**
- `src/train_trafficplanner.py` — 학습 루프
- `src/models/trafficplanner_model.py` — 메인 모델
- `src/losses/trafficplanner_loss.py` — 로스 계산
- `src/datasets/fit_dataset.py` — 데이터셋
- `src/datasets/fit_map_env.py` — 맵 환경
- `src/models/interaction_net.py` — GCN scene encoder (SceneInteractionNet)
- `src/models/individual_interaction_net.py` — Individual agent encoder
- `src/models/common.py` — MLP, car_dynamics (bicycle model)
- `src/utils/transforms.py` — 좌표 변환 (transform2frame 등)
- `src/losses/common.py` — kl_normal, log_normal

**유틸리티 (간접 참조):**
- `src/utils/torch.py` — save_state, load_state, get_device
- `src/utils/common.py` — dict2obj, Struct, mkdir
- `src/utils/logger.py` — Logger
- `src/utils/config.py` — argument parsing
- `src/datasets/utils.py` — MeanStdNormalizer, normalize_scene_graph
- `src/datasets/nuscenes_utils.py` — collision/map utilities
- `src/datasets/fit_utils.py` — map rasterization
- `src/datasets/map_env.py` — NuScenesMapEnv base class

**설정 파일:**
- `configs/train_trafficplanner.cfg` — Phase 1 학습 config
- `configs/train_finetune_trafficplanner.cfg` — Phase 2 finetune config
- `configs/test_trafficplanner.cfg`
- `configs/test_finetune_trafficplanner.cfg`

---

## Rule 2: TODO.md와 redesign 계획을 항상 참조
작업 전 반드시 TODO.md를 확인하여 현재 어떤 단계에 있는지 파악.
redesign_context.md를 참조하여 전체 목적과 방향성을 잃지 않도록.

관련 파일:
- `/home/hj/RACE_STRIVE/for_claude/TODO.md` — 현재 작업 진행 상황
- `/home/hj/.claude/projects/-home-hj-RACE-STRIVE/memory/redesign_context.md` — 전체 맥락

---

## Rule 3: Ego/Sur 최적화 시 분리 보장
- 학습 시: ego/sur weight 공유 OK (GCN, encoder 등 공유가 효율적)
- **최적화 시**: z_global은 freeze, sur trajectory만 최적화 변수
  - z_global = "운전자의 초기 의도" → sur 최적화로 변하면 안 됨
  - ego는 고정된 z_global + z_local(sur 반응)로 decode
  - model weight 전체 freeze, input(sur trajectory)만 최적화
- 이 구조를 항상 염두에 두고 설계할 것

---

## Rule 4: 변경 불가 (constraints)
다음은 절대 바꾸지 않음:
- z_global + CVAE (prior/posterior/KL) 구조
- GCN (SceneInteractionNet) agent interaction
- Map encoding (CNN 기반)
- Bicycle model output (acceleration, yaw rate)
- 12-step future prediction (0.5s interval)
- Visualization 코드 (viz 관련 함수/파일은 건드리지 않음)

---

## Rule 5: 변경 가능
다음은 자유롭게 변경 가능:
- Decoder 구조 (ego/sur 분리 방식, GRU, attention 등)
- z_local 설계 (cross-attention, window, computation 방식)
- Encoder 구조
- Teacher Forcing 전략 또는 대체 방안
- Loss 구성 및 weight (recon, potential 등)
- Training loop / optimizer 전략
- Data loading pipeline
- Phase 1/2 분리 방식

---

## Rule 6: Autoregressive 유지
Parallel decoding은 상호작용(ego가 sur에 반응)을 반영 못 함.
Autoregressive(step-by-step)는 유지하되, loop body를 최대한 경량화.
- 매 스텝 Python 객체 조작 최소화
- 매 스텝 scene_graph 재구성 최소화
- Pre-allocate tensor, list append 금지

---

## Rule 7: 효율성 우선
- GPU에서 최대한 모든 연산 수행
- Python for-loop body 최소화
- .item()/.cpu()/.tolist() 제거
- Pre-allocate tensor (list append → torch.stack 금지)
- 목표: 현재 3일 → 반나절~하루 이내

---

## Rule 8: 세션 간 연속성 보장
- 중요한 결정/변경사항은 memory/ 폴더의 md에 기록
- 사용자가 요청하면 현재 상태를 md로 요약
- 새 세션에서도 rules.md + TODO.md + memory/*.md로 복원 가능하게

---

## Rule 9: 사용자에게 방향 확인
대규모 구조 변경 전 반드시 사용자에게 방향 확인.
작은 수정(버그 fix, 오타)은 자율 진행 가능.

---

## Rule 10: Python 실행 시 가상환경 활성화
Bash로 Python 명령 실행 시 반드시 `source ~/venvs/strive_env/bin/activate &&` 를 앞에 붙이고, `python3`으로 실행할 것.
```bash
source ~/venvs/strive_env/bin/activate && python3 src/xxx.py
```

---

## Rule 11: GPU 최대 활용
모델 구조 변경 없이 GPU 최대 활용:
- ego/sur attention weight 분리 유지 (공유하여 병렬화하지 않음)
- Python overhead 최소화: pre-allocate buffer, in-place update, list append 금지
- Loss 계산: for-loop 제거, batched tensor operation으로 vectorize
- .item()/.cpu()/.tolist() 사용 금지 (학습 루프 내)
- torch 연산으로 병렬 처리 가능한 것은 모두 torch로
