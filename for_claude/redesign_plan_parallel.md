# GRU TF Decoder 병렬화 설계 문서

> 작성일: 2026-02-24
> 브랜치: 260222_trial
> 상태: 구현 완료, 검증 통과

---

## 0. 목적

GRU 기반 모델의 Teacher Forcing(TF) 학습이 for loop 12회로 실행되어 느림.
**모델 구조/의도는 변경하지 않고**, 실행 방식만 배치화하여 학습 속도 향상.

### 불변 사항 (절대 변경 금지)
- z_global + VAE (prior/posterior/KL) 구조
- GCN (IndividualSceneInteractionNet) message passing
- GRU decoder (ego/sur 분리)
- Bicycle model output
- Intent codebook (ego/sur 각각 flag로 관리)
- History/Map cross-attention 모듈
- AR inference 경로 (autoregressive_decoder) — 일절 변경 없음

### 핵심 원리
TF 모드에서는 모든 timestep의 GT 상태를 미리 알고 있으므로:
- GCN: 12개 graph를 Batch로 합쳐 1회 호출
- Attention/Intent: GT 기반 입력을 미리 준비하여 배치 계산
- GRU 1-step: timestep을 batch 차원에 합쳐 (N*FT, 1, D) 1회 호출
- 수학적으로 기존 순차 실행과 동일한 결과

---

## 1. 현재 TF 구조 (3-step)

### Step A: GCN 12회 호출 → gt_gcn_cache
```
for t in range(FT=12):
    scene_graph.x = gcn_input[t]  # GT 기반
    scene_graph.pos = gt_pos[t]
    ego_feat, sur_feat = interaction_gcn(scene_graph, ego_mask)
    gt_gcn_cache[:, t, :] = feat
```

### Step B: GRU hidden snapshot 순차 구축
```
h[0] = warmup(past)
for t in range(FT):
    gru_input[t] = [hist_ctx, map_ctx, z_global, z_local, lw, sem]
    _, h[t+1] = GRU(gru_input[t], h[t])  ← 순차 필수
    snapshots.append(h[t+1])
```

### Step C: 독립 1-step prediction (12회 loop)
```
for t in range(FT):
    restore h[t]
    hist_ctx = history_attn(h_last, gcn_cache[:t+1])  ← 누적 참조
    map_ctx = map_attn(h_last, map_tokens[t])
    intent = codebook(state[t])
    gru_out = GRU(input, h[t])  ← 독립 (t간 의존성 없음)
    pred[t] = bicycle(output_head(gru_out), prev_state[t])
```

---

## 2. 병렬화 설계

### Step A 변경: Batch GCN (1회 호출)
- `_build_temporal_graph()`로 12개 graph를 수동 결합 (Batch.from_data_list() 미사용 — torch_geometric 1.7.1 호환 문제)
- edge_index에 `+NA*t` offset → 서로 다른 timestep의 노드 간 edge 없음 → message passing 완전 격리
- ego_mask.repeat(FT) → ego/sur feature 분리
- reshape → (N, FT, D) = gt_gcn_cache

### Step B 변경: GRU hidden snapshot 구축
- GRU loop는 유지 (hidden 누적 불가피, `h[t]` → `h[t+1]` 의존)
- loop 내부에서 GT 기반 input 구성 (History Attn, Map Attn, Intent)
- Step A의 gt_gcn_cache 활용하여 GCN 추가 호출 불필요
- 출력: `gt_hidden_snapshots[0..FT-1]`

### Step C 변경: 완전 배치화 (1회)
- timestep 차원을 batch에 합침: (N, ...) → (N*FT, ...)
- **모든 flat 텐서를 time-major layout으로 통일** (중요!)
  - time-major: `(FT, N, D).reshape(N*FT, D)` → [t0a0, t0a1, ..., t1a0, ...]
  - `(N, FT, D)` 형태는 `.permute(1,0,2).reshape(N*FT, D)` 적용
- History Attn: causal mask로 가변 길이 처리
  - t=0: zero output, t=k: 0~k-1만 attend
- Map Attn, Intent, GRU, Output, Bicycle: 모두 (N*FT, ...) 배치
- 결과 reshape: `.reshape(FT, N, 4).permute(1,0,2)` → (N, FT, 4)
- 분석 저장: reshape 후 list 분할 (loss 호환)

---

## 3. 핵심 모듈 변경

### DecoderHistoryAttention.forward_batched (신규)
- queries: (N*FT, D), history_buffer: (N, FT, D)
- expand → (N*FT, FT, D) + causal float mask (-inf)
- 1회 cross_attn → (N*FT, D) → t=0 zero out
- PE(positional encoding): learnable nn.Embedding, 기존과 동일하게 적용

### MapCrossAttention — 변경 없음
- 기존 forward가 이미 임의 batch 지원
- 호출 시 (N*FT, tokens, ch) 입력만 하면 됨

### IntentCodebook — 변경 없음
- (N*FT, input_dim) 입력 가능

### _recompute_map_tokens — mult_samp 지원 추가
- mult_samp=True 시 mapixes expand

---

## 4. TF vs AR 일관성 보장

| 항목 | TF (학습) | AR (추론) |
|------|----------|----------|
| GCN | GT states → 1회 Batch | predicted states → step별 |
| History Attn | batched + causal mask | 기존 forward(query, buf, t) |
| Map Attn | batched map_tokens | step별 map_tokens |
| Intent | batched states | step별 state |
| GRU | Step C: batched 1-step | carried forward |
| Output | batched | step별 |
| Bicycle | batched | step별 |
| z_global | ego/sur 분리, 각자 decoder에만 | 동일 |
| z_local(intent) | use_ego/sur_intent flag | 동일 |

**AR 코드 변경: 없음.** autoregressive_decoder() 함수는 일절 수정하지 않음.

---

## 5. Auxiliary Loss 호환

- `_ego_pred_outputs`: list of (N, pred_dim) × FT → 유지
- `_sur_pred_outputs`: 동일
- `_intent_weights_outputs`: 동일
- `_ego_map_attn_weights_outputs`: 동일
- 배치 결과를 reshape 후 list로 분할하여 기존 format 보존

---

## 6. 버그 수정 (debugged 브랜치 참조)

| 항목 | 파일 | 내용 |
|------|------|------|
| Data split | fit_dataset.py | 50/16.7/33.3 → 85/10/5 |
| save/load | torch.py | global_step 추가, 3개 반환 |
| LR scheduler | train_trafficplanner.py | step LambdaLR + warmup floor |
| Config | train_trafficplanner.cfg | batch 4, lr_max 3e-4, kl 50 |
| Grad clip | train_trafficplanner.py | max_norm=1.0 |
| map_recrop | trafficplanner_model.py | mult_samp 지원 |

---

## 7. 구현 중 발견한 버그 및 해결

### Bug 1: torch_geometric Batch.from_data_list 호환
- **증상**: Data 객체에 batch/ptr 있으면 AssertionError (torch_geometric 1.7.1)
- **해결**: `_build_temporal_graph()` 수동 구현 — x, pos, sem concat + edge_index offset + batch/ptr 직접 생성

### Bug 2: History Attention t=0 NaN
- **증상**: TF loss NaN, backward에서 전파
- **원인**: t=0에서 참조할 history 없음 → causal mask 전부 `-inf` → softmax NaN
- **해결**: forward_batched에서 t=0 제외, t=1..FT-1만 MHA 처리, t=0 output = zero

### Bug 3: pos reshape 순서 불일치 (Step A)
- **증상**: GCN 출력 max_diff ~0.49
- **원인**: `all_states[:,:,:4].reshape(NA*FT,4)` = agent-major, x/sem = time-major
- **해결**: `.permute(1,0,2).reshape(NA*FT,4)`로 time-major 통일

### Bug 4: Layout mismatch (Step C) — 가장 심각
- **증상**: 동일 cache/snapshot으로도 순차 vs 배치 결과 다름 (CPU max_diff ~1e-3)
- **원인**: flat 텐서 reshape 순서 혼재
  - `hist_ctx_flat` = `(N,FT,D).reshape(N*FT,D)` → agent-major [a0t0, a0t1, ..., a1t0, ...]
  - `all_hidden_gru` = `(3,FT,N,D).reshape(3,N*FT,D)` → time-major [t0a0, t0a1, ..., t1a0, ...]
  - batch index `i`에서 서로 다른 (agent, timestep) 쌍 참조
- **해결**: 모든 flat 텐서를 time-major로 통일
  - `(N,FT,D)` → `.permute(1,0,2).reshape(N*FT,D)`
  - per_t 저장: `.reshape(FT,N,...).permute(1,0,2)` → `(N,FT,...)`
  - 최종: `.reshape(FT,N,4).permute(1,0,2)` → `(N,FT,4)`

---

## 8. 검증 결과

### test_forward_pass.py — 12개 테스트 ALL PASSED
- Model instantiation, Forward, TF, Reconstruct, Sample, Loss, Phase2, Map Attn, Intent CE 등

### test_layout_fix.py — Step C 수치 동일성 (CPU)
- 순차 loop vs 배치화, 동일 gt_gcn_cache + gt_hidden_snapshots 공유
- 중간 텐서 비교:
  - gru_h_last: 0.00 (완벽)
  - hist_ctx: ~4e-7 (MHA float32 연산 순서 차이)
  - map_ctx: ~3e-8
  - gru_out: ~3e-7
  - hidden state: 0.00 (완벽)
- **최종 trajectory: 12 timestep 전부 max_diff = 0.00 (완벽 동일)**

### Step A GCN 배치화 검증 (CPU)
- 12회 loop vs 1회 배치: max_diff = 1e-7 (float32 연산 순서)

---

## 9. Map Attention Soft Label 재설계

### 기존 방식 (discrete)
- 각 GT future step → grid cell 1개에 discrete 매핑 (841개 중 6개만 nonzero)
- KL divergence로 맞추기 거의 불가능 (attention이 정확히 6개 셀에 확률 집중 필요)

### 새 방식 (trajectory-based continuous)
- GT future polyline (agent 위치 + K개 미래 점)을 따라 연속 분포 생성
- 각 grid cell에서:
  1. Polyline의 가장 가까운 segment까지 수직 거리(d)와 arc length(s) 계산
  2. Longitudinal: `exp(-λ * s)` — 가까울수록 높음 (multiplicative decay)
  3. Lateral: `exp(-0.5 * d² / σ_d²)` — trajectory 중심선 Gaussian spread
  4. Behind cutoff: agent 뒤쪽 cell은 0 (첫 segment proj < 0)
  5. Normalize to probability distribution

### Hyperparameters
| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `map_gt_steps` | 6 | 미래 참조 step 수 |
| `map_gt_decay_lambda` | 0.3 | 종방향 exponential decay rate |
| `map_gauss_sigma_d` | 0.8 | 횡방향 Gaussian σ (grid cells, ~2.1m) |
| `loss_map_attn` | 0.1 | Map attn guidance loss weight |

### Grid 해상도
- 1 grid cell = 2.66m (77m / 29 cells)
- 차선 폭 3.5m ≈ 1.3 cells → 세밀한 차선 guidance는 해상도 한계

### Map Attention Loss Annealing
- Config: `map_attn_anneal: True`, `map_attn_anneal_epochs: 200`
- Epoch 기반 cosine decay: `weight = initial * 0.5 * (1 + cos(π * epoch / anneal_epochs))`
- 500 epoch 기준: 0~200 epoch guide → 200~500 epoch 자유 학습

### 수정 파일
- `src/losses/trafficplanner_loss.py` — `_make_soft_label` 전면 재작성
- `src/train_trafficplanner.py` — annealing 로직, args 추가
- `configs/train_trafficplanner.cfg` — sigma_d, anneal 설정 추가
