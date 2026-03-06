# TODO — 260306_no_tf: TF/Blending 제거로 z_global 살리기

## 핵심 발견: Teacher Forcing이 z collapse의 근본 원인

### 실험 결과 (260306)

**TF+Blending 있을 때** (이전 모든 학습):
- KL: 0.1 → 0.0000003 (완전 collapse)
- z_mdist: 0.31 → 0.00002
- z_global/std: 0.62 → 0.03 (z 죽음)
- train-val gap: 5x

**TF/Blending 끈 후** (full AR):
- KL: 0.1 → **17.6** (계속 상승)
- z_mdist: 0.38 → **3.81**
- z_global/std: 0.82 → **1.10** (z 살아있음)
- train-val gap: **거의 없음** (0.15 vs 0.14)

### 왜 TF가 z를 죽이는가

```
TF/Blending ON (blend_alpha < 1):
  pred_action → blend → (1-α)*GT_action + α*pred → bicycle_model → next_state
                         ^^^^^^^^^^^^^^^^
                         GT action이 방향 정보를 줌
  → next_state ≈ GT state → GCN 입력에 GT 방향 정보 포함
  → decoder가 GCN만으로 방향 파악 가능 → z 불필요 → KL=0

Full AR (TF OFF):
  pred_action → bicycle_model → next_state
  → next_state = 순수 예측 결과 → GCN에 GT 정보 없음
  → 방향 정보는 z가 유일한 경로 → z 학습 필연
```

### z=0 실험으로 검증

z_samp을 0으로 고정하고 학습:
- **train loss**: 줄어듦 (blend=0.05 → 95% GT action 보정)
- **val loss (full AR)**: 안 줄어듦 → z 없이는 inference 불가

→ **구조상 z가 필요하지만, TF가 학습 시 z의 필요성을 제거**하는 것이 문제

---

## 구조 (변경 없음 — TF/Blend만 제거)

### 전체 흐름

```
[1회 실행 — Encoding]

(1) Past GCN: past 4 steps → past_tokens (4, 128)     ← temporal PE 부여
(2) Map CNN:  map image → map_tokens (57×57)
              map_pool → map_summary (8, 128)           ← PE 없음
(3) Context Encoder:
    [past_tokens + map_summary] → self-attn 2L → context (12, 128)
    ※ z 미포함 — past-map 관계만 정제 ("이 도로에서 감속" 등)

(4) CVAE:
    posterior(past, future, map) → z ~ N(μ, σ)         ← z 생성 (32d)
    prior(past, map) → z_prior                         ← KL 학습용

(5) z Cross-Attention:
    z_tokens = z_proj(z_global) → (4, 128)             ← z를 4개 토큰으로 분할
    z_context = cross_attn(Q=z_tokens, KV=context) → (4, 128)
    ※ z가 query → z가 "뭘 읽을지" 결정
    ※ z가 의미 없으면 엉뚱한 정보 읽음 → 예측 실패


[AR loop — Decoding, step t = 0..11]

(6) GCN(current_state) → gcn_feat (N_agents, 64)
(7) decoder_token = proj(gcn_feat ⊕ lw ⊕ sem) + step_PE(t)
    ※ query에 z 없음 — z_global은 A2Z로만, z_local은 Layer 0→1 사이 삽입

(8) Decoder Layer 0:
      A2A(decoder_token)                          → agent 상호작용
      A2Z(Q=위 결과, KV=z_context, 4토큰)         → z/past trend 정보 흡수
      A2S(Q=위 결과, KV=map_tokens_recrop)         → map 고해상도 정보
      FFN

    Intent codebook: Layer 0 출력 ego token → codebook → z_local(32d)
      ego_token = cat([ego_token, z_local]) → proj(160→128) → ego token 교체

(8') Decoder Layer 1:
      A2A (z_local 반영된 ego token이 query)      → agent 상호작용
      A2Z (z_local + z_global 정보 결합)           → z/past trend 정보 흡수
      A2S                                          → map 고해상도 정보
      FFN

(9) output_head → action (a, ddh)
(10) bicycle_model → next_state
     ※ TF/blending 없음 — full AR (자기 예측으로만 진행)
```

### 정보 경로 정리

| 정보 종류 | 경로 | z 의존 |
|-----------|------|--------|
| Past trend (가속/감속/커브) | posterior → z → z_context → A2Z | **필수** |
| Future 의도 (좌/우회전) | posterior → z → z_context → A2Z | **필수** |
| Map 환경 (도로/차선) | Map CNN → A2S | 불필요 |
| Agent 상호작용 | GCN → A2A | 불필요 |
| 현재 상태 (속도/heading) | GCN → decoder_token | 불필요 |

→ past trend + future 의도는 **z_context가 유일한 경로** → z 의존 필연
→ TF가 이 "유일한 경로" 제약을 **GCN 경유 GT leak으로 우회**시켰던 것이 근본 원인

---

## Config 변경 (2줄만)

```
use_teacher_forcing: False    # 기존 True
action_blending: False        # 기존 True
```

모델 코드 변경: **없음**

---

## 제거 항목

| 제거 대상 | 이유 |
|-----------|------|
| **Teacher Forcing** | GT action → bicycle → GCN → 방향 정보 leak → z collapse 근본 원인 |
| **Action Blending** | blend_alpha < 1이면 GT action이 z를 대체. α=0.05면 95% GT |

## 유지 항목

| 유지 대상 | 비고 |
|-----------|------|
| Context Encoder (past+map self-attn) | z만 빠짐, past-map 관계 정제 역할 유지 |
| CVAE (prior/posterior/KL) | z 생성 구조 동일 |
| ZCrossAttention | z(Q) × context(KV) → z_context |
| A2Z in decoder | Q=decoder_token, KV=z_context (4토큰) |
| GCN (encoder + decoder) | agent 상호작용 |
| Map CNN + A2S | 환경 정보 (고해상도) |
| A2A + RelBias | agent 상호작용 |
| Intent codebook + z_local | Layer 0→1 사이 삽입 |
| Bicycle model | action → state |
| 모든 loss 종류 | recon, KL, map_attn, intent_ce, sur/ego_pred, potential |

---

## 검증 결과 (epoch 0-11, full AR)

### z 분포 건강 지표

| 지표 | TF 있을 때 (collapse) | TF 없을 때 (현재) | 의미 |
|------|----------------------|-------------------|------|
| KL loss | 0.0000003 | **17.6** | posterior ≠ prior, z가 정보 담음 |
| z_mdist | 0.00002 | **3.81** | posterior가 prior에서 3.8σ 벗어남 |
| z_global/std | 0.03 | **1.10** | 시나리오마다 다른 z 생성 |
| z_global/norm | 3.8 | **7.95** | z 벡터 크기 유지 |
| prior_post_gap | 0.017 | **0.85** | posterior가 future 정보 인코딩 |
| kl_per_dim_mean | 0.026 | **0.68** | 32차원 전부 의미있는 정보 |
| posterior_var | 0.10 | **0.41** | 적절한 불확실성 유지 |
| active_dims | 32 (형식적) | **32** (실질적) | 모든 차원 활성 |

### 학습 성능

| 지표 | TF 있을 때 | TF 없을 때 | 비교 |
|------|-----------|-----------|------|
| val pos_loss | 0.34 (ep9) | **0.14** (ep9) | 2.4x 개선 |
| val recon | 4.30 | **3.95** | 개선 |
| train-val gap | 5x | **~1x** | overfitting 없음 |
| ang_err (val) | 13.2° | **13.1°** | 비슷 |
| intent_ce | 1.03 (랜덤) | **1.10** (학습 중) | 개선 중 |

---

## 브랜치 정보

| 브랜치 | 내용 |
|--------|------|
| `0306_trans_encoder` | base — TF+blend ON, 원래 구조 |
| `0306_no_tf_trans_encoder` | **이 문서** — TF/blend OFF만, 모델 변경 없음 |
| `0306_force_z_add` | 참고용 — z_direct_proj + residual add + TF OFF (z 살아있음 확인) |

---

## 후속 과제

1. **장기 학습 안정성 확인**: full AR이 epoch 50+ 이후에도 안정적인지
2. **error accumulation 대응**: 필요 시 scheduled sampling (blend를 나중에 서서히 도입)
3. **map_attn_loss 실험**: TF 없는 상태에서 map_attn_loss가 z에 미치는 영향 재확인
4. **z 품질 정성 평가**: z 변경 시 trajectory 변화 시각화 (적대적 최적화)

---

## TODO (260306 세션 후반)

### viz_attn_intent.py 좌표 수정 필요 (0306_force_z_add 브랜치 stash)
- `_overlay_heatmap`: 기존 단순 zoom → RF-aware canvas 매핑으로 변경 (작업 중)
- `compute_soft_label_grid`: `pix2grid = grid_size/pix_size` → RF 기반 `_m2token()` 변경 완료
- **아직 얼라인 안 맞음**: 축 방향(long/lat ↔ x/y) 매핑 재검증 필요
- 이론상 `grid[i,j]` i=long, j=lat → `canvas[py,px]` px=long, py=lat 맞지만 실제 이미지에서 어긋남
- `0306_force_z_add` 브랜치에 stash 되어 있음

### Loss weight 밸런스 분석 결과
- recon_loss가 total의 99.4% 독점 (log_normal 상수 텀 포함)
- KL weight=0.001 → 0.2% (너무 작지만, 구조적으로 z 살아있음)
- intent_ce: 2.05→1.08 (학습 중, 9slot 중 6개 활용)
- map_attn: loss=0 → A2S uniform (학습 안 됨, map 활용 못 함)
- map_attn_loss 넣으면 A2S 부트스트랩 가능 (weight 0.005 수준)

### 학습 상태 (epoch 33 기준)
- pos_err: train=2.4m, val=2.3m (gap ≈ 1.0x)
- KL: 17.6→7.0 (하락 추세, 아직 건강 범위)
- z_global/std ≈ 1.0 (살아있음)
- active_dims: 32/32
