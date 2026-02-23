# TODO_trans.md — Transformer Decoder 구현 계획

> 기반: redesign_plan_trans_decoder.md + rules.md
> 작성일: 2026-02-23
> 상태: 구현 시작 전
> GRU 버전: git에 백업 완료 → GRU 코드 완전 제거

---

## 개요

GRU decoder → Transformer decoder 완전 교체 (fallback 없음).
Layer 4개, config에서 layer 수 조절 가능.
Training: 12 step 병렬 (causal mask) / Inference: autoregressive.

---

## Phase 0: 사전 준비

### 0-1. Config 업데이트
- [ ] `configs/train_trafficplanner.cfg`에 Transformer 파라미터 추가:
  ```
  trans_num_layers = 4
  trans_d_model = 128
  trans_nhead = 8
  trans_ffn_dim = 512
  trans_dropout = 0.1
  use_ego_z_local = True
  use_sur_z_local = False
  ```

---

## Phase 1: 새 모듈 구현 (trafficplanner_model.py)

### 1-1. TransformerDecoderLayer 클래스
- [ ] 순서: A2T → A2A → A2S → FFN (Pre-norm)
- [ ] ego/sur 분리는 token 단위: ego_mask로 분리 → 각자 처리 → 결합

- [ ] **A2T**: self-attention 1개 (ego/sur 구분 없음)
  - reshape (B, T, N, D) → (B*N, T, D), D=128
  - Q = K = V = LayerNorm(x)
  - causal mask: 하삼각
  - residual: x = x + A2T(x)

- [ ] **A2A**: K/V 공유 1개 + ego_Q/O, sur_Q/O 분리
  - reshape (B, T, N, D) → (B*T, N, D)
  - K/V = shared_KV_proj(LayerNorm(x))  — 1개, ego/sur 공통
  - Q = ego_Q_proj(LayerNorm(x)) 또는 sur_Q_proj(LayerNorm(x))
  - O = ego_O_proj(attn_out) 또는 sur_O_proj(attn_out)
  - residual: x = x + A2A(x)

- [ ] **A2S**: K/V 공유 1개 + ego_Q/O, sur_Q/O 분리
  - K/V = shared_map_KV_proj(map_tokens)  — map_tokens는 에이전트별 위치 기준 crop
  - Q = ego_Q_proj(LayerNorm(x)) 또는 sur_Q_proj(LayerNorm(x))
  - O = ego_O_proj(attn_out) 또는 sur_O_proj(attn_out)
  - attn_weights 반환 (map guidance loss용)
  - residual: x = x + A2S(x)

- [ ] **FFN**: ego_FFN, sur_FFN 완전 분리
  - Linear(128→512) → ReLU → Dropout → Linear(512→128)
  - residual: x = x + FFN(x)

### 1-2. IntentCodebook 수정
- [ ] input_dim: 72 → 128 (Layer 0 출력이 입력, d_model=128)
- [ ] ego/sur z_local 개별 flag로 제어:
  - `use_ego_z_local = True`: ego codebook 활성, z_local concat+proj, intent CE loss 작동
  - `use_ego_z_local = False`: ego z_local=0, concat(token, zeros)→proj, intent CE 무의미
  - `use_sur_z_local = True`: sur codebook 생성+활성, concat+proj
  - `use_sur_z_local = False`: sur z_local 없음, token 그대로 Layer 1로 전달
  - Phase와 무관하게 flag로만 제어 (IntentCodebook 내부에 phase 체크 없음)
- [ ] sur_intent_ce_head: Linear(32 → num_intents), use_sur_z_local=True 시에만 생성
- [ ] 위치 고정: Layer 0 ~ Layer 1 사이
  - ego_token → ego_codebook → z_local_ego → concat(token, z_local) → Linear(160→128)
  - sur_token → (use_sur_z_local 시) sur_codebook → z_local_sur → concat → Linear(160→128)
  - sur_token → (disable 시) 그대로 Layer 1 입력

### 1-3. Token 구성 함수
- [ ] `build_decoder_tokens()` 함수
  - GCN 출력(64) → Linear(64→128) projection
  - token[t] = Linear([GCN_feat_proj(128) + z_global(32) + lw(2) + sem(NC)]) → 128dim
  - temporal_pe: nn.Embedding(16, 128) learnable, 더하기
  - Output: (B, T_total, N, 128), T_total = PT(4) + FT(12) = 16

### 1-4. Causal Mask 생성
- [ ] `build_causal_mask(T_total=16)` 함수
  - A2T에서만 사용
  - 하삼각 mask (16×16)

### 1-5. Pred Loss Head 위치 고정
- [ ] sur_pred_head: Linear(128→2), ego_pred_head: Linear(128→2)
- [ ] Layer 0의 A2A 직후에서 추출 (고정)

---

## Phase 2: Forward 구현

### 2-1. Training Forward (병렬)
- [ ] `transformer_decoder_training()` 구현
  ```
  1. GT state 16 step → GCN 병렬 → gcn_feats (B, 16, N, 64)
  2. build_decoder_tokens() → tokens (B, 16, N, 128)
  3. Layer 0:
     A2T(causal) → A2A → [pred loss 추출] → A2S → [map loss 추출] → FFN
  4. Intent 선택 (ego/sur 각각)
     concat(token, z_local) → Linear(160→128)
  5. Layer 1~3:
     A2T → A2A → A2S → FFN
  6. Output:
     ego_token[future_steps] → ego_output_head(Linear 128→2) → (acc, yaw_rate)
     sur_token[future_steps] → sur_output_head(Linear 128→2) → (acc, yaw_rate)
  7. Bicycle model → predicted trajectory (NA, FT, 4)
  8. Loss: future 12 step에 대해서만
  ```

### 2-2. GCN 16 step 병렬 처리
- [ ] Training 시 GT state가 있으므로 16 step scene graph 한번에 구성
  - 각 step의 state로 scene_graph 구성 → GCN forward
  - for loop 최소화, 가능하면 batch로

### 2-3. Inference Forward (autoregressive)
- [ ] `transformer_decoder_inference()` 구현
  ```
  1. Past 4 step → GCN → past tokens (B, 4, N, 128)
  2. Transformer layers → future step 0 예측
  3. bicycle model → new state → GCN → 새 token append
  4. Transformer layers (B, 5, N, 128) → future step 1 예측
  5. 반복 12회
  ```
- [ ] 매 step에서 전체 sequence를 다시 Transformer에 넣음 (KV-cache는 추후)

### 2-4. forward() 메서드 수정
- [ ] 기존 GRU 경로 제거
- [ ] teacher_forcing 파라미터에 따라:
  - True → transformer_decoder_training()
  - False → transformer_decoder_inference()
- [ ] 출력 형식: (pred_traj, analysis_dict)

---

## Phase 3: Loss 연결

### 3-1. Auxiliary Loss 추출 (analysis_dict에 저장)
- [ ] Sur/Ego Pred: Layer 0 A2A 직후
- [ ] Ego Map Attn Guidance: Layer 0 A2S ego attn_weights
- [ ] Sur Map Attn Guidance: Layer 0 A2S sur attn_weights
- [ ] Ego Intent: Layer 0~1 사이 ego_intent_weights, ego_z_local
- [ ] Sur Intent: Layer 0~1 사이 sur_intent_weights, sur_z_local (use_sur_z_local 시)

### 3-2. Loss Config 정리
- [ ] 각 auxiliary loss를 ego/sur 별도 weight로 관리:
  ```
  # Reconstruction
  loss_recon = 1.0              # ego + sur 합산 MSE

  # KL
  loss_kl = 0.004               # z_global KL

  # Pred (A2A 품질 감독)
  loss_sur_pred = 1.0           # ego→sur delta 예측 MSE
  loss_ego_pred = 1.0           # sur→ego delta 예측 MSE

  # Map Attention Guidance
  loss_ego_map_attn = 0.1       # ego map attn KL
  loss_sur_map_attn = 0.1       # sur map attn KL

  # Intent CE (Phase 2)
  loss_ego_intent_ce = 0.0      # ego intent KL (Phase 2에서 활성화)
  loss_sur_intent_ce = 0.0      # sur intent KL (use_sur_z_local=True 시)

  # Potential
  loss_potential_env = 0.1
  ```

### 3-3. Reconstruction Loss 수정
- [ ] TF segment 방식 제거 → future 12 step tensor 직접 MSE
- [ ] `_forward_teacher_forcing()` 단순화 또는 제거

### 3-4. TrafficPlannerLoss 수정
- [ ] Transformer 출력 (NA, FT, 4) tensor에 맞게 loss 계산
- [ ] segment list 처리 로직 제거
- [ ] ego/sur 별도 auxiliary loss 계산 로직 추가

---

## Phase 4: 제거 & 정리

### 4-1. 제거할 모듈/메서드
- [ ] `ego_decoder_gru`, `sur_decoder_gru`
- [ ] `ego_warmup_gru`, `sur_warmup_gru`
- [ ] `ego_history_attn`, `sur_history_attn` (DecoderHistoryAttention 클래스)
- [ ] `ego_map_attn`, `sur_map_attn` (MapCrossAttention 클래스)
- [ ] `history_buffer` 관련 로직
- [ ] `_warmup_gru_hidden()`
- [ ] `_sur_loop_decoder()`, `_ego_loop_decoder()`
- [ ] TF segment 관련 코드
- [ ] PositionalEncoding 클래스 (Transformer encoder용이었다면 → temporal PE로 교체)

### 4-2. 유지할 모듈
- [ ] Encoder 전체 (prior, posterior, temporal GCN, map CNN)
- [ ] `interaction_gcn` — 토큰 생성용
- [ ] `encode_map()` — conv1-6, map_tokens
- [ ] `_apply_dynamics()` — bicycle model
- [ ] Auxiliary loss heads (위치만 변경)
- [ ] Public API (forward, reconstruct, sample, embed, decode_embedding)
- [ ] Phase control, Analysis getters

---

## Phase 5: 테스트

### 5-1. Forward Pass 테스트 수정
- [ ] model instantiation (Transformer 파라미터)
- [ ] training forward: output shape (NA, FT, 4) 확인
- [ ] inference forward: autoregressive 12 step 확인
- [ ] auxiliary outputs 존재 확인
- [ ] loss backward: gradient flow 확인

### 5-2. 병렬 vs 순차 일관성
- [ ] Training(병렬)과 Inference(AR)가 동일 GT input일 때 같은 결과
- [ ] causal mask 정상 작동 검증

---

## Phase 6: 학습 실행

### 6-1. Phase 1 학습
- [ ] 학습 실행
- [ ] 목표: val_loss ≤ 3.70
- [ ] train/val gap ≤ 0.03

### 6-2. 모니터링 & 조정
- [ ] A2A attention weight 시각화 → mask 필요 여부 판단
- [ ] Map attention weight 시각화
- [ ] layer 수 실험 (config에서 2~6 조절)

---

## 구현 순서

```
0-1              Config 준비
 ↓
1-1 → 1-2 → 1-3 → 1-4 → 1-5   새 모듈 (Layer, Intent, Token, Mask, Pred)
 ↓
2-1 → 2-2 → 2-3 → 2-4         Forward (Training병렬 → GCN병렬 → Inference → 통합)
 ↓
3-1 → 3-2 → 3-3               Loss 연결
 ↓
4-1 → 4-2                     정리 (제거 & 유지)
 ↓
5-1 → 5-2                     테스트
 ↓
6-1 → 6-2                     학습 & 모니터링
```

---

## ego/sur 분리 전략 (확정)

| Module | 구조 | Phase 2 Freeze |
|--------|------|----------------|
| A2T | self-attention 1개 (구분 없음) | 전체 freeze |
| A2A | K/V 공유 + Q/O 분리 | K/V + sur_Q/O freeze, ego_Q/O 학습 |
| A2S | K/V 공유 + Q/O 분리 | K/V + sur_Q/O freeze, ego_Q/O 학습 |
| FFN | 완전 분리 | sur freeze, ego 학습 |
| Intent | 별도 codebook | sur freeze, ego 학습 |

---

## 학습 전 검토 사항

- [ ] Config 점검: 기존 config 파라미터 중 Transformer와 충돌/불필요한 항목 정리
- [ ] Dropout 전략 검토: Adv-BMT는 dropout=0.0 (480K data), 우리는 5K data → 0.1로 시작하되 과적합 양상 보면서 조절
- [ ] LR scheduling 고려: cosine warmup (Adv-BMT 방식) 도입 검토
  - Adv-BMT: lr=3e-4, warmup 2000 steps, cosine decay, 10M steps, batch 2
  - 우리: 데이터/스텝 규모 다르므로 warmup steps 조절 필요
  - 구현 우선순위 낮음 — 기본 학습 돌려보고 필요시 추가
- [ ] Optimizer 검토:
  - Adv-BMT: AdamW (lr=3e-4, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.0)
  - beta2=0.95는 Transformer 학습 표준 (GPT, LLaMA 등), PyTorch 기본 0.999보다 낮음
  - gradient clipping=1.0 사용
  - weight_decay=0.0이면 사실상 Adam과 동일 → 우리는 0.01 정도 시도 고려
- [ ] KL annealing 단위 검토:
  - 현재 코드: epoch 단위 (compute_kl_weight(cur_epoch, end_epoch, ...))
  - step 단위로 바꿀 수도 있으나, 현재 구조로도 충분 → 유지
- [ ] Adv-BMT 학습 규모 참고:
  - Forward: 10M steps, 185시간 / Reverse: 15M steps, 310시간
  - epoch 없이 step 단위로 학습 관리
  - 480K data, batch 2 → 1 epoch = 240K steps → ~42 epochs

---

## 미결 사항

- d_model=128, nhead=8, FFN=512 (확정)
- A2A mask → attention weight 시각화 후 결정
- KV-cache → 학습 잘 되면 추후 최적화
