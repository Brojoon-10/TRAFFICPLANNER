# Enc-Dec Cross-Attention 구조 전환 설계

> 작성일: 2026-03-04
> 상태: 설계 완료, 구현 대기
> 기반: 현재 GPT-style Transformer decoder (trafficplanner_model.py)

---

## 0. 배경 — 왜 이 변경이 필요한가

### 현상: z_global Posterior Collapse
- KL loss: epoch 0에서 4.0 → epoch 5에서 0.1 → epoch 40+ 에서 0.03 (사실상 prior ≈ posterior)
- z_mdist ≈ 0.85 (posterior mean ≈ prior mean)
- z_logprob: -33 → +26 (z가 점점 N(0,1)에 수렴)
- **결과**: z_global을 바꿔도 trajectory 변화 없음 → 적대적 최적화 불가능

### 근본 원인: 디코더가 z 없이 해결 가능
- **A2T (temporal self-attention)**: causal mask로 이전 토큰(past + 이전 predicted)을 직접 attend
- step 11을 예측할 때 past₀..₃ + pred₀..₁₀의 패턴을 보고 외삽 가능
- z_global보다 직접적이고 정확한 정보원 → z를 무시하는 법을 학습
- **map도 동일**: A2T로 trajectory 패턴을 직접 보면 도로 구조 몰라도 예측 가능 → A2S 무시 가능

### 이전 시도들 (효과 없었음)
- `dec_past_dropout=0.5`: past tokens를 50% 드랍 → A2T가 여전히 남은 토큰 + pred 토큰으로 해결
- `kl_free_bits=0.05`: KL 하한선 설정 → 구조적 문제를 loss 가중치로 해결 불가
- `loss_z_aux`: z에서 trajectory 복원 보조 loss → 디코더가 안 쓰는 건 변하지 않음

### 검토한 대안들과 선택 근거

| 대안 | 설명 | 기각 이유 |
|------|------|----------|
| **인코더 강화** | 인코더를 더 크게/강하게 | 문제는 인코더가 약한 게 아니라 디코더가 z를 안 씀. 인코더가 아무리 좋은 z를 만들어도 디코더가 무시하면 무의미 |
| **디코더 축소** | d_model/FFN/layer 줄이기 | 주행 모사 능력도 같이 줄어듦. 특히 multi-agent interaction, map 활용이 약화 |
| **Flow/Diffusion 기반** | 디코더 자체를 교체 | 실시간 시나리오 대응 불가. z_global 최적화 구조와 비호환 |
| **인코더-디코더 구분 없는 통합 구조** | 전체를 하나의 Transformer로 | AutoBots와 유사. 결국 동일한 collapse 문제 발생 가능 |
| **Enc-Dec Cross-Attention** ✅ | 디코더→인코더 정보 경로 제한 | 구조적으로 z bypass 차단. 기존 모듈 최대 재사용 |

### 260222 모델 (GRU 기반) 비교
- 이전 GRU 모델도 `hist_ctx + map_ctx`로 past/map 직접 접근 → z bypass 가능
- History Attention이 A2T와 동일한 역할 (과거 trajectory 직접 참조)
- **결론**: GRU든 Transformer든, past 직접 접근이 가능하면 collapse 발생

### 해결 원리
- 디코더가 past trajectory를 **직접 접근 못하게** 하고
- z_global이 포함된 encoder context를 통해서만 접근하도록 구조 변경
- z를 빼면 context 품질 저하 → recon 악화 → z에 정보 담을 gradient 유지
- **map도 동일 원리**: context에 map_summary 포함 → map 없이는 context 품질 저하

---

## 1. 전체 아키텍처

### 1a. 현재 (GPT-style) — 문제 있는 구조
```
토큰: [past₀, past₁, past₂, past₃, pred₀, pred₁, ..., pred₁₁]
                    ↕ A2T causal self-attention ↕
                    pred₁₁이 past₀..₃ + pred₀..₁₀ 전부 직접 attend
                    → z 없이도 "다음 step" 외삽 가능
                    → z collapse
```

### 1b. 변경 후 (Enc-Dec Cross-Attention)
```
┌─────────────────── Context Encoder (1회) ───────────────────┐
│                                                               │
│  past_gcn = GCN(past 4 steps) → (NA, 4, 64) → proj(128)     │
│  z_tokens = MLP(z_global)     → (NA, K, 128)  K=4 default   │
│  map_summary = MapSummaryPooling(map_tokens) → (NA, M, 128)  │
│                                                               │
│  context_input = [past₀+PE₀, ..., past₃+PE₃,                │
│                   z₀, ..., z_{K-1},                           │
│                   map_s₀, ..., map_s_{M-1}]                   │
│                                                               │
│  TransformerEncoder(self-attention, num_layers=config)        │
│  → context_tokens (NA, 4+K+M, 128)                           │
│  ※ past + z + map이 self-attend → 서로 분리 불가             │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                    │ K, V (고정)
                    ▼
┌─────────────────── Decoder AR Loop (12회) ───────────────────┐
│                                                               │
│  매 step t:                                                   │
│    1. GCN(현재 agent 상태) → interaction_feat (NA, 64)        │
│    2. intent_codebook(ego_feat) → z_local                     │
│    3. query = proj(gcn ⊕ z_local ⊕ lw ⊕ sem) + step_PE(t)   │
│                                                               │
│    4. Decoder Layer (×2):                                     │
│         A2A: agent 상호작용 (ego-sur)                         │
│         A2C: context cross-attn (query→context) ← ★핵심★     │
│         A2S: map cross-attn (query→map_tokens)                │
│         FFN: ego/sur 분리                                     │
│                                                               │
│    5. output_head → action (acc, hdot)                        │
│    6. blend: (1-α)*gt + α*pred                                │
│    7. bicycle_model → next_state                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. 디코더가 보는 정보 비교

| 정보 | 현재 (GPT) | 변경 후 (Enc-Dec) |
|------|-----------|-------------------|
| 이전 predicted 토큰 | ✅ A2T causal attn으로 직접 | ❌ 접근 불가 |
| past trajectory 패턴 | ✅ A2T로 past₀..₃ 직접 attend | ❌ context 경유만 (z와 혼합) |
| z_global | AdaLN + A2Z + concat (3중) | context에 녹아있음 (past/map과 혼합) |
| map 구조 (거시적) | A2S로 접근 가능 (안 써도 됨) | context에 녹아있음 (past/z와 혼합) |
| map 구조 (미시적) | ✅ A2S | ✅ A2S (유지, 의도 기반 정밀 참조) |
| 현재 step agent 상태 | ✅ GCN | ✅ GCN (유지) |
| 다른 agent 상호작용 | ✅ A2A | ✅ A2A (유지) |
| 현재 step 번호 | temporal PE | step_PE(t) (유지) |

**핵심**: past/z/map이 context에서 혼합 → 어느 하나라도 빠지면 context 품질 저하 → 모두 활용 강제

---

## 3. 신규 모듈

### 3a. MapSummaryPooling — map 축소 모듈

#### 설계 배경
- map_tokens = 57×57 = 3249개 → context self-attention에 직접 넣으면 3249² ≈ 10.6M attention scores
- NA=5, batch=2, 8heads → ~3.2GB (attention matrix만) → 24GB VRAM 초과 위험
- 또한 3249개 중 past/z는 8개 → map에 묻혀서 정보 혼합 효과 약화
- **따라서 learnable query pooling으로 8~16 tokens으로 축소 필요**

#### 검토한 축소 방법

| 방법 | 설명 | 판단 |
|------|------|------|
| Learnable Query Pooling ✅ | K개 학습 query가 map에서 cross-attn | Perceiver/Set Transformer에서 검증. 학습으로 정보 손실 최소화 |
| Spatial AvgPool | 57×57 → 4×4 등 고정 pooling | 중요 영역 구분 불가. 공간 정보 균등 손실 |
| 추가 Conv stride | conv layer 추가로 해상도 축소 | RF 너무 커짐. 세밀한 도로 정보 손실 |

#### 구조
```python
class MapSummaryPooling(nn.Module):
    def __init__(self, num_queries=8, d_model=128, map_ch=64, nhead=4):
        self.queries = nn.Parameter(torch.randn(num_queries, d_model))
        self.map_proj = nn.Linear(map_ch, d_model)           # 64 → 128
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

    def forward(self, map_tokens):
        # map_tokens: (NA, 3249, 64) — CNN conv3 spatial features
        kv = self.map_proj(map_tokens)                        # (NA, 3249, 128)
        q = self.queries.unsqueeze(0).expand(NA, -1, -1)      # (NA, 8, 128)
        summary, _ = self.cross_attn(q, kv, kv)               # (NA, 8, 128)
        return summary
```

- 파라미터: ~66K (전체 대비 ~2%)
- num_queries=8 (config: `map_summary_tokens`)

#### map_summary vs A2S 역할 분담

| | map_summary (context) | A2S (decoder) |
|---|---|---|
| **해상도** | 8 tokens (축소) | 3249 tokens (원본) |
| **역할** | "커브 도로인지, 교차로인지" 거시적 구조 | "어느 레인, 어느 경계" 미시적 위치 |
| **접근 방식** | context self-attn으로 past/z와 혼합 | A2C 출력이 query → 의도 기반 정밀 참조 |
| **강제성** | context에 녹아있어 무시 불가 | A2C에서 의도 파악 후 참조 → 의미있는 활용 |

### 3b. Context Encoder

#### 구조
```python
class ContextEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2,
                 gcn_feat_dim=64, z_size=32, num_z_tokens=4):
        self.past_proj = nn.Linear(gcn_feat_dim, d_model)      # 64 → 128
        self.z_proj = nn.Linear(z_size, num_z_tokens * d_model) # 32 → K×128
        self.past_pe = nn.Embedding(4, d_model)                 # past temporal PE
        # map_summary는 외부에서 주입 (이미 d_model 차원)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2),
            num_layers=num_layers
        )
```

#### num_layers 결정 근거
- context tokens = 4(past) + 4(z) + 8(map) = 16개 → 작은 시퀀스
- 1 layer: self-attention 1회 → 직접 이웃만 혼합, 깊은 상호작용 부족
- **2 layers (default)**: 충분한 혼합. 2차 관계까지 학습 (past↔z, z↔map, past↔map 모두)
- 3+ layers: 16 tokens에 과도. config `context_num_layers`로 조절 가능

#### Forward
```python
def forward(self, past_gcn_feats, z_global, map_summary):
    """
    past_gcn_feats: (NA, PT=4, 64)
    z_global: (NA, 32)
    map_summary: (NA, M, 128) — MapSummaryPooling 출력
    returns: context_tokens (NA, 4+K+M, 128)
    """
    past_tokens = self.past_proj(past_gcn_feats)              # (NA, 4, 128)
    past_tokens += self.past_pe(torch.arange(4))              # temporal PE
    z_tokens = self.z_proj(z_global).view(NA, K, d_model)     # (NA, K, 128)
    # map_summary는 이미 (NA, M, 128)
    context = torch.cat([past_tokens, z_tokens, map_summary], dim=1)  # (NA, 4+K+M, 128)
    context = self.encoder(context)                            # self-attn
    return context
```

#### 정보 혼합 원리
```
Self-attention 후:
  past₀' = attn(past₀, [past, z, map]) → past + z + map 혼합
  z₁'    = attn(z₁, [past, z, map])     → z + past + map 혼합
  map₂'  = attn(map₂, [past, z, map])   → map + past + z 혼합

→ 모든 context token에 past/z/map 정보가 분산
→ z를 빼면 → past/map만으로 구성 → trajectory 의도 정보 손실
→ map을 빼면 → past/z만으로 구성 → 도로 구조 정보 손실
→ 세 정보 모두 활용이 강제됨
```

---

## 4. 변경된 TransDecoderLayer

### 4a. 블록 비교
| 블록 | 현재 | 변경 후 | 비고 |
|------|------|---------|------|
| A2T | causal self-attention | **제거** | z/map bypass 주범 |
| A2A | SharedKV, ego/sur Q/O | **유지** | 상호작용 필수 |
| A2Z | z multi-token cross-attn | **제거** | z가 context에 포함 |
| A2C | 없음 | **신규** | context cross-attn |
| A2S | map cross-attn, ego/sur Q/O | **유지** | per-step recrop, 고해상도 |
| AdaLN | z modulation (6곳) | **제거 → LayerNorm** | z가 context에 포함 |
| FFN | ego/sur 분리 | **유지** | |

### 4b. A2C (Context Cross-Attention) 상세
```python
# A2S와 동일한 패턴: shared K/V + ego/sur 분리 Q/O
a2c_norm = LayerNorm(d_model)
a2c_ctx_k_proj = Linear(d_model, d_model)   # context → K
a2c_ctx_v_proj = Linear(d_model, d_model)   # context → V
a2c_ego_q_proj = Linear(d_model, d_model)   # ego query
a2c_ego_o_proj = Linear(d_model, d_model)   # ego output
a2c_sur_q_proj = Linear(d_model, d_model)   # sur query
a2c_sur_o_proj = Linear(d_model, d_model)   # sur output

# Forward:
#   Q: decoder tokens (1, N, D) — 현재 step
#   K, V: context tokens (N, C, D) — encoder 출력 (고정)
#   → ego/sur 각각 cross-attend → output
```

### 4c. A2C → A2S 연결 (decoder layer 내 정보 흐름)
```
입력: x (1, 1, N, D) — 현재 step의 agent 토큰들

A2A: x → norm → SharedKVAttn(x, x, ego_mask) + residual
     (같은 timestep agent끼리 상호작용)

A2C: x → norm → CrossAttn(Q=x, K/V=context) + residual
     (context에서 past+z+map_summary 정보 획득)
     → x에 "의도 + 거시적 도로 구조" 정보가 담김

A2S: x → norm → CrossAttn(Q=x, K/V=map_tokens) + residual
     (의도를 아는 상태에서 고해상도 map 정밀 참조) ← ★A2C 출력이 query★
     → A2C 없이는 맹목적 map 참조, A2C 있으면 의도 기반 참조

FFN: x → ego_norm → ego_FFN + residual
     x → sur_norm → sur_FFN + residual

출력: x (1, 1, N, D)
```

### 4d. A2C → A2S 상보적 작동 원리

**A2C 출력의 의미**:
```
ego_Q: "나는 지금 이런 상태인데, context에서 뭘 가져와야 해?"
context KV: "과거엔 이렇게 움직였고, z 의도는 이렇고, 도로는 대충 이런 구조야"
→ attention이 context 16 tokens 중 관련 있는 것에 높은 weight
→ 출력 = 현재 상태에 맞는 "의도 + 과거 패턴 + 거시적 도로 요약"

예시:
  직진 중 step t=3 → z tokens + 직진 past에 높은 weight → "계속 직진, 속도 유지"
  커브 진입 step t=5 → map_summary(커브) + z(감속)에 높은 weight → "좌회전, 감속"
```

**A2C 출력이 A2S를 조건화**:
```
A2C 후 x = 원래 query + "직진 의도, 속도 유지" (residual)
    ↓
A2S: Q=x → 3249 map tokens에 attention
    → "직진 의도"를 아니까 → 전방 도로 tokens에 높은 weight
    → 출력: "전방 3m 차선 경계, 5m 커브 시작" (미시적 정밀 정보)
```

**A2C 없이 A2S만 있을 경우** (비교):
```
x = 원래 query (GCN + z_local + lw + sem만, 의도 모름)
    ↓
A2S: Q=x → 3249 map tokens에 attention
    → 어디를 봐야 할지 모름 → 전체 map에 균등한 attention → 약한 정보
```

**map_attn_loss (soft label)와의 시너지**:
```
A2C: "어디로 갈 의도인지" → query에 방향성 부여 (학습된, 암묵적 가이드)
soft label: "실제로 어디를 지나가는지" → attention에 직접 가이드 (명시적 가이드)
→ 둘이 같이 A2S를 훈련시킴
→ inference에서는 soft label 없어도 A2C의 의도만으로 올바른 곳을 attend
```

**요약**: context의 map_summary(거시)와 A2S의 map_tokens(미시)가 상보적.
A2C가 A2S의 query를 "의도로 조건화"하여, 3249 tokens 중 어디를 봐야 하는지 안내.

---

## 5. Positional Encoding 설계

### 5a. Context Encoder PE
```
past tokens: nn.Embedding(PT=4, 128) — temporal 순서
  past₀ + PE(0), past₁ + PE(1), past₂ + PE(2), past₃ + PE(3)

z tokens: PE 없음 — 순서 개념 없는 latent

map_summary tokens: PE 없음 — 공간 정보는 이미 map_tokens에서 pooling 시 학습됨
```

### 5b. Decoder Step PE
```
nn.Embedding(FT=12, 128) — 미래 step 번호
  step 0 query += step_PE(0)  → "0.5초 후"
  step 5 query += step_PE(5)  → "3.0초 후"
  step 11 query += step_PE(11) → "6.0초 후"

→ 디코더가 "지금 몇 번째 step인지" 인식
→ 초반/후반 행동 차이 학습 가능
```

---

## 6. Intent Codebook 처리 (유지)

```
Layer 0 → ego token → intent_codebook → z_local → concat → proj → Layer 1 입력
```
현재와 동일한 삽입 구조. Layer가 2개이므로 Layer 0-1 사이에 intent 처리.

---

## 7. AR Loop 변경 상세

### 7a. 현재 AR Loop (GPT-style)
```python
all_tokens = past_tokens  # (1, 4, NA, D) → 누적됨
for t in range(FT):
    causal_mask = build_causal_mask(PT+t)
    for layer in trans_layers:
        x = layer(all_tokens, causal_mask, map, z_global, ...)  # 전체 시퀀스 처리
    last_token = x[:, -1]  # 마지막 토큰만 사용
    action = output_head(last_token)
    # ... bicycle, blend ...
    new_token = build_token(next_state, t_pos=PT+t)
    all_tokens = cat([all_tokens, new_token])  # 누적! O(T²) 연산
```

### 7b. 변경 후 AR Loop
```python
# Context 생성 (1회)
map_summary = map_summary_pooling(map_tokens)              # (NA, M, 128)
context = context_encoder(past_gcn_feats, z_global, map_summary)  # (NA, 4+K+M, 128)

prev_state = scene_graph.past[:, -1, :]
for t in range(FT):
    # 현재 상태로 GCN
    gcn_feat = run_gcn(prev_state, scene_graph)       # (NA, 64)

    # Query 구성 (단일 토큰)
    query = query_proj(cat([gcn_feat, z_local, lw, sem]))  # (NA, 128)
    query = query + step_pe(t)                              # step PE 추가
    query = query.unsqueeze(0).unsqueeze(0)                 # (1, 1, NA, 128)

    # map recrop (A2S용)
    if map_recrop:
        cur_map_tokens = recompute_map_tokens(prev_state)

    # Decoder layers (context는 고정, map_tokens는 매 step 갱신 가능)
    for layer_idx, layer in enumerate(trans_layers):
        query = layer(query, ego_mask, context, cur_map_tokens, ...)
        if layer_idx == 0:  # intent between layer 0-1
            # intent codebook 처리 (기존과 동일)

    # Output
    action = output_head(query.squeeze())                   # (NA, 2)
    blended = (1-α)*gt + α*action
    next_state = bicycle_model(blended, prev_state)
    prev_state = next_state
```

**핵심 차이**: 토큰 누적 없음. 매 step 독립적 query. O(T) 연산.

---

## 8. 제거/유지 총정리

### 제거
| 항목 | 코드 위치 | 이유 |
|------|----------|------|
| A2T (temporal self-attn) | TransDecoderLayer init/forward | z/map bypass 주범 |
| A2Z (z cross-attn) | TransDecoderLayer init/forward/_a2z_attention | z가 context에 포함 |
| AdaLN (6곳) | TransDecoderLayer 전체 | z가 context에 포함 |
| _build_causal_mask() | model | A2T 제거로 불필요 |
| _run_decoder_parallel() | model | TF parallel 불필요 |
| dec_past_dropout | model init, _build_decoder_tokens | context encoder가 대체 |
| use_teacher_forcing 분기 | train loop | AR-only |

### 유지
| 항목 | 비고 |
|------|------|
| CVAE prior/posterior | z_global 생성 |
| GCN (SceneInteractionNet) | encoder + decoder 양쪽 |
| Map CNN (conv1~3) | map_tokens 생성 + MapSummaryPooling 입력 |
| Bicycle model | action → state |
| z_local intent codebook | Layer 0→1 사이 |
| A2A + RelativeBias | agent 상호작용 |
| A2S (map cross-attn) | per-step recrop, 고해상도 가이던스 |
| FFN ego/sur 분리 | |
| sur/ego_pred_head | A2A auxiliary loss |
| Action Blending | α schedule |
| 모든 Loss | recon, KL, map_attn, intent_ce, pred, potential |

### 신규
| 항목 | 비고 |
|------|------|
| MapSummaryPooling | map_tokens → 8 summary tokens (learnable query pooling) |
| ContextEncoder | past+z+map_summary → self-attn → context tokens |
| A2C (context cross-attn) | decoder에서 context 참조 |
| step_PE | nn.Embedding(12, 128), decoder step 번호 |

---

## 9. GCN과 Map의 관계 (참고)

### GCN은 map을 전혀 사용하지 않음
SceneInteractionNet(GCN)의 입력:
- `x`: agent state features (position, speed, heading)
- `pos`: agent 위치/방향 (상대 변환 계산용)
- `sem`: semantic class (ego/sur 구분)
- `edge_index`: agent 간 연결

→ **Map 정보 없음**. GCN은 순수하게 agent 간 상호작용만 처리.

### 현재 모델의 map 정보 경로
| 경로 | 내용 |
|------|------|
| CVAE Encoder | CNN → map_feat(64d) → prior/posterior에 concat |
| A2S (Decoder) | map_tokens(3249) cross-attn |
| GCN | map 없음 |

### 이전 GRU 모델도 동일
- map은 CNN → MLP → map_feat, 그리고 CNN tokens → MapCrossAttention으로만 사용
- GCN은 map과 완전히 독립된 경로

**따라서**: context에 map_summary를 명시적으로 넣지 않으면, 디코더가 A2S를 무시해도
context(past+z)만으로 trajectory를 만들 수 있음 → map 활용이 강제되지 않음

---

## 10. 변경 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| **trafficplanner_model.py** | MapSummaryPooling 추가, ContextEncoder 추가, TransDecoderLayer 수정(A2T/A2Z/AdaLN 제거, A2C 추가), AR loop 수정, TF parallel 제거 |
| **train_trafficplanner.py** | use_adaln/use_z_cross_attn 파라미터 제거, context encoder + map summary 파라미터 추가, TF 분기 제거 |
| **train_trafficplanner.cfg** | use_adaln/use_z_cross_attn 제거, context_num_layers/map_summary_tokens 추가, dec_past_dropout 제거 |
| **test_trafficplanner.py** | AR inference 경로 업데이트 |
| **trafficplanner_loss.py** | 변경 없음 |

---

## 11. Config 변경 사항

### 제거
```
use_adaln: True              → 제거
use_z_cross_attn: True       → 제거
dec_past_dropout: 0.5        → 제거
use_teacher_forcing: True    → 제거 (항상 AR)
```

### 추가
```
context_num_layers: 2        # Context Encoder self-attn layers
map_summary_tokens: 8        # MapSummaryPooling query 수
```

### 유지 (값 변경 없음)
```
trans_d_model: 128           # decoder + context encoder 공유
trans_nhead: 8
trans_ffn_dim: 256
trans_num_layers: 2          # decoder layers
num_z_tokens: 4              # context encoder용 z tokens
action_blending: True
blend_anneal_steps: 65000
blend_target_alpha: 1.0
blend_alpha_floor: 0.05
```

### d_model=128 유지 근거
- nhead=8 → head당 dim=16 (이미 작음)
- d_model=64로 줄이면 head당 dim=8 → attention 표현력 부족
- A2A, A2C, A2S 세 cross-attn이 모두 같은 d_model 사용
- 특히 A2S: 3249 map tokens에서 정보 추출 → 충분한 표현력 필요
- A2T/A2Z/AdaLN 제거로 이미 3.5M → ~2.3M 절감 → 추가 축소 불필요

---

## 12. 파라미터 수 추정

| 모듈 | 현재 | 변경 후 |
|------|------|---------|
| CVAE Encoder | ~300K | 300K |
| GCN | ~200K | 200K |
| Map CNN | ~500K | 500K |
| MapSummaryPooling (신규) | 0 | ~66K |
| Context Encoder (신규) | 0 | ~300K |
| TransDecoderLayer ×2 | ~1.5M | ~600K |
| Token proj / Output heads | ~200K | ~150K |
| Intent | ~100K | 100K |
| **합계** | **~3.5M** | **~2.3M** |

---

## 13. z/map collapse 방지 메커니즘

### z collapse 방지
```
현재:
  decoder → A2T → past₀..₃ + pred₀..ₜ 직접 attend
  → z 없이 trajectory 외삽 가능
  → KL gradient: "z에 정보 담지 마라"
  → collapse

변경 후:
  decoder → A2C → context (past+z+map 혼합) cross-attend
  → past 정보를 얻으려면 context를 통해야 함
  → context에 z가 녹아있음 (self-attn으로 분리 불가)
  → z를 빼면 context 품질 저하 → recon 악화
  → recon gradient: "z에 정보 담아라"
  → KL gradient: "z에 정보 담지 마라"
  → 두 gradient가 균형 → z 활성 유지
```

### map 활용 강제
```
현재:
  A2T로 trajectory 패턴 직접 접근 가능
  → map(A2S) 안 봐도 다음 step 예측 가능
  → map_attn_loss가 가이드 줘도 구조적으로 무시 가능

변경 후:
  context에 map_summary가 past/z와 혼합
  → "이 도로 구조에서 이 의도면 이렇게" 관계가 context에 학습
  → A2S의 query(=A2C 출력)에 이미 의도+도로구조 포함
  → A2S가 "어디를 봐야 하는지 아는 상태"에서 고해상도 참조
  → 맹목적 map 참조가 아닌 의도 기반 정밀 참조
```

---

## 14. 검증 계획

1. **Shape 테스트**:
   - map_summary (NA, 8, 128)
   - context (NA, 16, 128)  [4+4+8]
   - decoder query (1, 1, NA, 128)
2. **학습 모니터링**:
   - KL loss: epoch 20 이후 > 0.5 유지 (현재 0.03) → 성공
   - z_mdist: > 1.5 유지 (현재 0.85)
   - pos_err train/val gap < 2x (현재 5x)
3. **적대적 최적화 테스트**: z_global 변경 시 ||∂traj/∂z|| > 0 확인

---

## 15. 선행 연구 참조

- **AutoBots (ICLR 2022)**: Latent variable + Transformer, learnable seed tokens
- **MTR (NeurIPS 2022)**: Learnable intention query → trajectory, Waymo 1위
- **Scene Transformer (Google Waymo)**: Unified multi-agent prediction
- **Perceiver (ICML 2021)**: Learnable query pooling으로 대규모 입력 축소 — MapSummaryPooling 근거
- **Set Transformer (ICML 2019)**: Induced Set Attention Block — pooling 이론적 배경
- **STRIVE (NVIDIA)**: CVAE latent optimization for adversarial scenario generation
