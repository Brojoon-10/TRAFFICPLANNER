# TrafficPlannerModel 전체 설계 문서

이 문서는 TrafficPlannerModel의 데이터 입력부터 디코더 출력까지 모든 네트워크 구조, 차원, 피처 흐름을 코드 기반으로 상세히 설명합니다.

---

## 목차

1. [모델 개요](#1-모델-개요)
2. [입력 데이터 구조](#2-입력-데이터-구조)
3. [Map Encoder](#3-map-encoder)
4. [Temporal GCN Encoder](#4-temporal-gcn-encoder)
5. [z_global Encoder (Prior/Posterior)](#5-z_global-encoder-priorposterior)
6. [Decoder](#6-decoder)
7. [z_local Encoder](#7-z_local-encoder)
8. [전체 Forward Pass 흐름](#8-전체-forward-pass-흐름)
9. [네트워크 파라미터 요약](#9-네트워크-파라미터-요약)

---

## 1. 모델 개요

### 1.1 핵심 특징

TrafficPlannerModel은 **Conditional VAE** 기반 다중 에이전트 궤적 예측 모델로, 다음과 같은 특징을 가집니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TrafficPlannerModel 개요                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  • 잠재 변수: z_global (전역 의도) + z_local (지역 반응, Ego 전용)              │
│  • 인코더: per-step GCN + Temporal Encoding (GRU 또는 Transformer)           │
│  • 디코더: Autoregressive, Ego/Other 분리 경로                               │
│  • 출력: Bicycle Model (가속도 a, 요율 hdot)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 하이퍼파라미터 (기본값)

```python
# trafficplanner_model.py:68-84
npast = 4                    # PT: past timesteps
nfuture = 12                 # FT: future timesteps
map_obs_size_pix = 200       # 맵 크롭 크기 (픽셀)
nclasses = 2                 # NC: 시맨틱 클래스 수 (car, truck 등)
map_feat_size = 64           # 맵 피처 차원
past_feat_size = 64          # GRU hidden 차원
latent_size = 32             # z_global 차원
z_local_size = 32            # z_local 차원
z_local_window = 4           # 슬라이딩 윈도우 크기
gcn_hidden_dim = 64          # GCN 출력 차원
transformer_nhead = 8        # Transformer attention heads
transformer_nlayer = 3       # Transformer layers
dt = 0.5                     # 시간 간격 (초)
```

---

## 2. 입력 데이터 구조

### 2.1 Scene Graph (PyTorch Geometric Batch)

데이터셋(`NuScenesDataset`)에서 로드된 `scene_graph`는 다음과 같은 구조를 가집니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Scene Graph 구조                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  scene_graph.past        : (NA, PT, 6)    # 과거 궤적                        │
│                            [x, y, hx, hy, s, hdot]                          │
│                                                                             │
│  scene_graph.past_vis    : (NA, PT)       # 과거 가시성 (0 or 1)             │
│                                                                             │
│  scene_graph.future      : (NA, FT, 6)    # 미래 궤적 (학습 시 GT)            │
│                            [x, y, hx, hy, s, hdot]                          │
│                                                                             │
│  scene_graph.future_vis  : (NA, FT)       # 미래 가시성 (0 or 1)             │
│                                                                             │
│  scene_graph.lw          : (NA, 2)        # 차량 크기 [length, width]        │
│                                                                             │
│  scene_graph.sem         : (NA, NC)       # 시맨틱 클래스 (one-hot)           │
│                                                                             │
│  scene_graph.edge_index  : (2, num_edges) # 그래프 연결 (fully connected)    │
│                                                                             │
│  scene_graph.batch       : (NA,)          # 배치 소속 인덱스                  │
│                                                                             │
│  scene_graph.ptr         : (B+1,)         # 배치 경계 포인터                  │
│                            [0, n1, n1+n2, ...]                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

NA = 배치 내 전체 에이전트 수 (모든 씬의 에이전트 합)
B  = 배치 크기 (씬 개수)
PT = 4 (past timesteps)
FT = 12 (future timesteps)
NC = 2 (semantic classes)
```

### 2.2 상태 벡터 설명

```
State Vector (6차원):
┌─────┬─────┬──────┬──────┬─────┬──────┐
│  x  │  y  │  hx  │  hy  │  s  │ hdot │
├─────┼─────┼──────┼──────┼─────┼──────┤
│위치x│위치y│heading│heading│속력│요율 │
│     │     │ cos  │ sin  │    │      │
└─────┴─────┴──────┴──────┴─────┴──────┘
```

### 2.3 Ego 에이전트 식별

```python
# trafficplanner_model.py:688-700
def _get_ego_mask(self, scene_graph):
    """
    Ego는 각 배치(씬)의 첫 번째 에이전트 (ptr[:-1] 인덱스)
    """
    NA = scene_graph.x.size(0)
    ego_mask = torch.zeros(NA, dtype=torch.bool, device=scene_graph.x.device)
    ego_inds = scene_graph.ptr[:-1]  # [0, n1, n1+n2, ...]
    ego_mask[ego_inds] = True
    return ego_mask
```

예시:
```
B=2, 씬1: 3개 에이전트, 씬2: 2개 에이전트
ptr = [0, 3, 5]
ego_mask = [True, False, False, True, False]
           └────씬1────┘  └───씬2───┘
```

---

## 3. Map Encoder

### 3.1 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Map Encoder                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력: scene_graph.pos (NA, 4) - 에이전트 현재 위치 [x, y, hx, hy]           │
│                                                                             │
│  1. Map Crop 추출 (map_env.get_map_crop)                                    │
│     ┌─────────────────────────────────────────┐                            │
│     │  pos로부터 200x200 픽셀 로컬 맵 크롭      │                            │
│     │  channels = 4 (drivable, lanes 등)       │                            │
│     └─────────────────────────────────────────┘                            │
│     출력: (NA, 4, 200, 200)                                                 │
│                                                                             │
│  2. CNN (6 layers)                                                         │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Conv2d(4→16, k=7, s=2) → GN(16) → ReLU                          │    │
│     │  Conv2d(16→32, k=5, s=2) → GN(32) → ReLU                         │    │
│     │  Conv2d(32→64, k=5, s=2) → GN(64) → ReLU                         │    │
│     │  Conv2d(64→64, k=3, s=2) → GN(64) → ReLU                         │    │
│     │  Conv2d(64→128, k=3, s=2) → GN(128) → ReLU                       │    │
│     │  Conv2d(128→128, k=3, s=2) → GN(128) → ReLU                      │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│     출력: (NA, 128, 1, 1)  [200→94→43→19→8→3→1]                            │
│                                                                             │
│  3. Flatten + Linear                                                        │
│     ┌─────────────────────────────────────────┐                            │
│     │  Flatten: 128*1*1 = 128                  │                            │
│     │  Linear(128 → 64)                        │                            │
│     └─────────────────────────────────────────┘                            │
│                                                                             │
│  출력: map_feat (NA, 64)                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 코드 근거

```python
# trafficplanner_model.py:123-141
conv_layer_list = []
final_conv_out = map_obs_size_pix  # 200
conv_filter_list = [conv_channel_in] + conv_filter_list  # [4, 16, 32, 64, 64, 128, 128]

for lidx in range(len(conv_kernel_list)):  # kernels: [7, 5, 5, 3, 3, 3]
    cur_conv = nn.Conv2d(conv_filter_list[lidx],
                         conv_filter_list[lidx+1],
                         kernel_size=conv_kernel_list[lidx],
                         stride=conv_stride_list[lidx],  # [2, 2, 2, 2, 2, 2]
                         padding=0)
    cur_gn = nn.GroupNorm(1, conv_filter_list[lidx+1])
    conv_layer_list.extend([cur_conv, cur_gn, nn.ReLU()])
    final_conv_out = calc_conv_out(final_conv_out, conv_kernel_list[lidx], conv_stride_list[lidx])

self.map_conv = nn.Sequential(*conv_layer_list)
self.map_feat_in_size = conv_filter_list[-1] * final_conv_out * final_conv_out  # 128*1*1=128
self.map_feat_out_size = map_feat_size  # 64
self.map_feature = nn.Linear(self.map_feat_in_size, self.map_feat_out_size)
```

---

## 4. Temporal GCN Encoder

### 4.1 전체 구조

`_run_temporal_encoder` 함수가 Prior와 Posterior 모두에서 사용됩니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Temporal GCN Encoder                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력: traj_data (NA, T, 6), vis_data (NA, T)                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  For each timestep t = 0 to T-1:                                    │   │
│  │                                                                     │   │
│  │  1. Step Feature 생성                                               │   │
│  │     step_in_feat = concat(                                          │   │
│  │         cur_state,    # (NA, 6)  - 현재 상태                         │   │
│  │         cur_lw,       # (NA, 2)  - 차량 크기                         │   │
│  │         cur_vis,      # (NA, 1)  - 가시성                            │   │
│  │         cur_sem       # (NA, NC) - 시맨틱 클래스                      │   │
│  │     )                                                               │   │
│  │     → (NA, 6+2+1+NC) = (NA, 11)                                     │   │
│  │                                                                     │   │
│  │  2. Step Feature Extractor (MLP)                                    │   │
│  │     ┌───────────────────────────────────────────┐                   │   │
│  │     │  Linear(11 → 128) → LayerNorm → ReLU      │                   │   │
│  │     │  Linear(128 → 64)                         │                   │   │
│  │     └───────────────────────────────────────────┘                   │   │
│  │     gcn_node_in → (NA, 64)                                          │   │
│  │                                                                     │   │
│  │  3. Per-step GCN (IndividualSceneInteractionNet)                    │   │
│  │     ┌───────────────────────────────────────────┐                   │   │
│  │     │  scene_graph.x = gcn_node_in              │                   │   │
│  │     │  scene_graph.pos = cur_state[:, :4]       │                   │   │
│  │     │  ego_feat_t, other_feat_t = temporal_gcn_encoder(...)        │   │
│  │     │  gcn_feat_t = merge(ego_feat_t, other_feat_t)                │   │
│  │     └───────────────────────────────────────────┘                   │   │
│  │     → (NA, 64) per timestep                                         │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Stack all timesteps: sequence_features → (NA, T, 64)                       │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────────────────┐    │
│  │  Prior (past)       │    │  Posterior (future)                      │    │
│  │  use_transformer=   │    │  use_transformer=True                    │    │
│  │  False              │    │                                          │    │
│  │                     │    │  1. Positional Encoding                  │    │
│  │  GRU                │    │     sequence_features += PE              │    │
│  │  (64→64, 1 layer)   │    │                                          │    │
│  │                     │    │  2. TransformerEncoder                   │    │
│  │  gru_out:           │    │     (d_model=64, nhead=8, nlayer=3)      │    │
│  │  (NA, T, 64)        │    │                                          │    │
│  │                     │    │  transformer_out: (NA, T, 64)            │    │
│  └─────────────────────┘    └─────────────────────────────────────────┘    │
│                                                                             │
│  출력:                                                                       │
│    - sequence_output: (NA, T, 64)                                           │
│    - context_vector: (NA, 64) ← sequence_output[:, -1, :]                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 temporal_gcn_encoder (IndividualSceneInteractionNet)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    IndividualSceneInteractionNet                             │
│                    (temporal_gcn_encoder)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력:                                                                       │
│    scene_graph.x        : (NA, 64)  - gcn_node_in                           │
│    scene_graph.pos      : (NA, 4)   - [x, y, hx, hy]                        │
│    scene_graph.sem      : (NA, NC)  - 시맨틱 클래스                          │
│    scene_graph.edge_index: (2, E)   - 그래프 연결                            │
│    ego_mask             : (NA,)     - Ego 마스크                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. mlp_in (공유, 모든 에이전트)                                      │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  Linear(64 → 128)                                         │   │   │
│  │     │  LayerNorm(128) → ReLU → Linear(128 → 128)               │   │   │
│  │     │  LayerNorm(128) → ReLU → Linear(128 → 128)               │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │     x → (NA, 128) [msg_node_channels]                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. Message Passing (AgentInteractionConv, k=1 layer)               │   │
│  │                                                                     │   │
│  │     message() 함수:                                                 │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  msg_in = concat(x_i, x_j, sem_i, sem_j, rel_trans)       │   │   │
│  │     │         = (128 + 128 + NC + NC + 4)                       │   │   │
│  │     │         = (264 + 2*NC) → (268) for NC=2                   │   │   │
│  │     │                                                           │   │   │
│  │     │  edge_mlp:                                                │   │   │
│  │     │    Linear(268 → 128)                                      │   │   │
│  │     │    LayerNorm → ReLU → Linear(128 → 128)                   │   │   │
│  │     │    LayerNorm → ReLU → Linear(128 → 128)                   │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │     aggregate(): max pooling over neighbors                         │   │
│  │     aggr_out → (NA, 128)                                            │   │
│  │                                                                     │   │
│  │     update() 함수:                                                  │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  update_in = concat(x, aggr_out, sem)                     │   │   │
│  │     │            = (128 + 128 + NC)                             │   │   │
│  │     │            = (258) for NC=2                               │   │   │
│  │     │                                                           │   │   │
│  │     │  update_mlp:                                              │   │   │
│  │     │    Linear(258 → 128)                                      │   │   │
│  │     │    LayerNorm → ReLU → Linear(128 → 128)                   │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │     x → (NA, 128)                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. mlp_out (Ego/Other 분리)                                        │   │
│  │                                                                     │   │
│  │     mlp_out_ego (Ego 에이전트):                                      │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  Linear(128 → 128)                                        │   │   │
│  │     │  LayerNorm → ReLU → Linear(128 → 128)                     │   │   │
│  │     │  LayerNorm → ReLU → Linear(128 → 64)                      │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │     ego_feat → (num_ego, 64)                                        │   │
│  │                                                                     │   │
│  │     mlp_out_other (Other 에이전트):                                  │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  Linear(128 → 128)                                        │   │   │
│  │     │  LayerNorm → ReLU → Linear(128 → 128)                     │   │   │
│  │     │  LayerNorm → ReLU → Linear(128 → 64)                      │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │     other_feat → (num_other, 64)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  출력:                                                                       │
│    ego_feat   : (num_ego, 64)                                               │
│    other_feat : (num_other, 64)                                             │
│    → merge하여 (NA, 64)                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 코드 근거

```python
# trafficplanner_model.py:148-160
step_input_size = self.state_size + self.att_feat_size + 1 + self.NC  # 6+2+1+NC = 11
self.step_feature_extractor = MLP([step_input_size, 128, gcn_hidden_dim])  # 11→128→64

self.temporal_gcn_encoder = IndividualSceneInteractionNet(
    gcn_hidden_dim,  # 64 - agent input feat size
    self.NC,         # NC - semantic feat size
    4,               # edge feat size (x,y,hx,hy)
    128,             # msg_node_channels - interaction node size
    gcn_hidden_dim,  # 64 - output feat size
)

# Prior: GRU
self.prior_temporal_gru = nn.GRU(gcn_hidden_dim, gcn_hidden_dim, 1, batch_first=True)

# Posterior: PE + Transformer
self.positional_encoding = PositionalEncoding(gcn_hidden_dim, max_len=max(self.PT, self.FT))
encoder_layer = TransformerEncoderLayer(d_model=gcn_hidden_dim, nhead=transformer_nhead, batch_first=True)
self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_nlayer)
```

---

## 5. z_global Encoder (Prior/Posterior)

### 5.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        z_global Encoder                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         PRIOR (Past → z_global)                      │   │
│  │                                                                     │   │
│  │  1. Temporal Encoding (Past)                                        │   │
│  │     past_seq_out, past_context = _run_temporal_encoder(             │   │
│  │         scene_graph.past,      # (NA, PT, 6)                        │   │
│  │         scene_graph.past_vis,  # (NA, PT)                           │   │
│  │         PT=4,                                                       │   │
│  │         use_transformer=False  # GRU 사용                           │   │
│  │     )                                                               │   │
│  │     past_seq_out: (NA, PT, 64)                                      │   │
│  │     past_context: (NA, 64) ← past_seq_out[:, -1, :]                 │   │
│  │                                                                     │   │
│  │  2. Prior Input                                                     │   │
│  │     prior_in = concat(                                              │   │
│  │         past_context,  # (NA, 64)                                   │   │
│  │         map_feat,      # (NA, 64)                                   │   │
│  │         sem            # (NA, NC)                                   │   │
│  │     )                                                               │   │
│  │     → (NA, 64+64+NC) = (NA, 130) for NC=2                           │   │
│  │                                                                     │   │
│  │  3. latent_prior_net (MLP)                                          │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  Linear(130 → 128)                                        │   │   │
│  │     │  LayerNorm → ReLU → Linear(128 → 64)                      │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │     prior_z → (NA, 64) = (NA, z_size*2)                             │   │
│  │                                                                     │   │
│  │  4. Split                                                           │   │
│  │     prior_mu  = prior_z[:, :32]  → (NA, 32)                         │   │
│  │     prior_var = exp(prior_z[:, 32:]) → (NA, 32)                     │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      POSTERIOR (Future → z_global)                   │   │
│  │                                                                     │   │
│  │  1. Temporal Encoding (Future)                                      │   │
│  │     _, future_context = _run_temporal_encoder(                      │   │
│  │         scene_graph.future,     # (NA, FT, 6)                       │   │
│  │         scene_graph.future_vis, # (NA, FT)                          │   │
│  │         FT=12,                                                      │   │
│  │         use_transformer=True    # PE + Transformer 사용             │   │
│  │     )                                                               │   │
│  │     future_context: (NA, 64)                                        │   │
│  │                                                                     │   │
│  │  2. Posterior Input                                                 │   │
│  │     posterior_in = concat(                                          │   │
│  │         past_context,   # (NA, 64) ← Prior에서 받음                  │   │
│  │         future_context, # (NA, 64)                                  │   │
│  │         map_feat,       # (NA, 64)                                  │   │
│  │         sem             # (NA, NC)                                  │   │
│  │     )                                                               │   │
│  │     → (NA, 64+64+64+NC) = (NA, 194) for NC=2                        │   │
│  │                                                                     │   │
│  │  3. latent_posterior_net (MLP)                                      │   │
│  │     ┌───────────────────────────────────────────────────────────┐   │   │
│  │     │  Linear(194 → 128)                                        │   │   │
│  │     │  LayerNorm → ReLU → Linear(128 → 64)                      │   │   │
│  │     └───────────────────────────────────────────────────────────┘   │   │
│  │     posterior_z → (NA, 64) = (NA, z_size*2)                         │   │
│  │                                                                     │   │
│  │  4. Split                                                           │   │
│  │     post_mu  = posterior_z[:, :32]  → (NA, 32)                      │   │
│  │     post_var = exp(posterior_z[:, 32:]) → (NA, 32)                  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Sampling (Reparameterization Trick):                                       │
│    z_global = post_mu + eps * sqrt(post_var)   # 학습 시                     │
│    z_global = prior_mu + eps * sqrt(prior_var) # 테스트 시                   │
│    eps ~ N(0, 1)                                                            │
│    → (NA, 32)                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 코드 근거

```python
# trafficplanner_model.py:183-199
self.z_size = latent_size  # 32

self.latent_prior_net = MLP([
    gcn_hidden_dim + self.map_feat_out_size + self.NC,  # 64+64+NC = 130
    128,
    self.z_size * 2  # 64 (mean + variance)
])

self.latent_posterior_net = MLP([
    gcn_hidden_dim * 2 + self.map_feat_out_size + self.NC,  # 64*2+64+NC = 194
    128,
    self.z_size * 2  # 64
])
```

---

## 6. Decoder

### 6.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Autoregressive Decoder                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력:                                                                       │
│    scene_graph : 그래프 구조                                                 │
│    map_feat    : (NA, 64) 맵 피처                                           │
│    past_seq_out: (NA, PT, 64) Prior GRU 시퀀스 출력                          │
│    z_global    : (NA, 32) 샘플링된 잠재 변수                                  │
│                                                                             │
│  초기화:                                                                     │
│    prev_state = scene_graph.past[:, -1, :]     # (NA, 6) 마지막 과거 상태    │
│    past_feat = past_seq_out[:, -1, :]          # (NA, 64) GRU 마지막 출력    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Warm-up Phase: z_local 슬라이딩 윈도우 초기화                         │   │
│  │                                                                     │   │
│  │  for pt in range(PT=4):                                             │   │
│  │      past_state_t = scene_graph.past[:, pt, :]  # (NA, 6)           │   │
│  │      z_local_gcn_in = concat(past_state_t, lw, sem)                 │   │
│  │                     = (NA, 6+2+NC) = (NA, 10)                        │   │
│  │      ego_feat, other_feat = z_local_gcn(z_local_gcn_in, ego_mask)   │   │
│  │      ego_feat_window.append(ego_feat)                               │   │
│  │      other_feat_window.append(other_feat)                           │   │
│  │                                                                     │   │
│  │  → ego_feat_window   : [feat_1, feat_2, feat_3, feat_4]  # 4 steps  │   │
│  │  → other_feat_window : [feat_1, feat_2, feat_3, feat_4]  # 4 steps  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Autoregressive Loop: for t = 0 to FT-1 (12 steps)                  │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Step 1: z_local 계산 (Ego 전용)                               │ │   │
│  │  │                                                               │ │   │
│  │  │  ego_current_feat = ego_feat_window[-1]  # (num_ego, 64)      │ │   │
│  │  │  z_local = _compute_z_local(                                  │ │   │
│  │  │      ego_current_feat,   # Query                              │ │   │
│  │  │      other_feat_window   # Key/Value (GRU temporal encoding)  │ │   │
│  │  │  )                                                            │ │   │
│  │  │  → z_local: (num_ego, 32)                                     │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Step 2: Other 에이전트 디코딩 (GCN 기반, STRIVE 스타일)        │ │   │
│  │  │                                                               │ │   │
│  │  │  decoder_in = concat(                                         │ │   │
│  │  │      cur_past_feat,  # (NA, 64)                               │ │   │
│  │  │      cur_map_feat,   # (NA, 64)                               │ │   │
│  │  │      cur_sem,        # (NA, NC)                               │ │   │
│  │  │      z_global,       # (NA, 32)                               │ │   │
│  │  │      cur_lw          # (NA, 2)                                │ │   │
│  │  │  )                                                            │ │   │
│  │  │  → (NA, 64+64+NC+32+2) = (NA, 164) for NC=2                   │ │   │
│  │  │                                                               │ │   │
│  │  │  scene_graph.x = decoder_in                                   │ │   │
│  │  │  _, other_traj_out = decoder_net(scene_graph, ego_mask)       │ │   │
│  │  │  → other_traj_out: (num_other, 2) [a, hdot]                   │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Step 3: Ego 에이전트 디코딩 (z_global + z_local → GRU)        │ │   │
│  │  │                                                               │ │   │
│  │  │  z_global_ego = z_global[ego_mask]  # (num_ego, 32)           │ │   │
│  │  │                                                               │ │   │
│  │  │  z_combined = z_combine_mlp(concat(z_global_ego, z_local))    │ │   │
│  │  │             = MLP([32+32, 64, 64])                            │ │   │
│  │  │  → z_combined: (num_ego, 64)                                  │ │   │
│  │  │                                                               │ │   │
│  │  │  ego_gru_in = concat(                                         │ │   │
│  │  │      z_combined,    # (num_ego, 64)                           │ │   │
│  │  │      ego_map_feat,  # (num_ego, 64)                           │ │   │
│  │  │      ego_sem,       # (num_ego, NC)                           │ │   │
│  │  │      ego_lw         # (num_ego, 2)                            │ │   │
│  │  │  )                                                            │ │   │
│  │  │  → (num_ego, 64+64+NC+2) = (num_ego, 132) for NC=2            │ │   │
│  │  │                                                               │ │   │
│  │  │  ego_decoder_gru: GRU(132→64, 3 layers)                       │ │   │
│  │  │  ego_gru_out → (num_ego, 64)                                  │ │   │
│  │  │                                                               │ │   │
│  │  │  ego_output_mlp: MLP([64, 64, 2])                             │ │   │
│  │  │  ego_traj_out → (num_ego, 2) [a, hdot]                        │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Step 4: 출력 병합 및 상태 업데이트                             │ │   │
│  │  │                                                               │ │   │
│  │  │  decoder_out = merge(ego_traj_out, other_traj_out)            │ │   │
│  │  │  → (NA, 2) [a, hdot]                                          │ │   │
│  │  │                                                               │ │   │
│  │  │  Bicycle Model Dynamics:                                      │ │   │
│  │  │  cur_state_global = sim_traj(prev_state, a, hdot, veh_len)    │ │   │
│  │  │  → (NA, 4) [x, y, hx, hy] in global frame                     │ │   │
│  │  │                                                               │ │   │
│  │  │  traj_out.append(cur_state_global)                            │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │  Step 5: 다음 스텝 준비 (t < FT-1)                             │ │   │
│  │  │                                                               │ │   │
│  │  │  Memory 업데이트 (decoder_memory GRU):                        │ │   │
│  │  │    other_past_feat_new = decoder_memory(other_state_local)    │ │   │
│  │  │    ego_past_feat_new = decoder_memory(ego_state_local)        │ │   │
│  │  │                                                               │ │   │
│  │  │  Map Feature 업데이트:                                         │ │   │
│  │  │    cur_map_feat = encode_map(cur_state_global)                │ │   │
│  │  │                                                               │ │   │
│  │  │  z_local Window 업데이트:                                      │ │   │
│  │  │    new_ego_feat, new_other_feat = z_local_gcn(cur_state)      │ │   │
│  │  │    ego_feat_window.pop(0); ego_feat_window.append(new_ego)    │ │   │
│  │  │    other_feat_window.pop(0); other_feat_window.append(new_other)│ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  출력: traj_out → (NA, FT, 4) [x, y, hx, hy] in global frame                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 decoder_net (IndividualSceneInteractionNet for Other Agents)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    decoder_net (Other 에이전트용)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력:                                                                       │
│    decode_in_size = z_size + past_feat_size + map_feat_out_size + NC + 2    │
│                   = 32 + 64 + 64 + NC + 2 = 164 (NC=2)                      │
│                                                                             │
│  IndividualSceneInteractionNet(                                              │
│      in_node_channels=164,    # 입력 노드 피처                               │
│      in_sem_channels=NC,      # 시맨틱 피처                                  │
│      in_edge_channels=4,      # 엣지 피처 (x,y,hx,hy)                        │
│      msg_node_channels=64,    # 메시지 패싱 차원                             │
│      out_channels=2           # 출력 (a, hdot)                              │
│  )                                                                          │
│                                                                             │
│  구조:                                                                       │
│    mlp_in: Linear(164→128)→LN→ReLU→Linear(128→128)→LN→ReLU→Linear(128→64)  │
│    msg: AgentInteractionConv(64, NC, 4, 64)                                 │
│    mlp_out_ego: Linear(64→128)→LN→ReLU→...→Linear(128→2)                   │
│    mlp_out_other: Linear(64→128)→LN→ReLU→...→Linear(128→2)                 │
│                                                                             │
│  출력:                                                                       │
│    _, other_traj_out = decoder_net(scene_graph, ego_mask)                   │
│    → other_traj_out: (num_other, 2) [a, hdot]                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 decoder_memory (GRU for State Memory)

```python
# trafficplanner_model.py:218-225
self.num_memory_layers = 3
self.decoder_memory = nn.GRU(
    4,                    # input size (x,y,hx,hy) - local frame 변환된 상태
    self.past_feat_size,  # hidden size = 64
    self.num_memory_layers,  # 3 layers
    batch_first=True,
)
```

### 6.4 Ego Decoder Components

```python
# trafficplanner_model.py:259-273

# z_combine_mlp: z_global + z_local 결합
z_combine_in_size = self.z_size + self.z_local_size  # 32+32=64
self.z_combine_mlp = MLP([z_combine_in_size, 64, 64])

# ego_decoder_gru: Ego 궤적 디코딩용 GRU
ego_gru_in_size = 64 + self.map_feat_out_size + self.NC + self.att_feat_size
# = 64 + 64 + NC + 2 = 132 (NC=2)
self.ego_decoder_gru = nn.GRU(
    ego_gru_in_size,       # 132
    64,                    # hidden size
    self.num_memory_layers,  # 3 layers
    batch_first=True,
)

# ego_output_mlp: GRU 출력 → 궤적 (a, hdot)
self.ego_output_mlp = MLP([64, 64, self.traj_out_size])  # 64→64→2
```

### 6.5 Ego GRU Hidden Warm-up

Ego decoder GRU의 hidden state를 **과거 trajectory context**로 초기화한다.

**네트워크 구조**:

```python
# trafficplanner_model.py:259-266
# Independent GRU for warm-up (shares NO parameters with ego_decoder_gru)
self.ego_warmup_gru = nn.GRU(
    gcn_hidden_dim,              # 64
    64,                          # hidden size
    self.num_memory_layers,      # 3 layers
    batch_first=True,
)
```

**Warm-up 프로세스**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Ego GRU Hidden Warm-up                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  목적: past_feat 초기화와 동일하게 GRU hidden을 temporal context로 warm-up   │
│                                                                             │
│  Window: 4 timesteps (PT=4)                                                 │
│    - Autoregressive: [t-4, t-3, t-2, t-1]  (all from past)                  │
│    - Teacher Forcing: depends on target_t                                   │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Process:                                                             │ │
│  │                                                                       │ │
│  │    For each timestep in window:                                      │ │
│  │      1. Get state (past or GT depending on target_t)                 │ │
│  │      2. Extract GCN features via temporal_gcn_encoder                │ │
│  │      3. Collect to sequence                                          │ │
│  │                                                                       │ │
│  │    ego_sequence: (num_ego, 4, 64)                                    │ │
│  │         ↓                                                             │ │
│  │    ego_warmup_gru                                                    │ │
│  │         ↓                                                             │ │
│  │    hidden: (3, num_ego, 64)  # 3 layers                              │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  핵심 특징:                                                                  │
│    - z_local 계산 없음 (past_feat 초기화 방식과 동일)                         │
│    - GCN 사용: interaction-aware features                                   │
│    - Autoregressive/Teacher Forcing 모두 지원                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Teacher Forcing 예시** (target_t 별 window):

```
Past:   [t-4, t-3, t-2, t-1]
Future: [t0, t1, t2, ..., t11]

target_t=0  → window: [t-4, t-3, t-2, t-1]  (all past)
target_t=3  → window: [t-1, t0, t1, t2]     (1 past + 3 GT)
target_t=6  → window: [t2, t3, t4, t5]      (all GT)
target_t=9  → window: [t5, t6, t7, t8]      (all GT)
```

### 6.6 Teacher Forcing with Cosine Annealing

Segment length를 1에서 12까지 curriculum learning 방식으로 증가.

**Annealing Schedule**:

```python
# Slow start, fast end
progress = current_epoch / tf_max_annealing_epoch  # 0.0 → 1.0
cos_out = 1 - cos(π/2 * progress)                  # 0.0 → 1.0
segment_len = 1 + (12 - 1) * cos_out
```

**예시** (tf_max_annealing_epoch=400):

```
Epoch   segment_len
─────────────────────
  0          1
 40          1
120          2
200          4
280          7
360         11
400         12
```

**동작**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Teacher Forcing Flow                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Example: segment_len = 3                                                   │
│                                                                             │
│  Segment 1 (target_t=0):                                                    │
│    Window: [past: t-4,t-3,t-2,t-1]          → warm-up hidden               │
│    Predict: [t0, t1, t2]                                                    │
│                                                                             │
│  Segment 2 (target_t=3):                                                    │
│    Window: [past: t-1] + [GT: t0,t1,t2]     → warm-up hidden               │
│    Predict: [t3, t4, t5]                                                    │
│                                                                             │
│  Segment 3 (target_t=6):                                                    │
│    Window: [GT: t2,t3,t4,t5]                → warm-up hidden               │
│    Predict: [t6, t7, t8]                                                    │
│                                                                             │
│  Segment 4 (target_t=9):                                                    │
│    Window: [GT: t5,t6,t7,t8]                → warm-up hidden               │
│    Predict: [t9, t10, t11]                                                  │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Key Feature: Early Termination                                            │
│    if target_t + segment_len > 12:                                          │
│        break  # Ensure consistent segment size                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Config**:

```yaml
use_teacher_forcing: True
tf_init_segment_len: 1        # Start with 1-step segments
tf_max_annealing_epoch: 400   # Reach 12-step at epoch 400
```

---

## 7. z_local Encoder

### 7.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          z_local Encoder                                     │
│                      (Ego 전용, 매 디코더 스텝)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  목적: 슬라이딩 윈도우 내 다른 에이전트들의 상태를 참고하여                      │
│        Ego의 reactive한 지역 잠재 변수 생성                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. z_local_gcn (IndividualSceneInteractionNet)                      │   │
│  │     per-step GCN으로 현재 상태에서 에이전트 상호작용 피처 추출             │   │
│  │                                                                     │   │
│  │     입력: concat(state, lw, sem)                                     │   │
│  │          = (NA, 6+2+NC) = (NA, 10) for NC=2                          │   │
│  │                                                                     │   │
│  │     IndividualSceneInteractionNet(                                   │   │
│  │         in_node_channels=10,    # state(6) + lw(2) + sem(NC)         │   │
│  │         in_sem_channels=NC,                                          │   │
│  │         in_edge_channels=4,                                          │   │
│  │         msg_node_channels=64,                                        │   │
│  │         out_channels=64         # z_local_gcn_feat_size              │   │
│  │     )                                                                │   │
│  │                                                                     │   │
│  │     출력:                                                             │   │
│  │       ego_feat   : (num_ego, 64)                                     │   │
│  │       other_feat : (num_other, 64)                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2. Sliding Window (4 timesteps)                                     │   │
│  │                                                                     │   │
│  │     ego_feat_window   = [ego_t-3, ego_t-2, ego_t-1, ego_t]          │   │
│  │     other_feat_window = [other_t-3, other_t-2, other_t-1, other_t]  │   │
│  │                                                                     │   │
│  │     매 스텝: pop(0) → append(new_feat)                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  3. _compute_z_local_dist() 함수 (확률 분포 출력)                     │   │
│  │                                                                     │   │
│  │     For each ego agent (batch별로):                                  │   │
│  │                                                                     │   │
│  │     a. Other Temporal Encoding                                       │   │
│  │        other_feat_seq = stack(other_feat_window)  # (W, num_other, 64)│  │
│  │        other_feat_seq = other_feat_seq.permute(1,0,2) # (num_other, W, 64)│
│  │                                                                     │   │
│  │        z_local_temporal_gru: GRU(64→64, 1 layer)                    │   │
│  │        temporal_out, _ = gru(other_feat_seq)                        │   │
│  │        temporal_encoded = temporal_out[:, -1, :]  # (num_other, 64)  │   │
│  │                                                                     │   │
│  │     b. Cross-Attention (Ego → Other)                                 │   │
│  │        Q = ego_current_feat  # (1, 64)                               │   │
│  │        K = V = temporal_encoded  # (num_other, 64)                   │   │
│  │                                                                     │   │
│  │        cross_attn: MultiheadAttention(                               │   │
│  │            embed_dim=64,                                             │   │
│  │            num_heads=4,                                              │   │
│  │            batch_first=True                                          │   │
│  │        )                                                            │   │
│  │        attn_out, _ = cross_attn(Q, K, V)  # (1, 64)                  │   │
│  │                                                                     │   │
│  │     c. z_local 분포 파라미터 (VAE-style)                              │   │
│  │        z_local_mean_mlp: MLP([64, 64, 32])                          │   │
│  │        z_local_var_mlp:  MLP([64, 64, 32])                          │   │
│  │                                                                     │   │
│  │        z_local_mean = z_local_mean_mlp(attn_out)  # (32,)           │   │
│  │        z_local_var = softplus(z_local_var_mlp(attn_out))  # (32,)   │   │
│  │                                                                     │   │
│  │     d. Reparameterization Trick (rsample)                            │   │
│  │        eps = torch.randn_like(z_local_mean)                         │   │
│  │        z_local = z_local_mean + eps * sqrt(z_local_var)             │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  출력: z_local_mean → (num_ego, 32)                                         │
│        z_local_var  → (num_ego, 32)                                         │
│        z_local (sampled) → (num_ego, 32)                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 코드 근거

```python
# trafficplanner_model.py:227-257

# z_local GCN
self.z_local_gcn_feat_size = 64
z_local_gcn_in_size = self.state_size + self.att_feat_size + self.NC  # 6+2+NC=10
self.z_local_gcn = IndividualSceneInteractionNet(
    z_local_gcn_in_size,  # 10
    self.NC,              # NC
    4,                    # edge feat
    64,                   # msg_node_channels
    self.z_local_gcn_feat_size  # 64
)

# Temporal GRU for sliding window
self.z_local_temporal_gru = nn.GRU(
    self.z_local_gcn_feat_size,  # 64
    self.z_local_gcn_feat_size,  # 64
    1,  # 1 layer
    batch_first=True,
)

# Cross-attention
self.cross_attn_heads = 4
self.cross_attn = nn.MultiheadAttention(
    embed_dim=self.z_local_gcn_feat_size,  # 64
    num_heads=self.cross_attn_heads,       # 4
    batch_first=True,
)

# z_local 분포 출력 MLP (VAE-style)
self.z_local_mean_mlp = MLP([self.z_local_gcn_feat_size, 64, self.z_local_size])  # 64→64→32
self.z_local_var_mlp = MLP([self.z_local_gcn_feat_size, 64, self.z_local_size])   # 64→64→32
```

### 7.3 z_local 샘플링 (Reparameterization)

z_local은 확률 분포에서 샘플링되며, gradient flow를 위해 reparameterization trick을 사용합니다:

```python
# _compute_z_local_dist() 내부
z_local_mean = self.z_local_mean_mlp(attn_out)              # (num_ego, 32)
z_local_var = F.softplus(self.z_local_var_mlp(attn_out))    # (num_ego, 32), positive

# Reparameterization trick for gradient flow
eps = torch.randn_like(z_local_mean)
z_local = z_local_mean + eps * torch.sqrt(z_local_var)      # rsample

return z_local_mean, z_local_var, z_local
```

**설계 이유:**
- z_global과 마찬가지로 z_local도 확률 분포로 모델링하여 불확실성 표현
- 매 디코더 스텝마다 새로운 z_local 샘플링 → reactive한 local adaptation
- softplus로 variance를 양수로 보장

### 7.4 Enhanced Features

#### 7.4.1 Temporal KV: 전체 시퀀스 활용

GRU 출력의 **모든 4 timesteps**을 Key/Value로 사용:

```python
# trafficplanner_model.py:1643-1685
temporal_out, _ = self.z_local_temporal_gru(other_feat_seq)  # (num_other, W, 64)
temporal_encoded = temporal_out.reshape(num_other * W, D)    # (num_other*4, 64)
KV = temporal_encoded.unsqueeze(0)                            # (1, num_other*4, 64)
```

구조적 특징:
- 주변 에이전트의 시간적 변화 패턴을 풍부하게 인코딩
- Cross-attention이 각 timestep별 정보에 독립적으로 접근
- Key/Value 시퀀스 크기: num_other × 4

**시각화** (3 agents, 4 timesteps):

```
Other Agents: [A₁, A₂, A₃]
Window: [t-3, t-2, t-1, t]

GRU Output → [[h_A₁(t-3), h_A₁(t-2), h_A₁(t-1), h_A₁(t)],
              [h_A₂(t-3), h_A₂(t-2), h_A₂(t-1), h_A₂(t)],
              [h_A₃(t-3), h_A₃(t-2), h_A₃(t-1), h_A₃(t)]]

Flatten → KV: (12, 64) = (3 agents × 4 timesteps, 64)

Ego attends to all 12 agent-timestep combinations
```

#### 7.4.2 Ego Query Mode

**ego_full_query** 파라미터 (default: False)

**Mode 1: Single-step Query (False)**

```python
ego_query = ego_feat[b:b + 1]  # (1, 64) - current timestep
Q = ego_query.unsqueeze(0)     # (1, 1, 64)
attn_out, _ = self.cross_attn(Q, KV, KV)  # (1, 1, 64)
```

**Mode 2: Multi-step Query (True)**

```python
# Ego temporal GRU
ego_window_stack = torch.stack(ego_feat_window, dim=0)  # (W, num_ego, D)
ego_feat_seq = ego_window_stack[:, b, :].unsqueeze(0)   # (1, W, D)
ego_temporal_out, _ = self.z_local_ego_temporal_gru(ego_feat_seq)
ego_query_encoded = ego_temporal_out.reshape(W, D)      # (W, D)

Q = ego_query_encoded.unsqueeze(0)  # (1, W, 64) - all 4 timesteps
attn_out, _ = self.cross_attn(Q, KV, KV)  # (1, W, 64)
attn_out = attn_out.mean(dim=1)  # (1, 64) - temporal average
```

**비교**:

```
Single-step (False):
  Q:  [ego(t)]                      → (1, 1, 64)
  KV: [A₁,A₂,A₃] × [t-3,t-2,t-1,t] → (1, 12, 64)
  Output: (1, 1, 64)

Multi-step (True):
  Q:  [h_ego(t-3), h_ego(t-2), h_ego(t-1), h_ego(t)] → (1, 4, 64)
  KV: [A₁,A₂,A₃] × [t-3,t-2,t-1,t]                   → (1, 12, 64)
  Attention: (1, 4, 64)
  Average: (1, 64)
```

**추가 네트워크** (ego_full_query=True):

```python
self.z_local_ego_temporal_gru = nn.GRU(64, 64, 1, batch_first=True)
```

**Config**:
```yaml
ego_full_query: False  # Single-step (default)
```

#### 7.4.3 전체 z_local 인코딩 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   z_local Encoding (Enhanced)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sliding Window (4 timesteps):                                              │
│    ego_feat_window   = [feat_t-3, feat_t-2, feat_t-1, feat_t]               │
│    other_feat_window = [feat_t-3, feat_t-2, feat_t-1, feat_t]               │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Other Temporal Encoding                                              │ │
│  │                                                                       │ │
│  │    other_seq: (num_other, 4, 64)                                     │ │
│  │         ↓                                                             │ │
│  │    z_local_temporal_gru                                              │ │
│  │         ↓                                                             │ │
│  │    temporal_out: (num_other, 4, 64)  ← ALL timesteps                 │ │
│  │         ↓                                                             │ │
│  │    Flatten: (num_other*4, 64)                                        │ │
│  │         ↓                                                             │ │
│  │    KV: (1, num_other*4, 64)                                          │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Ego Query (depends on ego_full_query)                               │ │
│  │                                                                       │ │
│  │    if ego_full_query == False:                                       │ │
│  │        Q: [ego_t]                    → (1, 1, 64)                     │ │
│  │                                                                       │ │
│  │    if ego_full_query == True:                                        │ │
│  │        ego_seq: (1, 4, 64)                                           │ │
│  │             ↓                                                         │ │
│  │        z_local_ego_temporal_gru                                      │ │
│  │             ↓                                                         │ │
│  │        Q: [h_ego_t-3, h_ego_t-2, h_ego_t-1, h_ego_t] → (1, 4, 64)    │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Cross-Attention                                                      │ │
│  │                                                                       │ │
│  │    attn_out, _ = cross_attn(Q, KV, KV)                               │ │
│  │                                                                       │ │
│  │    if ego_full_query:                                                │ │
│  │        attn_out = attn_out.mean(dim=1)  # (1, 64)                    │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  z_local Distribution                                                 │ │
│  │                                                                       │ │
│  │    z_local_mean = mean_mlp(attn_out)  → (32,)                        │ │
│  │    z_local_var = softplus(var_mlp(attn_out)) → (32,)                 │ │
│  │    z_local = z_local_mean + eps * sqrt(z_local_var)                  │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 전체 Forward Pass 흐름
│                                                                             │
│    Other Agents: [A₁, A₂, A₃] (3 agents)                                    │
│    Window: [t-3, t-2, t-1, t]  (4 timesteps)                                │
│                                                                             │
│    GRU → [[h_A₁(t-3), h_A₁(t-2), h_A₁(t-1), h_A₁(t)],                       │
│           [h_A₂(t-3), h_A₂(t-2), h_A₂(t-1), h_A₂(t)],                       │
│           [h_A₃(t-3), h_A₃(t-2), h_A₃(t-1), h_A₃(t)]]                       │
│                                                                             │
│    Flatten → KV: (12, 64) = (3 agents × 4 timesteps, 64)                    │
│                                                                             │
│    Ego can attend to:                                                       │
│      - Each agent's temporal evolution                                      │
│      - Different timesteps with different attention weights                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 7.4.2 Ego Full Query Option

**매개변수**: `ego_full_query` (default: False)

**False (기본)**: Ego의 현재 timestep feature만 Query로 사용
```python
# ego_full_query = False
ego_query = ego_feat[b:b + 1]  # (1, 64) - 현재 스텝만
Q = ego_query.unsqueeze(0)     # (1, 1, 64)
attn_out, _ = self.cross_attn(Q, KV, KV)  # (1, 1, 64)
```

**True**: Ego의 **전체 윈도우 (4 timesteps)** feature를 Query로 사용
```python
# ego_full_query = True
# Ego도 temporal encoding
ego_window_stack = torch.stack(ego_feat_window, dim=0)  # (W, num_ego, D)
ego_feat_seq = ego_window_stack[:, b, :].unsqueeze(0)   # (1, W, D)
ego_temporal_out, _ = self.z_local_ego_temporal_gru(ego_feat_seq)
ego_query_encoded = ego_temporal_out.reshape(W, D)      # (W, D)

Q = ego_query_encoded.unsqueeze(0)  # (1, W, 64) - 4 timesteps 전부
attn_out, _ = self.cross_attn(Q, KV, KV)  # (1, W, 64)

# Temporal average
attn_out = attn_out.mean(dim=1)  # (1, 64)
```

**비교**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ego_full_query Comparison                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ego_full_query = False (Default):                                          │
│  ─────────────────────────────────                                          │
│                                                                             │
│    Ego Query:  [ego_feat(t)]           → Q: (1, 1, 64)                      │
│    Other K/V:  [all_agents_all_times]  → KV: (1, num_other*4, 64)           │
│                                                                             │
│    Attention Output: (1, 1, 64) → single context vector                     │
│                                                                             │
│    특징:                                                                     │
│      ✓ 간단하고 빠름                                                         │
│      ✓ 현재 시점의 ego 상태에 집중                                            │
│      - Ego의 시간적 변화 패턴 미활용                                          │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ego_full_query = True:                                                     │
│  ──────────────────────                                                     │
│                                                                             │
│    Ego GRU:    [ego_t-3, ego_t-2, ego_t-1, ego_t]                           │
│                   ↓                                                         │
│    Ego Query:  [h_ego(t-3), h_ego(t-2), h_ego(t-1), h_ego(t)]               │
│                → Q: (1, 4, 64)                                               │
│                                                                             │
│    Other K/V:  [all_agents_all_times]  → KV: (1, num_other*4, 64)           │
│                                                                             │
│    Attention:  각 ego timestep이 독립적으로 attend                            │
│                → (1, 4, 64)                                                  │
│                                                                             │
│    Average:    mean over 4 timesteps → (1, 64)                              │
│                                                                             │
│    특징:                                                                     │
│      ✓ Ego의 시간적 진화 패턴 활용                                            │
│      ✓ 각 timestep별 다른 attention 가능                                      │
│      - 약간의 추가 계산 비용 (ego_temporal_gru)                               │
│      - 평균화로 인한 정보 손실 가능성                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**추가 네트워크 컴포넌트** (ego_full_query=True일 때만):

```python
# trafficplanner_model.py:249-254
if self.ego_full_query:
    self.z_local_ego_temporal_gru = nn.GRU(
        self.z_local_gcn_feat_size,  # 64
        self.z_local_gcn_feat_size,  # 64
        1,  # 1 layer
        batch_first=True,
    )
```

**설정 방법**:
```
# configs/train_trafficplanner.cfg
ego_full_query: False  # or True
```

---

## 8. 전체 Forward Pass 흐름

### 8.1 학습 시 (forward)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Forward Pass                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력: scene_graph, map_idx, map_env                                        │
│                                                                             │
│  Step 1: Map Feature 추출                                                   │
│  ────────────────────────                                                   │
│  scene_graph.pos = scene_graph.past[:, -1, :4]  # 마지막 past 위치          │
│  map_feat = encode_map(scene_graph, map_idx, map_env)                       │
│  → map_feat: (NA, 64)                                                       │
│                                                                             │
│  Step 2: PRIOR (Past → z_global)                                            │
│  ───────────────────────────────                                            │
│  prior_mu, prior_var, past_seq_out = prior(scene_graph, map_feat)           │
│  → prior_mu: (NA, 32), prior_var: (NA, 32), past_seq_out: (NA, PT, 64)      │
│                                                                             │
│  Step 3: POSTERIOR (Future → z_global)                                      │
│  ─────────────────────────────────────                                      │
│  past_context = past_seq_out[:, -1, :]  # (NA, 64)                          │
│  post_mu, post_var = encoder(scene_graph, map_feat, past_context)           │
│  → post_mu: (NA, 32), post_var: (NA, 32)                                    │
│                                                                             │
│  Step 4: z_global 샘플링 (Reparameterization)                               │
│  ─────────────────────────────────────────────                              │
│  z_samp = rsample(post_mu, post_var)  # posterior에서 샘플                   │
│  → z_samp: (NA, 32)                                                         │
│                                                                             │
│  Step 5: DECODER                                                            │
│  ───────────────                                                            │
│  future_pred = decoder(scene_graph, map_feat, past_seq_out, z_samp, ...)    │
│  → future_pred: (NA, FT, 4)                                                 │
│                                                                             │
│  출력:                                                                       │
│  {                                                                          │
│      'prior_out': (prior_mu, prior_var),                                    │
│      'posterior_out': (post_mu, post_var),                                  │
│      'future_pred': future_pred,                                            │
│      'past_seq_out': past_seq_out                                           │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 테스트 시 (sample_batched)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Test Sampling (sample_batched)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력: scene_graph, map_idx, map_env, num_samples=NS                        │
│                                                                             │
│  Step 1-2: Map Feature + PRIOR (동일)                                       │
│  ──────────────────────────────────                                         │
│  prior_mu, prior_var, past_seq_out = prior(scene_graph, map_feat)           │
│                                                                             │
│  Step 3: 다중 샘플 (NS개)                                                    │
│  ────────────────────────                                                   │
│  samp_mu = prior_mu.expand(NS, NA, 32)                                      │
│  samp_var = prior_var.expand(NS, NA, 32)                                    │
│  z_samp = rsample(samp_mu, samp_var)                                        │
│  → z_samp: (NS, NA, 32) → transpose → (NA, NS, 32)                          │
│                                                                             │
│  Step 4: Batched Decoding                                                   │
│  ────────────────────────                                                   │
│  future_pred = decoder(..., z_samp.transpose(0,1), ...)                     │
│  → future_pred: (NA, NS, FT, 4)                                             │
│                                                                             │
│  Note: mult_samp=True일 때 모든 텐서가 3D로 확장됨                            │
│        decoder 내부에서 (NA*NS) 형태로 reshape하여 처리                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 네트워크 파라미터 요약

### 9.1 Encoder 모듈

| 모듈 | 입력 차원 | 출력 차원 | 구조 |
|------|----------|----------|------|
| `step_feature_extractor` | 6+2+1+NC=11 | 64 | MLP: 11→128→64 |
| `temporal_gcn_encoder` | 64 | 64 | IndividualSceneInteractionNet |
| `prior_temporal_gru` | 64 | 64 | GRU(64→64, 1 layer) |
| `positional_encoding` | 64 | 64 | Sinusoidal PE |
| `transformer_encoder` | 64 | 64 | TransformerEncoder(nhead=8, nlayer=3) |
| `latent_prior_net` | 64+64+NC=130 | 64 | MLP: 130→128→64 |
| `latent_posterior_net` | 128+64+NC=194 | 64 | MLP: 194→128→64 |

### 9.2 Decoder 모듈

| 모듈 | 입력 차원 | 출력 차원 | 구조 |
|------|----------|----------|------|
| `decoder_net` | 32+64+64+NC+2=164 | 2 | IndividualSceneInteractionNet |
| `decoder_memory` | 4 | 64 | GRU(4→64, 3 layers) |
| `z_combine_mlp` | 32+32=64 | 64 | MLP: 64→64→64 |
| `ego_decoder_gru` | 64+64+NC+2=132 | 64 | GRU(132→64, 3 layers) |
| `ego_output_mlp` | 64 | 2 | MLP: 64→64→2 |

### 9.3 z_local 모듈

| 모듈 | 입력 차원 | 출력 차원 | 구조 |
|------|----------|----------|------|
| `z_local_gcn` | 6+2+NC=10 | 64 | IndividualSceneInteractionNet |
| `z_local_temporal_gru` | 64 | 64 | GRU(64→64, 1 layer) |
| `cross_attn` | 64 | 64 | MultiheadAttention(64, nhead=4) |
| `z_local_mlp` | 64 | 32 | MLP: 64→64→32 |

### 9.4 Map Encoder

| 모듈 | 입력 차원 | 출력 차원 | 구조 |
|------|----------|----------|------|
| `map_conv` | (4, 200, 200) | (128, 1, 1) | 6-layer CNN |
| `map_feature` | 128 | 64 | Linear |

### 9.5 IndividualSceneInteractionNet 내부 (공통)

```
┌────────────────────────────────────────────────────────────────┐
│  mlp_in      : Linear(in→128)→LN→ReLU→Linear(128→128)→LN→ReLU  │
│                →Linear(128→128)                                │
│  edge_mlp    : Linear(edge_in→128)→LN→ReLU→Linear(128→128)     │
│                →LN→ReLU→Linear(128→out)                        │
│  update_mlp  : Linear(update_in→128)→LN→ReLU→Linear(128→out)   │
│  mlp_out_ego : Linear(128→128)→LN→ReLU→...→Linear(128→out)     │
│  mlp_out_other: Linear(128→128)→LN→ReLU→...→Linear(128→out)    │
└────────────────────────────────────────────────────────────────┘
```

---

## 부록: 전체 데이터 흐름 다이어그램

### A.1 전체 흐름도 (네트워크 타입 명시)

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TrafficPlannerModel 전체 흐름                                         │
│                              (각 화살표에 네트워크 타입 명시)                                        │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘

     scene_graph
         │
         │  past: (NA,4,6)      ← timestep 0~3 (총 4 steps)
         │  future: (NA,12,6)   ← timestep 0~11 (총 12 steps)
         │  lw: (NA,2)
         │  sem: (NA,NC)
         ▼
┌──────────────────────┐
│     Map Encoder      │◄─── scene_graph.pos (NA,4)
│                      │
│  ┌────────────────┐  │
│  │ 6-layer CNN    │  │
│  │ (4ch→128ch)    │  │
│  └───────┬────────┘  │
│          │           │
│  ┌───────▼────────┐  │
│  │ Linear (MLP)   │  │
│  │ 128 → 64       │  │
│  └───────┬────────┘  │
└──────────┼───────────┘
           │ map_feat: (NA,64)
           ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         ENCODER                                                    │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                   │
│  ┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐      │
│  │              PRIOR                       │    │            POSTERIOR                    │      │
│  │                                          │    │                                         │      │
│  │  past: (NA, 4, 6)                        │    │  future: (NA, 12, 6)                    │      │
│  │         │                                │    │         │                               │      │
│  │         │ for t = 0..3:                  │    │         │ for t = 0..11:                │      │
│  │         ▼                                │    │         ▼                               │      │
│  │  ┌─────────────────────────────────┐    │    │  ┌─────────────────────────────────┐   │      │
│  │  │  step_feature_extractor (MLP)   │    │    │  │  step_feature_extractor (MLP)   │   │      │
│  │  │  11 → 128 → 64                  │    │    │  │  11 → 128 → 64                  │   │      │
│  │  └────────────┬────────────────────┘    │    │  └────────────┬────────────────────┘   │      │
│  │               │ (NA, 64)                 │    │               │ (NA, 64)               │      │
│  │               ▼                          │    │               ▼                        │      │
│  │  ┌─────────────────────────────────┐    │    │  ┌─────────────────────────────────┐   │      │
│  │  │  temporal_gcn_encoder           │    │    │  │  temporal_gcn_encoder           │   │      │
│  │  │  (IndividualSceneInteractionNet)│    │    │  │  (IndividualSceneInteractionNet)│   │      │
│  │  │                                 │    │    │  │                                 │   │      │
│  │  │  ┌───────────────────────────┐  │    │    │  │  ┌───────────────────────────┐  │   │      │
│  │  │  │ mlp_in (MLP)              │  │    │    │  │  │ mlp_in (MLP)              │  │   │      │
│  │  │  │ 64→128→128→128            │  │    │    │  │  │ 64→128→128→128            │  │   │      │
│  │  │  └─────────────┬─────────────┘  │    │    │  │  └─────────────┬─────────────┘  │   │      │
│  │  │                │                │    │    │  │                │                │   │      │
│  │  │  ┌─────────────▼─────────────┐  │    │    │  │  ┌─────────────▼─────────────┐  │   │      │
│  │  │  │ AgentInteractionConv      │  │    │    │  │  │ AgentInteractionConv      │  │   │      │
│  │  │  │ (Message Passing + MLP)   │  │    │    │  │  │ (Message Passing + MLP)   │  │   │      │
│  │  │  └─────────────┬─────────────┘  │    │    │  │  └─────────────┬─────────────┘  │   │      │
│  │  │        ┌───────┴───────┐        │    │    │  │        ┌───────┴───────┐        │   │      │
│  │  │        ▼               ▼        │    │    │  │        ▼               ▼        │   │      │
│  │  │  ┌──────────┐   ┌──────────┐   │    │    │  │  ┌──────────┐   ┌──────────┐   │   │      │
│  │  │  │mlp_out_  │   │mlp_out_  │   │    │    │  │  │mlp_out_  │   │mlp_out_  │   │   │      │
│  │  │  │ego (MLP) │   │other(MLP)│   │    │    │  │  │ego (MLP) │   │other(MLP)│   │   │      │
│  │  │  │128→..→64 │   │128→..→64 │   │    │    │  │  │128→..→64 │   │128→..→64 │   │   │      │
│  │  │  └────┬─────┘   └────┬─────┘   │    │    │  │  └────┬─────┘   └────┬─────┘   │   │      │
│  │  │       │              │         │    │    │  │       │              │         │   │      │
│  │  └───────┼──────────────┼─────────┘    │    │  └───────┼──────────────┼─────────┘   │      │
│  │          │              │               │    │          │              │             │      │
│  │          └──────┬───────┘               │    │          └──────┬───────┘             │      │
│  │                 │ merge                  │    │                 │ merge               │      │
│  │                 ▼                        │    │                 ▼                     │      │
│  │          (NA, 64) per step              │    │          (NA, 64) per step            │      │
│  │                 │                        │    │                 │                     │      │
│  │         stack → (NA, 4, 64)             │    │         stack → (NA, 12, 64)          │      │
│  │                 │                        │    │                 │                     │      │
│  │                 ▼                        │    │                 ▼                     │      │
│  │  ┌─────────────────────────────────┐    │    │  ┌─────────────────────────────────┐   │      │
│  │  │  prior_temporal_gru (GRU)       │    │    │  │  positional_encoding (PE)       │   │      │
│  │  │  GRU(64→64, 1 layer)            │    │    │  │  Sinusoidal                     │   │      │
│  │  │  (NA,4,64) → (NA,4,64)          │    │    │  └────────────┬────────────────────┘   │      │
│  │  └────────────┬────────────────────┘    │    │               │                        │      │
│  │               │                          │    │               ▼                        │      │
│  │               │                          │    │  ┌─────────────────────────────────┐   │      │
│  │               │                          │    │  │  transformer_encoder            │   │      │
│  │               │                          │    │  │  TransformerEncoder             │   │      │
│  │               │                          │    │  │  (nhead=8, nlayer=3)            │   │      │
│  │               │                          │    │  │  (NA,12,64) → (NA,12,64)        │   │      │
│  │               │                          │    │  └────────────┬────────────────────┘   │      │
│  │               │                          │    │               │                        │      │
│  │               ▼                          │    │               ▼                        │      │
│  │  past_seq_out: (NA,4,64)                │    │  transformer_out: (NA,12,64)          │      │
│  │  past_context = [:,-1,:] → (NA,64)      │    │  future_context = [:,-1,:] → (NA,64)  │      │
│  │               │                          │    │               │                        │      │
│  │               │ concat(past_context,     │    │               │ concat(past_context,   │      │
│  │               │        map_feat, sem)    │    │               │   future_context,      │      │
│  │               │ → (NA, 130)              │    │               │   map_feat, sem)       │      │
│  │               │                          │    │               │ → (NA, 194)            │      │
│  │               ▼                          │    │               ▼                        │      │
│  │  ┌─────────────────────────────────┐    │    │  ┌─────────────────────────────────┐   │      │
│  │  │  latent_prior_net (MLP)         │    │    │  │  latent_posterior_net (MLP)     │   │      │
│  │  │  130 → 128 → 64                 │    │    │  │  194 → 128 → 64                 │   │      │
│  │  └────────────┬────────────────────┘    │    │  └────────────┬────────────────────┘   │      │
│  │               │                          │    │               │                        │      │
│  │               ▼                          │    │               ▼                        │      │
│  │  split → prior_mu (NA,32)               │    │  split → post_mu (NA,32)              │      │
│  │          prior_var (NA,32)              │    │          post_var (NA,32)             │      │
│  │                                          │    │                                        │      │
│  └──────────────────────────────────────────┘    └────────────────────────────────────────┘      │
│                                                               │                                   │
│                                                               ▼                                   │
│                                              z_global = rsample(post_mu, post_var)               │
│                                              → (NA, 32)                                          │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           DECODER                                                  │
├───────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                   │
│  초기화:                                                                                           │
│    prev_state = past[:,-1,:]  (NA,6)                                                              │
│    past_feat = past_seq_out[:,-1,:]  (NA,64)                                                      │
│                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Warm-up Phase: z_local window 초기화 (past timestep 0~3)                                    │ │
│  │                                                                                              │ │
│  │  for pt = 0..3:                                                                             │ │
│  │    past_state_t = past[:,pt,:]  (NA,6)                                                      │ │
│  │           │                                                                                  │ │
│  │           ▼                                                                                  │ │
│  │    ┌──────────────────────────────────────────┐                                             │ │
│  │    │  z_local_gcn (IndividualSceneInteractionNet)                                           │ │
│  │    │                                          │                                             │ │
│  │    │  ┌────────────────────────────────────┐  │                                             │ │
│  │    │  │ mlp_in (MLP): 10→128→128→64        │  │                                             │ │
│  │    │  └─────────────┬──────────────────────┘  │                                             │ │
│  │    │                │                         │                                             │ │
│  │    │  ┌─────────────▼──────────────────────┐  │                                             │ │
│  │    │  │ AgentInteractionConv (Msg Passing)  │  │                                             │ │
│  │    │  └─────────────┬──────────────────────┘  │                                             │ │
│  │    │        ┌───────┴───────┐                 │                                             │ │
│  │    │        ▼               ▼                 │                                             │ │
│  │    │  ┌──────────┐   ┌──────────┐            │                                             │ │
│  │    │  │mlp_out_  │   │mlp_out_  │            │                                             │ │
│  │    │  │ego (MLP) │   │other(MLP)│            │                                             │ │
│  │    │  └────┬─────┘   └────┬─────┘            │                                             │ │
│  │    └───────┼──────────────┼───────────────────┘                                             │ │
│  │            ▼              ▼                                                                  │ │
│  │    ego_feat_window    other_feat_window                                                     │ │
│  │    .append(ego_feat)  .append(other_feat)                                                   │ │
│  │                                                                                              │ │
│  │  → ego_feat_window = [feat_0, feat_1, feat_2, feat_3]  (4 steps)                            │ │
│  │  → other_feat_window = [feat_0, feat_1, feat_2, feat_3]  (4 steps)                          │ │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Autoregressive Loop: for t = 0..11 (총 12 steps 예측)                                      │ │
│  │                                                                                              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                          z_local 계산 (Ego 전용)                                       │  │ │
│  │  │                                                                                       │  │ │
│  │  │  ego_current = ego_feat_window[-1]  (num_ego, 64)  ← 현재 스텝만 query로 사용          │  │ │
│  │  │  other_window = stack(other_feat_window)  (4, num_other, 64)                          │  │ │
│  │  │               │                                                                        │  │ │
│  │  │               │ permute → (num_other, 4, 64)                                           │  │ │
│  │  │               ▼                                                                        │  │ │
│  │  │  ┌─────────────────────────────────────────────┐                                      │  │ │
│  │  │  │  z_local_temporal_gru (GRU)                 │                                      │  │ │
│  │  │  │  GRU(64→64, 1 layer)                        │                                      │  │ │
│  │  │  │  (num_other, 4, 64) → [:,-1,:] → (num_other, 64)                                   │  │ │
│  │  │  └────────────────────────┬────────────────────┘                                      │  │ │
│  │  │                           │ temporal_encoded                                           │  │ │
│  │  │                           ▼                                                            │  │ │
│  │  │  ┌─────────────────────────────────────────────┐                                      │  │ │
│  │  │  │  cross_attn (MultiheadAttention)            │                                      │  │ │
│  │  │  │  Q = ego_current     (1, 64)                │                                      │  │ │
│  │  │  │  K = V = temporal_encoded  (num_other, 64)  │                                      │  │ │
│  │  │  │  nhead=4                                    │                                      │  │ │
│  │  │  └────────────────────────┬────────────────────┘                                      │  │ │
│  │  │                           │ attn_out (1, 64)                                           │  │ │
│  │  │                           ▼                                                            │  │ │
│  │  │  ┌─────────────────────────────────────────────┐                                      │  │ │
│  │  │  │  z_local_mlp (MLP)                          │                                      │  │ │
│  │  │  │  64 → 64 → 32                               │                                      │  │ │
│  │  │  └────────────────────────┬────────────────────┘                                      │  │ │
│  │  │                           │                                                            │  │ │
│  │  │                           ▼                                                            │  │ │
│  │  │                    z_local (num_ego, 32)                                               │  │ │
│  │  └───────────────────────────┼───────────────────────────────────────────────────────────┘  │ │
│  │                              │                                                               │ │
│  │  ┌───────────────────────────┼───────────────────────────────────────────────────────────┐  │ │
│  │  │                           │      Other Agent 디코딩 (GCN 기반)                         │  │ │
│  │  │                           │                                                            │  │ │
│  │  │  decoder_in = concat(past_feat, map_feat, sem, z_global, lw) → (NA, 164)              │  │ │
│  │  │               │                                                                        │  │ │
│  │  │               ▼                                                                        │  │ │
│  │  │  ┌─────────────────────────────────────────────────────────────────────┐              │  │ │
│  │  │  │  decoder_net (IndividualSceneInteractionNet)                         │              │  │ │
│  │  │  │                                                                      │              │  │ │
│  │  │  │  ┌────────────────────────────────────┐                              │              │  │ │
│  │  │  │  │ mlp_in (MLP): 164→128→128→64       │                              │              │  │ │
│  │  │  │  └─────────────┬──────────────────────┘                              │              │  │ │
│  │  │  │                │                                                     │              │  │ │
│  │  │  │  ┌─────────────▼──────────────────────┐                              │              │  │ │
│  │  │  │  │ AgentInteractionConv (Msg Passing)  │                              │              │  │ │
│  │  │  │  └─────────────┬──────────────────────┘                              │              │  │ │
│  │  │  │        ┌───────┴───────┐                                             │              │  │ │
│  │  │  │        ▼               ▼                                             │              │  │ │
│  │  │  │  ┌──────────┐   ┌──────────┐                                        │              │  │ │
│  │  │  │  │mlp_out_  │   │mlp_out_  │                                        │              │  │ │
│  │  │  │  │ego (MLP) │   │other(MLP)│  ← Other만 사용                         │              │  │ │
│  │  │  │  │(unused)  │   │64→...→2  │                                        │              │  │ │
│  │  │  │  └──────────┘   └────┬─────┘                                        │              │  │ │
│  │  │  └──────────────────────┼──────────────────────────────────────────────┘              │  │ │
│  │  │                         │                                                              │  │ │
│  │  │                         ▼                                                              │  │ │
│  │  │                  other_traj_out: (num_other, 2) [a, hdot]                              │  │ │
│  │  └────────────────────────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                                              │ │
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │                          Ego Agent 디코딩 (z_global + z_local → GRU)                  │  │ │
│  │  │                                                                                       │  │ │
│  │  │  z_global_ego = z_global[ego_mask]  (num_ego, 32)                                     │  │ │
│  │  │               │                                                                        │  │ │
│  │  │               │ concat(z_global_ego, z_local) → (num_ego, 64)                          │  │ │
│  │  │               ▼                                                                        │  │ │
│  │  │  ┌─────────────────────────────────────────────┐                                      │  │ │
│  │  │  │  z_combine_mlp (MLP)                        │                                      │  │ │
│  │  │  │  64 → 64 → 64                               │                                      │  │ │
│  │  │  └────────────────────────┬────────────────────┘                                      │  │ │
│  │  │                           │ z_combined (num_ego, 64)                                   │  │ │
│  │  │                           │                                                            │  │ │
│  │  │               │ concat(z_combined, map_feat, sem, lw) → (num_ego, 132)                 │  │ │
│  │  │               ▼                                                                        │  │ │
│  │  │  ┌─────────────────────────────────────────────┐                                      │  │ │
│  │  │  │  ego_decoder_gru (GRU)                      │                                      │  │ │
│  │  │  │  GRU(132→64, 3 layers)                      │                                      │  │ │
│  │  │  └────────────────────────┬────────────────────┘                                      │  │ │
│  │  │                           │ ego_gru_out (num_ego, 64)                                  │  │ │
│  │  │                           ▼                                                            │  │ │
│  │  │  ┌─────────────────────────────────────────────┐                                      │  │ │
│  │  │  │  ego_output_mlp (MLP)                       │                                      │  │ │
│  │  │  │  64 → 64 → 2                                │                                      │  │ │
│  │  │  └────────────────────────┬────────────────────┘                                      │  │ │
│  │  │                           │                                                            │  │ │
│  │  │                           ▼                                                            │  │ │
│  │  │                    ego_traj_out: (num_ego, 2) [a, hdot]                                │  │ │
│  │  └───────────────────────────┼───────────────────────────────────────────────────────────┘  │ │
│  │                              │                                                               │ │
│  │          ┌───────────────────┴───────────────────────────────┐                              │ │
│  │          │                                                   │                              │ │
│  │          ▼                                                   ▼                              │ │
│  │   ego_traj_out                                        other_traj_out                        │ │
│  │   (num_ego, 2)                                        (num_other, 2)                        │ │
│  │          │                                                   │                              │ │
│  │          └───────────────────┬───────────────────────────────┘                              │ │
│  │                              │ merge (index로 재배치)                                        │ │
│  │                              ▼                                                               │ │
│  │                       decoder_out: (NA, 2) [a, hdot]                                        │ │
│  │                              │                                                               │ │
│  │                              ▼                                                               │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │  Bicycle Model (sim_traj)                                                            │   │ │
│  │  │  car_dynamics 함수로 (a, hdot) → 다음 상태 계산                                       │   │ │
│  │  └──────────────────────────────────────┬──────────────────────────────────────────────┘   │ │
│  │                                          │                                                  │ │
│  │                                          ▼                                                  │ │
│  │                               cur_state_global: (NA, 4) [x, y, hx, hy]                     │ │
│  │                                          │                                                  │ │
│  │                                          ▼                                                  │ │
│  │                               traj_out.append(cur_state_global)                            │ │
│  │                                          │                                                  │ │
│  │                                          ▼                                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │ │
│  │  │  상태 업데이트 (t < 11일 때만)                                                        │   │ │
│  │  │                                                                                      │   │ │
│  │  │  1. decoder_memory (GRU) 로 past_feat 업데이트 (Ego + Other 모두)                    │   │ │
│  │  │     GRU(4→64, 3 layers), input: cur_state_local (x,y,hx,hy)                          │   │ │
│  │  │     - Other: decoder_net 입력으로 직접 사용 (trajectory 출력)                        │   │ │
│  │  │     - Ego: decoder_net message passing용 (Other가 Ego 정보 참조, 출력은 미사용)      │   │ │
│  │  │                                                                                      │   │ │
│  │  │  2. encode_map (CNN+Linear) 로 map_feat 업데이트                                     │   │ │
│  │  │                                                                                      │   │ │
│  │  │  3. z_local window 슬라이드                                                          │   │ │
│  │  │     new_ego_feat, new_other_feat = z_local_gcn(cur_state)                            │   │ │
│  │  │     ego_feat_window.pop(0); ego_feat_window.append(new_ego_feat)                     │   │ │
│  │  │     other_feat_window.pop(0); other_feat_window.append(new_other_feat)               │   │ │
│  │  │                                                                                      │   │ │
│  │  │  ┌──────────────────── 다음 iteration (t+1)에서 재사용 ────────────────────┐         │   │ │
│  │  │  │                                                                        │         │   │ │
│  │  │  │  cur_state_local (4dim) ─→ decoder_memory ─→ cur_past_feat ─→ decoder  │         │   │ │
│  │  │  │                                                                        │         │   │ │
│  │  │  │  cur_state_global (4dim) ─→ scene_graph.pos ─→ encode_map ─→ map_feat  │         │   │ │
│  │  │  │                                                                        │         │   │ │
│  │  │  │  cur_bike_state (6dim) ─→ z_local_gcn ─→ window slide ─→ z_local       │         │   │ │
│  │  │  │                                                                        │         │   │ │
│  │  │  └────────────────────────────────────────────────────────────────────────┘         │   │ │
│  │  └─────────────────────────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                                              │ │
│  └──────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                   │
│  출력: stack(traj_out) → future_pred: (NA, 12, 4)                                                │
│                                                                                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### A.2 네트워크 타입 범례

| 표기 | 네트워크 타입 | 설명 |
|------|-------------|------|
| MLP | Multi-Layer Perceptron | Linear → LayerNorm → ReLU → Linear → ... |
| GRU | Gated Recurrent Unit | 시퀀스 처리용 RNN |
| CNN | Convolutional Neural Network | 2D 이미지 처리 (맵 인코딩) |
| PE | Positional Encoding | Sinusoidal 위치 인코딩 |
| Transformer | TransformerEncoder | Self-attention 기반 시퀀스 모델 |
| MultiheadAttention | Cross-Attention | Q/K/V 기반 어텐션 |
| IndividualSceneInteractionNet | GCN | mlp_in + Message Passing + mlp_out_ego/other |
| AgentInteractionConv | Message Passing | edge_mlp → aggregate → update_mlp |

---

## B. Teacher Forcing 듀얼 루프 구조

Teacher Forcing(TF)은 autoregressive 디코딩에서 에러 누적을 방지하기 위한 학습 기법입니다.
**모델 구조(가중치)는 동일**하며, 학습 시 **Ego와 Sur를 분리된 루프**에서 처리합니다.

### B.1 핵심 개념: 듀얼 루프

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Teacher Forcing Dual-Loop Architecture                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  문제: 기존 단일 루프 TF의 한계                                               │
│  ─────────────────────────────────                                          │
│  - Ego/Sur가 같은 루프에서 동시 처리 → 에러가 상호 전파                       │
│  - Sur 예측 오류가 Ego의 z_local 계산에 영향                                 │
│  - Ego 예측 오류가 Sur의 GCN message passing에 영향                          │
│                                                                             │
│  해결: Ego와 Sur를 완전히 분리된 루프에서 처리                                │
│  ─────────────────────────────────────────────                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Sur Loop: ego=GT, sur=autoregressive                               │   │
│  │  ──────────────────────────────────                                  │   │
│  │  - Ego 위치는 항상 GT 사용 (예측 안 함)                               │   │
│  │  - Sur는 autoregressive하게 자체 예측값 사용                         │   │
│  │  - 목적: Sur가 정확한 Ego 움직임 기반으로 z_global 학습              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Ego Loop: sur=GT, ego=TF segmented                                  │   │
│  │  ──────────────────────────────────                                  │   │
│  │  - Sur 위치는 항상 GT 사용 (예측 안 함)                               │   │
│  │  - Ego는 segment 시작마다 GT로 리셋 후 autoregressive                │   │
│  │  - 목적: Ego가 정확한 Sur 움직임 기반으로 z_local 학습               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  최종 출력: Sur Loop의 sur + Ego Loop의 ego를 merge                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.2 TF=True vs TF=False 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TF=True vs TF=False                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  모델 구조(가중치): 완전히 동일                                              │
│  ──────────────────────────────                                             │
│  - z_local_mlp, ego_decoder_gru, ego_output_mlp                             │
│  - decoder_net, z_local_gcn 등 모든 레이어 동일                              │
│  - 체크포인트 100% 호환                                                      │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │       TF=False              │    │       TF=True               │        │
│  │     (autoregressive)        │    │     (dual-loop)             │        │
│  ├─────────────────────────────┤    ├─────────────────────────────┤        │
│  │                             │    │                             │        │
│  │  [Single Loop]              │    │  [Sur Loop]                 │        │
│  │  ego/sur 동시 처리          │    │  ego=GT, sur=autoregressive │        │
│  │                             │    │                             │        │
│  │  ego: pred → pred → ...    │    │  [Ego Loop]                 │        │
│  │  sur: pred → pred → ...    │    │  sur=GT, ego=TF segmented   │        │
│  │                             │    │                             │        │
│  │  출력: (NA, FT, 4) 텐서     │    │  출력: 12 segments 리스트    │        │
│  │                             │    │  각 segment: [(NA,4), ...]  │        │
│  └─────────────────────────────┘    └─────────────────────────────┘        │
│                                                                             │
│  사용 시점:                                                                  │
│  - Training: TF=True (에러 격리 학습)                                       │
│  - Validation/Test: TF=False (실제 autoregressive 성능 평가)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.3 Sur Loop 상세 (_sur_loop_decoder)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Sur Loop: ego=GT, sur=autoregressive                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  목적: z_global 학습을 위해 Sur가 정확한 Ego 움직임 참조                     │
│  ────────────────────────────────────────────────────                       │
│                                                                             │
│  For t = 0 to FT-1:                                                         │
│                                                                             │
│    1. Ego 위치 = GT (항상)                                                  │
│       ─────────────────────                                                 │
│       if t == 0:                                                            │
│           ego_pos = scene_graph.past[:, -1, :4]                             │
│       else:                                                                 │
│           ego_pos = gt_future[:, t-1, :4]                                   │
│                                                                             │
│    2. Combined Position 생성                                                │
│       ─────────────────────                                                 │
│       combined_pos = sur_prev_state[:, :4].clone()                          │
│       combined_pos[ego_mask] = ego_pos   # Ego는 GT로 덮어쓰기              │
│                                                                             │
│    3. Sur 예측 (decoder_net GCN)                                            │
│       ────────────────────────                                              │
│       decoder_in = concat(cur_past_feat, cur_map_feat, sem, z_global, lw)   │
│       _, sur_traj_step = decoder_net(scene_graph, ego_mask)                 │
│       → sur_traj_step: (num_other, 2) [a, hdot]                             │
│                                                                             │
│    4. Bicycle Model로 Sur 상태 업데이트                                     │
│       ────────────────────────────────                                      │
│       sur_state_global = sim_traj(sur_prev_state, a, hdot, veh_len)         │
│       sur_traj_out.append(sur_state_global)                                 │
│                                                                             │
│    5. 다음 스텝 준비 (t < FT-1)                                             │
│       ─────────────────────────                                             │
│       - Sur prev_state = 자체 예측값                                        │
│       - Ego prev_state = GT                                                 │
│       - Sur/Ego past_feat 업데이트 (decoder_memory GRU)                     │
│                                                                             │
│  출력: sur_traj_all: (num_other, FT, 4)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.4 Ego Loop 상세 (_ego_loop_decoder)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Ego Loop: sur=GT, ego=TF segmented                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  목적: z_local 학습을 위해 Ego가 정확한 Sur 움직임 참조                       │
│  ────────────────────────────────────────────────────                       │
│                                                                             │
│  z_local 윈도우 초기화 (Past GT 기반)                                        │
│  ────────────────────────────────────                                       │
│  for pt in range(PT=4):                                                     │
│      past_state_t = scene_graph.past[:, pt, :]                              │
│      ego_feat, other_feat = z_local_gcn(past_state_t)                       │
│      ego_feat_window.append(ego_feat)                                       │
│      other_feat_window.append(other_feat)                                   │
│                                                                             │
│  For target_t = 0 to FT-1 (12 segments):                                    │
│                                                                             │
│    1. Ego 상태 = GT로 리셋 (매 segment 시작)                                │
│       ─────────────────────────────────────                                 │
│       if target_t == 0:                                                     │
│           ego_prev_state = scene_graph.past[:, -1, :]                       │
│       else:                                                                 │
│           ego_prev_state = gt_future[:, target_t-1, :]                      │
│                                                                             │
│       ego_gru_hidden = zeros  # GRU hidden도 리셋                           │
│       z_local 윈도우 = GT 기반으로 재구축                                   │
│                                                                             │
│    2. Autoregressive tf_segment_len steps 예측                              │
│       ─────────────────────────────────────                                 │
│       For step = 0 to min(tf_segment_len, FT-target_t)-1:                   │
│                                                                             │
│         a. Combined Position 생성                                           │
│            combined_pos = ego_prev_state[:, :4].clone()                     │
│            sur_gt_pos = gt_future[:, actual_t, :4] if needed                │
│            combined_pos[~ego_mask] = sur_gt_pos   # Sur는 GT                │
│                                                                             │
│         b. z_local 계산 (Ego 전용, Other는 참조만)                          │
│            z_local = _compute_z_local_dist(ego_feat, other_feat_window)     │
│                                                                             │
│         c. Ego 예측 (z_global + z_local → GRU → MLP)                        │
│            z_combined = z_combine_mlp(concat(z_global_ego, z_local))        │
│            ego_gru_in = concat(z_combined, map_feat, sem, lw)               │
│            ego_gru_out, ego_gru_hidden = ego_decoder_gru(ego_gru_in, h)     │
│            ego_traj_out = ego_output_mlp(ego_gru_out)                       │
│                                                                             │
│         d. Bicycle Model로 Ego 상태 업데이트                                │
│            ego_state_global = sim_traj(ego_prev, a, hdot, veh_len)          │
│            segment_preds.append(ego_state_global)                           │
│                                                                             │
│         e. 다음 스텝 입력 (autoregressive)                                  │
│            ego_prev_state = ego_state_global (자체 예측값)                  │
│            z_local 윈도우 슬라이드                                          │
│                                                                             │
│    3. Segment 예측 저장                                                     │
│       ─────────────────                                                     │
│       ego_segments.append(segment_preds)                                    │
│                                                                             │
│  출력: ego_segments: list of FT segments                                    │
│        각 segment: list of (num_ego, 4) predictions                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.5 듀얼 루프 Merge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    teacher_forcing_decoder: Merge                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  두 루프의 결과를 Segment 단위로 병합                                        │
│  ─────────────────────────────────────                                      │
│                                                                             │
│  sur_traj_all = _sur_loop_decoder(...)   # (num_other, FT, 4)               │
│  ego_segments = _ego_loop_decoder(...)   # list of FT segments              │
│                                                                             │
│  For seg_idx in range(FT):                                                  │
│      segment_preds = []                                                     │
│                                                                             │
│      For step, ego_pred in enumerate(ego_segments[seg_idx]):                │
│          actual_t = seg_idx + step                                          │
│          if actual_t >= FT: break                                           │
│                                                                             │
│          # Merge: Ego from Ego Loop, Sur from Sur Loop                      │
│          full_pred = zeros(NA, 4)                                           │
│          full_pred[ego_mask] = ego_pred                                     │
│          full_pred[~ego_mask] = sur_traj_all[:, actual_t, :]                │
│                                                                             │
│          segment_preds.append(full_pred)                                    │
│                                                                             │
│      all_segment_preds.append(segment_preds)                                │
│                                                                             │
│  출력 형태:                                                                  │
│  ───────────                                                                │
│  all_segment_preds = [                                                      │
│      seg0: [(NA,4), (NA,4), (NA,4)],  # t=0,1,2 predictions                 │
│      seg1: [(NA,4), (NA,4), (NA,4)],  # t=1,2,3 predictions                 │
│      ...                                                                    │
│      seg11: [(NA,4)]                   # t=11 prediction only               │
│  ]                                                                          │
│                                                                             │
│  핵심 포인트:                                                                │
│  ─────────────                                                              │
│  - Sur 예측: Sur Loop에서 전체 FT timesteps 한 번에 생성                    │
│  - Ego 예측: Ego Loop에서 segment별로 생성                                  │
│  - Merge 시점: 각 segment의 actual timestep에 맞춰 sur_traj_all 인덱싱     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.6 듀얼 루프의 장점

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Dual-Loop의 학습 효과                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  기존 단일 루프 TF 문제:                                                     │
│  ──────────────────────                                                     │
│  - Ego/Sur가 같은 루프에서 동시 예측                                        │
│  - Sur 예측 오류 → Ego z_local 계산 오염 → Ego 예측 오류                    │
│  - Ego 예측 오류 → Sur GCN message 오염 → Sur 예측 오류                     │
│  - 에러가 상호 증폭되는 악순환                                               │
│                                                                             │
│  듀얼 루프 해결:                                                             │
│  ──────────────                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Sur Loop                                                              │ │
│  │  ─────────                                                             │ │
│  │  - Sur가 "정확한 Ego GT"를 보고 z_global 기반 예측 학습                │ │
│  │  - Ego 예측 오류에 영향받지 않음                                       │ │
│  │  - 순수하게 자신의 예측 능력만 학습                                    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Ego Loop                                                              │ │
│  │  ─────────                                                             │ │
│  │  - Ego가 "정확한 Sur GT"를 보고 z_local 학습                          │ │
│  │  - Sur 예측 오류에 영향받지 않음                                       │ │
│  │  - 순수하게 reactive behavior만 학습                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  결과:                                                                      │
│  ─────                                                                      │
│  - Ego와 Sur가 각각 최적의 조건에서 학습                                    │
│  - 추론 시(TF=False) 둘 다 강건한 예측 가능                                 │
│  - 에러 발생해도 빠르게 recovery                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.7 Loss 계산 (_forward_teacher_forcing)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Teacher Forcing Loss Computation                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력: all_segment_preds (12 segments, 각 segment는 list of (NA,4))         │
│                                                                             │
│  For seg_idx, segment_preds in enumerate(all_segment_preds):                │
│                                                                             │
│    1. Segment 예측을 텐서로 스택                                            │
│       seg_traj = stack(segment_preds)  # (NA, seg_len, 4)                   │
│                                                                             │
│    2. 해당 GT segment 추출                                                  │
│       gt_start = seg_idx                                                    │
│       gt_end = min(seg_idx + seg_len, FT)                                   │
│       gt_seg = future_gt[:, gt_start:gt_end, :4]                            │
│                                                                             │
│    3. Reconstruction loss (visibility mask 적용)                            │
│       valid_mask = future_vis[:, gt_start:gt_end] == 1.0                    │
│       seg_recon_loss = -log_normal(pred[valid], gt[valid])                  │
│                                                                             │
│    4. Potential loss (segment별로 계산)                                     │
│       seg_potential_veh = veh_potential_loss(seg_traj)                      │
│       seg_potential_env = env_potential_loss(seg_traj)                      │
│                                                                             │
│  최종 Loss:                                                                 │
│  ───────────                                                                │
│  loss = recon_weight * mean(total_recon)                                    │
│       + kl_weight * kl_loss(z_global)                                       │
│       + potential_veh_weight * mean(total_potential_veh)                    │
│       + potential_env_weight * mean(total_potential_env)                    │
│       + sparse_weight * sparsity_loss(z_local)  # if enabled                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.8 설정 파라미터

```python
# train_trafficplanner.cfg

# Teacher Forcing 활성화
use_teacher_forcing: True    # False면 기존 단일 루프 autoregressive
tf_segment_len: 3            # Ego Loop에서 각 segment autoregressive 스텝 수
```

### B.9 Training vs Validation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Training vs Validation                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training (use_teacher_forcing=True):                                       │
│  ─────────────────────────────────────                                      │
│  - 듀얼 루프 구조 사용                                                       │
│  - Sur: ego=GT 기반 전체 FT 예측                                            │
│  - Ego: sur=GT 기반 segment별 TF 예측                                       │
│  - 출력: 12 segments 리스트                                                  │
│  - Loss: segment별 평균                                                     │
│                                                                             │
│  Validation (use_teacher_forcing=False):                                    │
│  ───────────────────────────────────────                                    │
│  - 기존 autoregressive_decoder 사용                                         │
│  - Ego/Sur 동시에 t=0부터 t=11까지 예측                                     │
│  - 출력: (NA, FT, 4) 텐서                                                   │
│  - 실제 추론 성능 평가 (에러 누적 포함)                                      │
│  - Best model 선정 기준                                                     │
│                                                                             │
│  핵심: 모델 가중치는 동일, 디코딩 방식만 다름                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.10 전체 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Teacher Forcing Dual-Loop 전체 흐름                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  teacher_forcing_decoder() 호출 시:                                         │
│  ────────────────────────────────                                           │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Step 1: Sur Loop 실행                                          │    │
│     │  ──────────────────────                                          │    │
│     │  sur_traj_all = _sur_loop_decoder(                              │    │
│     │      scene_graph, map_feat, past_seq_out, z_global,             │    │
│     │      map_idx, map_env, ego_mask, gt_future                      │    │
│     │  )                                                               │    │
│     │  → sur_traj_all: (num_other, FT, 4)                             │    │
│     │                                                                  │    │
│     │  Sur는 ego=GT 환경에서 전체 12 timesteps autoregressive 예측    │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Step 2: Ego Loop 실행                                          │    │
│     │  ──────────────────────                                          │    │
│     │  ego_segments, z_local_outputs = _ego_loop_decoder(             │    │
│     │      scene_graph, map_feat, past_seq_out, z_global,             │    │
│     │      map_idx, map_env, ego_mask, gt_future, tf_segment_len      │    │
│     │  )                                                               │    │
│     │  → ego_segments: list of 12 segments                            │    │
│     │    각 segment: list of (num_ego, 4) predictions                 │    │
│     │                                                                  │    │
│     │  Ego는 sur=GT 환경에서 segment별 TF 예측                        │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │  Step 3: Merge                                                   │    │
│     │  ──────────                                                      │    │
│     │  For each segment:                                               │    │
│     │      full_pred[ego_mask] = ego_pred                             │    │
│     │      full_pred[~ego_mask] = sur_traj_all[:, actual_t, :]        │    │
│     │      all_segment_preds[seg_idx].append(full_pred)               │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│     출력: all_segment_preds = [seg0, seg1, ..., seg11]                     │
│           각 seg: [(NA,4), (NA,4), ...] up to tf_segment_len predictions   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### B.11 주의사항

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    구현 시 주의사항                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. pred['future_pred'] 형태 차이                                           │
│     ─────────────────────────────                                           │
│     - TF=False: Tensor (NA, FT, 4)                                         │
│     - TF=True:  List of 12 segments, each segment is list of (NA, 4)       │
│     → Loss function에서 형태에 따라 분기 처리 필요                          │
│                                                                             │
│  2. compute_err() 스킵                                                      │
│     ─────────────────────                                                   │
│     - TF 모드에서는 빈 dict 반환                                            │
│     - Validation은 TF=False로 별도 측정                                     │
│     if isinstance(pred_future, list): return {}                             │
│                                                                             │
│  3. z_local 저장                                                            │
│     ───────────                                                             │
│     - Ego Loop에서 z_local 샘플, mean, var 저장                             │
│     - _z_local_outputs, _z_local_mean_outputs, _z_local_var_outputs         │
│     - Sparsity loss 계산에 사용                                             │
│                                                                             │
│  4. 체크포인트 호환성                                                        │
│     ─────────────────                                                       │
│     - TF=True로 학습한 체크포인트 → TF=False로 테스트 가능                  │
│     - 모델 가중치 구조 100% 동일                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 참고: 코드 파일 위치

- 모델: `src/models/trafficplanner_model.py`
- GCN: `src/models/individual_interaction_net.py`
- 공통 모듈 (MLP, car_dynamics): `src/models/common.py`
- 데이터셋: `src/datasets/nuscenes_dataset.py`
- 학습: `src/train_trafficplanner.py`
- 손실 함수: `src/losses/trafficplanner_loss.py`
- 테스트: `src/test_trafficplanner.py`
