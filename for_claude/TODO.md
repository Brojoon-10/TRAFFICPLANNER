# TODO.md — TrafficPlanner 모델 재설계 작업 목록

> 상태: [ ] 미시작 | [~] 진행중 | [x] 완료

---

## Phase A: 분석 및 설계
- [x] 현재 모델 구조 파악 및 문제점 분석
- [x] 데이터 현황 파악 (CARLA 시나리오)
- [x] CPU-GPU 병목 전수 조사
- [x] 재설계 아키텍처 확정 (v5)
- [x] CARLA 정규화 전환 (nuScenes → CARLA stats)

## Phase B: 핵심 모델 구현 (v5)
- [x] AdaLN conditioning (ego/sur 분리)
- [x] Transformer Decoder (2L, FFN256, d_model=128)
- [x] A2A RelBias (상대 거리/방향 bias)
- [x] A2Z CrossAttn (map cross-attention)
- [x] Intent Codebook (K=9, Gumbel-Softmax)
- [x] z_aux head (z_global → trajectory)
- [x] map_dist_aux head
- [x] Auxiliary losses (intent_ce, sur_pred, ego_pred, map_attn)
- [x] Action Blending (TF 대체)
- [x] Potential-based collision loss

## Phase C: 2-Phase 학습
- [x] Phase 1 구현: encoder-only 학습 (z_aux + KL + map_dist_aux)
- [x] Phase 1 학습 완료 (30 epochs) — z_aux ADE=0.79m, corr=0.93
- [x] Phase 1 viz 비활성화 (decoder 미사용)
- [x] Phase 2 구현: frozen encoder + fresh decoder
- [~] Phase 2 학습 진행 중 (80 epochs)

## Phase D: Viz 개선
- [x] Viz 폴더 구조 정리 (map_attention/, intent/, intent_grid/)
- [x] Intent viz FT 버그 수정 (torch.cat → torch.stack, 24→12 steps)
- [x] Intent x축 라벨 수정 (t*0.5 → (t+1)*0.5)
- [x] Intent grid viz 추가 (2D scatter: GT vs Model on prototype grid)
  - [x] 축 스왑 (y=Acc, x=Yaw Rate)
  - [x] 코너 방향 라벨 (Accel+Left, Decel+Right 등)
  - [x] Reds colormap + colorbar 중앙 배치
- [x] Map+Traj 패널 추가 (intent, intent_grid 둘 다)
- [x] z_aux viz 스크립트 (viz_z_aux.py)

## Phase E: 향후 개선 (미착수)
- [ ] Prior 네트워크 분리 (현재 prior/posterior weight 공유)
- [ ] Posterior에 future map 인코딩 ([past|future|map] self-attention)
- [ ] Phase 1 내 2-stage: Stage A(posterior 학습) → Stage B(prior만 KL 피팅)
- [ ] Prior가 posterior 분포 모방 학습 (posterior 고정 + prior unfreeze)
- [ ] 데이터 증강 (CARLA 시나리오 추가 생성)

---

## 현재 상태 요약
Phase 2 학습 진행 중. Viz 개선 완료 (intent grid, map+traj 패널 등).
향후: prior 분리 + future map 인코딩으로 z_global 품질 향상 계획.
