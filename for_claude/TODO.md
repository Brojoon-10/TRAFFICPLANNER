# TODO.md — TrafficPlanner 모델 재설계 작업 목록

> 상태: [ ] 미시작 | [~] 진행중 | [x] 완료

---

## Phase A: 분석 및 설계 (현재 단계)
- [x] 현재 모델 구조 파악 및 문제점 분석
- [x] 데이터 현황 파악 (1000 시나리오, train 500, val 166)
- [x] CPU-GPU 병목 전수 조사
- [x] 실제 사용 파일 의존성 트리 구축
- [x] rules.md 작성
- [x] redesign_plan.md v2 작성 (Decoder History Attention + z_local CVAE)
- [~] 재설계 아키텍처 확정 (사용자 승인 대기)

## Phase B: 핵심 모델 재구현
- [ ] DecoderHistoryAttention 모듈 구현
- [ ] Ego decoder 재설계 (history attention + 풍부한 MLP 입력)
- [ ] z_local Phase 1 zero-masking 구현
- [ ] history_buffer pre-allocation 및 in-place update
- [ ] scene_graph 업데이트 최적화 (pos만 in-place)

## Phase C: 학습 파이프라인 개선
- [ ] TF → Scheduled Sampling 전환
- [ ] SS annealing schedule 구현
- [ ] Loss vectorization (VehPotential, EnvPotential, VehColl)
- [ ] LR warmup + gradient clipping 추가
- [ ] 데이터 증강 전략 (시나리오 추가 뽑기)

## Phase D: Phase 1 학습 및 검증
- [ ] Phase 1 학습 실행
- [ ] Val loss ≤ 3.70 달성 확인
- [ ] Train/Val gap ≤ 0.03 확인

## Phase E: Phase 2 (z_local CVAE)
- [ ] ZLocalCVAE 모듈 구현 (prior/posterior)
- [ ] KL_local loss 추가
- [ ] Phase 2 finetune 실행 (z_global freeze + z_local + ego decoder)
- [ ] Ego reactive behavior 검증 (sur 접근 시 감속 등)

---

## 현재 상태 요약
Phase A 거의 완료. redesign_plan.md v2 작성 완료, 사용자 승인 대기 중.
핵심 변경: past_seq_out cross-attn → Decoder History Attention, z_local → CVAE self-organizing intent.
