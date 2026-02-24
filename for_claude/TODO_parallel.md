# TODO — GRU TF 병렬화 작업 목록

> 상태: [ ] 미시작 | [~] 진행중 | [x] 완료

---

## Phase 1: 버그 수정 (독립적, 낮은 위험)
- [ ] data split 85/10/5 (fit_dataset.py)
- [ ] save_state/load_state global_step (torch.py + 호출부)
- [ ] LR scheduler step 기반 + warmup floor (train_trafficplanner.py)
- [ ] Config 업데이트 (train_trafficplanner.cfg)
- [ ] Grad clipping (train_trafficplanner.py)

## Phase 2: History Attention batched
- [ ] DecoderHistoryAttention.forward_batched 구현
- [ ] causal mask + t=0 zero out
- [ ] 수치 동일성 테스트 (forward vs forward_batched)

## Phase 3: Step A GCN 배치화
- [ ] _sur_loop_decoder Step A: Batch.from_data_list 1회 호출
- [ ] _ego_loop_decoder Step A: 동일
- [ ] ego/sur GCN 결과가 순차 실행과 동일한지 검증

## Phase 4: Step B GRU input 사전 배치 계산
- [ ] History Attn / Map Attn / Intent를 미리 배치 계산
- [ ] GRU loop 내부 → 사전계산 결과 인덱싱만
- [ ] hidden snapshot 결과 동일성 검증

## Phase 5: Step C 완전 배치화
- [ ] Sur loop: N*FT 배치 처리
- [ ] Ego loop: N*FT 배치 처리
- [ ] 분석 저장 (loss 호환 format)
- [ ] map_recrop 배치 처리

## Phase 6: map_recrop mult_samp
- [ ] _recompute_map_tokens mult_samp 지원

## Phase 7: 검증
- [ ] 수치 동일성: 순차 vs 병렬 (torch.allclose)
- [ ] Gradient 흐름: loss.backward() → 주요 모듈 .grad 확인
- [ ] AR 경로 무변경 확인
- [ ] 학습 5-10 epoch: loss 감소, NaN 없음
- [ ] TF/AR 일관성 확인

## Phase 8: 커밋 및 푸시
- [ ] 커밋
- [ ] 푸시

---

## 현재 상태
Phase 1 시작 예정
