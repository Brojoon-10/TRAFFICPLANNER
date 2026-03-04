# Model Simplification Plan

> 작성일: 2026-03-03
> 상태: 검토용 메모 (미적용)
> 목적: 데이터 1만개 학습 후에도 문제가 있을 경우 참고

---

## 0. 전제

- 현재 모델: 3.5M params, task 대비 적절한 크기
- 과적합 원인은 300개 데이터 (모델 크기 아님)
- **데이터 1만개 학습이 최우선**, 그래도 문제 시 아래 검토

---

## 1. 제거 대상

### 1a. A2Z Cross-Attention (제거)
- z_global이 token concat + AdaLN으로 이미 2중 주입
- A2Z까지 하면 3중 → 중복
- 제거 시 절감: ~100K params

### 1b. AdaLN → LayerNorm (제거)
- token concat으로 z_global conditioning 충분
- AdaLN MLP 6개 (A2T, A2A, A2Z, A2S, ego_FFN, sur_FFN) 전부 제거
- 제거 시 절감: ~150K params

### 1c. Ego/Sur Q/O 분리 → 통합 (제거)
- A2A: ego_Q/O + sur_Q/O → 단일 Q/O
- A2S: ego_Q/O + sur_Q/O → 단일 Q/O
- FFN: ego_FFN + sur_FFN → 단일 FFN
- Phase 2 fine-tuning 불가해짐 (주의)
- 제거 시 절감: ~400K params

### 1d. Sur Intent Codebook (제거)
- Sur agent intent 분류 불필요
- Ego intent codebook만 유지
- 제거 시 절감: ~30K params

### 1e. z_aux Decoder (제거)
- z_global 품질 검증용 보조 모듈
- 학습 안정화 확인 후 제거 가능
- 제거 시 절감: ~50K params

---

## 2. 축소 대상

### 2a. d_model 128 → 64, ffn_dim 256 → 128
- Layer 수는 2 유지 (1이면 intent 후 처리 부족)
- Aux heads 입력도 64로 축소

---

## 3. 유지 대상 (변경 불가)

- GCN (~200K) — agent interaction 필수
- Encoder prior/posterior (~300K) — CVAE 구조 유지
- Map CNN (~500K) — 도로 구조 인식 필수
- Ego Intent Codebook — z_local 학습 필요
- sur_pred_head, ego_pred_head — A2A 가이드용 유지
- Bicycle model output

---

## 4. 축소 후 Transformer Layer 구조

```
현재:  A2T → A2A(ego/sur) → A2Z → A2S(ego/sur) → ego_FFN/sur_FFN + AdaLN×6
축소:  A2T → A2A(통합) → A2S(통합) → FFN + LayerNorm×4
```

---

## 5. 예상 파라미터

| 모듈 | 현재 | 축소 후 |
|------|------|---------|
| GCN | ~200K | 200K |
| Encoder | ~300K | 300K |
| Map CNN | ~500K | 500K |
| Transformer (2L) | ~1.5M | ~250K |
| Aux/Intent | ~100K | ~50K |
| 기타 (token proj 등) | ~900K | ~200K |
| **합계** | **~3.5M** | **~1.5M** |

---

## 6. 주의사항

- Ego/Sur 통합 시 Phase 2 fine-tuning 재설계 필요
- AdaLN 제거 시 z_global → decoder 영향력 약화 가능
- 모델 축소보다 데이터 증가가 훨씬 효과적
