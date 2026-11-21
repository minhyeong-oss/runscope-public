# landing_failure_audit_v1

힘판 전체 접지 이벤트를 분모로 Landing 파이프라인의 실패 원인을 분해하는 후향적 감사 실험이다. 기존 모델을 재학습하거나 임계값을 재조정하지 않는다.

## 실행

```bash
python experiments/landing_failure_audit_v1/src/audit_landing_failures.py
```

## 입력

- Twente 개발 7명 pose·힘판 자료
- 개발 참가자 분리 OOF `ridge_alpha1000` 예측
- 역사적 참조 Subj04·Subj11의 동결 모델 예측
- 기존 marker–video 동기화 함수와 힘판 event 추출 함수

## 규칙

- [사전 고정 프로토콜](PREREGISTERED_PROTOCOL_KO.md)
- 기계 판독 설정: `config/protocol.json`
- 주원인 우선순위: `POSE_MISS → FOOT_SWAP → SYNC_LOW → STRIDE_MISS → MODEL_ERROR → PASS`
- 중복 원인은 이벤트 CSV의 `flag_*` 열에 별도로 보존

## 출력

- `reports/AUDIT_REPORT_KO.md`: 사람이 읽는 핵심 보고서
- `reports/AUDIT_SUMMARY.json`: 전체 요약과 프로토콜 해시
- `reports/force_event_failure_audit.csv`: 힘판 이벤트 단위 판정과 원시 flag
- `reports/trial_end_to_end_summary.csv`: trial 단위 coverage와 원인 집계
- `reports/primary_cause_summary.csv`: 분할별 주원인 집계

## 주의

현재 자료에 대한 규칙은 기존 집계 결과가 알려진 뒤 만든 후향적 감사 규칙이다. 이후 합성 데이터 비교에는 전향적으로 고정해 사용한다. 역사적 참조 결과로 모델을 다시 선택하거나 조정하면 안 된다.
