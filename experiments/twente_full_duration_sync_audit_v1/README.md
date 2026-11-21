# Twente full-duration synchronization drift audit v1

Twente 개발 참가자의 약 60초 측면 영상 전체를 학습에 추가하기 전에,
영상과 마커 시간축 사이의 고정 offset이 trial 전체에서 유지되는지 확인하는
독립 감사다. 기존 10~22초 실험과 모델·산출물은 수정하지 않는다.

## 고정 원칙

- 개발 참가자 `Subj05,06,07,08,09,10,12`만 사용한다.
- `Subj04`, `Subj11`, `test2`, `test3`는 접근하지 않는다.
- 정렬은 MediaPipe heel/foot-index 궤적과 3D 마커 궤적만 사용한다.
- 힘판 Initial Contact는 정렬 offset 선택에 사용하지 않는다.
- 각 trial의 `5~15, 15~25, 25~35, 35~45, 45~55초`를 독립적으로 처리한다.
- offset 검색 범위는 `-120~+120 ms`, 간격은 `0.5 ms`다.
- 구간 offset 범위가 `15 ms`를 넘으면 고정 offset 안정성을 통과시키지 않는다.

## 출력

- `reports/window_sync_audit.csv`: 구간별 offset, 상관, 포즈 품질
- `reports/trial_drift_summary.csv`: trial별 offset 범위, 선형 drift 적합
- `reports/AUDIT_SUMMARY.json`: 전체 요약과 프로토콜 SHA-256
- `reports/AUDIT_REPORT_KO.md`: 해석용 한국어 보고서
- `artifacts/raw_pose/`: 재실행을 피하기 위한 구간별 MediaPipe 33점 CSV

이 감사 결과만으로 모델 성능 향상을 주장하지 않는다. 통과한 구간으로 확장
기준선을 새로 만들고, 그 이후에 후보 생성·선택·정밀화 모델을 비교한다.
