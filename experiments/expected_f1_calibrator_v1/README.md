# Expected F1 calibrator v1

힘판 정답이 없는 신규 영상에서도 과거 라벨 영상의 관계를 이용해 Hybrid
Macro F1의 기대값을 추정하는 별도 성능 보정 모델이다. 4단계 분류 모델을
덮어쓰거나 새 영상을 재학습에 넣지 않는다.

- 학습 단위: 기존 개발 참가자 7명, 21개 trial
- 입력: Hybrid 평균 최대확률
- 정답: 각 trial의 힘판 라벨 기반 OOF Hybrid Macro F1
- 검증: 참가자 단위 LOSO
- 출력: 예상 Macro F1과 LOSO 잔차 기반 90% 경험적 범위

이 출력은 실제 정답과 비교한 실측 F1이 아니라 `predicted expected F1`이다.
