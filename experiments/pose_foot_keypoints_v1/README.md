# pose_foot_keypoints_v1

기존 YOLO COCO-17 기준선을 수정하지 않고, 뒤꿈치와 발끝 관절을 추가하는 독립 포즈 실험입니다.

## 목적

- MediaPipe Pose Landmarker의 33개 관절 사용
- 왼쪽·오른쪽 heel 및 foot index 추출
- Twente 개발군의 3D `LCAL/RCAL`, `L1MT/R1MT` 마커와 궤적 상관 검증
- 이 실험에서는 4단계 분류 모델을 학습하지 않음

## 데이터 규칙

- 개발 확인 영상: Subj05만 사용
- 잠금 test Subj04·Subj11은 사용하지 않음
- 기존 `video_v1/` 출력과 모델을 덮어쓰지 않음

## 경로

- `config/`: 실행 설정
- `src/`: 독립 추출·검증 코드
- `models/`: 로컬 MediaPipe task 파일(Git 제외)
- `artifacts/`: 추출 CSV(Git 포함 가능)
- `reports/`: 품질 보고서

## 판정 기준

heel과 foot-index의 영상 x 궤적을 골반 중심 기준으로 정규화한 뒤, 동기화된 3D 마커의 시상면 궤적과 절대 Pearson 상관을 계산합니다. 양발 평균 상관이 0.90 이상이면 다음 경계 회귀 실험의 입력 후보로 통과시킵니다.

