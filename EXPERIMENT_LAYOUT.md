# RunScope 실험 버전 관리 규칙

## 불변 기준선

- `model_v1/`
- `video_v1/`
- `model_v1/twente_subject_split_v3/`

위 경로의 기존 결과는 비교 기준으로 보존한다. 후속 실험에서 파일을 덮어쓰지 않는다.

## 후속 독립 실험 경로

- `experiments/pose_foot_keypoints_v1/`
- `experiments/boundary_regression_v1/`
- `experiments/tcn_phase_v1/`
- `experiments/hybrid_v1/`

각 경로는 다음 구조를 사용한다.

```text
experiment/
├── README.md
├── config/
├── src/
├── artifacts/
└── reports/
```

## 평가 규칙

1. 모델 선택과 임계값 결정은 개발군에서만 수행한다.
2. 기존 잠금 test Subj04·Subj11은 재학습에 사용하지 않는다.
3. 이미 결과를 확인한 Subj04·Subj11은 후속 모델의 새로운 미사용 final test로 주장하지 않는다.
4. 각 모델은 독립적으로 평가한 후 마지막에만 하이브리드 후보로 결합한다.
5. 하이브리드 모델도 별도 버전과 아티팩트를 사용한다.
