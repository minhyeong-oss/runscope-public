# RunScope 4단계 구간 분류 모델 카드

## 모델 개요

- 버전: 0.1.0
- 목적: 완료된 한 보폭(stride)의 각 프레임을 4개 달리기 단계로 분류
- 모델: `HistGradientBoostingClassifier` + 단계 순서 제약 디코더
- 지원 범위: 오프라인 분석, 1.8–2.4 m/s
- 직접 학습 데이터: 5번 Reznick
- 보조 기준 데이터: 3번 Fukuchi, 4번 King, 7번 Baker

3·4·5·7 외 데이터는 이 모델의 학습·선정·테스트에 사용하지 않았다.

## 4단계 정답 정의

1. `landing_window`: force 기반 heel strike 직후 30 ms
2. `braking`: landing window 종료부터 전후방 지면반력(AP force)의 양→음 전환까지
3. `propulsion`: AP force 전환점부터 toe-off까지
4. `recovery`: toe-off부터 같은 발의 다음 heel strike까지

1단계는 원래 한 시점인 heel strike를 프레임 분류에 맞게 30 ms 구간으로 확장한 운영상 정의다. 따라서 1단계 F1 1.0을 독립적인 착지 검출 성능으로 해석하면 안 된다.

## 데이터별 사용 방식

| 번호 | 데이터 | 모델에서 맡은 역할 | 이번 모델에 직접 투입 여부 |
|---:|---|---|---|
| 3 | Fukuchi RBDS | 2.5–4.5 m/s 고속 phase 기준 및 향후 속도 외삽 점검 | 아니요 — feature schema와 속도 범위가 다름 |
| 4 | King event timing | IC·TO 시각 오차 벤치마크 | 아니요 — 2→3 경계 정답이 없음 |
| 5 | Reznick continuous locomotion | 100 Hz 관절각 입력과 1000 Hz force 기반 4단계 정답 | 예 — 학습·검증·잠금 테스트 |
| 7 | Baker running environments | 실외·트레드밀 force timing 외부 검증 기준 | 아니요 — 동일 관절각 feature가 없고 recovery가 일부만 관측됨 |

즉, 3·4·7은 아무 데이터나 섞어 학습한 것이 아니라 각 데이터가 실제로 증명할 수 있는 항목에만 사용했다. 서로 다른 센서 열을 억지로 결합하면 데이터 번호를 학습하는 모델이 될 위험이 있어, 첫 모델은 공통된 프레임 입력을 가진 5번으로 학습했다.

## 입력 특성

총 14개 열을 사용한다.

- 속도와 stride 진행률: `nominal_speed_m_s`, `cycle_fraction`
- 각도: 고관절, 무릎, 발목 굽힘 및 골반 기울기
- 각속도: 위 4개 각도의 1차 미분
- 각가속도: 위 4개 각도의 2차 미분

정확한 열 이름과 단위는 `artifacts/feature_schema.json`에 있다.

## 참가자 단위 분할

- 학습: AB01, AB02, AB03, AB05, AB06, AB09 — 1,480 strides
- 검증: AB07, AB08 — 484 strides
- 잠금 테스트: AB04, AB10 — 353 strides

같은 참가자가 학습과 테스트에 동시에 들어가지 않도록 참가자 단위로 분리했다.

## 잠금 테스트 결과

| 지표 | ML 모델 | 시간 기준선 |
|---|---:|---:|
| 프레임 정확도 | 0.9521 | 0.8648 |
| Macro F1 | 0.9585 | 0.8558 |
| 2→3 경계 MAE | 16.53 ms | 31.27 ms |
| 3→4 경계 MAE | 23.35 ms | 70.05 ms |

단계별 F1은 landing 1.0000, braking 0.9574, propulsion 0.9072, recovery 0.9695다.

## 사용 방법

입력 CSV에 `feature_schema.json`의 열을 준비한 뒤 아래처럼 실행한다.

```bash
python predict_phase_csv.py input.csv output.csv
```

출력에는 `phase_label`과 `phase_name` 열이 추가된다. 디코더는 한 stride 안에서 단계가 1→2→3→4 순서로만 진행되도록 제한한다.

## 사용하면 안 되는 경우와 한계

- 현재 성능은 5번 데이터 내부의 잠금 참가자 테스트 결과이며, 3번·7번의 외부 검증 결과가 아니다.
- `cycle_fraction`은 완료된 stride 길이를 알아야 계산되므로 현재 모델은 실시간 모델이 아니다.
- 영상에서 사용하려면 MediaPipe 등의 landmark로 동일한 각도·각속도·각가속도 열을 만드는 전처리기가 추가로 필요하다.
- 2.4 m/s를 넘는 속도, 다른 카메라 각도, 가림, 실외 영상에 현재 수치를 그대로 적용하면 안 된다.
- `good/review`는 데이터 추출 신뢰도다. 사람의 러닝 자세가 좋다/나쁘다는 판정이 아니다.
- 의료 진단, 부상 위험 판정 또는 치료 결정에 사용하지 않는다.

## 재현성과 파일

- 모델: `artifacts/runscope_phase_model_v1.joblib`
- 스키마: `artifacts/feature_schema.json`
- 학습 코드: `train_runscope_v1.py`
- 추론 코드: `predict_phase_csv.py`
- 상세 지표: `reports/metrics.json`
- 샘플 입력·출력: `reports/sample_input_stride.csv`, `reports/sample_prediction_stride.csv`
- 모델 SHA-256: `272d43814418a6b17fb05c0fc851c0727461f1b3e7066bb134b3028afc789109`
