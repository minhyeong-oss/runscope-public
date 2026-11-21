# RunScope 영상 분석 행동강령

## 1. 촬영할 때

1. 카메라는 달리는 방향과 직각인 측면에 고정한다.
2. 머리부터 발끝까지 전신이 영상 안에 들어오게 한다.
3. 최소 30 fps로 촬영하고, 가능하면 60 fps를 사용한다.
4. 시작·정지·방향 전환보다 일정한 속도로 달리는 구간을 길게 촬영한다.
5. 러닝머신 표시값이나 측정 장비로 실제 속도를 기록한다. 단안 영상만으로는 m/s 단위 속도를 확정할 수 없다.

## 2. 분석할 때

아래 명령에서 영상 경로와 속도만 바꾼다.

```bash
YOLO_CONFIG_DIR=/private/tmp/ultralytics \
MPLCONFIGDIR=/private/tmp/runscope_mpl \
  .venv_yolo/bin/python video_v1/analyze_video_end_to_end.py \
  /absolute/path/to/running_video.mp4 \
  --speed 2.0 \
  --motion-mode treadmill \
  --facing right \
  --side-mapping direct
```

- 일반 영상은 `direct`를 먼저 사용한다.
- BML-MoVi Subject 1 PG1 영상은 V3D 대조 결과에 따라 `swapped`를 사용한다.
- 현재 모델 학습 속도 범위는 1.8–2.4 m/s이다. 이 범위 밖 결과는 실험 결과로만 취급한다.
- 러닝머신 영상은 `--motion-mode treadmill`을 사용한다.
- 사람이 화면 오른쪽을 바라보면 `--facing right`, 왼쪽을 바라보면 `--facing left`를 사용한다.

## 2-1. 나중에 본인 영상을 보낼 때

다음 네 가지를 함께 제공한다.

1. 편집하지 않은 원본 MP4 또는 MOV 영상
2. 러닝머신 표시 속도와 단위(km/h 또는 m/s)
3. 영상에서 바라보는 방향(왼쪽 또는 오른쪽)
4. 가능하면 촬영 fps와 스마트폰 슬로모션 사용 여부

분석 결과로 다음을 만든다.

- 양발 4단계 전체 오버레이 영상
- stride별 착지·버티기·밀어내기·회수 대표 장면 4분할 이미지
- 4단계 반복 GIF
- 단계 시간, 케이던스 추정, 좌우 차이, 반복 일관성 표
- Good extraction / Review required 기술 품질 판정

## 3. 결과를 판정할 때

1. `*_full_two_foot_phase_overlay.mp4`에서 두 발의 단계 표시를 확인한다.
2. `*_auto_phase_summary.csv`에서 stride별 3단계·4단계 시작 시점을 확인한다.
3. `*_four_stage_scenes` 폴더에서 각 stride의 4분할 장면과 GIF를 확인한다.
4. `AUTO_OK`는 해당 프레임의 기술적 모델 확률이 기준 이상이라는 뜻이다.
5. `REVIEW_TRANSITION`은 단계 전환 부근에서 모델 확률이 0.60 미만이므로 사람이 확인해야 한다는 뜻이다.
6. `OUTSIDE_ACCEPTED_STRIDE`는 측면성·회전·신뢰도·stride 길이 조건을 통과하지 못한 구간이다. 억지로 단계 결과를 만들지 않는다.

## 4. Good/Bad를 구분할 때

- `AUTO_OK`를 좋은 달리기 자세로 해석하지 않는다.
- `REVIEW_TRANSITION`을 나쁜 달리기 자세로 해석하지 않는다.
- 현재 Good/Bad는 영상 처리 품질에 대해서만 다음처럼 사용한다.
  - Good extraction: 전신 측면, 관절 신뢰도 0.85 이상, stride 0.65–1.15초, 회전 없음.
  - Review required: 관절 가림, 방향 전환, 정면에 가까운 자세, 비정상적으로 짧거나 긴 stride.
- 생체역학적 Good/Bad 자세 판정은 부상·효율 기준의 별도 정답 라벨이 필요하다.

## 5. 발표에서 말할 수 있는 범위

- 말할 수 있음: 사람 영상에서 pose를 추출하고, stride를 자동 검출하고, 발별 4단계를 표시하는 오프라인 프로토타입을 만들었다.
- 말할 수 있음: 채택된 착지 이벤트 9개의 V3D 대비 평균 오차는 1.78프레임, 약 59ms였다.
- 말하면 안 됨: BML-MoVi에서 4단계 정확도가 검증됐다.
- 말하면 안 됨: 영상만으로 좋은 자세와 나쁜 자세를 의학적으로 판정한다.
