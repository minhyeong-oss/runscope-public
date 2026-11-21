# RunScope one-frame four-stage form v1

기존 모델과 결과를 덮어쓰지 않고 다음 평가 정책을 통합한 실험이다.

- 이벤트 허용 오차: 영상별 `±1 frame`
- Twente 영상: 33 fps, 즉 약 `±30.3 ms`
- 4단계: 착지, 제동, 추진, 회복
- 자세 표시: `GOOD` 초록, `IMPROVE` 빨강, `UNKNOWN` 회색
- 자세 평가는 최소 5보폭과 관절 신뢰도 품질문을 적용한다.
- 추진·회복은 검증된 측면 2D 절단값이 없어 v1에서 `UNKNOWN`으로 유지한다.

착지 이벤트 F1, 4단계 인식 F1, 자세 Good/Bad는 서로 다른 평가 문제다.
자세 정답 라벨이 없으므로 자세 F1은 계산하지 않는다.

```bash
.venv_yolo/bin/python experiments/one_frame_four_stage_form_v1/src/run_locked_trials.py
```

