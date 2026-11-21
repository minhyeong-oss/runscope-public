# RunScope running-form feedback v1

기존 4단계 분류 모델의 결과를 덮어쓰지 않고, 측면 러닝 영상에서 자세를
근거 기반으로 스크리닝하는 별도 실험입니다.

## 출력 의미

- `GOOD / 양호`: 선택한 스크리닝 기준을 반복 보폭에서 충족했습니다.
- `IMPROVE / 개선 권장`: 반복 보폭에서 기준을 벗어났습니다. 진단이 아니라 촬영 영상을 바탕으로 한 운동 피드백입니다.
- `UNKNOWN / 판정 불가`: 연구 근거, 시점, 영상 품질 중 하나가 부족합니다.

개선 영상은 `IMPROVE`일 때만, 사람이 검토한 허용 목록에서 제공합니다.
검색 결과를 실행 중에 임의로 추천하지 않습니다.

## 현재 v1 규칙

1. 착지: 착지 단계의 지지 다리 정강이가 수직에 가까운지 확인합니다.
   절대 정강이 기울기 중앙값 5도 이하이고, 80% 이상의 보폭이 8도 이하면
   `GOOD`으로 표시합니다. 이 수치는 의료 진단 절단값이 아닌 보수적인 제품
   스크리닝 규칙입니다.
2. 제동: 제동 단계의 보폭별 최대 무릎 굽힘을 계산합니다. 5보폭 이상에서
   중앙값이 40도 미만이면 `IMPROVE`, 40도 이상이면 `GOOD`입니다. 문헌은
   지지기 최대 무릎 굽힘을 약 45도로 설명하지만 명확한 임상 절단값은 아직
   확립되지 않았다고 명시합니다.
3. 추진: 고관절 신전 대리값은 표시하지만, 속도와 개인차를 반영한 합의 기준이
   없어 `UNKNOWN`입니다.
4. 회복: 측면 2D 영상만으로 사용할 검증된 v1 규칙을 선택하지 않아
   `UNKNOWN`입니다.

모든 판정은 최소 5개 보폭과 관련 관절의 중앙 신뢰도 0.50 이상을 요구합니다.
후면에서 평가해야 하는 무릎 외반, 골반 하강, 발뒤꿈치 외번은 측면 영상에서
판정하지 않습니다.

## 실행

저장소 루트에서 실행합니다.

```bash
.venv_yolo/bin/python experiments/running_form_feedback_v1/src/analyze_form.py
.venv_yolo/bin/python experiments/running_form_feedback_v1/src/render_feedback.py
.venv_yolo/bin/python -m unittest discover -s experiments/running_form_feedback_v1/tests -v
```

기본 입력은 `Subj05_run81_side_10s_to_22s.mp4`이며 결과는 이 실험 폴더의
`reports/` 아래에 새로 생성됩니다.

## 근거와 한계

- Souza R. *An Evidence-Based Videotaped Running Biomechanics Analysis*.
  PM&R, 2016. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4714754/>
- Heiderscheit BC, et al. *Effects of step rate manipulation on joint mechanics
  during running*. Med Sci Sports Exerc, 2011.
  <https://pubmed.ncbi.nlm.nih.gov/20581720/>
- Tripodi N, et al. *Running retraining to treat lower limb injuries: a mixed-methods
  masterclass*. J Sci Med Sport, 2024.
  <https://vuir.vu.edu.au/48172/1/1-s2.0-S1746068923000421-main.pdf>

2D 마커리스 포즈는 가려짐, 카메라 각도, 의복, 속도에 민감합니다. 이 출력은
의학적 진단이나 부상 위험 예측이 아니며, 통증이 있으면 의료 전문가의 평가가
우선입니다. 자세 Good/Bad 정답 라벨이 없으므로 이 규칙 출력에는 F1-score를
붙이지 않습니다. 4단계 인식 F1과 자세 스크리닝은 서로 다른 평가 문제입니다.
