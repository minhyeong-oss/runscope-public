# Landing ±1 frame + four-stage frame classification + form feedback v2

v1의 평가 정책을 다음처럼 정정한 별도 실험이다.

- Landing 최초 접지: 순간 이벤트이므로 `±1 native video frame` 허용
- 제동·추진·회복: 시간 구간이므로 허용오차 없이 프레임 분류 F1 평가
- 4단계 전체: 프레임 단위 Macro F1 평가
- 자세 Good/Improve/Unknown: 규칙 기반 스크리닝이며 정답 라벨이 없어 F1 없음

기존 실험과 출력은 덮어쓰지 않는다.

