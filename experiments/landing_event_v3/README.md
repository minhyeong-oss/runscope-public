# landing_event_v3

Landing을 4단계의 희소 프레임 클래스가 아니라 최초 접지 이벤트로 분리하기
위한 독립 실험 버전이다. 기존 모델·결과를 덮어쓰지 않는다.

## 준비 결과

- Twente 개발 7명, 힘판 접지 정답 460보폭
- 64시점 × 26개 관절 특징
- 단일 접지 위치 hard target
- 15 ms Gaussian soft target
- 접지 후 30 ms 표시용 Landing window target
- 영상–힘판 동기 품질을 high/medium/low로 분리
- 참가자 LOSO와 ±30/±60 ms 평가 규칙 고정
- test2/test3 및 Subj04/Subj11 학습 제외

## 학습·평가 상태

- 개발 대상자 7명 LOSO로 Ridge 2종과 Landing TCN 2종을 비교했다.
- 개발 결과만으로 `ridge_alpha1000`을 선택해 동결했다.
- 동결 뒤 Subj04·Subj11은 재학습·재조정 없이 역사적 참조 테스트에만 사용한다.
- 이 두 대상자는 이미 과거 실험에서 사용됐으므로 새 미접촉 test로 주장하지 않는다.
