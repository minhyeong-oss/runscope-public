# Landing event v3 중단 체크포인트

- 중단 시각: 2026-11-15 13:52:17 KST
- 상태: 사용자 요청으로 개발 LOSO 학습 중 정상 중단
- 프로세스: 종료됨
- 준비 데이터: 완료·검증됨
- 모델 선택/동결: 아직 안 됨
- Subj04·Subj11 v3 테스트: 아직 안 함

## 진행 위치

Subj05, 06, 07, 08, 09 fold는 메모리에서 완료됐고 Subj10 학습 도중
중단됐다. 현재 스크립트는 전체 fold가 끝난 뒤 결과를 저장하므로 부분 OOF
예측은 파일로 남지 않았다. 따라서 다음 실행은 결정론적 seed를 유지한 채
Subj05부터 다시 시작한다.

## 재시작 명령

```bash
MPLCONFIGDIR=/private/tmp/runscope-mpl \
XDG_CACHE_HOME=/private/tmp/runscope-cache \
.venv_yolo/bin/python experiments/landing_event_v3/src/train_candidates.py
```

학습이 끝나 `artifacts/FROZEN_SELECTION.json`이 생성되기 전에는 Subj04·11
테스트를 실행하지 않는다.

정확한 해시와 남은 작업은 `CHECKPOINT_STOPPED_2026-11-15.json`에 기록했다.
