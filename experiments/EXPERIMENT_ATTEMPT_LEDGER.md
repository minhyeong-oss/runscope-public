# RunScope experiment attempt ledger

이 장부는 같은 개발 참가자에서 반복 실험한 횟수와 결과를 숨기지 않기 위해
유지한다. 개발 결과를 여러 번 본 뒤 선택한 모델은 untouched test 결과로
부르지 않는다.

| 날짜 | 실험 | 목적 | 데이터 접근 | 결과 | 모델 교체 |
|---|---|---|---|---|---|
| 2026-11-21 | `twente_full_duration_sync_audit_v1` | 60초 영상의 구간별 marker-pose offset 안정성 감사 | 개발 7명, force는 offset 선택에 미사용 | 21 trial 중 stable 7; 승인 window 74/105 | 아니오 |
| 2026-11-21 | `twente_expanded_alignment_v1` | 승인 구간으로 보폭 확장 후 4단계 Ridge 개발 LOSO | 개발 6명, 1,642보폭 | Macro F1 0.701; 사전 기준 미달 | 아니오 |
| 2026-11-21 | `twente_expanded_weighting_ablation_v1` | 참가자 균형 가중 영향 분리 | 같은 개발 6명 | 가중 0.701, 비가중 0.704; 가중치가 주원인 아님 | 아니오 |
| 2026-11-21 | `twente_sample_participant_curve_v1` | 참가자 수와 참가자당 보폭 수 효과 분리 | 같은 승인 개발 6명; nested alpha | 보폭 3.88배 효과 -0.0022(CI 0 포함), 참가자 2→5명 효과 +0.0197(CI 0 초과) | 아니오 |

Subj04·11 및 test2·3은 위 네 실험의 학습·튜닝·재평가에 사용하지 않았다.
