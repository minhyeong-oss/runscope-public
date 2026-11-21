# RunScope 관절별 Good/Bad 근거 설계 v0.1

## 결론

RunScope의 색은 의료 진단이 아니라 **현재 속도·환경에서의 러닝 폼 피드백**을 뜻한다.

- 초록(GOOD): 같은 트레드밀·속도 조건의 참조범위 안이며, 반복 실험에서 낮은 제동/관절 하중과 같은 유리한 방향이 확인되고 영상 측정 신뢰도가 충분함
- 빨강(BAD): 참조범위를 측정오차보다 크게 벗어나고, 반복 실험에서 불리한 방향이 확인되며 여러 보폭에서 반복됨
- 회색(UNKNOWN): 경계값, 낮은 포즈 신뢰도, 가림, 지원 속도 밖, 측면 영상으로 판정 불가능, 또는 논문 간 방향이 일치하지 않음

건강한 사람의 평균에서 벗어났다는 이유만으로 BAD로 만들지 않는다. 부상자 데이터도 참가자 수준의 진단 라벨이므로 개별 보폭의 BAD 정답으로 직접 사용하지 않는다.

## 근거를 합치는 순서

1. Reznick 데이터로 4단계 사건과 속도별 3D 관절각/힘 기준을 계산한다.
2. 대규모 Ferber 데이터로 성별·속도·건강/부상 집단에서 기준이 재현되는지 확인한다.
3. cadence, overstride, trunk lean, foot strike를 조작한 트레드밀 실험에서 변화 방향을 확인한다.
4. 실제 측면 영상과 3D 모션캡처를 동시에 촬영하여 RunScope 2D 각도의 편향과 측정오차를 계산한다.
5. 두 명 이상의 평가자가 만든 실제 영상 라벨로 Good/Bad F1-score를 계산한다.

## 4단계별 표시 항목

| 단계 | 화면에 표시 | 1차 판정 근거 | 초기 상태 |
|---|---|---|---|
| 착지 | 무릎 굴곡, 발-골반 수평거리/다리길이, 몸통각 | 속도별 참조 + overstride/step-rate 실험 | 판정 가능 |
| 제동 | 무릎 굴곡 변화량, 제동시간, 정강이각 | force 기반 제동구간 + step-rate 실험 | 판정 가능 |
| 추진 | 고관절 신전, 무릎 신전, 발목각 | 속도별 참조 + 추진력 자료 | 발목은 신뢰도 부족 시 회색 |
| 회복 | 최대 무릎 굴곡, 고관절 굴곡, 좌우 시간차 | 속도별 참조 + 반복 보폭 변동 | 판정 가능 |

측면 한 대로는 골반 하강, 무릎 외반, 고관절 내전/회전, 발 회내를 판정하지 않는다.

## 공통성이 높은 기준

### A급: 직접 색상 판정 후보

1. 개인 선호 케이던스 대비 과도하게 낮은 케이던스
   - 같은 속도에서 케이던스를 5~10% 높이면 보폭, 제동 충격량, 무릎 에너지 흡수가 감소한 트레드밀 실험 결과가 있다.
   - 절대값 180 spm을 모든 사람에게 강요하지 않고 개인 기준 대비 변화율을 쓴다.
2. 과도한 overstride
   - 착지 발목과 골반/몸 중심의 수평거리를 다리길이로 나눈다.
   - 한 프레임이 아니라 최소 5보폭의 중앙값으로 판정한다.
3. 속도별 무릎 굴곡 변화량
   - 착지각 하나보다 착지에서 지지기 최대 굴곡까지의 변화량을 우선한다.
   - Reznick/Ferber의 동일 속도 참조범위와 영상 측정오차를 함께 쓴다.

### B급: 조건부 판정

1. 몸통 전경각
   - 전경은 무릎 하중을 줄일 수 있지만 고관절 하중을 늘리거나 충격 관련 지표를 증가시킬 수 있다.
   - 고정된 이상각을 두지 않고 과도한 후경/전경만 경고하며 목적별 피드백으로 표시한다.
2. 고관절 신전과 회복기 무릎 굴곡
   - 속도 의존성이 크므로 속도별 참조범위를 사용한다.
3. 좌우 비대칭
   - 영상 신뢰도가 충분하고 5보폭 이상에서 반복될 때만 표시한다.

### C급: 단독 Good/Bad 금지

1. 뒤꿈치/중족/앞꿈치 착지 유형
   - 앞꿈치 착지는 일부 무릎 하중을 줄여도 발목·종아리 쪽으로 하중을 이동시킬 수 있다.
   - 현재는 유형 정보만 표시하고 빨강/초록으로 평가하지 않는다.
2. 단일 관절각 한 번의 이탈
   - 다음 보폭에서 사라지면 오류 또는 자연 변동일 수 있으므로 회색 처리한다.
3. 부상 여부
   - 건강 참가자=GOOD, 부상 참가자=BAD로 직접 학습하지 않는다.

## 색상 판정식

측정값을 `x`, 영상 측정오차를 `e`, 속도별 참조구간을 `[L,U]`, 임상/실용적 여유값을 `m`이라 한다.

- GOOD: `[x-e, x+e]`가 `[L,U]` 안에 있고 최소 5보폭 중 4보폭 이상에서 재현
- BAD: `x+e < L-m` 또는 `x-e > U+m`이며 최소 5보폭 중 4보폭 이상에서 재현
- UNKNOWN: 나머지 모두

단계 점수는 판정 가능한 항목만 계산하고, 유효 항목 비율도 함께 보여준다. 단계 분류 확률은 자세 점수에 합치지 않는다.

## 우선 사용할 공개 데이터와 논문

1. Reznick et al. (2021), *Lower-limb kinematics and kinetics during continuously varying human locomotion*, Scientific Data. 10명, 계측 트레드밀, 달리기 1.8/2.0/2.2/2.4 m/s, 3D 관절각과 힘.
   - https://doi.org/10.1038/s41597-021-01057-9
2. Ferber et al. (2024), *A Biomechanical Dataset of 1,798 Healthy and Injured Subjects During Treadmill Walking and Running*, Scientific Data. 1.1~4.9 m/s, 20~60초, 건강/부상 메타데이터와 3D 마커.
   - https://doi.org/10.1038/s41597-024-04011-7
3. Fukuchi et al. (2017), *A public dataset of running biomechanics and the effects of running speed on lower extremity kinematics and kinetics*, PeerJ. 속도별 관절각/힘 기준 확장용.
   - https://doi.org/10.7717/peerj.3298
4. Heiderscheit et al. (2011), *Effects of step rate manipulation on joint mechanics during running*. 같은 속도에서 선호 케이던스 ±5/±10%를 실험한 원 연구.
   - https://doi.org/10.1249/MSS.0b013e3181ebedf4
5. dos Santos et al. (2019), *Patellofemoral joint stress measured across three different running techniques*. 앞꿈치 착지, 케이던스 +10%, 전방 몸통 기울기를 같은 참가자에게 비교.
   - https://doi.org/10.1016/j.gaitpost.2018.11.002
6. Willwacher et al. (2024), *Managing lower extremity loading in distance running by altering sagittal plane trunk leaning*. 몸통각 변화가 고관절과 무릎 하중을 서로 다르게 이동시킨다는 실험.
   - https://pubmed.ncbi.nlm.nih.gov/39251186/
7. Kipp et al. (2024), *Dataset of running kinematics, kinetics and muscle activation at different speeds, surface gradients, cadences and with forward trunk lean*. 19명, 13개 계측 트레드밀 조건.
   - https://doi.org/10.1016/j.dib.2024.110312
8. Peebles et al. (2021), *Validity of Using Automated Two-Dimensional Video Analysis to Measure Continuous Sagittal Plane Running Kinematics*. 2D 영상 오차와 사용 가능한 관절지표 검증용.
   - https://doi.org/10.1007/s10439-020-02569-y

## 구현할 키노그램

- 보폭 하나당 착지/제동/추진/회복 4장
- 왼발과 오른발을 분리하고 여러 보폭을 생성
- 각 관절 옆에 `측정값 / 속도별 기준 / 상태` 표시
- 무릎은 `외각 157° (굴곡 23°)`처럼 두 표기법을 함께 표시
- 전체 뼈대를 한 색으로 칠하지 않고 평가 대상 관절선만 초록/빨강/회색으로 표시
- 대표 보폭, 중앙값 보폭, 가장 큰 이탈 보폭을 별도로 제시
- 최종 화면에 자세 점수와 별도로 4단계 인식 신뢰도를 표시
