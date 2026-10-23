# Research Experiments Plan (v0.1) — 비교군 실험

- 작성일: 2026-10-20
- 목적: 제안 방법 개발 전에, **코드가 공개된 비교군의 본 데이터 성능(현재 수준)을 먼저 측정**한다. 결과는 논문 실험 표에 사용한다.
- 관련 문서: `docs/research_plan.md` (연구 전제·가정·한계), `docs/data_dictionary.md` (컬럼 설명)

---

## 1. 1차 비교군 (8개)

| 구분 | 방법 | 논문 | PDF / 원문 | 코드 |
|---|---|---|---|---|
| 통계 SSD | R-test | Cao & Rhinehart, J. Process Control 5(6), 1995 | https://doi.org/10.1016/0959-1524(95)00009-F (구독 필요) | 직접 구현 |
| 통계 SSD | 이동창 표준편차 + 기울기 임계값 | 특정 원 논문 없음 (단순 기준선) | – | 직접 구현 |
| 대조학습 | TS2Vec | Yue et al., AAAI 2022 | https://arxiv.org/pdf/2106.10466 | https://github.com/yuezhihan/ts2vec |
| 대조학습 | SoftCLT | Lee et al., ICLR 2024 | https://arxiv.org/pdf/2312.16424 | https://github.com/seunghan96/softclt |
| 대조학습 | TS-TCC | Eldele et al., IJCAI 2021 | https://arxiv.org/pdf/2106.14112 | https://github.com/emadeldeen24/TS-TCC |
| 대조학습 | CoST | Woo et al., ICLR 2022 | https://arxiv.org/pdf/2202.01575 | https://github.com/salesforce/CoST |
| 생성/복원 | SimMTM | Dong et al., NeurIPS 2023 | https://arxiv.org/pdf/2302.00861 | https://github.com/thuml/SimMTM |
| 파운데이션 모델 | MOMENT (small 또는 base) | Goswami et al., ICML 2024 | https://arxiv.org/pdf/2402.03885 | https://github.com/moment-timeseries-foundation-model/moment (+ Hugging Face 가중치) |

- 코드 저장소는 `git ls-remote`로 존재를 확인함 (내용·라이선스·실행 가능 여부는 미확인).
- 외부 코드는 실행 전 내용을 검토한다.

### 1.1 제외한 방법

| 방법 | 제외 사유 |
|---|---|
| TNC (Tonekaboni et al., ICLR 2021) | 음성 쌍 샘플러가 `delta = 5·epsilon·W` (epsilon ≤ 3) 로 이웃 밖 거리를 정하므로, 기록 길이가 이웃 범위보다 훨씬 길 것을 전제한다. 본 데이터의 run 길이 1,288 샘플(140 s)에서는 음성 쌍이 run 양 끝으로 몰린다 — 창 64 에서 23.6%(균등 기준선 10.5% 의 2.2배), 창 128 에서 82.5%, 창 192 에서 99.5%. run 맨 앞은 본 데이터에서 시동 과도구간이므로, `research_plan.md` 0절이 금지한 **위치 가정과 동등한 정보가 비교군에만 주입**된다. 측정 근거는 `worklog.md` 2026-10-23 §7 |

## 2. 2차 비교군 (1차 결과 확인 후 결정)

| 구분 | 방법 | 비고 |
|---|---|---|
| 대조학습 | TF-C (NeurIPS 2022), MHCCL (AAAI 2023), REBAR (ICLR 2024), CaTT (ECAI 2025 추정), Di-COT (ICML 2026) | 공식 코드 확인 |
| 대조학습 | TimesURL (AAAI 2024) | 공식 저장소 여부 확인 필요 |
| 프리텍스트 | T-Rep (ICLR 2024) | PyTorch 2.0 비호환 — Python 3.13 환경 설치 위험 |
| 비대조 | VICReg식 적용 (ICLR 2022) | 시계열용 공식 코드 없음, 직접 적용 필요 |
| 통계/변화점 | Kelly–Hedengren (2013), PELT (2012), BOCPD (2007) | PELT는 `ruptures` 필요 |
| 관련 연구로만 (코드 없음) | StatioCL (CIKM 2024), AdaTS (NeurIPS 2025) | 실험 제외 |

## 3. 공통 설정

| 항목 | 설정 |
|---|---|
| 입력 | 단일 채널 e = `Film Tension #1 (kg)` − `Film-Tension Setting Value (kg)` |
| 표현 → 시점별 판정 | 시점 단위 표현 방법(TS2Vec, SoftCLT)은 그대로; 나머지는 방법별로 변환 — MOMENT 는 patch(8 샘플) 표현을 해당 시점에 복제, CoST 는 마지막 시점만 내놓으므로 이동창(`sliding_length=1`, causal), TS-TCC·SimMTM 은 이동창 |
| 평가 방식 | **트랙 A만 사용 (라벨 없음)**: 표현 → 2-군집 GMM → 정의 규칙(e 평균 ≈ 0, 기울기 ≈ 0)으로 steady 군집 지정 → 판정 |
| 채점 | `Trigger 4` (0 = non-steady, 1 = steady); ambiguous(2)는 채점에서 제외 |
| 하이퍼파라미터 | 각 저장소 기본값 (라벨로 조정하지 않음) |
| 별도 검증 세트 | 두지 않음 (하이퍼파라미터 조정이 없으므로 학습/평가 2분할) |

## 4. 교차검증

조건당 run이 1개이므로 run 단위 분할만으로도 조건 조합은 달라지지만, **검증 조건의 설정값이 학습에 나오지 않도록 설정값 수준 단위로 분할**한다.

### 4.1 주 실험 (본문 표): Film Tension 수준별 5-fold

| Fold | 검증 (Film Tension) | 학습 (Film Tension) | 학습 / 검증 runs | 성격 |
|---|---|---|---|---|
| 1 | 3 kg | 4, 5, 6, 7 kg | 168 / 42 | 외삽 |
| 2 | 4 kg | 3, 5, 6, 7 kg | 168 / 42 | 내삽 |
| 3 | 5 kg | 3, 4, 6, 7 kg | 168 / 42 | 내삽 |
| 4 | 6 kg | 3, 4, 5, 7 kg | 168 / 42 | 내삽 |
| 5 | 7 kg | 3, 4, 5, 6 kg | 168 / 42 | 외삽 |

- fold별 난이도가 다르므로 **평균 ± 표준편차와 fold별 값을 함께 보고**한다.

### 4.2 보조 실험: 속도 수준별 7-fold

| Fold | 검증 (Speed, mm/s) | 학습 / 검증 runs |
|---|---|---|
| 1–7 | 50, 75, 100, 125, 150, 175, 200 중 하나 | 180 / 30 |

- 목적: 속도 일반화 확인, **속도 150 문제**(steady 편차 증가, 1.2절 참조) 확인.
- 1차 비교군 결과 확인 후 수행.

### 4.3 fold 내 학습·평가 절차 (트랙 A)

| 단계 | 학습 fold | 검증 fold |
|---|---|---|
| 인코더 자기지도 학습 | 학습 fold 신호만 사용 (라벨 없음) | – |
| 2-군집 GMM 학습 | 학습 fold 표현으로 학습 | – |
| steady 군집 지정 규칙 | 학습 fold에서 결정 | – |
| 판정 | – | 학습된 인코더·GMM을 그대로 적용 |
| 채점 | – | `Trigger 4`로 채점 (ambiguous 제외) |

- 통계 SSD: 학습 불필요, 원 논문·기본 임계값 사용, 동일 검증 fold에서 평가.
- MOMENT: 추가 학습 없음, GMM과 steady 군집 지정만 학습 fold에서 수행.
- seed 반복 횟수(예: 3회)는 실행 시간 측정 후 결정.

## 5. 지표
| 지표 | 정의 |
|---|---|
| 오염률 (최우선) | steady로 판정한 시점 중 실제 non-steady 비율 |
| 보존률 | 실제 steady 시점 중 steady로 판정한 비율 |
| steady 진입 시점 오차 (s) | run별 판정 진입 시점 − `Trigger 4` 첫 steady 시점 |
| AUROC / AUPRC | non-steady = positive (점수형 출력이 있는 경우: GMM 사후확률) |
| F1 | non-steady = positive |
| 조건별 heatmap | 검증 fold 내 (ReWinder × Speed) 조건별 성능 |

## 6. 결과물 구성 (예정)
- 코드: `code/03_baselines/` — 방법별 실행 코드(어댑터 포함), 공통 평가 코드
- 결과: `outputs/03_baselines/` — fold별·방법별 결과 표(CSV), 조건별 heatmap

## 7. 실행 전제와 위험
| 항목 | 내용 |
|---|---|
| 패키지 설치 | PyTorch(CPU), scikit-learn, matplotlib — `pypi.org`, `files.pythonhosted.org` 허용 필요 |
| MOMENT 가중치 | `huggingface.co` 허용 필요 |
| 저장소 복제 | `git clone`은 현재 가능 |
| 계산 자원 | CPU 4코어, GPU 없음 — 1차 비교군은 감당 가능하다고 추정(실행 시간 미측정) |
| 어댑터 작업 | 저장소마다 데이터 입력 형식이 달라 방법별 변환 코드 필요 |
