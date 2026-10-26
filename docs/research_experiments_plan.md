# Research Experiments Plan (v0.1) — 비교군 실험

- 작성일: 2026-10-20
- 목적: 제안 방법 개발 전에, **코드가 공개된 비교군의 본 데이터 성능(현재 수준)을 먼저 측정**한다. 결과는 논문 실험 표에 사용한다.
- 관련 문서: `docs/research_plan.md`(연구 전제·가정·한계), `docs/data_dictionary.md`(컬럼 설명), `docs/hyperparameters.md`(설정값), `docs/outputs_guide.md`(산출물 읽는 법), `docs/periodic_disturbance.md`(창 길이 유도), `docs/baseline_cases.md`(23종이 각각 무엇인가)

> **진행 상태 (2026-10-26): 1차 비교군 8개 전부 실행 완료** (변형 포함 23종).
> 결과는 `outputs/03_baselines/_paper_table.csv` 외 집계 표 4개. 각 행의 정체는 `baseline_cases.md`.
>
> **단, 이 문서의 실험(4절 장력 5-fold)은 "A 실험" 이고 최종 목표가 아니다.** 최종 목표는
> **저속 학습 -> 미학습 고속 테스트**(C 실험)이며 분할이 다르다 — `hyperparameters.md` §13.0 참조.
> 4.2 의 속도 7-fold 는 **leave-one-level-out 이라 내삽**이어서 그 목표를 달성하지 못한다.

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


#### LaST (Wang et al., NeurIPS 2022) — 검토 후 제외 (2026-10-26)

- 논문: *Learning Latent Seasonal-Trend Representations for Time Series Forecasting*
- 코드: https://github.com/zhycs/LaST (Apache-2.0, 커밋 `5f60862`) — **공개돼 있고 본 환경에서
  수정 없이 동작한다** (torch 2.11+cu128 에서 import·forward 확인, 파라미터 5.7만개, 10 step 0.2초)
- **왜 검토했나**: 현재 1위인 CoST 와 같은 **추세/계절 분리**를 다루는데, CoST 는 대조학습이고
  LaST 는 변분추론(VAE)이다. 그리고 **LaST 에는 증강이 전혀 없다** — §3.1 (3-1) ④ 가 기록한
  "CoST 의 scaling/shift 증강이 진폭 정보를 지운다" 와 분리해 볼 수 있는 비교군이었다.

**제외 사유 — 논문이 제안한 '잠재 표현' 을 트랙 A 후단에 쓸 수 없다**

`VarUnit`(`models/LaST.py:44`)이 **`nn.Linear(1, 128)` 한 층**이다. 가중치 shape 가 `(128, 1)` 이다.

| 실측 검증 | 결과 |
|---|---|
| 문맥 독립성 — x 순서를 뒤집어도 같은 값의 표현이 동일한가 | **True** → `qz[t]` 가 `x[t]` **하나에만** 의존. 시간 문맥 없음 |
| 평균(loc) 표현의 rank (float64) | 유의 특이값 **2개** — `z = x·Wᵀ + b` 이므로 행공간이 `{W, b}` 2차원 아핀 |
| 재매개화(`qz = mean + var·eps`) 포함 rank | 128 — **차원을 채우는 것은 노이즈** |
| 시간 모델링 위치 | `FourierNet(qz_s)`: (B,144,128) → (B,**288**,128). **잠재 이후**의 예측 헤드 |

즉 `cat([qz_s, qz_t])` 256차원의 **신호 성분은 `e[t]` 스칼라의 아핀 사상**이다.
트랙 A 후단(표현 → 2군집 GMM)에 넣으면 **GMM 이 보는 정보가 `e[t]` 1차원**이 되어
MovStd-Slope 의 2차원(std, |기울기|)보다도 적다. 결과가 나쁠 것이 **구조적으로 확실**하다.

**대안을 쓰면 제안 방식이 아니게 된다**: 시간 모델링을 통과한 분기 출력
(`FourierNet(qz_s)`, `t_pred_1(qz_t)`)을 떼어 쓰면 CoST 의 `cat([trend, season])` 과 구조가
대응하지만, 논문이 "표현" 이라 부르는 것이 아니고 추출 지점(입력 구간 끝 / 첫 예측 step)을
우리가 정해야 한다. **As-published 가 아니게 되므로 제외한다.**

**기록해 둘 것**: LaST 가 나쁜 방법이라는 뜻이 아니다. 예측 과제에서는 그 구조가 작동하며,
`qz` 를 "계절적/추세적" 으로 만드는 것은 보조 목적함수(`period_sim` 자기상관 / `trend_sim` 형상)다.
**잠재를 떼어내 다른 과제에 쓰는 용도로 설계되지 않았다.** 이는 `research_plan.md` 3.1 (3-1) ②
("표현학습 7종 전부 상태 구분을 목적함수에 두지 않는다")의 추가 사례다.

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
- **실행하지 않는다** (연구자 결정 2026-10-26). leave-one-level-out 은 **내삽**이라
  최종 목표(저속 학습 -> 고속 외삽)를 달성하지 못한다. 속도 100 을 빼면 학습에 125~200 이
  들어가 "고속을 이미 학습한 상태" 가 된다. 대신 4.4 를 수행한다.

### 4.4 C 실험: 저속 학습 -> 고속 테스트 (3분할)

실제 적용이 **"새로운 고속 신호가 들어왔을 때"** 이므로 속도 축의 **한 방향 외삽**이다.
저속 **50/75/100/125** (120 run) 으로 학습하고 고속 **150/175/200** (90 run) 은 최종 test 에만 쓴다.

| fold | 학습 | 검증 | 미사용 | 묻는 것 |
|---|---|---|---|---|
| `cv_T1`~`cv_T5` | 96 | 24 | 90 | 저속 안에서 **새 장력** 에 되는가 |
| `cv_S1` | 60 (50,75) | 30 (100) | 120 | **1수준 속도 외삽** |
| `cv_S2` | 90 (50,75,100) | 30 (125) | 90 | **1수준 속도 외삽** |
| `final_test` | 120 | 90 | 0 | **최종 목표.** 한 번만 쓴다 |

방법당 8 fold x 5 seed = 40회. 배정은 `outputs/04_speed_extrap/folds_c.csv`,
결과는 `outputs/04_speed_extrap/` 아래. 설계 근거와 쟁점은 `hyperparameters.md` §13.0.

**A 실험(4.1)은 그대로 둔다** - 답하는 질문이 다르다. A 는 "속도 전 범위를 학습한 상태에서
새 장력 설정값에 일반화되는가" 이고 속도 외삽을 주장하지 않는다. **보고할 때 혼용하지 않는다.**

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

## 6. 결과물 구성 (2026-10-26 실제)

**코드** `code/03_baselines/`

| 파일 | 내용 |
|---|---|
| `00_folds.py` | fold 배정 (1회 생성 후 고정) |
| `evaluation.py` | 공통 평가 — 트랙 A 후단·채점·그림. **라벨을 읽는 유일한 지점** |
| `01_stat_ssd.py` | R-test 4종(2x2) + 이동창 std/기울기 4종 |
| `02_ts2vec.py` `03_moment.py` `04_cost.py` `05_softclt.py` `07_tstcc.py` `08_simmtm.py` | 외부 저장소 어댑터 6개 |
| `06_cost_ablation.py` | CoST 성분 분해 2종 + 증강 제거 1종 |
| `09_summary_tables.py` | 집계 표 4개 생성 |
| `external_versions.csv` | 외부 저장소 커밋 hash·환경·shim 기록 |

**결과** `outputs/03_baselines/`

| 파일 | 내용 |
|---|---|
| `_paper_table.csv` | **논문 표에 바로 붙이는 형태** (평균±표준편차 + fold별 값) |
| `_all_methods_full_table.csv` | 전 지표 (`__mean`/`__std`/`__foldstd`/`__seedstd`) + 설정값 |
| `_all_methods_by_fold.csv` | 방법 x fold — 4.1 이 요구한 fold별 값 |
| `_all_methods_by_fold_seed.csv` | 방법 x fold x seed (원자료) |
| `01_stat_ssd/` ~ `08_simmtm/` | 방법별 run/fold/summary/config CSV + 조건별 heatmap + 손실 곡선 |

읽는 법은 `outputs_guide.md` §16·§17.

## 7. 실행 전제와 위험
| 항목 | 내용 |
|---|---|
| 패키지 설치 | PyTorch(CPU), scikit-learn, matplotlib — `pypi.org`, `files.pythonhosted.org` 허용 필요 |
| MOMENT 가중치 | `huggingface.co` 허용 필요 |
| 저장소 복제 | `git clone`은 현재 가능 |
| 계산 자원 | CPU 4코어, GPU 없음 — 1차 비교군은 감당 가능하다고 추정(실행 시간 미측정) |
| 어댑터 작업 | 저장소마다 데이터 입력 형식이 달라 방법별 변환 코드 필요 |

---

## 8. Ablation 실험 — 왜 하는가

1차 비교군이 "어느 방법이 더 나은가" 를 측정한다면, ablation 은 **"왜 그런가"** 를 묻는다.
세 가지를 보는데, 각각 `research_plan.md` 3.1 의 차별점 가설(D1–D6)과 직접 연결된다.
제안 방법 설계의 근거가 되려면 "기존 설계가 실제로 깨짐을 실험으로 입증" 해야 하기 때문이다
(`research_plan.md` 3.1 (4)).

**창 길이는 모든 ablation 에서 144 로 고정한다** (규칙 1b, `docs/periodic_disturbance.md` §7).
창 길이 민감도는 CoST 5종·MovStd-Slope 2쌍에 이미 있으므로 ablation 에서 다시 탐색하지 않는다.

### 8.1 CoST 표현 성분 — 추세만 / 계절만 / 결합

| 항목 | 내용 |
|---|---|
| **무엇을 바꾸나** | 표현을 슬라이스만 달리한다. CoST 표현은 `cat([trend, season])` 로 앞 160차원이 추세, 뒤 160차원이 계절이다(`cost.py:334`) |
| **왜 하나** | **어떤 종류의 정보가 실제로 상태 구분에 기여하는지** 알기 위해서다. CoST 가 현재 1위인데, 그 성능이 추세(저주파 수렴 거동)에서 오는지 계절(회전 외란 등 주기 성분)에서 오는지 결합 효과인지 구분되지 않는다 |
| **무엇을 알 수 있나** | 추세만으로 충분하면 "steady 판별은 저주파 수렴 거동 문제" 이고, 계절이 기여하면 "주기 성분에도 상태 정보가 있다" 는 뜻이다. 후자면 **주기 성분을 제거(residual)하는 설계(D3)가 정보를 버리는 것**이 되므로 설계 방향이 바뀐다 |
| **연결** | `research_plan.md` 3.1 **D3**(설정값 보정 정상성: 주기 성분 제거·모델링 후 잔차로 판단), 4.1 (d)(신호 분해 구조) |
| **비용** | **재학습 없음.** 체크포인트로 한 번 encode 하고 슬라이스만 달리해 같은 트랙 A 후단에 넣는다 |

### 8.2 CoST scaling/offset 증강 제거

| 항목 | 내용 |
|---|---|
| **무엇을 바꾸나** | CoST 증강 `jitter(shift(scale(x)))` 에서 **scale(곱셈)·shift(덧셈)를 빼고 jitter 만** 남긴 뒤 재학습한다. 원본은 수정하지 않고 런타임 shim 으로 교체한다 |
| **왜 하나** | **실험별 수준·크기 차이에 대한 학습이 일반화에 기여했는지** 검토하기 위해서다. `scale`·`shift` 는 "신호의 크기와 오프셋이 달라도 같은 상태" 라고 가르친다. 그런데 본 과제의 구분 정보는 **e 의 진폭 자체**에 있다 — 미안정 0.209 kg vs steady 0.041 kg |
| **문제의 크기** | sigma = 0.5 다. steady 구간 e 표준편차 0.041 kg 의 **12배** 크기 교란이고, `scale` 은 약 2.3% 확률로 **부호를 뒤집는다.** 즉 구분 정보보다 훨씬 큰 잡음을 덮어씌우고 있다 |
| **무엇을 알 수 있나** | 증강을 뺐을 때 **좋아지면** → 그 증강이 구분 정보를 지우고 있었다는 직접 증거(**D2 확인**). **나빠지면** → 수준 차이 불변성이 미학습 조건 일반화에 실제로 기여했다는 뜻이고, D2 를 재검토해야 한다 |
| **연결** | `research_plan.md` 3.1 **D2**(도메인 불변성 정의 증강: 설정값 수준에는 불변, 오차 진폭은 보존) |
| **비용** | 재학습 필요. 창 144 에서 25회 |

### 8.3 TS-TCC permutation 제거 (예정)

| 항목 | 내용 |
|---|---|
| **무엇을 바꾸나** | TS-TCC 의 strong augmentation `jitter(permutation(x))` 에서 **permutation 만** 빼고 재학습한다. weak augmentation(`scaling`)은 그대로 둔다 |
| **permutation 이 하는 일** | 시계열을 최대 `max_seg`(기본 8) 개 구간으로 임의 분할한 뒤 **구간 순서를 무작위로 섞는다**(`augmentations.py:permutation`). 즉 "구간 순서가 바뀌어도 같은 상태" 라고 가르친다 |
| **왜 하나** | **입력 구간의 수렴 순서를 보존하는 것이 중요한지** 검토하기 위해서다. 본 데이터는 **과도 → 정상으로 한 방향 수렴**하는 구조이고(`research_plan.md` 3.1 (2)), steady 가 86% 다. 구간을 섞으면 그 수렴 순서가 파괴되는데, 이는 상태를 정의하는 구조 자체를 지우는 것일 수 있다 |
| **무엇을 알 수 있나** | permutation 을 뺐을 때 **좋아지면** → 순서 불변성 가정이 본 데이터에서 해롭다는 증거이고, 시간 순서를 보존하는 증강 설계가 필요하다는 근거가 된다. **차이가 없으면** → 상태 판별이 구간 순서와 무관한 국소 통계(진폭·분산)로 결정된다는 뜻이다 |
| **연결** | `research_plan.md` 3.1 **D2**(증강 설계), **(2) 상태 구조**(한 방향 수렴) |
| **선행 조건** | ~~TS-TCC 어댑터~~ -> **해소 (2026-10-26).** 어댑터·baseline 실행 완료 (`code/03_baselines/07_tstcc.py`, AUPRC 0.290) |
| **비용** | 재학습 필요. baseline 과 같은 25회 (약 1.5시간) |
| **착수 전 확인 사항** | Epilepsy config 에서는 **두 view 차이가 거의 전부 permutation 에서 온다** (weak = 원본 x 2.0 거의 고정, jitter sigma=0.001 은 무시 가능 — `tstcc_config_choice.md` §4). 그래서 permutation 을 **완전히 제거하면 두 view 가 "2배 vs 원본" 이 되어 §8.4 가 경고한 자명화에 해당**한다. **A6 을 돌리기 전에 손실 곡선으로 자명화 여부를 먼저 확인할 것** |

### 8.4 공통 설계 원칙

- **한 번에 하나만 바꾼다.** 나머지 설정(창 144, 반복수 600, 시드 5회, 트랙 A 후단)은 baseline 과 완전히 동일하게 둔다.
- **원본 저장소는 수정하지 않는다.** 증강 변경은 런타임 shim 으로 하고 `external_versions.csv` 에 기록한다.
- **증강을 전부 제거하지는 않는다.** CoST·TS-TCC 모두 두 view 가 증강으로만 달라지므로, 증강을 전부 끄면 양성 쌍이 같아져 대조 과제가 자명해진다. 그래서 CoST 는 jitter 를, TS-TCC 는 weak augmentation 과 jitter 를 남긴다.
- **baseline 과 같은 창·같은 후단으로 비교한다.** ablation 결과는 baseline 결과와 나란히 그림으로 저장한다.
