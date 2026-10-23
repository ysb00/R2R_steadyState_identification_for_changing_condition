# 산출물 설명서 — CSV 컬럼과 그림 읽는 법

- 작성일: 2026-10-23
- 목적: `outputs/03_baselines/` 의 모든 CSV 컬럼과 그림의 축·범례·colorbar 의미, 각 그림이 무엇을 보기 위한 것인지, 수치를 어떻게 해석하는지를 적어둔다.
- 관련: `docs/hyperparameters.md`(설정값), `docs/data_dictionary.md`(원본 데이터 컬럼), `docs/worklog.md`(결정 이력)

## 0. 먼저 알아야 할 용어 3개

| 용어 | 뜻 |
|---|---|
| **run** | 운전 조건 1개의 측정 1회. 조건당 run 1개뿐이라 run = 조건이다. 총 210개, 약 1,288 샘플(140 s) |
| **fold** | 교차검증 분할. 주 실험은 Film Tension 수준별 5-fold (fold 1 = 장력 3 kg 검증, fold 5 = 7 kg 검증). 검증 42 run / 학습 168 run |
| **seed** | 난수 시드 반복. 0~4 의 5회. 학습이 없는 방법(R-test)은 결과가 같지만 기록 형식을 맞추려 반복한다 |

**steady 가 86%, non-steady 가 8%, ambiguous 가 6% 다.** 모든 지표를 읽을 때 이 비율이 기준선이다.
ambiguous(`Trigger 4 == 2`)는 지표 계산에서 제외하므로, 채점 대상 중 non-steady 는 **8.13%** 다.
즉 아무 정보 없이 "전부 steady" 라고 하면 `precision_steady = 0.919`, "전부 non-steady" 라고 하면
`precision_nonsteady = 0.081` 이 나온다. **이 값과 비교해야 의미가 있다.**

## 1. 파일 이름 규칙

```
outputs/03_baselines/<단계>/<방법이름>_<종류>.<확장자>
```

| 단계 폴더 | 방법 |
|---|---|
| `01_stat_ssd/` | `R-test`, `R-test-decimated`, `MovStd-Slope-fixed`, `MovStd-Slope-speed` |
| `02_ts2vec/` | `TS2Vec` |
| `03_moment/` | `MOMENT` |
| `04_cost/` | `CoST-W160`, `CoST-W256`, `CoST-W306`, `CoST-W512` (W = 창 길이 samples) |
| `05_softclt/` | `SoftCLT-fs`(도메인 유도 tau), `SoftCLT-ref`(저자 표 중앙값 tau) |

| 종류 | 내용 | 행의 단위 |
|---|---|---|
| `_run_metrics.csv` | run별 지표 | 1행 = run 1개 × seed 1개 (210 × 5 = 1,050행) |
| `_fold_metrics.csv` | fold별 지표 | 1행 = fold 1개 × seed 1개 (5 × 5 = 25행) |
| `_summary.csv` | fold별 평균±표준편차 요약 | 1행 = fold (+ `all_folds` 행) |
| `_config.csv` | 실행에 사용한 설정값 | 1행 = 설정값 1개 |
| `_progress_run.csv`, `_progress_fold.csv` | 실행 중 중간 저장본 (내용은 위와 동일) | — |
| `_loss.csv` | 자기지도 학습 손실 | 1행 = iteration 1개 (학습이 있는 방법만) |
| `_heatmap_*.png/.pdf` | 조건별 성능 heatmap | — |
| `_loss.png/.pdf` | 학습 손실 곡선 | — |

`_progress_*` 는 장시간 실행이 끊겨도 결과가 남게 하는 중간 저장본이다. 내용이 최종본과 같으므로
**분석에는 `_run_metrics.csv` / `_fold_metrics.csv` 를 쓴다.**

---

## 2. `_run_metrics.csv` — run별 지표

### 2.1 식별 컬럼

| 컬럼 | 의미 |
|---|---|
| `method` | 방법 이름 (파일 이름과 동일) |
| `fold` | 이 run 이 속한 검증 fold 번호 (1~5) |
| `seed` | 난수 시드 (0~4) |
| `run_id` | run 식별자. `T<장력>_R<리와인더장력>_V<속도>` 예: `T5_R4_V100` |
| `Film-Tension Setting Value (kg)` | 필름 장력 설정값 3~7. **fold 를 결정하는 축** |
| `ReWinder-Tension Setting Value (kg)` | 리와인더 장력 설정값 2~7. heatmap 의 **세로축** |
| `Web Speed setting value (mm/sec)` | 웹 속도 설정값 50~200. heatmap 의 **가로축** |

### 2.2 개수 컬럼

| 컬럼 | 의미 | 읽는 법 |
|---|---|---|
| `n` | 이 run 의 전체 시점 수 | 1,275~1,300 |
| `n_scored` | 채점 대상 시점 수 (ambiguous 제외) | `n` 보다 작다. 차이가 ambiguous 개수 |
| `n_nonsteady` | 채점 대상 중 실제 non-steady 시점 수 | 0 인 run 이 26개 있다 (과도구간이 없는 run) → 그 run 은 non-steady 관련 지표가 빈칸 |

### 2.3 판정 품질 컬럼

**positive 클래스가 지표마다 다르다.** 이름 뒤에 어느 클래스인지 붙어 있다.

| 컬럼 | 정의 | 높을수록 좋은가 | 의미 |
|---|---|---|---|
| `contamination_rate` | steady 로 판정한 시점 중 실제 non-steady 비율 | **낮을수록** | **오염률.** 계획 문서의 최우선 지표. "선별한 steady 데이터에 불량이 얼마나 섞였나" |
| `retention_rate` | 실제 steady 시점 중 steady 로 판정한 비율 | 높을수록 | **보존률.** "쓸 수 있는 steady 데이터를 얼마나 살렸나" |
| `precision_steady` | = 1 − `contamination_rate` | 높을수록 | 같은 양을 precision 이름으로 적은 것 (중복) |
| `recall_steady` | = `retention_rate` | 높을수록 | 같은 양을 recall 이름으로 적은 것 (중복) |
| `precision_nonsteady` | non-steady 로 판정한 시점 중 실제 non-steady 비율 | 높을수록 | **기준선 0.081 과 비교해야 한다.** 0.081 이면 정보가 전혀 없다는 뜻 |
| `recall_nonsteady` | 실제 non-steady 시점 중 non-steady 로 판정한 비율 | 높을수록 | 1.0 에 가까우면서 precision 이 낮으면 "전부 non-steady 라고 외치는" 상태 |
| `f1_nonsteady` | non-steady 를 positive 로 본 F1 | 높을수록 | 위 두 값의 조화평균 |
| `auroc` | non-steady = positive 인 AUROC | 높을수록 | **점수의 순위 품질.** 판정(임계값)과 무관하다 |
| `auprc` | non-steady = positive 인 AUPRC | 높을수록 | 소수 클래스(8%)에서 AUROC 보다 변별력이 높다 |

> **중요**: `auroc` 가 높은데 `precision_nonsteady` 가 낮으면 **"순위는 잘 매기는데 임계값이 틀렸다"** 는 뜻이다.
> R-test 다운샘플판이 그 예다 (AUROC 0.901, `precision_nonsteady` 0.092).

점수는 GMM 의 **non-steady 사후확률**(= 1 − steady 군집 사후확률)이고, 판정은 그 값이 0.5 이상인지로 한다.

### 2.4 steady 진입 시점 컬럼

| 컬럼 | 의미 | 단위 |
|---|---|---|
| `entry_pred_k` | 방법이 판정한 steady 진입 시점의 샘플 번호. 연속 10개가 steady 로 판정된 첫 시점. 끝까지 없으면 **−1** | 샘플 index |
| `entry_true_k` | `Trigger 4` 가 처음 1 이 되는 샘플 번호 | 샘플 index |
| `entry_err_s` | `(entry_pred_k − entry_true_k) × 0.109` | **초** |

`entry_err_s` 부호의 뜻:
- **음수 = 전문가보다 빨리** steady 선언 (성급함)
- **양수 = 전문가보다 늦게** steady 선언 (보수적)
- `entry_pred_k` 가 −1 이면 `entry_err_s` 는 빈칸이다

---

## 3. `_fold_metrics.csv` — fold별 지표

1행 = fold 1개 × seed 1개. **같은 지표를 두 방식으로 집계한다.**

| 접두사 | 집계 방식 | 왜 둘 다 보는가 |
|---|---|---|
| `pooled_` | fold 안의 모든 시점을 한 덩어리로 모아 한 번 계산 | 시점이 많은 run 이 더 큰 영향을 준다. 전체 데이터 관점 |
| `runmean_` | run 별로 계산한 뒤 산술평균 | 모든 run 이 동등하게 기여한다. 조건별 관점 |

**두 값이 크게 갈리는 것 자체가 정보다.** 예: R-test 다운샘플판의 `precision_steady` 는
pooled 0.979 인데 runmean 0.756 이다. steady 로 판정한 시점이 run 당 몇 개뿐이라
run 단위 분모가 극히 작아 생기는 차이다.

| 컬럼 묶음 | 의미 |
|---|---|
| `pooled_<지표>` | 위 방식으로 계산한 9개 지표 |
| `runmean_<지표>` | run 평균 |
| `runstd_<지표>` | run 간 표준편차 (조건에 따라 성능이 얼마나 흔들리나) |
| `n_runs_<지표>` | 그 지표가 정의된 run 수. 42 보다 작으면 일부 run 에서 계산 불가 (예: non-steady 가 없는 run) |
| `runmean_entry_err_s` | 진입 오차의 **부호 포함 평균** = bias. 방법이 성급한지 보수적인지 |
| `runmae_entry_err_s` | 진입 오차의 **절댓값 평균** = MAE. 방향 무관한 정확도 |
| `runstd_entry_err_s` | 진입 오차의 표준편차 |
| `n_runs_no_entry` | **steady 를 끝까지 판정하지 못한 run 수.** 0 이 아니면 그 방법은 해당 run 에서 쓸 수 없다 |

### 3.1 학습·실행 기록 컬럼 (방법에 따라 있고 없음)

| 컬럼 | 의미 | 주의 |
|---|---|---|
| `window` | 창 길이 (CoST) | |
| `tau_inst`, `tau_temp`, `n_eff_median/min/max` | SoftCLT soft label 설정과 앵커당 유효 양성 수 | `n_eff` 가 학습 run 수(167)에 가까우면 음성이 사라진 퇴화 상태 |
| `fit_sec`, `total_sec` | 학습 시간 / 전체 시간 (초) | 다른 작업과 동시에 돌렸으면 늘어난다 |
| `n_epochs`, `n_iters` | 완료 epoch 수 / 총 iteration 수 | |
| `iters_per_epoch` | 1 epoch 의 iteration 수 | **CoST 기존 결과의 이 값은 틀렸다** (`n_iters // n_epochs` 로 계산해 과대평가). 참값은 168/105/84/42 |
| `data_passes`, `samples_seen` | 학습 데이터 통과 횟수 / 본 시점 수 | 코드 수정 이후 실행분에만 있다. **창·방법 간 학습량 비교에 쓴다** |
| `epoch_mean_loss_last` | 마지막 완료 epoch 의 평균 손실 | |
| `final_loss` | 마지막 20 iteration 의 평균 손실 | `epoch_mean_loss_last` 보다 낮게 나온다 (epoch 평균은 초반 큰 손실을 포함) |
| `steady_cluster` | GMM 2군집 중 steady 로 지정된 번호 (0 또는 1) | **seed 마다 0/1 이 뒤바뀌는 것은 정상이다** (군집 번호는 임의) |
| `e_mean_c0`, `e_mean_c1` | 군집 0·1 의 e 평균 (kg) | 절댓값이 작은 쪽이 steady 로 지정된다. 두 값이 비슷하면 군집이 steady 와 무관하게 갈렸다는 신호 |

> `n_iters` 가 비어 있으면(`NaN`) 저장소 내부 규칙에 맡겼다는 뜻이고, 본 데이터에서는 600 이 된다.

---

## 4. `_summary.csv` — fold별 요약

2단 헤더다. `(지표이름, mean)` / `(지표이름, std)` 쌍으로 되어 있고, 행은 fold 1~5 와 `all_folds` 다.

| 행 | 의미 |
|---|---|
| `1`~`5` | 해당 fold 의 **seed 5회에 대한** 평균과 표준편차 → `std` 는 **seed 민감도** |
| `all_folds` | 25개 행(fold×seed) 전체의 평균과 표준편차 → `std` 는 **fold 차이 + seed 차이가 섞인 값** |

`std` 를 읽을 때 행에 따라 뜻이 달라지는 점을 주의한다. fold 간 차이만 보려면
fold별 `mean` 5개의 표준편차를 따로 계산해야 한다.

---

## 5. `_config.csv` — 실행 설정값

`name`, `value` 두 컬럼이다. 공통 평가 설정(`DT`, `ENTRY_MIN_CONSEC`, `N_GMM_COMPONENTS`,
`GMM_COVARIANCE_TYPE`)에 방법별 설정이 덧붙고, 마지막에 `repo` 와 `commit`(외부 저장소 커밋 해시)이 들어간다.
각 값의 출처와 역할은 `docs/hyperparameters.md` 에 정리돼 있다.

---

## 6. `_loss.csv` + `_loss.png/.pdf` — 학습 손실 곡선

학습이 있는 방법(TS2Vec, CoST, SoftCLT)만 있다. MOMENT·R-test·MovStd-Slope 는 학습이 없어 없다.

| 컬럼 | 의미 |
|---|---|
| `fold`, `seed` | 어느 실행인지 |
| `iter` | iteration 번호 (1~600) |
| `loss` | 그 iteration 의 손실 |

**TS2Vec 만 `iter` 대신 `epoch` 컬럼이다** (재실행 보류로 코드·산출물 일치를 유지했다).
환산은 `iter = epoch × 21` 이다.

### 그림 읽는 법

| 요소 | 내용 |
|---|---|
| subplot | fold 하나 (제목 `Fold 1` ~ `Fold 5`) |
| **가로축 `Iteration`** | 학습 iteration. epoch 이 아니다 — epoch 당 iteration 수가 방법·창마다 달라(TS2Vec·SoftCLT 21, CoST 42~168) **epoch 축으로는 비교할 수 없다** |
| **세로축** | 손실. 이름은 방법마다 다르다 — `Hierarchical contrastive loss`(TS2Vec), `CoST contrastive loss`, `Soft hierarchical contrastive loss`(SoftCLT) |
| 옅은 선 5개 | seed 0~4 각각 |
| 진한 선 1개 | seed 평균 |
| 축 범위 | 모든 subplot 이 공유 (`sharex`, `sharey`) → fold 간 직접 비교 가능 |

### 무엇을 보기 위한 그림인가

1. **학습이 실제로 진행됐는가.** 평평하면 사실상 학습 전 상태다. TS2Vec 은 25회 중 11회가 손실이 0.3 도 떨어지지 않았다.
2. **seed 간 흔들림.** 옅은 선 5개가 벌어지면 시드에 따라 다른 해로 간다는 뜻이고, 지표의 seed 표준편차와 함께 읽는다.
3. **수렴 여부 — 단, 이 그림만으로 판정할 수 없다.** CoST 는 cosine LR 감쇠라서 학습 끝에서 lr → 0 이 되어 **어떤 반복수에서도 곡선 끝이 평탄해진다.** 평탄함은 수렴의 증거가 아니다. 수렴을 보려면 반복수를 2배로 늘려 최종 손실이 더 내려가는지 봐야 한다 (`worklog.md` 2026-10-23 §2).

### 손실값 비교의 한계

- **방법 간 손실값은 비교할 수 없다.** 목적함수가 다르다 (TS2Vec 약 3, CoST 약 3.3, SoftCLT-fs 약 82, SoftCLT-ref 약 13).
- **같은 방법 안에서도 설정이 바뀌면 비교할 수 없다.** SoftCLT 는 tau 가 바뀌면 soft label 가중치의 총합이 바뀌어 손실 스케일 자체가 달라진다.

---

## 7. `_heatmap_contamination_rate` / `_heatmap_retention_rate` — 조건별 성능

### 그림 구조

| 요소 | 내용 |
|---|---|
| subplot | 검증 fold 하나. 제목 `Fold 1: film tension 3 kg` = **이 fold 에서 검증(미학습)한 장력 조건** |
| **가로축 `Web speed setting (mm/s)`** | 웹 속도 설정값 7수준 (50, 75, 100, 125, 150, 175, 200) |
| **세로축 `ReWinder tension setting (kg)`** | 리와인더 장력 설정값 6수준 (2~7) |
| 셀 1칸 | run 1개 (= 조건 1개). seed 5회를 평균한 값 |
| 셀 안 숫자 | 그 값. 소수점 앞 0 을 떼고 표기 (`.03` = 0.03) |
| **colorbar** | 그림 전체에 하나. 이름은 `Contamination rate` 또는 `Retention rate` |
| 색 스케일 | **단일 색상(파랑) 밝음 → 어두움.** `vmin`/`vmax` 는 그림 안 전체 fold 의 최소·최대로 맞춘다 → **subplot 끼리 색을 직접 비교할 수 있다** |
| 셀 글자색 | 배경이 어두우면 흰색, 밝으면 검정 (가독성) |

> **색의 방향에 주의한다.** colorbar 는 "값이 크면 진하다" 일 뿐이고, 좋고 나쁨과는 무관하다.
> - 오염률 heatmap: **진한 칸 = 나쁘다**
> - 보존률 heatmap: **진한 칸 = 좋다**

### 무엇을 보기 위한 그림인가

**조건에 따라 성능이 구조적으로 갈리는 지점을 찾는 것**이다. 숫자 하나(평균)로는 안 보이는 것을 본다.

| 보는 것 | 읽는 법 |
|---|---|
| 속도 의존성 | 가로 방향 패턴. 특정 속도 열이 전부 나쁘면 그 속도에서 방법이 깨진다 |
| **속도 150 문제** | `research_plan.md` 1.2절에서 steady 편차가 속도 150 에서 체계적으로 증가한다고 확인했다. V=150 열이 유독 나쁜지 확인한다 |
| 리와인더 장력 의존성 | 세로 방향 패턴 |
| 외삽 난이도 | Fold 1(3 kg)과 Fold 5(7 kg)는 격자 끝이라 **외삽**이다. 이 두 subplot 이 내삽 fold(2·3·4)보다 나쁜 것이 정상이며, 그래야 실험 설계가 의도대로 작동한다는 신호다 |
| 빈 칸 | 그 조건의 run 에서 지표가 정의되지 않았다는 뜻 (예: steady 로 판정한 시점이 0개라 오염률 계산 불가) |

### 실제로 읽힌 예

- MovStd-Slope: 오염률이 **저속(50–100 mm/s)에 몰린다.** 특히 장력 5–7 kg fold 의 V=50 열이 0.12–0.16 으로 뚜렷하다.
- CoST: Fold 1(장력 3 kg, 외삽)만 뚜렷하게 어렵다 — AUPRC 0.494 vs 내삽 fold 0.80–0.93.

---

## 8. 그림에서 쓰는 방법별 색·marker

여러 방법을 한 그림에 겹칠 때 쓰는 고정 식별자다 (현재는 비교 그림을 아직 만들지 않았다).
방법마다 **고유 색 + 고유 marker + 선 종류**를 갖는다. 색은 OKLab 거리와 색각이상 시뮬레이션으로 계산해 골랐고,
전체 66쌍에서 색각이상 ΔE 7.6 / 정상시각 15.6 이다. 목록은 `docs/hyperparameters.md` §9 에 있다.

변형이 2개인 방법은 **같은 색상의 명암 쌍**으로 묶었다 (R-test 하늘색/파랑, MovStd-Slope 초록/진초록,
TS2Vec 주황 / SoftCLT 갈색). 혼동이 생기더라도 같은 계열 안에서만 생기게 한 것이다.

---

## 9. 방법 이름이 뜻하는 것

| 이름 | 뜻 |
|---|---|
| `R-test` | 원 샘플링(9 Hz) + 원 논문 임계값. **원 논문 전제(잡음 무상관)가 깨진 조건** |
| `R-test-decimated` | 1/10 다운샘플링 후 적용. 원 논문 전제를 맞춘 버전 |
| `MovStd-Slope-fixed` | 이동창 표준편차+기울기, 창 길이 고정(64 samples ≈ 7.0 s) |
| `MovStd-Slope-speed` | 같은 방법, 창 길이를 속도에 반비례하게 조정 (`W(V) = 64 × 125 / V`) |
| `TS2Vec` | 저장소 기본 설정 |
| `MOMENT` | 사전학습 가중치 그대로(zero-shot), 추가 학습 없음 |
| `CoST-W<숫자>` | 숫자 = 이동창 길이(samples). 160 ≈ 17.4 s, 256 ≈ 27.9 s, 306 ≈ 33.4 s, 512 ≈ 55.8 s |
| `SoftCLT-fs` | tau 를 **본 데이터에서 유도** (`tau_inst` 는 앵커당 유효 양성 수 5 를 맞추는 값, `tau_temp = 0.034`) |
| `SoftCLT-ref` | tau 를 **저자 표 중앙값** 그대로 (`tau_inst = 5`, `tau_temp = 1.0`) |

---

## 10. 결과를 읽을 때의 주의 4가지

1. **"거의 전부 steady" 라는 자명해가 꽤 좋은 숫자를 낸다.** steady 가 86% 이므로 오염률 3–5% / 보존률 99.9% 가 나온다(공통 평가 모듈 smoke test 로 확인). 오염률·보존률만 보고 판단하면 안 되고, `precision_nonsteady` 를 기준선 0.081 과 비교해야 한다.
2. **오염률이 낮다고 좋은 게 아니다.** steady 로 인정하는 양 자체가 적으면 오염률은 자동으로 낮아진다. TS2Vec 이 오염률 0.002(전체 최저)인데 AUPRC 0.344(하위)인 이유다. 반드시 보존률과 함께 읽는다.
3. **현재 학습량(600 iter)이 수렴 전이다.** TS2Vec 에서 **학습이 덜 된 쪽이 모든 지표에서 더 좋았다**(최종손실↔AUPRC 상관 +0.61). 따라서 학습 있는 방법의 성능이 설계 덕인지 미학습 덕인지 아직 구분되지 않는다.
4. **CoST 창 길이 비교는 학습량과 교란되어 있다.** `max_train_length = W` 때문에 같은 600 iter 에서 데이터 통과 횟수가 창별로 3.57~14.29 회로 4배 차이 난다. "창이 길면 나빠진다" 를 창 효과로 단정할 수 없다. 상세는 `hyperparameters.md` §11 문제 9.
