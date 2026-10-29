# Proposed (final v1) — 제안 방법 확정판과 논문 비교표

연구자 결정 2026-10-29. **현재까지의 최종판이고 추후 검증으로 개선해 나간다.**

---

## 1. Proposed (final v1) 이란

> **baseline `CoST-W144-noscaleshift`**
> **+ trunk 분리** (trend trunk / seasonal trunk. 창은 **둘 다 144** — 창 분리 아님)
> **+ seasonal 분기 전체 동결** (trunk + 차수 `sfd` 가 무작위 초기값, gradient 0)
> **+ 차수 정규화** (order tracking, $N_{\rm rot}=1$, $L_{\rm ORD}=36$)

| | |
|---|---|
| **방법 이름** | `CoST-dual-T144-S144-frz` |
| **코드** | [`code/04_proposed/04_cost_dual_trunk.py`](../../code/04_proposed/04_cost_dual_trunk.py) — 파일 맨 위에 `★★★ Proposed (final v1)` 표식 |
| **실행 명령** | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze` |
| **정본 결과** | `outputs/03_baselines/12_proposed_cost_dual_trunk/CoST-dual-T144-S144-frz_*` |
| **사본** | 이 폴더의 `final_v1_outputs/` |
| **표 생성 코드** | [`code/04_proposed/06_final_v1_tables.py`](../../code/04_proposed/06_final_v1_tables.py) |
| **설계·해석** | [`docs/idea/proposed_ideas_already_implemented.md`](../../docs/idea/proposed_ideas_already_implemented.md) 의 "전체 결과 요약" 과 §4 |

### A 실험 성능 (5 fold × 5 seed = 25회)

| | 값 | baseline 대비 |
|---|---|---|
| F1(ns) | **.9303** | **+0.0171** (22/3, Wilcoxon **p<0.0001**) |
| AUROC | **.9807** | +0.0025 |
| 보존률 | .9932 | +0.0029 |
| gradient 를 받는 파라미터 | **13.69M** | **−21%** (baseline 17.44M) |

**학습 용량을 21% 줄이면서 성능이 가장 좋다** — "파라미터를 늘려서 좋아진 것" 이라는 반론이
이 변형에서는 성립하지 않는다.

### 바꾸지 않은 것 (baseline 과 동일)

창 144 · jitter-only 증강 · 표현 320차원 · GMM 후단(`StandardScaler` + `GaussianMixture(2, diag)`) ·
분할(A: 장력 leave-one-level-out 5-fold) · 지표 · **학습량** (600 iteration, `data_passes` 3.17)

속도 설정값 $V$ 는 **차수 창 길이 결정에만** 쓰고 표현의 특징으로 넣지 않는다
(선례: `MovStd-Slope-speed-rule1b`. 비교군 23종 중 같은 정보를 쓰는 것은 2종뿐이므로 논문에 명시).

---

## 2. 이 폴더의 파일

| 파일 | 내용 |
|---|---|
| `table1_model_comparison_final_v1.csv` | **표 (1)** 모델별 성능 비교 (8행) |
| `table2_ablation_study_final_v1.csv` | **표 (2)** 제안 방법 ablation (7행) |
| **`baselines_A/`** | **표 (1) 8종**의 실제 산출물 사본 (csv + plot) |
| **`baselines_A_ablations/`** | **표 (2) ablation 7종**의 실제 산출물 사본 (csv + plot) |
| **`code/`** | **final v1 코드 사본** + 그 README |
| **`code_ablations/`** | **ablation 코드 사본** (이름에 ` (ablation)`) + 그 README |

`CoST-W144` 와 `CoST-dual-T144-S144-frz` 는 두 표에 모두 들어가므로 **두 폴더에 각각 들어 있다**
(각 폴더가 자족적이도록). 두 사본 폴더는 스크립트를 다시 돌릴 때마다 비우고 다시 채운다.

### `baselines_A/` 와 `baselines_A_ablations/`

**여기 결과는 모두 A 실험**(장력 leave-one-level-out 5-fold × 5 seed)이다. 그래서 폴더 이름에
`_A` 를 붙였다. `outputs/03_baselines/<단계>/` 와 같은 구조로 **평탄하게** 넣었고,
파일 접두사가 방법 이름이라 섞이지 않는다.

**final v1 은 `CoST-dual-T144-S144-frz_*`** 로 찾으면 된다.

산출물은 방법마다 다음을 포함한다.

| 확장자 | 파일 | 내용 |
|---|---|---|
| csv | `*_run_metrics.csv` | run × seed 1,050행 (속도·장력 포함) — 속도별 분석용 |
| csv | `*_fold_metrics.csv` | fold × seed 25행 — **표의 평균·표준편차가 여기서 나온다** |
| csv | `*_summary.csv` | 전체 집계 |
| csv | `*_config.csv` | 하이퍼파라미터·설계 기록 (재현용) |
| csv | `*_loss.csv` | 손실 곡선 원자료 (딥러닝 방법만) |
| png/pdf | `*_loss.*` | 손실 곡선 |
| png/pdf | `*_heatmap_contamination_rate.*` | 조건별 오염률 |
| png/pdf | `*_heatmap_retention_rate.*` | 조건별 보존률 |

`_progress_run.csv` / `_progress_fold.csv` 는 **복사하지 않았다** — 실행 중 append 하는
중단 복구용 로그이고 `run_metrics` / `fold_metrics` 와 내용이 중복된다.
**정본은 `outputs/03_baselines/<단계>/` 이고 이 폴더는 사본이다** (`source_folder` 열로 추적).

두 표 모두 **A 실험 25회의 pooled 지표 평균과 표준편차**이고, 모든 지표가 들어 있다
(`*_mean` / `*_std` 쌍). `source_method`·`source_folder` 열로 어느 결과에서 나온 값인지 추적된다.

**수록 지표** — 보존률, 오염률, precision/recall/F1(steady), AUPRC(steady),
precision/recall/F1(non-steady), AUPRC(non-steady), AUROC, 진입 오차 MAE·bias(초), 진입 미검출 run 수

### 누락 열을 복원한 것 — `derived_from_confusion_matrix` 열을 볼 것

`evaluation.py` 가 precision/recall/F1 열을 **나중에** 추가해서, 먼저 돌린 방법들
(`MovStd-Slope-fixed`, `TS2Vec`, `MOMENT`)의 `fold_metrics.csv` 에 그 열이 없다.
**혼동행렬 항등식으로 정확히 복원했다.**

$$N = \text{n\_scored},\quad P_{ns} = \text{n\_nonsteady},\quad
TP_{st} = \text{보존률}\times(N-P_{ns}),\quad \text{precision}_{st} = 1-\text{오염률}$$
$$\widehat{N_{st}} = TP_{st}/\text{precision}_{st},\quad TP_{ns} = P_{ns}-(\widehat{N_{st}}-TP_{st}),\quad
\text{precision}_{ns} = \frac{TP_{ns}}{N-\widehat{N_{st}}},\quad \text{recall}_{ns} = \frac{TP_{ns}}{P_{ns}}$$

**검증**: 열이 **있는** `CoST-W144` 에 같은 식을 적용하니 기록된 값과 최대 오차 **2.1e-15**
(부동소수점 수준)였고, 복원한 precision·recall 로 다시 만든 F1 이 기록된 F1 과 소수점까지 일치했다.
복원한 열 이름은 `derived_from_confusion_matrix` 에 기록된다 (복원 없으면 빈칸).

**`auprc_steady` 는 복원할 수 없다** — 점수(확률)가 필요하고 fold_metrics 에 남지 않는다.
없는 방법은 공란이고 `unavailable_metrics` 열에 표시된다. 논문 주 지표는 AUPRC(non-steady) 이고
`evaluation.py` 도 "AUPRC 는 비대칭이라 두 값을 같은 잣대로 비교하면 안 된다" 고 적어 두었다.

---

## 3. 표 (1) 모델별 성능 비교

행 순서는 연구자 지정. `Proposed method` 가 final v1 이다.

| 표의 이름 | 실제 방법 | 폴더 |
|---|---|---|
| Statistics-based | `MovStd-Slope-fixed` | `01_stat_ssd` |
| TS-TCC (2021) | `TS-TCC` | `07_tstcc` |
| TS2Vec (2022) | `TS2Vec` | `02_ts2vec` |
| CoST (2022) | `CoST-W144` | `04_cost` |
| SimMTM (2023) | `SimMTM` | `08_simmtm` |
| SoftCLT (2024) | `SoftCLT-ref` | `05_softclt` |
| MOMENT (2024) | `MOMENT` | `03_moment` |
| **Proposed method** | **`CoST-dual-T144-S144-frz`** | `12_proposed_cost_dual_trunk` |

> **변형 선택을 적어둔다 (바꾸려면 `06_final_v1_tables.py` 의 `TABLE1` 한 줄)**
> `01_stat_ssd` 에는 통계 기준선이 8종, `05_softclt` 에는 3종이 있다. 각 계열에서
> **F1(ns) 가 가장 높은 것**을 골랐다 — 비교군을 가장 강하게 두는 보수적 선택이다.
> `MovStd-Slope-fixed` (F1 .6598, 8종 중 1위) / `SoftCLT-ref` (F1 .4014, 3종 중 1위).
> `CoST (2022)` 는 연구자 지정대로 **원 논문 설정** `CoST-W144` 이고
> baseline `CoST-W144-noscaleshift` 가 아니다.

---

## 4. 표 (2) 제안 방법 ablation

`trunk_separation` / `seasonal_frozen` / `order_tracking` 열에 `O` 로 구성 요소를 표시했다.
`delta_f1_nonsteady_vs_base`·`win_loss_vs_base`·`wilcoxon_p_vs_base` 는
**`CoST (no scale/shift)` 기준** 짝지은 비교다.

| 표의 이름 | 분리 | 동결 | 차수 | 실제 방법 |
|---|---|---|---|---|
| CoST | | | | `CoST-W144` |
| CoST (no scale/shift) | | | | `CoST-W144-noscaleshift` ← ablation 기준 |
| + trunk separation | O | | | `CoST-dual-T144-S144-noord` |
| + order tracking | | | O | `CoST-W144-order` |
| + trunk separation + freeze | O | O | | `CoST-dual-T144-S144-noord-frz` |
| + trunk separation + order tracking | O | | O | `CoST-dual-T144-S144` |
| **Proposed method** | **O** | **O** | **O** | **`CoST-dual-T144-S144-frz`** |

### "+ freeze" 단독 행이 없는 이유

**동결은 분리된 seasonal trunk 를 얼리는 것이므로 분리가 전제다.** 공유 trunk 를 얼리면
trend 분기까지 얼어 다른 실험이 된다. 그래서 `동결만` / `동결 + 차수(분리 없이)` 조합은
**구조적으로 존재할 수 없다.** 위 7행이 가능한 조합의 전부다.

### 재실험

| 행 | 명령 |
|---|---|
| CoST | `python code/03_baselines/04_cost.py` |
| CoST (no scale/shift) | `python code/03_baselines/06_cost_ablation.py` |
| + trunk separation | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 noorder` |
| + order tracking | `python code/04_proposed/01_cost_order.py tension nrot=1` |
| + trunk separation + freeze | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 noorder freeze` |
| + trunk separation + order tracking | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1` |
| **Proposed method** | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze` |

같은 명령을 다시 돌리면 `{방법}_fold_metrics.csv` 가 있으면 **건너뛴다** (재실행하려면 먼저 지운다).
전체 변형 목록과 명령은 `docs/hyperparameters.md` §13.10 의 재실험 명령표에 있다.

---

## 5. 남은 검증 (추후 개선 항목)

| # | 항목 | 왜 |
|---|---|---|
| **—** | **final v1 체크포인트 재생성 + 임베딩 그림** | **2026-10-29 체크포인트 삭제에 final v1 것도 포함됐다.** 임베딩 그림을 만들려면 ① `10_embeddings.py` 에 `dual` 로더 추가(제안방법이 등록된 적 없음) ② 재학습 16분 ③ 그림 생성. `docs/hyperparameters.md` §13.10 참조 |
| **—** | **`outputs/` 폴더 구조 재정비** | 축이 섞였다 — `03_baselines/` 가 A 전용인데 이름에 표시 없음, 제안방법이 `03_baselines/09_~13_` 에 비교군과 섞임, 단계 번호 두 체계 중복, 로그 19개가 최상위에 흩어짐. `docs/hyperparameters.md` §13.10 에 7개 항목으로 정리 |
| P12 | 부분 공유 깊이 스캔 | trunk 10층 중 앞 $k$층만 공유 → "어느 깊이까지 공유해도 되는가" |
| P14 | order 재표본화 앤티앨리어싱 | 저속 4배 다운샘플에서 차수 18~72 가 0~18 로 접힌다 |
| — | 오프셋 기준선을 비교군에 추가 | `|창평균 e|` 단독 AUROC 0.9842 > CoST 0.9782. **비교군에 오프셋을 쓰는 방법이 없다** |
| — | C 실험 (저속→고속 외삽) | 연구자 결정으로 제안 방법은 **A 까지만**. 차수의 본래 동기는 A 에서 직접 검증되지 않는다 |

**기각된 변형과 반증된 가설**은 `docs/idea/proposed_ideas_already_implemented.md` 의
"전체 결과 요약" 에 전부 기록돼 있다 (19개 변형, 반증된 가설 4개).
