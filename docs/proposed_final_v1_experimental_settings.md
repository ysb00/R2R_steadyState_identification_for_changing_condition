# Proposed (final v1) — Experimental Settings

이 문서는 **확정된 제안 방법 `CoST-dual-T144-S144-frz`** 의 실험 설정을 논문 *Experimental setup*
절에 그대로 쓸 수 있는 형태로 정리한다. 수치의 출처는 전부
`outputs/03_baselines/12_proposed_cost_dual_trunk (final v1)/CoST-dual-T144-S144-frz_config.csv`
와 `code/03_baselines/evaluation.py`, `code/03_baselines/splits.py` 다.

| | |
|---|---|
| 방법 이름 | `CoST-dual-T144-S144-frz` |
| 코드 | `code/04_proposed/04_cost_dual_trunk.py` |
| 실행 (A) | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze` |
| 실행 (C) | `python code/04_proposed/04_cost_dual_trunk.py c nrot=1 freeze` |
| 기반 저장소 | `salesforce/CoST`, commit `7bb5271c2b66cd5a1d634c27f5df308f190b50ec` (**원본 미수정**, 런타임 서브클래스 shim) |

---

## 1. 데이터

### 1.1 입력 신호

단일 채널이다.

$$e(t) \;=\; \underbrace{T_1(t)}_{\text{Film Tension \#1 (kg)}} \;-\; \underbrace{T^{\rm set}}_{\text{Film-Tension Setting Value (kg)}}$$

설정값 기준 **절대 오차**이고, 창마다 기준점을 다시 잡지 않는다 (창 평균 제거 없음).
sag(처짐)는 음수, overshoot 은 양수로 부호가 유지된다.
전 방법이 같은 입력을 쓴다 (`code/03_baselines/*.py` 의 `df['e'] = df[T1] - df[TS]`).

| | |
|---|---|
| 샘플 간격 $\Delta t$ | **0.109 s** (약 9.174 Hz) |
| run 1개 | 하나의 운전 조건. 길이 **1,275–1,300 샘플** (약 139–142 s) |
| 전체 | **210 run, 268,933 시점** |
| 조건 격자 | Film Tension 5수준 × ReWinder Tension 6수준 × Web Speed 7수준, 조건당 run 정확히 1개 |
| Web Speed 수준 | 50 / 75 / 100 / 125 / 150 / 175 / 200 mm/s |

### 1.2 라벨 (`Trigger 4`) 과 채점 대상

| 값 | 뜻 | 시점 수 | 비율 |
|---|---|---|---|
| 1 | steady | 230,787 | **0.858** |
| 0 | non-steady | 20,429 | **0.076** |
| 2 | ambiguous | 17,717 | **0.066** |

**`Trigger 4 == 2`(ambiguous)는 채점에서 제외한다** (기본 모드). 따라서 채점 대상은
251,216 시점이고 그 안에서 non-steady 유병률은 **0.0813** 이다 — AUPRC(non-steady) 의 기준선이
이 값이다.

> 라벨은 **표현 학습에 전혀 쓰지 않는다.** `Trigger 4` 를 읽는 코드는
> `evaluation.py` 의 `_labels()` 한 곳뿐이고, 모델 어댑터는 `usecols` 에서 라벨을 제외한다.
> 평가는 라벨 없는 2-군집 GMM 판정 뒤 채점에만 라벨을 쓴다 (§3).

### 1.3 분할 — A 실험 (논문 표 1·2 의 기준)

**장력 leave-one-level-out 5-fold × seed 5 = 25회.** Film Tension 설정값 5수준 중 하나를
검증으로 빼고 나머지 4수준으로 학습한다. **속도 7수준은 학습·검증에 모두 등장한다.**

| fold | 학습 run | 검증 run | 학습 시점 | 검증 시점 | 학습:검증 run 비 |
|---|---|---|---|---|---|
| 1 | 168 | 42 | 215,148 | 53,785 | 0.80 : 0.20 |
| 2 | 168 | 42 | 215,126 | 53,807 | 0.80 : 0.20 |
| 3 | 168 | 42 | 215,156 | 53,777 | 0.80 : 0.20 |
| 4 | 168 | 42 | 215,151 | 53,782 | 0.80 : 0.20 |
| 5 | 168 | 42 | 215,151 | 53,782 | 0.80 : 0.20 |

seed 는 0, 1, 2, 3, 4 다 (`SEED=0`, `N_SEEDS=5`).
fold 배정은 1회 생성 후 고정이다 (`outputs/03_baselines/folds.csv`).
fold 별 조건 구성은 `docs/split_description.md` 에 있다.

### 1.4 분할 — C 실험 (저속 → 고속 외삽)

**저속 50/75/100/125 (120 run) = 학습 + CV / 고속 150/175/200 (90 run) = 최종 test.**
fold 8개 × seed 5 = **40회**. C 의 CV fold 에서는 고속 run 이 **학습도 검증도 아니다**(미사용).

| fold | 학습 run | 검증 run | 미사용 | 묻는 것 |
|---|---|---|---|---|
| `cv_T1`~`cv_T5` | 96 (저속, 장력 4수준) | 24 (저속, 장력 1수준) | 90 (고속) | 저속 안에서 **새 장력** (A 와 같은 질문) |
| `cv_S1` | 60 (50, 75) | 30 (100) | 120 | **1수준 속도 외삽** |
| `cv_S2` | 90 (50, 75, 100) | 30 (125) | 90 | **1수준 속도 외삽** (학습량 증가) |
| `final_test` | 120 (저속 전부) | 90 (고속 전부) | 0 | **최종 목표. 한 번만 쓴다** |

> **C 는 네 단위를 가로질러 평균하지 않는다** — `cv_T` / `cv_S1` / `cv_S2` / `final_test` 는
> 묻는 질문이 서로 다르고 학습량도 다르다. 표에서도 단위별로 따로 싣는다.

---

## 2. 모델 하이퍼파라미터

### 2.1 저장소 기본값을 그대로 쓴 것 (As-published)

비교군과 공정하게 비교하기 위해 CoST 저장소의 기본값을 바꾸지 않았다.

| 기호 | 값 | 설명 |
|---|---|---|
| $d_{\rm repr}$ | **320** | 표현 차원. trend 160 + seasonal 160 |
| $d_{\rm hidden}$ | **64** | trunk 내부 채널 폭 |
| depth | **10** | `DilatedConvEncoder` 층 수 (dilation $2^i$, kernel 3) |
| kernels | **[1, 2, 4, 8, 16, 32, 64, 128]** | trend 분리기(`tfd`)의 Conv1d 커널 8종. 출력은 8개 평균 |
| $\alpha$ | **0.0005** | seasonal 손실 가중치 |
| MoCo $K$ | 256 | 음성 큐 길이 |
| MoCo $m$ | 0.999 | 모멘텀 계수 |
| MoCo $T$ | 0.07 | 온도 |
| optimizer | SGD | momentum 0.9, weight decay 1e-4 |
| learning rate | **0.001** | cosine 스케줄 (`adjust_learning_rate`) |
| batch size | **8** | |
| `repr_dropout` | 0.1 | seasonal 출력에만 적용 |

### 2.2 제안 방법이 정한 값

| 기호 | 값 | 유도 근거 |
|---|---|---|
| 창 $W_T$ (trend) | **144** | baseline 과 동일. $V=50$ 의 **1회전** = $\lceil C/(V\Delta t)\rceil = 144$ 샘플 (15.7 s) |
| 창 $W_S$ (seasonal) | **144** | $N_{\rm rot}=1$ 이라 $W_T$ 와 같다 (창 분리는 기각됨) |
| $N_{\rm rot}$ | **1** | seasonal 분기가 담는 회전 수. 차수 해상도 $1/N_{\rm rot} = 1.00$ 차수/bin |
| $L_{\rm ORD}$ | **36** | 차수영역 길이 $= N_{\rm rot}\cdot\mathrm{round}(C/(V_{\max}\Delta t))$. **가장 빠른 속도 기준** |
| $C$ (롤러 둘레) | **783.5 mm** | 205 run 평균, 지름 249.4 mm (`docs/periodic_disturbance.md`) |
| 증강 | **jitter only** | 원본의 `scale`·`shift` 제거 (§2.3) |
| seasonal 분기 | **동결** | 전체 `requires_grad=False` (무작위 초기값 고정) |
| trunk | **분리 (2개)** | trend trunk / seasonal trunk 각 637,248 파라미터 |

**속도별 order window** (차수 변환이 자르는 뒤쪽 길이, $N_{\rm rot}=1$):

| $V$ (mm/s) | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| 샘플 | 144 | 96 | 72 | 58 | 48 | 42 | 36 |

> **속도 설정값 $V$ 의 사용** — $V$ 는 **order window 길이를 정하는 데만** 쓰고 표현의 특징으로
> 넣지 않는다. 입력 2채널 중 채널 1 로 실어 보내고 trunk 진입 직전에 떼어낸다.
> 선례가 있다 (`MovStd-Slope-speed-rule1b`, `01_stat_ssd.py:163`). 다만 **비교군 23종 중 같은
> 정보를 쓰는 것은 2종뿐**이므로 논문의 공정성 서술에 반드시 적는다.

### 2.3 학습량 — baseline 과 동등하게 맞춘다

`max_train_length = W` 가 run 을 $\lfloor 1300/W \rfloor$ 조각으로 쪼개므로 1 epoch 의 iteration
수가 창에 따라 달라진다. 같은 iteration 수를 쓰면 **데이터 통과 횟수가 어긋나 비교가 깨진다.**

$$n_{\rm iters} \;=\; n_{\rm iters}^{\rm base}\cdot\frac{\text{조각수}_{\rm 변형}}{\text{조각수}_{\rm baseline}}$$

| | 조각 수 | iter/epoch | $n_{\rm iters}$ | 데이터 통과 |
|---|---|---|---|---|
| baseline (창 144) | 9 | 189 | 600 | **3.17** |
| **final v1** (창 144) | 9 | 189 | **600** | **3.17** |

창을 바꾸지 않으므로 유도값이 600 으로 baseline 과 같다. `fold_metrics.csv` 에
`data_passes` 와 `data_passes_baseline` 을 나란히 기록해 동등성을 사후 감사할 수 있다.

### 2.4 파라미터 수

| | 전체 | **gradient 를 받는 것** |
|---|---|---|
| baseline `CoST-W144-noscaleshift` | 17.44 M | **17,443,808** |
| `CoST-W144-order` (공유 trunk) | 14.67 M | 14,670,368 |
| `CoST-dual-T144-S144` (분리, 학습) | 33.1 M | 15,307,616 |
| **`CoST-dual-T144-S144-frz`** (final v1) | 33.1 M | **13,694,528 (baseline 대비 −21%)** |

부품별: trunk 637,248 / `tfd` 13,057,280 / Hz `sfd` 3,749,280 / 차수 `sfd`($L_{\rm ORD}$=36) 975,840.
이원 trunk 의 "전체 33.1 M" 중 **16.8 M 은 forward 에서 계산만 되고 버려져 gradient 가 0** 이다
(trend trunk 의 `sfd`, seasonal trunk 의 `tfd`). final v1 은 seasonal 인코더 전체가 동결이라
**학습 용량이 baseline 보다 21% 적다.**

> `config.csv` 의 `trainable_params_note` 는 동결 **이전** 판(15.31 M)을 적어 둔 것이다.
> final v1 의 정확한 값은 위 표의 13,694,528 이다.

---

## 3. 평가 절차

라벨을 쓰지 않는 2-군집 GMM 판정 → `Trigger 4` 채점의 2단계다. **전 방법이 동일한 후단**
(`evaluation.py`)을 쓴다.

```
표현 z (320차원, 시점별)
  → StandardScaler + GaussianMixture(n=2, covariance_type='diag')   ← 학습 fold 로만 적합
  → |e 평균| 이 작은 군집 = steady 로 지정                            ← 라벨 미사용
  → 검증 fold 시점을 판정 (사후확률 ≥ 0.5)
  → Trigger 4 로 채점 (ambiguous 제외)
```

| 설정 | 값 |
|---|---|
| GMM 성분 수 | 2 |
| GMM 공분산 | `diag` |
| 표준화 | `StandardScaler`, **학습 fold 로만** 적합 |
| steady 군집 지정 | 군집별 $e$ 평균의 **절댓값이 작은 쪽** |
| non-steady 점수 | $\text{score} = 1 - P(\text{steady})$ |
| 판정 | $P(\text{steady}) \ge 0.5$ → steady |
| 표현 추출 | 인과 이동창. `sliding_padding = W-1`, `sliding_length = 1`, `casual=True` → 시점 $t$ 의 표현은 **직전 $W$ 샘플만** 본다 |

---

## 4. 평가 지표 — 정의와 수식

### 4.1 혼동행렬 기호

채점 대상 시점(ambiguous 제외)에서 정의한다.

| | 실제 steady | 실제 non-steady |
|---|---|---|
| **steady 로 판정** | $TP_{st}$ | $FP_{st}$ |
| **non-steady 로 판정** | $FN_{st}$ | $TP_{ns}$ |

$N = TP_{st}+FP_{st}+FN_{st}+TP_{ns}$, $P_{ns} = FP_{st}+TP_{ns}$, $P_{st} = TP_{st}+FN_{st}$.

### 4.2 표 1·2 에 들어간 모든 지표

| CSV 열 | 표기 | 수식 | 방향 | 비고 |
|---|---|---|---|---|
| `retention_rate` | 보존률 Retention | $\dfrac{TP_{st}}{P_{st}}$ | ↑ | $=\text{recall}_{st}$. **도메인 이름** — 실제 steady 시점을 얼마나 지켜냈나 |
| `contamination_rate` | 오염률 Contamination | $\dfrac{FP_{st}}{TP_{st}+FP_{st}}$ | ↓ | $=1-\text{precision}_{st}$. steady 로 판정한 것 중 실제로는 non-steady 인 비율 |
| `precision_steady` | Precision (st) | $\dfrac{TP_{st}}{TP_{st}+FP_{st}}$ | ↑ | $=1-$ 오염률 |
| `recall_steady` | Recall (st) | $\dfrac{TP_{st}}{P_{st}}$ | ↑ | $=$ 보존률 |
| `f1_steady` | F1 (st) | $\dfrac{2\,\text{P}_{st}\text{R}_{st}}{\text{P}_{st}+\text{R}_{st}}$ | ↑ | |
| `auprc_steady` | AUPRC (st) | steady 를 positive 로 한 PR 곡선 아래 면적 | ↑ | 점수는 $1-\text{score}$. **기준선 0.919** |
| `precision_nonsteady` | Prec (ns) | $\dfrac{TP_{ns}}{FN_{st}+TP_{ns}}$ | ↑ | |
| `recall_nonsteady` | Rec (ns) | $\dfrac{TP_{ns}}{P_{ns}}$ | ↑ | |
| **`f1_nonsteady`** | **F1 (ns)** | $\dfrac{2\,\text{P}_{ns}\text{R}_{ns}}{\text{P}_{ns}+\text{R}_{ns}}$ | ↑ | **주 지표** |
| **`auprc`** | **AUPRC (ns)** | non-steady 를 positive 로 한 PR 곡선 아래 면적 | ↑ | **기준선 0.0813** (유병률) |
| **`auroc`** | **AUROC** | ROC 곡선 아래 면적 | ↑ | **클래스 대칭** — positive 를 바꾸고 점수를 뒤집으면 같은 값이라 하나만 기록 |
| `entry_mae_s` | 진입 MAE (s) | $\frac{1}{n_{\rm run}}\sum_{\rm run}\lvert \hat k_{\rm entry}-k_{\rm entry}\rvert\,\Delta t$ | ↓ | run 별 오차의 **절댓값** 평균 |
| `entry_bias_s` | 진입 bias (s) | $\frac{1}{n_{\rm run}}\sum_{\rm run}(\hat k_{\rm entry}-k_{\rm entry})\,\Delta t$ | →0 | 부호 있는 평균. **음수 = 이르게 판정** |
| `n_runs_no_entry` | 진입 미검출 run | $\hat k_{\rm entry} = -1$ 인 run 수 | ↓ | 끝까지 steady 판정이 없던 run |

**진입 시점 정의**

$$\hat k_{\rm entry} \;=\; \min\Big\{k \;:\; \text{pred}(k)=\text{pred}(k{+}1)=\cdots=\text{pred}(k{+}9)=\text{steady}\Big\}$$

**연속 10 시점**(`ENTRY_MIN_CONSEC = 10`, 약 1.09 s)이 steady 로 판정된 첫 시점이다. 없으면 $-1$.
참 진입 시점 $k_{\rm entry}$ 는 `Trigger 4 == 1` 인 첫 $k$ 다.
진입 시점은 **ambiguous 를 포함한 전체 시계열**에서 읽는다 (구간이 끊기지 않게).

### 4.3 `pooled` vs `runmean` — 두 가지 집계

| 접두사 | 뜻 | 표에 쓰는 것 |
|---|---|---|
| `pooled_` | fold 의 **전체 시점을 한 덩어리로** 계산 | **표 1·2 가 이것을 쓴다** |
| `runmean_` / `runstd_` | run 별로 계산한 뒤 평균·표준편차 | 조건별 heatmap, 진입 오차 |

표의 각 칸은 **`pooled_` 지표를 25회(A: 5 fold × 5 seed) 또는 40회(C: 8 fold × 5 seed)에 대해
평균 ± 표준편차**로 적는다.

### 4.4 유의성 검정

같은 **fold × seed** 로 짝지은 **Wilcoxon signed-rank test** (양측, $n=25$).
표 2 의 `delta_f1_nonsteady_vs_base` · `win_loss_vs_base` · `wilcoxon_p_vs_base` 가 그것이고
기준은 `CoST (no scale/shift)` 다.

> **F1(ns) 해석 주의** — fold 간 F1(ns) 차이의 대부분이 **유병률 차이**다
> (A fold 1 은 42 run 중 16개가 non-steady 0개이고, ns 비율과 F1 의 상관이 0.784).
> 그래서 유병률과 무관한 **AUROC** 와 **보존률**을 함께 본다.

### 4.5 복원한 지표 — `derived_from_confusion_matrix` 열

`evaluation.py` 가 precision/recall/F1 열을 나중에 추가해서, 먼저 돌린 방법
(`MovStd-Slope-fixed`, `TS2Vec`, `MOMENT`)의 `fold_metrics.csv` 에 그 열이 없다.
**혼동행렬 항등식으로 정확히 복원했다.**

$$TP_{st} = \text{보존률}\cdot P_{st},\qquad \widehat{N_{st}} = \frac{TP_{st}}{1-\text{오염률}},\qquad
TP_{ns} = P_{ns}-(\widehat{N_{st}}-TP_{st})$$
$$\text{precision}_{ns} = \frac{TP_{ns}}{N-\widehat{N_{st}}},\qquad \text{recall}_{ns} = \frac{TP_{ns}}{P_{ns}}$$

**검증**: 열이 **있는** `CoST-W144` 에 같은 식을 적용해 기록값과 최대 오차 **2.1e-15**
(부동소수점 수준)였고, 복원한 precision·recall 로 다시 만든 F1 이 기록된 F1 과 일치했다.
`auprc_steady` 는 점수가 필요해 복원할 수 없어 해당 방법은 공란이고 `unavailable_metrics` 에 표시된다.

---

## 5. 그림 — 축·라벨·컬러바

공통: 글꼴 크기 **9 pt**, PNG **300 dpi**, PDF 동시 저장, `savefig.bbox='tight'`,
격자선 0.5 pt `#e1e0d9`, 축선 0.8 pt `#c3c2b7`.

### 5.1 손실 곡선 `*_loss.png|pdf`

| 요소 | 내용 |
|---|---|
| subplot | **fold 하나 = subplot 하나**. 제목은 A 는 `Fold 1`…`Fold 5`, C 는 fold 이름(`cv_T1`, `final_test` 등) |
| x축 (`supxlabel`) | **`Iteration`** |
| y축 (`supylabel`) | **`CoST contrastive loss (dual trunk)`** (스크립트가 넘기는 문자열) |
| 선 | seed 5개를 **옅은 선** (`#2a78d6`, 1.0 pt, alpha 0.35), **seed 평균을 진한 선** (`#2a78d6`, 2.0 pt) |
| 원자료 | `*_loss.csv` (`fold`, `seed`, `iter`, `loss`) |

### 5.2 조건별 heatmap `*_heatmap_{contamination,retention}_rate.png|pdf`

운전 조건 격자에서 지표가 어디서 나빠지는지 보는 그림이다.

| 요소 | 내용 |
|---|---|
| subplot | fold × (조건 패널). 제목은 `{fold}: {부제}` |
| x축 (`supxlabel`) | **`Film tension setting (kg)`** 또는 **`Web speed setting (mm/s)`** (패널에 따라) |
| y축 (`supylabel`) | **`ReWinder tension setting (kg)`** |
| 셀 값 | 숫자를 셀 안에 직접 표기 (밝은 배경은 검정 `#0b0b0b`, 어두운 배경은 흰색 `#ffffff`) |
| colormap | `seq_blue` — 단일 색상 sequential, 밝음 `#cde2fb` → 어두움 `#0d366b` (13단계) |
| **colorbar 라벨** | **`Contamination rate`** / **`Retention rate`** |
| **colorbar 범위** | 오염률 **0.0–0.3** / 보존률 **0.0–1.0** — **방법 전체에서 고정**해 panel·방법 간 비교가 가능하다 |
| colorbar 화살표 | 범위를 넘는 값이 있으면 끝에 화살표(`extend`)로 표시. 값 자체는 셀에 찍혀 있다 |

> **색 범위를 고정한 이유** — 보존률 0–1 은 비율의 전 범위라 포화가 없다. 오염률은 0–1 로 두면
> 중앙값이 0.000, 90분위가 0.04–0.05 라서 거의 모든 칸이 같은 색이 되어 방법 간 비교가 불가능하다.
> **0–0.3 이 셀의 99%** 를 담는다.

---

## 6. 산출물

| 경로 | 내용 |
|---|---|
| `outputs/03_baselines/12_proposed_cost_dual_trunk (final v1)/` | A 실험 정본 |
| `outputs/04_speed_extrap/12_proposed_cost_dual_trunk (final v1)/` | C 실험 정본 |
| `outputs/07_proposed_final_v1/table1_model_comparison_final_v1.csv` | 논문 표 1 (8행) |
| `outputs/07_proposed_final_v1/table2_ablation_study_final_v1.csv` | 논문 표 2 (7행) |
| `outputs/07_proposed_final_v1/baselines_A/` | final v1 산출물 사본 |
| `outputs/07_proposed_final_v1/baselines_A_ablations/` | ablation 산출물 사본 (코드별 하위 폴더) |

방법마다 `*_run_metrics.csv` (run × seed), `*_fold_metrics.csv` (fold × seed),
`*_summary.csv`, `*_config.csv`, `*_loss.csv`, 그림 6개(png/pdf × 3)가 생성된다.

**관련 문서**: 모듈별 정의·수식·코드는 `proposed_final_v1_module_desciption.md`,
ablation 효과는 `proposed_final_v1_ablation_effect.md`, 전체 시행착오 기록은
`idea/proposed_ideas_already_implemented.md`.
