# 아이디어 02 — 설정값 방향 게이트 (trend branch)

**제안 방법 이름**: `CoST-W144-dirgate`
**기준(baseline)**: `CoST-W144-noscaleshift`  — **창이 같다. 대조군이 필요 없다**
**구현**: `code/04_proposed/02_cost_dirgate.py`
**결과**: `outputs/<평가방식>/10_proposed_cost_setpoint_direction_gate/`

---

## 1. 한 줄 요약

> **"설정값에 다가가는 중인가"를 시점마다 계산해, CoST 의 trend 표현을 그 값으로 조건화한다.**

CoST 의 손실·구조·창 길이는 그대로 두고, trend 분기 출력에 FiLM 게이트 한 층을 더한다.

---

## 2. 무엇이 문제인가

transient 는 "설정값으로 돌아가는 운동" 이고 steady 는 "설정값 주변의 방향 없는 요동" 이다.
그런데 **추세의 부호만으로는 이 둘을 가를 수 없다.**

- 아래로 처진 뒤 회복 → 추세가 **양(+)**
- 위로 overshoot 된 뒤 회복 → 추세가 **음(−)**

둘 다 transient 인데 부호가 반대다. 실측으로 두 경우가 모두 존재한다 —
184 run 중 **아래 처짐 181개**(`e` 최솟값 중앙값 −0.265 kg), **위 overshoot 48개**(최댓값 > +0.1 kg).

방향을 **부호가 아니라 "설정값 기준 접근/이탈"** 로 정의하면 둘이 하나로 묶인다.

CoST 의 trend 분기는 다중 커널 인과 conv 로 느린 성분을 뽑지만,
**"그 느린 성분이 설정값을 향하는가" 라는 물음은 목적함수에 없다.**
`research_plan.md` §3.1 (3-1) 이 기록한 *"기존 표현학습에는 상태 구분 과제가 없다"* 의 한 사례다.

---

## 3. 제안 — 설정값 방향 게이트

### 3.1 방향 지표

오차 에너지 $e^2$ 의 변화율로 방향을 정의한다.

$$r(t) \;=\; -\tfrac{1}{2}\frac{d}{dt}e(t)^2 \;=\; -\,e(t)\,\dot e(t)$$

$$r>0 \;\Longleftrightarrow\; |e| \text{ 가 줄어드는 중} \;=\; \textbf{설정값에 접근}$$

위상평면 $(e,\dot e)$ 에서 **회복 사분면**($e<0,\dot e>0$ 또는 $e>0,\dot e<0$)에 있다는 뜻이고,
sag 회복과 overshoot 회복이 **같은 부호**로 묶인다.

### 3.2 평활이 필수다 — 이것이 이 아이디어의 핵심 발견

$r$ 을 **샘플 단위**로 계산하면 신호가 없다. 실측:

| 평활 길이 | steady `frac_toward` | transient | AUROC |
|---|---|---|---|
| **없음** | 0.502 | **0.428** | 0.743 |
| 21 (2.3 s) | 0.505 | 0.305 | 0.832 |
| **$P(V)$ = 1회전** | 0.518 | **0.205** | **0.912** |

이 데이터의 **lag-1 자기상관이 0.922** 이고 회전 외란이 겹쳐 있어, 샘플 단위 미분은
회복 추세가 아니라 **요동이 지배**한다. 1회전 길이로 평활하면 회전 외란이 정확히 지워지고
회복 방향만 남는다.

평활 길이를 **$P(V) = \frac{1}{\Delta t}\frac{C}{V}$ (1회전 샘플 수, $C=783.5$ mm)** 로 두면
**임의값이 0개**다. 아이디어 01 과 같은 물리 상수에서 나온다.

### 3.3 시점별 게이트 값

시점 $t$ 의 창 $[t-W+1,\;t]$ (전부 과거 — **인과**) 안에서

$$\bar s(t) \;=\; \frac{1}{W}\sum_{\tau \in [t-W+1,\,t]} \mathbb{1}\big[\,r_{\text{smooth}}(\tau) > 0\,\big]
\;\in\; [0,1]$$

$\bar s = 0.5$ 는 "접근과 이탈이 반반 = 방향 없음", $\bar s$ 가 작을수록 "회복 중".

> **인과성** — 창이 전부 과거이므로 창 **안에서** 중심 평활을 써도 시점 $t$ 기준으로 인과다.
> 인과 이동평균을 쓰면 지연 $P/2$ 가 생겨 고속($P=36$)에서 반주기와 겹쳐 부호가 뒤집힌다
> (실측: V=200 에서 AUROC 0.912 → 0.599). **창 내 중심 평활을 써야 한다.**

### 3.4 CoST 접목 — trend 표현에 FiLM 게이트

$$\mathbf{h}_T'(t) \;=\; \mathbf{h}_T(t)\odot\big(1+\gamma(\bar s(t))\big) \;+\; \beta(\bar s(t))$$

$\gamma,\beta:\mathbb{R}\to\mathbb{R}^{160}$ 은 **영 초기화된 선형층 하나**다.
따라서 학습 시작 시점에 $\gamma=0,\beta=0$ 이고 $\mathbf{h}_T'=\mathbf{h}_T$ —
**baseline 과 정확히 같은 상태에서 출발**한다.

- 표현 차원은 320 그대로 (trend 160 + season 160). 비교군과 같은 후단을 쓴다
- seasonal 분기·trunk·증강·손실은 **건드리지 않는다**
- 시간 반전은 진폭 스펙트럼을 바꾸지 않으므로($|{\rm DFT}|$ 불변) 이 정보는 seasonal 손실로
  얻을 수 없다. **trend 쪽에 두어야 하는 구조적 이유**다

### 3.5 왜 (a) 시간 화살표 보조 과제가 아니라 (b) 게이트인가

| | (a) 보조 과제 | **(b) 게이트** |
|---|---|---|
| 구조 | 반전 crop + 분류 head + $\mathcal{L}_{\rm aot}$ | FiLM 층 1개 |
| 새 하이퍼파라미터 | $\beta$ (손실 가중) | **없음** |
| 측정 근거 | 간접 | **직접** — $\bar s$ 의 AUROC 0.912 를 쟀다 |

측정이 (b) 의 재료를 직접 뒷받침하므로 (b) 를 택한다.

---

## 4. 고정값

| 기호 | 값 | 근거 |
|---|---|---|
| $C$ | 783.5 mm | 기존 측정값 (`periodic_disturbance.md`) |
| 평활 길이 | $P(V)$ = 1회전 | **유도값.** 회전 외란을 정확히 지운다 |
| 집계 창 $W$ | **144** | baseline 과 동일 → **창 대조군 불필요** |
| 게이트 초기화 | 영 초기화 | baseline 과 같은 지점에서 출발 |
| 나머지 | `04_cost.py` 와 동일 | kernels [1..128], alpha 5e-4, lr 1e-3, batch 8, repr 320 |

$V$ 는 평활 길이 결정에만 쓰고 표현에 특징으로 넣지 않는다.
선례: `MovStd-Slope-speed-rule1b` (`01_stat_ssd.py:163`).

---

## 5. 사전 진단 (구현 전 측정, 재학습 없음)

`frac_toward` 단독의 변별력. 창 144, 전 시점(앞은 0 패딩), stride 10.

### 5.1 진폭 기준선을 넘는다 — **지금까지 검토한 것 중 처음**

| 특질 | AUROC |
|---|---|
| **`frac_toward` ($P(V)$ 평활)** | **0.912** |
| [기준선] `movstd` | 0.789 |

### 5.2 속도별로 고르게 작동한다

| 속도 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| `frac_toward` | 0.89 | 0.93 | **0.97** | 0.96 | **0.93** | 0.94 | 0.84 |
| `movstd` | 0.78 | 0.85 | 0.80 | 0.78 | **0.70** | 0.79 | 0.86 |

속도 150 — 통계 기준선이 무너지는 조건 — 에서 0.695 → **0.932** 로 가장 크게 벌어진다.

### 5.3 진폭을 통제해도 유지된다 — 가장 중요한 결과

앞서 검토한 아이디어들이 모두 여기서 무너졌다.

| `movstd` 5분위 | ns 비율 | `frac_toward` AUROC |
|---|---|---|
| Q1 | 0.072 | **0.965** |
| Q3 | 0.007 | 0.884 |
| Q4 | 0.020 | 0.862 |
| **Q5** (non-steady 대부분) | 0.322 | **0.931** |

| 비교 | 단독 AUROC | 진폭 Q5 통제 후 |
|---|---|---|
| 위상 정렬 (아이디어 기각분) | 0.698 | 0.524 |
| 추세-요동 의존성 (기각분) | 0.586 | 0.524 |
| **설정값 방향 게이트** | **0.912** | **0.931** |

### 5.4 한계 — 속도 외삽 아이디어는 아니다

저속 0.931 / 고속 0.894 로 전이에서 약간 떨어진다 (`movstd` 는 0.800 → 0.793).
**상태 변별력 자체를 올리는 아이디어**이지 조건 일반화 아이디어가 아니다.
속도 축은 아이디어 01 이 담당한다.

---

## 6. 코드 변경점

원본 저장소는 한 줄도 수정하지 않는다. 런타임 서브클래스 shim 이다.

### S1. 게이트를 채널로 실어 보낸다

아이디어 01 과 같은 방식. 인코더 맨 앞에서 떼어내므로 **trunk 는 e 1채널만 본다.**

```python
x: (n_run, T, 2)      # 채널 0 = e,  채널 1 = s̄(t) (시점별 게이트, 사전 계산)
```

$\bar s(t)$ 는 run 마다 한 번 계산해 두므로 학습·추론에서 추가 비용이 없다.

### S2. `CoSTEncoder` — trend 에만 FiLM 게이트

```python
class DirGateCoSTEncoder(CoSTEncoder):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.film = nn.Linear(1, 2 * self.component_dims)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # 시작 = baseline

    def forward(self, x, tcn_output=False, mask='all_true'):
        s = x[..., 1:2]                                   # (B, T, 1) 게이트
        out = super().forward(x[..., :1].contiguous(), tcn_output, mask)
        if tcn_output:
            return out
        trend, season = out
        g = self.film(torch.nan_to_num(s, nan=0.5))       # 패딩은 중립값 0.5
        gamma, beta = g.chunk(2, dim=-1)
        return trend * (1 + gamma) + beta, season
```

### S3. `PretrainDataset` — jitter 를 채널 0 에만

게이트 채널이 증강 잡음에 오염되지 않게 한다 (아이디어 01 과 동일).

### 변경 요약

| 구성요소 | baseline | 제안 | 변경 |
|---|---|---|---|
| 입력 | e 1채널 | e 1채널 (+ 게이트 운반 채널) | 운반만 |
| 창 | 144 | **144 (동일)** | — |
| trunk | | 동일 | — |
| **trend 분기** | 다중 커널 인과 conv | **+ FiLM 게이트 (영 초기화)** | ★ |
| seasonal 분기 | | 동일 | — |
| 손실 | trend MoCo + α·season | **동일** | — |
| 평가 후단 | GMM | 동일 | — |

---

## 7. 신규성

"시간 비가역성으로 자기지도 학습" 은 선행연구가 있다 (Agrawal et al., AISTATS 2022 —
구간 쌍의 순서를 맞히는 사전학습). 차별점은 세 가지다.

1. 방향을 **부호가 아니라 설정값 기준 접근/이탈**로 정의 — 폐루프 제어 공정의 구조를 쓴다.
   sag 와 overshoot 를 하나로 묶는다
2. **회전 주기로 평활**해야 신호가 드러난다는 것 — 이 데이터의 자기상관 0.922 에서 나온
   구체적 발견이고, 평활 길이가 **유도값**이다
3. 상태 **구간 분할**에 적용하고 **미학습 조건**에서 평가

---

## 8. 실험 설계

| 방법 | 역할 |
|---|---|
| `CoST-W144-noscaleshift` | baseline (이미 실행됨, `06_cost_ablation`) |
| **`CoST-W144-dirgate`** | **제안 방법** |

**창이 같으므로 창 대조군이 필요 없다.** 아이디어 01 과 다른 점이다.

평가는 비교군과 완전히 동일하다 — `evaluation.py` 후단, `splits.py` 분할, 같은 지표·그림.

| 평가 방식 | 출력 |
|---|---|
| A (ambiguous 제외) | `outputs/03_baselines/10_proposed_cost_setpoint_direction_gate/` |
| C (ambiguous 제외) | `outputs/04_speed_extrap/10_proposed_cost_setpoint_direction_gate/` |
| A (ambiguous 포함) | `outputs/05_ambiguous_included_A/10_proposed_cost_setpoint_direction_gate/` |
| C (ambiguous 포함) | `outputs/06_ambiguous_included_C/10_proposed_cost_setpoint_direction_gate/` |

```
python code/04_proposed/02_cost_dirgate.py tension        # A
python code/04_proposed/02_cost_dirgate.py c              # C
python code/04_proposed/02_cost_dirgate.py tension amb    # A, ambiguous 포함 (추론만)
python code/04_proposed/02_cost_dirgate.py c amb          # C, ambiguous 포함 (추론만)
```

변형 선택은 C `cv_S1`·`cv_S2` 에서 하고 `final_test` 는 확정 후 한 번만 본다.

---

## 9. 아이디어 01 과의 관계

둘은 **서로 다른 분기**에 있어 구조적으로 충돌하지 않는다.

| | 분기 | 겨누는 축 | 창 |
|---|---|---|---|
| 01 차수 정규화 | seasonal | 속도 일반화 | 288 강제 |
| **02 방향 게이트** | **trend** | **상태 변별력** | **144 (baseline 동일)** |

둘 다 같은 물리 상수 $C=783.5$ mm 에서 나온다 — 01 은 회전 수, 02 는 평활 길이.
각각 단독으로 검증한 뒤 결합을 세 번째 실험으로 둔다
(`research_experiments_plan.md` §8.4 "한 번에 하나만 바꾼다").


---

## 10. A 실험 결과 (2026-10-28) — **효과 0**

| 방법 | 보존률 | Prec(ns) | F1(ns) | AUROC | AUPRC(ns) |
|---|---|---|---|---|---|
| `CoST-W144-noscaleshift` | .990 | .877 | .913 | .978 | .853 |
| `CoST-W144-dirgate` | .990 | .878 | **.914** | .979 | .854 |
| 차이 | +.0001 | +.0004 | **+.0009** | +.0006 | +.0017 |

전 지표가 소수 셋째 자리 차이로, 표준편차(±0.099)의 1% 수준이다. fold 별로도 동일하다.

### 10.1 원인 — 학습이 게이트를 쓰지 않았다

| | 값 |
|---|---|
| FiLM `\|weight\|` 평균 | **0.00133** |
| fold 별 편차 | 0.00132 ~ 0.00134 (**거의 동일**) |

영 초기화에서 거의 안 움직였고, **fold 가 달라도 값이 같다** — 데이터에 반응한 것이 아니라
gradient noise 만 누적된 것이다.

### 10.2 교훈

> **상태 정보를 구조로 넣어도, 목적함수가 요구하지 않으면 학습이 버린다.**

CoST 손실은 "같은 run 의 두 증강본을 가깝게" 만 요구하는데, s̄ 는 두 증강본에서 거의 같은 값이라
(jitter 만 걸므로) 인스턴스 판별에 **도움이 0** 이다. 오히려 FiLM 을 움직이면 표현이 흔들려
손실이 나빠진다. 그래서 학습이 0 에 머문다.

재료 자체는 강하다 — s̄ 단독 AUROC **0.912**, 진폭 Q5 통제 후 **0.931**.
**정보가 없어서가 아니라 쓸 이유를 못 줘서** 안 썼다.

이 결과가 **아이디어 03**(같은 s̄ 를 구조가 아니라 **손실**에 넣는다)의 직접적 근거다.
`docs/idea/03_direction_aux_loss.md` 참조.
