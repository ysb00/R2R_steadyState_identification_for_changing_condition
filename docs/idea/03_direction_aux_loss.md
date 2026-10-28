# 아이디어 03 — 설정값 방향을 예측하는 보조 손실 (trend branch)

**제안 방법 이름**: `CoST-W144-saux-point` / `CoST-W144-saux-mean`
**기준(baseline)**: `CoST-W144-noscaleshift` — **창이 같다. 창 대조군 불필요**
**구현**: `code/04_proposed/03_cost_saux.py`
**결과**: `outputs/<평가방식>/11_proposed_cost_direction_aux_loss/`

---

## 1. 왜 아이디어 02 가 아니라 이것인가

아이디어 02 는 **같은 양**(설정값 접근 비율 $\bar s$)을 FiLM 게이트로 trend 표현에 주입했다.
결과는 **효과 0** 이었다.

| | baseline | dirgate | 차이 |
|---|---|---|---|
| F1(ns) | 0.913 | 0.914 | **+0.0009** |
| AUPRC(ns) | 0.853 | 0.854 | +0.0017 |
| FiLM `\|weight\|` (영 초기화) | — | **0.00133** | 거의 안 움직임 |

**원인은 구조가 아니라 목적함수였다.** CoST 의 손실은

$$\mathcal{L} = \mathcal{L}^{\text{MoCo}}_{\text{trend}} + \alpha\cdot\tfrac12\mathcal{L}_{\text{season}}$$

로 **"같은 run 의 두 증강본을 가깝게"** 만 요구한다. $\bar s$ 는 두 증강본에서 거의 같은 값이라
인스턴스 판별에 도움이 **0** 이고, FiLM 을 움직이면 오히려 손실이 나빠진다. 그래서 학습이 버렸다.

> **교훈**: 상태 정보를 구조로 넣어도, 손실이 그것을 요구하지 않으면 학습이 쓰지 않는다.
> `research_plan.md` §3.1 (3-1) 의 *"기존 표현학습에는 상태 구분 과제가 없다"* 를 직접 보여준 결과다.
>
> 그리고 재료 자체는 강하다 — $\bar s$ 단독 AUROC **0.912** (진폭 기준선 0.789, 진폭 통제 후 0.86–0.97).
> **정보가 없어서가 아니라 쓸 이유를 못 줘서** 안 쓴 것이다.

**아이디어 03 은 같은 양을 손실에 넣는다.**

---

## 2. 제안

$$\mathcal{L} \;=\; \mathcal{L}^{\text{MoCo}}_{\text{trend}}
\;+\; \alpha\cdot\tfrac12\mathcal{L}_{\text{season}}
\;+\; \boldsymbol{\beta\cdot\mathcal{L}_{\text{aux}}}$$

$$\mathcal{L}_{\text{aux}} \;=\; \big\lVert\, \mathbf{w}^\top \mathbf{h}_T \;-\; \bar s \,\big\rVert_2^2$$

trend 표현 $\mathbf{h}_T\in\mathbb{R}^{160}$ 에서 **선형 head 하나**로 $\bar s$ 를 예측하게 한다.
예측하려면 표현이 방향 정보를 **담을 수밖에 없다.**

head 를 선형으로 최소화하는 것이 핵심이다 — MLP 를 쓰면 head 가 혼자 일을 해버려
**표현이 안 바뀐다.** 선형이면 표현이 직접 선형 분리 가능해져야 한다.

### 2.1 $\bar s$ 의 정의 (아이디어 02 와 동일)

$$r(t) = -\,e(t)\,\dot e(t),\qquad
\bar s(t) = \frac{1}{W}\sum_{\tau\in[t-W+1,\,t]} \mathbb{1}\big[r_{\text{smooth}}(\tau) > 0\big]$$

- $r>0$ ⟺ $|e|$ 감소 중 **= 설정값 접근**. sag 회복과 overshoot 회복이 같은 부호로 묶인다
- 평활 길이 = **1회전** $P(V) = \frac{1}{\Delta t}\frac{C}{V}$ ($C=783.5$ mm) — 유도값, 임의값 0개
- 창 $[t-W+1,t]$ 가 전부 과거이므로 창 **안에서** 중심 평활을 써도 인과다
  (인과 이동평균은 지연 $P/2$ 때문에 V=200 에서 AUROC 0.912 → 0.599 로 붕괴. 실측)

측정값: $\bar s$ 평균 0.487, 표준편차 0.161. **steady 0.518 / transient 0.197**.

### 2.2 읽기 시점 2종 — 이번 실험의 비교 대상

| 변형 | $\mathcal{L}_{\text{aux}}$ | 성격 |
|---|---|---|
| **`point`** | $\big(\mathbf{w}^\top\mathbf{h}_T(t_r) - \bar s(t_r)\big)^2$, $t_r$ = MoCo 가 쓰는 무작위 시점 | MoCo 와 **같은 시점**만 제약. 약한 압력 |
| **`mean`** | $\frac{1}{T}\sum_t \big(\mathbf{w}^\top\mathbf{h}_T(t) - \bar s(t)\big)^2$ | **전 시점**을 제약. 강한 압력, 시간 해상도 유지 |

`mean` 은 "창 평균끼리 비교" 가 아니라 **시점별 오차를 전 시점에서 평균**한 것이다.
선형 head 에서는 $\overline{\mathbf{w}^\top\mathbf{h}} = \mathbf{w}^\top\overline{\mathbf{h}}$ 이므로
"평균끼리 MSE" 는 평균만 맞추면 되는 약한 제약이 된다 — 그래서 시점별 제약을 쓴다.

### 2.3 $\beta$ — 크기를 재서 정했다

| 항 | 실측 크기 |
|---|---|
| $\mathcal{L}^{\text{MoCo}}_{\text{trend}}$ | 0.045 → **3.25** (MoCo 큐가 차면서 70배 증가) |
| $\alpha\cdot\mathcal{L}_{\text{season}}$ | $\alpha = 0.0005$ — 저자가 이미 매우 작게 눌러 둠 |
| $\mathcal{L}_{\text{aux}}$ 초기값 | $\approx \mathrm{Var}(\bar s) = 0.026$ |

| 목표 | $\beta$ |
|---|---|
| 기존과 같은 크기 | ≈ 125 |
| **기존의 10%** | **≈ 12** ← 기본값 |
| 기존의 1% | ≈ 1.2 |

$\beta = 1$ 은 전체의 0.8% 밖에 안 되어 아이디어 02 와 같은 결과가 날 것이다.
**기본값을 12 로 둔다** — CoST 저자가 보조 항(seasonal)을 작게 눌러 둔 설계 철학과도 덜 충돌한다.

> ⚠️ $\mathcal{L}^{\text{MoCo}}$ 가 학습 중 70배 변하므로, 고정 $\beta$ 에서는 **초반에 보조 항이
> 지배하고 후반에 묻힌다.** 적응 가중을 쓰면 임의값이 줄지만 구현이 복잡하고 CoST 원본과 멀어져
> **고정 $\beta$ 를 쓴다.** $\beta$ 는 CLI 인자라 민감도 확인이 쉽다.

---

## 3. 코드 변경점

원본 저장소는 한 줄도 수정하지 않는다. `CoSTModel` 서브클래스 shim 이다.

### S1. $\bar s$ 를 채널로 실어 보낸다 (아이디어 02 와 동일)

```python
x: (n_run, T, 2)      # 채널 0 = e,  채널 1 = s̄(t)
```
인코더 맨 앞에서 떼어내므로 **trunk 는 e 1채널만 본다.**
증강 shim 이 jitter 를 채널 0 에만 적용해 $\bar s$ 가 오염되지 않는다.

### S2. `CoSTModel.forward` 에 보조 항 추가 ★ 본체

원본 `forward` 본문을 그대로 두고 **끝에 두 줄**을 더한다. 추가되는 계산은 선형 head 하나뿐이고
**인코더 forward 추가 호출이 없다** (화살표 판별 방식은 반전 입력으로 +1회가 필요하다).

```python
class SAuxCoSTModel(CoSTModel):
    def __init__(self, *a, beta, readout, **kw):
        super().__init__(*a, **kw)
        self.beta, self.readout = beta, readout
        self.aux_head = nn.Linear(kw['dim'], 1)      # 160 -> 1, 선형 최소 구조

    def forward(self, x_q, x_k):
        # ... 원본 본문 그대로 (trend MoCo + seasonal) ...
        # 추가 : trend 원시 표현에서 s̄ 를 예측
        s = x_q[..., 1]                               # (B, T) 목표값
        if self.readout == 'point':
            pred, tgt = self.aux_head(t_raw[:, rand_idx]).squeeze(-1), s[:, rand_idx]
        else:                                         # 'mean' — 전 시점
            pred, tgt = self.aux_head(t_raw).squeeze(-1), s
        m = torch.isfinite(tgt)                       # 패딩(NaN) 제외
        aux = F.mse_loss(pred[m], tgt[m])
        loss = loss + self.beta * aux
```

### 변경 요약

| 구성요소 | baseline | 제안 | 변경 |
|---|---|---|---|
| 입력 | e 1채널 | e 1채널 (+ $\bar s$ 운반 채널) | 운반만 |
| 창 | 144 | **144 (동일)** | — |
| trunk / trend / seasonal 분기 | | **전부 동일** | — |
| **손실** | MoCo + α·season | **+ β·MSE(선형(trend), $\bar s$)** | ★ |
| 평가 후단 | GMM | 동일 | — |
| 표현 차원 | 320 | **320 (동일)** | — |

표현 차원이 그대로라 **비교군과 완전히 같은 후단**을 쓴다.

### S3. 학습됐는지 측정하는 장치

아이디어 02 에서 `film_weight_absmean` 이 "안 썼다" 를 드러냈듯, 여기서는
**보조 손실의 수렴값**을 `fold_metrics.csv` 에 기록한다.

| 기록 | 뜻 |
|---|---|
| `aux_loss_first` | 초기값. $\approx \mathrm{Var}(\bar s) = 0.026$ 이어야 정상 |
| `aux_loss_final` | 마지막 20 iter 평균 |
| `aux_r2` | $1 - \mathcal{L}_{\text{aux}}/\mathrm{Var}(\bar s)$ — **표현이 $\bar s$ 를 얼마나 담았나** |

`aux_r2` 가 0 근처면 "표현이 안 담았다", 1 에 가까우면 "담았다" 가 확정된다.
성능이 안 올라도 **원인 구분이 된다.**

---

## 4. 고정값

| 항목 | 값 | 근거 |
|---|---|---|
| $C$ | 783.5 mm | 기존 측정값 |
| 평활 길이 | $P(V)$ = 1회전 | 유도값 |
| 창 $W$ | **144** | baseline 동일 |
| head | `Linear(160, 1)` | 최소 구조 (연구자 지시) |
| $\beta$ | **12** (CLI 인자) | $\mathcal{L}_{\text{aux}}$ 를 기존의 10% 로 두는 값 |
| readout | `point` / `mean` 2종 | 비교 대상 |
| 나머지 | `04_cost.py` 와 동일 | |

$V$ 는 평활 길이 결정에만 쓴다. 선례: `MovStd-Slope-speed-rule1b` (`01_stat_ssd.py:163`).

---

## 5. 위험 (미리 기록)

1. **보조 과제가 풀려도 다운스트림이 안 좋아질 수 있다.** 표현이 $\bar s$ 를 담아도 GMM 이 보는 축은
   steady/non-steady 다. 둘이 정렬된다는 보장은 없다 — 다만 $\bar s$ 의 AUROC 가 0.912 라
   상당히 정렬돼 있을 것으로 본다
2. **기존 손실과 경쟁한다.** $\beta$ 가 크면 인스턴스 판별이 망가져 표현 전체가 나빠질 수 있다
3. **$\beta$ 선택 절차가 비교군과 다르다.** 비교군은 As-published 였는데 $\beta$ 는 기본값이 없다.
   값을 바꿔 고를 경우 **C `cv_S1`·`cv_S2` 에서만** 고르고 `final_test` 는 보지 않는다.
   논문에 절차를 명시해야 한다
4. **속도 외삽 아이디어가 아니다.** $\bar s$ 는 저속 0.931 / 고속 0.894 로 전이에서 약간 떨어진다.
   상태 변별력 자체를 올리는 아이디어이고, 속도 축은 아이디어 01 이 담당한다

---

## 6. 실험 설계

| 방법 | 역할 |
|---|---|
| `CoST-W144-noscaleshift` | baseline (이미 실행됨, `06_cost_ablation`) |
| `CoST-W144-dirgate` | 아이디어 02 (구조만 주입 — 효과 0) |
| **`CoST-W144-saux-point`** | 제안. MoCo 와 같은 시점만 제약 |
| **`CoST-W144-saux-mean`** | 제안. 전 시점 제약 |

```
python code/04_proposed/03_cost_saux.py tension                  # A, beta=12 기본
python code/04_proposed/03_cost_saux.py tension beta=30          # 민감도
python code/04_proposed/03_cost_saux.py c                        # C
python code/04_proposed/03_cost_saux.py tension amb              # ambiguous 포함 (추론만)
```

| 평가 방식 | 출력 |
|---|---|
| A (ambiguous 제외) | `outputs/03_baselines/11_proposed_cost_direction_aux_loss/` |
| C (ambiguous 제외) | `outputs/04_speed_extrap/11_proposed_cost_direction_aux_loss/` |
| A·C (ambiguous 포함) | `outputs/0{5,6}_ambiguous_included_{A,C}/11_proposed_cost_direction_aux_loss/` |

$\beta$ 가 기본값이 아니면 방법 이름에 붙는다 (`CoST-W144-saux-point-b30`) — 같은 폴더에서 섞이지 않는다.
