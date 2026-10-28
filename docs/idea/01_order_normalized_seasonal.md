# 아이디어 01 — 회전 차수(order) 축으로 재색인한 seasonal 분기

**제안 방법 이름**: `CoST-W288-order`
**기준(baseline)**: `CoST-W144-noscaleshift`
**구현**: `code/04_proposed/01_cost_order.py`
**결과**: `outputs/<평가방식>/09_proposed_cost_order_normalized_seasonal/`

---

## 1. 한 줄 요약

> **CoST 의 seasonal 분기에서 주파수 축을 Hz 대신 "회전 차수"로 재색인한다.**

새 손실 항도, 새 분기도 없다. FFT 직전에 시간축을 한 번 비트는 것이 전부이고,
나머지 변화는 모두 그 결과다.

---

## 2. 무엇이 문제인가

CoST 의 `BandedFourierLayer` 는 **주파수 bin 마다 독립적인 복소 가중치**를 학습한다.
이것은 "bin $k$ 가 모든 샘플에서 같은 것을 뜻한다" 는 가정 위에 서 있다.
**본 공정에서는 이 가정이 깨진다.**

steady 구간에는 지름 약 249 mm(둘레 $C = 783.5$ mm) 고정 회전체의 1회전 외란이 있고,
그 주파수는 웹 속도 $V$ 에 비례한다.

$$f_{\text{rot}}(V) = \frac{V}{C}$$

창 길이 144 샘플($\Delta t = 0.109$ s, 분해능 $1/(144\,\Delta t) = 0.0637$ Hz)에서
이 외란이 놓이는 bin 은 다음과 같다.

| $V$ (mm/s) | $f_{\text{rot}}$ (Hz) | 현재 bin | 차수 축 bin |
|---|---|---|---|
| 50 | 0.0638 | **1.00** | 2 |
| 75 | 0.0957 | 1.50 | 2 |
| 100 | 0.1276 | **2.00** | 2 |
| 125 | 0.1595 | 2.50 | 2 |
| 150 | 0.1914 | **3.00** | 2 |
| 175 | 0.2234 | 3.51 | 2 |
| 200 | 0.2553 | **4.01** | 2 |

**같은 물리적 외란이 bin 1 에서 bin 4 로 옮겨 다닌다.** 저속에서 학습한 bin 1 가중치가
고속에서는 엉뚱한 성분에 걸린다. 계절 대조손실 $\mathcal{L}_{\text{amp}}$, $\mathcal{L}_{\text{phase}}$ 도
bin 단위로 짝을 맞추므로 같은 문제를 겪는다.

이것은 `research_plan.md` §3.1 의 **D1**(속도 반영 구간 길이)과 **D3**(설정값 보정 정상성)이
지목한 지점이다.

---

## 3. 제안 — 차수 축 재색인

속도가 run 안에서 일정하므로, 각도(회전) 축으로의 재표본화는 **시간축의 배율 변환**과 같다.
회전기계 진단의 order tracking 을 정속 조건으로 단순화한 것이다.

### 3.1 수식

1회전에 해당하는 샘플 수:

$$P(V) = \frac{1}{\Delta t}\cdot\frac{C}{V}
\qquad P(50) = 143.8,\quad P(200) = 35.9$$

seasonal 분기가 보는 창 길이 (회전 수 $N_{\text{rot}}$ 고정):

$$W(V) = \lceil N_{\text{rot}}\, P(V) \rceil
\qquad W(50) = 288,\quad W(200) = 72$$

trunk 출력 $\mathbf{h} \in \mathbb{R}^{B\times T\times C}$ 의 **뒤쪽** $W(V)$ 시점을 잘라
(인과성 유지) 고정 길이 $L$ 로 선형 재표본화한다.

$$\tilde{\mathbf{h}}[b,\,m,\,c] \;=\; \mathrm{interp}\Big(\mathbf{h}\big[b,\,T-W(V_b):T,\,c\big]\Big)\Big|_{m},
\qquad m = 0,\dots,L-1$$

$$\tau(m) \;=\; m\cdot\frac{W(V_b)-1}{L-1}$$

재표본화 후 DFT 의 bin $k$ 는 **차수** $o$ 에 대응한다. 속도에 무관하다.

$$o(k) \;=\; \frac{k}{N_{\text{rot}}}
\qquad\Longrightarrow\qquad
f_{\text{rot}} \text{ 는 항상 } k = N_{\text{rot}}$$

그 위에 원본과 **같은** 연산을 수행한다.

$$\mathbf{s} \;=\; \mathcal{F}^{-1}\Big( \mathbf{W}\odot \mathcal{F}(\tilde{\mathbf{h}}) + \mathbf{b} \Big),
\qquad \mathbf{W}\in\mathbb{C}^{(L/2+1)\times C_{\text{in}}\times C_{\text{out}}}$$

즉 **가중치 $\mathbf{W}$ 의 $k$ 번째 성분이 "차수 $k/N_{\text{rot}}$" 라는 물리량에 묶인다.**

### 3.2 되돌리지 않는다

최종 표현은 `cat([trend[:, -1], season[:, -1]])` 로 **마지막 시점만** 쓴다.
차수 영역의 마지막 샘플은 원 시간축의 마지막 샘플과 같은 시점이므로
($\tau(L-1) = W(V)-1$), **시간축으로 되돌릴 필요가 없다.**

부수 효과로 계절 대조손실도 차수 영역에서 계산된다.

$$\mathcal{L}_{\text{season}} =
\tfrac12\Big[\mathcal{L}_{\text{inst}}\big(|\mathcal{F}(\mathbf{s}_q)|,\,|\mathcal{F}(\mathbf{s}_k)|\big)
+\mathcal{L}_{\text{inst}}\big(\angle\mathcal{F}(\mathbf{s}_q),\,\angle\mathcal{F}(\mathbf{s}_k)\big)\Big]$$

$\mathbf{s}\in\mathbb{R}^{B\times L\times 160}$ 이므로 bin 은 차수다 — **같은 차수끼리 대조**된다.

### 3.3 trend 분기는 건드리지 않는다

과도구간은 회전 수가 아니라 **시간**으로 정의된다(7–11 s 회복). 그래서 trend 는 원 시간축의
고정 창 288 을 그대로 본다. 변경은 seasonal 분기에만 국한된다.

---

## 4. 고정값과 그 근거

| 기호 | 값 | 근거 |
|---|---|---|
| $C$ | 783.5 mm | 기존 측정값 (205 run 평균, `periodic_disturbance.md`) |
| $N_{\text{rot}}$ | **2** | 진단 선택. 1 은 고속에서 1회전 = 3.9 s 뿐이라 상태 신호 부족(고속 AUPRC 0.249), 3 은 저속 창이 47 s 로 과도 |
| $W_{\text{in}}$ | **288** | $= \lceil 2P(50) \rceil$. 가장 느린 속도에서 2회전을 담는 길이 |
| $L$ | **72** | $= 2 \times \lfloor P(200) \rceil = 2\times 36$. **가장 빠른 속도 기준** — 업샘플로 bin 을 만들면 그 흔적 자체가 속도 단서가 된다(§6 측정) |
| 보간 | 선형 | 임의값을 늘리지 않는다 |
| 나머지 | `04_cost.py` 와 동일 | kernels [1..128], alpha 5e-4, lr 1e-3, batch 8, repr 320, n_iters 600 |

$V$ 는 **창 길이를 정하는 데만** 쓰이고 표현에 특징으로 들어가지 않는다.
같은 용법의 선례가 저장소에 있다 — `MovStd-Slope-speed-rule1b` 가
`w_rot = ceil((1/Δt)·C/V)` 로 창을 정한다 (`01_stat_ssd.py:163`).

---

## 5. 코드 변경점

원본 저장소는 **한 줄도 수정하지 않는다.** 전부 런타임 서브클래스 shim 이다.

### S1. 속도 채널을 실어 보낸다

배치 구성을 건드리지 않으려고 입력 텐서에 채널을 하나 더 둔다.
인코더 맨 앞에서 떼어내므로 **conv trunk 는 기존과 똑같이 e 1채널만 본다.**

```python
x: (n_run, T, 2)      # 채널 0 = e,  채널 1 = V (run 내 상수, 패딩 구간은 둘 다 NaN)
```

### S2. `PretrainDataset` — jitter 를 채널 0 에만

```python
class JitterCh0Dataset(cost_module.PretrainDataset):
    """baseline 과 동일하게 scale·shift 를 빼고 jitter 만 쓰되, 속도 채널은 건드리지 않는다."""
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])
        return out
```

### S3. `BandedFourierLayer` — 차수 축에서 동작

```python
class OrderBandedFourierLayer(BandedFourierLayer):
    #  length=L 로 생성 -> weight.shape = (L//2+1, 320, 160)
    def forward(self, input):                     # input: B x T x Cin
        v = self.speeds                           # (B,)  S4 가 넣어 준다
        out = input.new_zeros(input.size(0), self.length, self.out_channels)
        for s in torch.unique(v):                 # 속도별로 묶어 처리 (창 길이가 다르므로)
            m = v == s
            w = min(win_len(float(s)), input.size(1))
            seg = input[m, -w:]                                   # 뒤에서 W(V) 만큼 (인과)
            seg = F.interpolate(seg.transpose(1, 2), size=self.length,
                                mode='linear', align_corners=True).transpose(1, 2)
            out[m] = super().forward(seg)         # 원본 연산 그대로: rfft -> W -> irfft
        return out
```

### S4. `CoSTEncoder` — 속도 분리 후 기존 forward 호출

```python
class OrderCoSTEncoder(CoSTEncoder):
    def __init__(self, *a, l_ord, **kw):
        super().__init__(*a, **kw)
        self.sfd = nn.ModuleList([OrderBandedFourierLayer(
            self.output_dims, self.component_dims, 0, 1, length=l_ord)])

    def forward(self, x, **kw):
        # 패딩이 NaN 이므로 시간축 최댓값으로 속도를 읽는다 (V > 0)
        self.sfd[0].speeds = torch.nan_to_num(x[:, :, 1], nan=0.0).amax(dim=1)
        return super().forward(x[:, :, :1].contiguous(), **kw)   # <- 여기부터 원본과 동일
```

### 변경 요약

| 구성요소 | baseline | 제안 | 변경 |
|---|---|---|---|
| 입력 | e 1채널 | e 1채널 (+ V 운반 채널, trunk 진입 전 제거) | 운반만 |
| 증강 | jitter only | 동일 (채널 0 한정) | — |
| trunk | 창 144 | 창 288 | 창 |
| trend 분기 | 다중 커널 인과 conv | **동일** | — |
| **seasonal 분기** | **시간축** rfft(144) -> bin별 가중 | **차수축** rfft(72) -> bin별 가중 | ★ |
| `BandedFourierLayer` 가중치 | bin = Hz (속도마다 다른 물리량) | bin = 차수 (전 속도 동일) | ★ 결과 |
| $\mathcal{L}_{\text{amp}}$ / $\mathcal{L}_{\text{phase}}$ | 시간축 bin 끼리 | 같은 차수끼리 | ★ 결과 |
| MoCo trend 손실 | | 동일 | — |
| 평가 후단 | GMM | 동일 | — |

---

## 6. 사전 진단 (구현 전 측정, 재학습 없음)

선형 프로브로 스펙트럼 수준에서만 확인한 결과다. 저속(50–125) run 으로 학습하고
저속 held-out / 미학습 고속(150–200)에서 평가, 표준화+L2, 반복 10회.

| 설정 | bin | 저속 AUROC | 저속 AUPRC | 고속 AUROC | 고속 AUPRC |
|---|---|---|---|---|---|
| 주파수 · 고정 144 (baseline) | 72 | 0.937±.015 | 0.790±.034 | 0.953±.012 | 0.676±.066 |
| 주파수 · 고정 288 (창 대조) | 144 | 0.987±.005 | 0.879±.032 | **0.826**±.055 | **0.295**±.119 |
| 주파수 · 고정 72 (창 대조) | 36 | 0.709±.019 | 0.395±.030 | 0.919±.009 | 0.716±.033 |
| 차수 · 가변 $N_{\text{rot}}$=1, L=36 | 18 | 0.773±.015 | 0.580±.046 | 0.726±.004 | 0.249±.012 |
| **차수 · 가변 $N_{\text{rot}}$=2, L=72** | 36 | **0.989**±.003 | **0.939**±.031 | **0.979**±.002 | **0.925**±.006 |
| 차수 · 가변 $N_{\text{rot}}$=3, L=108 | 54 | 0.996±.002 | 0.946±.036 | 0.986±.002 | 0.916±.010 |

AUPRC 기준선(유병률): 저속 0.079 / 고속 0.084.

**확인된 것**

1. 회전 외란 실재 — 예측 $V/C$ 와 실측 피크가 전 속도에서 0.006 Hz 이내 일치,
   피크 두드러짐(최대 bin 파워 / 중앙값) **600–1300배**
2. 차수 정렬 작동 — 피크 bin 이 1~4 에서 **2 하나로** 모임
3. **창 길이 효과가 아니다** — 같은 길이(288) 고정창은 고속 AUPRC 0.295 로 오히려 붕괴.
   같은 길이를 저속에만 쓰는 차수 변형은 0.925
4. 분산 감소 — 고속 AUPRC 표준편차 ±0.066 → ±0.006

**기대치를 낮춰야 하는 이유**

실제 `CoST-W144-noscaleshift` 는 C `final_test` 에서 이미 AUPRC **0.908**, AUROC 0.990 이다.
위 진단의 차수 프로브(0.925)는 **라벨을 쓴 선형 판별**인데도 0.017 앞설 뿐이다.
즉 CoST 의 비선형 인코더는 주파수 영역에서도 이미 비슷한 수준에 도달해 있고,
차수 정규화가 그 위에 무엇을 더할지는 이 진단으로 알 수 없다. 두 시나리오가 가능하다.

- (A) CoST 가 이미 다른 경로로 그 정보를 얻고 있다 → 거의 안 오른다
- (B) bin 불일치로 손해를 보고 있다 → 오른다. 특히 학습 효율과 안정성 쪽

**가르려면 실제로 돌려보는 수밖에 없다.** 그것이 이 구현의 목적이다.

---

## 7. 신규성에 대한 솔직한 평가

**차수 추적 자체는 새롭지 않다.** 회전기계 진단의 표준 기법이고, 정속에서 시간축 배율
변환과 같다는 것도 알려져 있다. 새로운 것은 조합이다.

1. **자기지도 대조학습의 주파수 분기 안에** 넣은 것 — 보통은 전처리로 쓰며,
   "학습되는 bin별 가중치를 물리량에 묶는다" 는 동기로 쓰지 않는다
2. **고장 진단이 아니라 상태 구간 분할**에 적용
3. **속도 외삽 평가** 아래에서 — 차수 정렬의 이득이 드러나는 설정

논문 서술은 *"CoST 의 설계 가정(bin 의 의미가 샘플 간 동일)이 운전조건이 변하는 공정에서
깨지며, 도메인 물리로 그 가정을 복원한다"* 가 정직하다.

### 공정성 — 반드시 명시할 것

제안 방법은 **속도 설정값을 창 길이 결정에 사용한다.** 비교군 23종 중 같은 정보를 쓰는 것은
`MovStd-Slope-speed`, `MovStd-Slope-speed-rule1b` 2종이고 나머지 21종은 쓰지 않는다.
논문 표에 이 구분을 적어야 한다.

---

## 8. 실험 설계

| 방법 | 역할 |
|---|---|
| `CoST-W144-noscaleshift` | 기존 baseline (이미 실행됨, `06_cost_ablation`) |
| **`CoST-W288-noscaleshift`** | **창 대조군** — 창 144→288 효과를 분리한다 |
| **`CoST-W288-order`** | **제안 방법** |

창이 바뀌므로 대조군이 없으면 이득이 "창 때문" 인지 "차수 때문" 인지 가를 수 없다.

평가는 비교군과 **완전히 동일**하다 — `evaluation.py` 후단(학습 fold 2군집 diag GMM,
`|e 평균|` 작은 군집 = steady), `splits.py` 분할, 같은 지표·그림.

| 평가 방식 | 출력 |
|---|---|
| A (ambiguous 제외) | `outputs/03_baselines/09_proposed_cost_order_normalized_seasonal/` |
| C (ambiguous 제외) | `outputs/04_speed_extrap/09_proposed_cost_order_normalized_seasonal/` |
| A (ambiguous 포함) | `outputs/05_ambiguous_included_A/09_proposed_cost_order_normalized_seasonal/` |
| C (ambiguous 포함) | `outputs/06_ambiguous_included_C/09_proposed_cost_order_normalized_seasonal/` |

```
python code/04_proposed/01_cost_order.py tension        # A
python code/04_proposed/01_cost_order.py c              # C
python code/04_proposed/01_cost_order.py tension amb    # A, ambiguous 포함 (추론만)
python code/04_proposed/01_cost_order.py c amb          # C, ambiguous 포함 (추론만)
```

**변형 선택은 C `cv_S1`·`cv_S2` 에서 한다.** `final_test` 는 확정 후 한 번만 본다
(`split_description.md` §3.3).

---

## 9. 학습량 교란 (기록)

`n_iters = 600` 을 유지한다. 창이 288 이면 run 이 `1281 // 288 = 4` 조각으로 쪼개져
(창 144 는 8 조각) `data_passes` 가 baseline 의 약 절반이 된다.
**창 대조군(`CoST-W288-noscaleshift`)이 이 교란을 흡수**하므로, 제안 방법과 대조군의 차이는
학습량 효과가 아니다. baseline(W144)과의 비교에는 이 교란이 섞여 있다 —
`hyperparameters.md` §11 문제 2·9 와 같은 성질이다.


---

## 10. A 실험 결과 (2026-10-28 실행 완료)

| 방법 | 창 | 보존률 | Prec(ns) | F1(ns) | AUROC | AUPRC(ns) | 진입MAE |
|---|---|---|---|---|---|---|---|
| `CoST-W144-noscaleshift` (baseline) | 144 | .990 | .877 | .913 | .978 | .853 | 5.1 |
| **`CoST-W144-order`** (N_rot=1) | **144** | **.994** | **.907** | **.926** | .974 | **.872** | 5.9 |
| `CoST-W288-noscaleshift` (창 대조군) | 288 | .900 | .464 | .622 | .945 | .462 | 12.8 |
| `CoST-W288-order` (N_rot=2) | 288 | .960 | .694 | .797 | .972 | .688 | 6.6 |

**재현 검증 통과** — 새 어댑터의 `CoST-W144-noscaleshift` 가 기존 `06_cost_ablation` 값과
소수 셋째 자리까지 동일하다 (F1 −0.000, AUPRC +0.000).

### 10.1 N_rot=1 이 baseline 을 넘는다 — 짝지은 비교 (같은 fold×seed, n=25)

| 지표 | 평균차 | 개선/악화 | Wilcoxon p |
|---|---|---|---|
| F1(ns) | **+0.0124** | 19 / 6 | **0.0056** |
| AUPRC(ns) | **+0.0191** | 20 / 5 | **0.0025** |
| 보존률 | +0.0033 | **25 / 0** | 0.0000 |
| AUROC | −0.0039 | 11 / 14 | 0.2304 (차이 없음) |

fold 별로 전부 개선이고 **가장 어려운 fold 1 에서 가장 크다**:
F1 0.724→0.765 (+0.040), AUPRC 0.556→0.607 (+0.051).
표준편차도 줄었다 — F1 ±0.099→±0.085, AUPRC ±0.157→±0.141.

### 10.2 창 길이가 결정적이다

| 창 | 차수의 순수 기여 (order − noscaleshift) | 최종 (baseline 대비) |
|---|---|---|
| 144 | F1 **+0.012** / AUPRC **+0.019** | **+0.012 / +0.019** |
| 288 | F1 **+0.175** / AUPRC **+0.225** | −0.116 / −0.165 |

창 288 에서 차수 효과가 14배 크지만 **최종 성적은 창 144 가 낫다.**
창 288 자체가 F1 을 0.291 깎기 때문이다 — 31.4초 창이 과도구간(7~18초)보다 길어
과도 시점의 창에 steady 가 절반 이상 섞인다 (Prec(ns) 0.877→0.464, 진입 MAE 5.1→12.8 s).

> **차수 정규화는 망가진 seasonal 을 복구하는 힘이 크지만, 창을 망가뜨리면서까지 쓸 이유는 없다.**

### 10.3 §4 의 N_rot 선택은 틀렸다 — 정정

§4 에서 사전진단(고속 AUPRC 0.249)을 근거로 N_rot=1 을 기각하고 2 를 택했다.
**그 진단은 스펙트럼 단독을 잰 것이었다.** 실제 CoST 에는 trend 160차원이 함께 있어
seasonal 이 약해도 받쳐 준다. 진단의 적용 범위를 과하게 잡은 것이고,
**N_rot=1(창 144)이 권장 설정이다.**

### 10.4 다음

A 에서 baseline 을 넘었으므로 **C 실험·ambiguous 포함으로 확장할 가치가 있다** (§13.10 P6).
차수 정규화는 본래 **속도 외삽**을 겨눈 설계이므로 C 에서 효과가 더 클 가능성이 있다.
