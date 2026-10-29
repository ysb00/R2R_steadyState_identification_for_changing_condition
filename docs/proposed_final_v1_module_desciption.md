# Proposed (final v1) — 모듈별 정의 · 수식 · 코드 · R2R 적용 의미

제안 방법은 CoST 에 **네 가지**를 더한 것이다. 이 문서는 각 모듈을 **이 연구를 처음 보는 사람이
이해할 수 있도록** 정의 → 수식 → 코드 → **Roll-to-Roll(R2R) 장비에서 왜 효과적인가** 순으로 설명한다.

$$\text{Proposed (final v1)} \;=\; \text{CoST} \;+\; \underbrace{\text{① 증강 제거}}_{\text{no scale/shift}}
\;+\; \underbrace{\text{② trunk 분리}}_{\text{dual trunk}}
\;+\; \underbrace{\text{③ 차수 추적}}_{\text{order tracking}}
\;+\; \underbrace{\text{④ seasonal 동결}}_{\text{frozen seasonal}}$$

---

## 0. 먼저 — 풀려는 문제와 CoST 의 구조

### 0.1 R2R 공정에서 "정상상태 식별" 이 왜 필요한가

Roll-to-Roll 은 필름을 롤에서 풀어 가공하고 다시 감는 연속 공정이다. 필름이 찢어지거나 주름지지
않도록 **웹 장력(web tension)** 을 설정값에 맞춰 제어한다. 장력 설정값이나 웹 속도를 바꾸면
제어 루프가 새 설정값으로 **다시 수렴**하는데, 그 과도구간(transient)에서는 장력이 설정값
아래로 처지거나(sag) 위로 넘어간다(overshoot).

**이 과도구간의 데이터를 품질 지표에 섞으면 안 된다.** 그래서 "지금이 정상상태인가" 를 자동으로
판정해야 하고, 그 판정의 두 가지 실패가 각각 산업적 의미를 갖는다.

| 지표 | 수식 | 공정에서의 뜻 |
|---|---|---|
| **오염률** Contamination | $FP_{st}/(TP_{st}+FP_{st})$ | 정상이라고 판정해 쓴 데이터 중 **실제로는 과도구간**인 비율 → 품질 판정이 오염된다 |
| **보존률** Retention | $TP_{st}/P_{st}$ | 실제 정상 구간 중 **살려낸** 비율 → 낮으면 쓸 수 있는 데이터를 버린다 |

입력은 **설정값 기준 장력 오차** 단일 채널이다.

$$e(t) \;=\; T_1(t) - T^{\rm set}$$

### 0.2 CoST 의 구조 — 왜 이 모델을 골랐나

CoST 는 시계열 표현을 **추세(trend)** 와 **주기(seasonal)** 로 분해해 학습하는 자기지도 모델이다.
R2R 장력 신호가 바로 그 두 성분으로 되어 있어서 구조가 맞는다.

- **추세** = 제어 루프가 설정값으로 수렴하는 궤적 (과도구간의 정체)
- **주기** = 롤러가 돌면서 만드는 반복 외란 (정상상태에도 항상 있다)

```
e(t) ──► input_fc ──► DilatedConvEncoder (depth 10)  ◄── "trunk" (= backbone)
                              │  x : (B, T, 320)
              ┌───────────────┴───────────────┐
          tfd (시간 영역)                  sfd (주파수 영역)
       Conv1d 8종 평균                 BandedFourierLayer
              │                                │
         trend (B,T,160)                 season (B,T,160)

표현 = concat( trend[:, -1], season[:, -1] )  = 320차원
손실 = L_MoCo(trend 만) + α·½·L_season(seasonal 만),   α = 0.0005
```

용어: `tfd` = **T**rend **F**eature **D**isentangler, `sfd` = **S**easonal **F**eature **D**isentangler.
**두 분리기가 trunk 하나를 공유한다** — 이것이 ② 가 건드리는 지점이다.

---

## ① 증강 제거 (no scale / shift)

### 정의

CoST 는 자기지도 학습을 위해 같은 구간에 **무작위 변형(augmentation)** 을 두 번 걸어 "같은 것"
쌍을 만든다. 원본은 세 가지를 순서대로 적용한다.

$$\mathcal{T}_{\rm orig}(x) \;=\; \text{jitter}\big(\text{shift}(\text{scale}(x))\big)$$

$$\text{scale}(x) = x\cdot(1+\sigma\varepsilon_1),\qquad
\text{shift}(x) = x + \sigma\varepsilon_2,\qquad
\text{jitter}(x) = x + \sigma\boldsymbol{\varepsilon}_3$$

$\varepsilon_1,\varepsilon_2 \sim \mathcal N(0,1)$ 는 **창 전체에 하나씩** 뽑는 스칼라,
$\boldsymbol\varepsilon_3 \sim \mathcal N(0,1)^T$ 는 **시점마다** 뽑는 벡터다 ($\sigma = 0.5$,
각 변형이 적용될 확률 $p=0.5$).

제안 방법은 **`scale` 과 `shift` 를 제거하고 `jitter` 만 남긴다.**

$$\mathcal{T}_{\rm ours}(x) \;=\; \text{jitter}(x)$$

### R2R 에서 왜 효과적인가 — **이 두 변형이 지우는 것이 바로 정답이다**

자기지도 학습의 증강은 "이 변형을 받아도 같은 것으로 보라" 는 **불변성(invariance)** 을 가르친다.
그래서 **무엇을 변형하느냐가 모델이 무엇을 무시하도록 배우느냐**를 결정한다.

| 변형 | 하는 일 | R2R 에서의 의미 |
|---|---|---|
| `shift` | 창 전체에 상수를 더한다 | **"설정값에서 얼마나 떨어져 있나" 를 지운다** |
| `scale` | 창 전체에 상수를 곱한다 | **처짐의 깊이(크기)를 지운다** |
| `jitter` | 시점마다 독립 잡음 | 센서 잡음에 대한 둔감함 — **지켜야 할 불변성** |

$e$ 는 **이미 설정값을 뺀 오차**다. 그래서 $e$ 의 크기 자체가 "지금 장력이 설정값에서 얼마나
벗어나 있는가" 라는 **물리량**이다. 0.15 kg 처짐과 0.02 kg 처짐은 **다른 장비 상태**이고,
전자는 과도구간, 후자는 정상상태다. `shift`·`scale` 불변성은 "모양만 보고 분류" 하는 과제
(예: 파형 종류 구분)에는 맞지만, **크기 자체가 정답인 이 과제에는 정반대**다.

평가 후단도 같은 가정 위에 있다 — 2-군집 GMM 에서 **$|e$ 평균$|$ 이 작은 군집을 steady 로 지정**
한다. 오프셋을 지운 표현에서는 이 규칙이 성립할 근거가 사라진다.

### 측정된 효과 — **저속에서 결정적이다**

| $V$ (mm/s) | 50 | 75 | 100 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|
| AUROC 변화 | **+0.0707** | +0.0177 | +0.0002 | −0.0002 | +0.0001 | +0.0025 |
| F1(ns) 변화 | **+0.0816** | +0.0174 | −0.0047 | −0.0024 | −0.0034 | +0.0011 |

**고속에서는 거의 변화가 없고 저속(V=50)에서만 압도적으로 개선된다.** 이유가 물리에 있다.

창 144 는 $V=50$ 에서 **정확히 롤러 1회전**이다. 회전 외란이 창 안에서 한 번만 돌므로 창 내
변동 통계가 회전 위상에 좌우되어, 과도구간의 요동과 구별되지 않는다. 실측:

| $V$ | 창 144 의 회전 수 | movstd steady | movstd 과도 | 대비 | **요동으로 가른 AUROC** | **오프셋으로 가른 AUROC** |
|---|---|---|---|---|---|---|
| **50** | **1.00** | 0.0388 | 0.0467 | **1.20배** | **0.890** | **0.9991** |
| 100 | 2.00 | 0.0388 | 0.0612 | 1.58배 | 0.968 | 0.9915 |
| 200 | 4.01 | 0.0381 | 0.1314 | 3.45배 | 0.996 | 0.9946 |

**저속에서는 요동으로 상태를 가를 수 없고 오프셋으로만 가를 수 있다.** `shift` 가 지우던 것이
바로 그 오프셋이다. 참고로 $V=50$ 에서 창 평균 오차의 분포는 steady 5% 분위 $-0.0519$,
과도 95% 분위 $-0.0963$ 로 **90% 구간이 겹치지 않는다.**

### 코드

원본 (`code/03_baselines/external/CoST/cost.py`, **수정하지 않는다**)

```python
class PretrainDataset(Dataset):
    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))     # ← scale, shift 가 오프셋을 지운다
```

제안 방법 — 런타임 서브클래스로 갈아 끼운다 (`04_cost_dual_trunk.py`)

```python
class JitterCh0Dataset(cost_module.PretrainDataset):
    # 역할 : baseline 과 같은 'jitter only' 증강을 쓰되, 입력 2번째 채널에
    #        실어 보낸 속도 값이 잡음에 오염되지 않게 **채널 0 에만** 적용한다.
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])            # 채널 0(오차 e)에만 jitter
        return out
...
cost_module.PretrainDataset = JitterCh0Dataset            # 학습 직전에만 바꿔 끼운다
```

채널 0 에만 적용하는 이유: 입력이 2채널(`e`, 속도 $V$)이고 ③ 이 $V$ 를 써야 하는데,
$V$ 에 잡음이 섞이면 order window 길이가 흔들린다.

---

## ② trunk 분리 (dual trunk)

### 정의

원본은 `tfd`(시간 영역)와 `sfd`(주파수 영역)가 **trunk 하나를 공유**한다. 제안 방법은
**trunk 를 두 개로 나누어** 각 분리기에 전용 trunk 를 준다.

$$\textbf{원본: } \quad x = g_\theta(e),\qquad
h_T = \mathrm{tfd}(x),\qquad h_S = \mathrm{sfd}(x)$$

$$\textbf{제안: } \quad h_T = \mathrm{tfd}\big(g_{\theta_T}(e)\big),\qquad
h_S = \mathrm{sfd}\big(g_{\theta_S}(e)\big),\qquad \theta_T \neq \theta_S$$

손실 구조는 **건드리지 않는다.** CoST 의 두 손실 항이 이미 분기별로 갈라져 있기 때문이다.

$$\mathcal{L} \;=\; \underbrace{\mathcal{L}^{\rm MoCo}(h_T)}_{\theta_T \text{ 만 학습}}
\;+\; \alpha\cdot\tfrac12\underbrace{\mathcal{L}_{\rm season}(h_S)}_{\theta_S \text{ 만 학습}}$$

최종 표현은 그대로 320차원이다 ($h_T[:,-1]$ 와 $h_S[:,-1]$ 를 이어 붙임) — **평가 후단이 동일**하다.

### R2R 에서 왜 효과적인가 — **두 성분이 물리적으로 다른 축에 산다**

R2R 장력 신호의 두 성분은 **고정되는 축이 서로 다르다.**

| 성분 | 물리적 원인 | **고정된 축** | 측정 근거 |
|---|---|---|---|
| 추세 (과도구간 정착) | 장력 제어 루프의 수렴 | **시간** | 지속시간 9.4–17.8 s. 속도와 거의 무관 |
| 주기 외란 | 롤러 편심 · 베어링 결함 · 접합부 | **각도 (회전)** | $f_{\rm rot} = V/C$ 가 실측 피크와 0.006 Hz 이내 일치 |

과도구간의 지속시간이 **각도 고정이 아니라 시간 고정**임을 측정으로 확인했다.

| $V$ | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| 과도구간 최장 길이 (s) | 17.78 | 14.31 | 11.60 | 9.40 | 9.45 | 10.04 | **13.67** |
| 각도 고정이면 ($\propto 1/V$) | 17.78 | 11.86 | 8.89 | 7.11 | 5.93 | 5.08 | **4.45** |

$V=200$ 에서 예측 4.45 s vs 실측 13.67 s — **각도 고정 가설은 기각**이다. 제어 루프의 정착
시간이 결정하는 양이니 당연하다.

**문제는 trunk 를 공유하면 둘 중 한 축에만 맞춰진다는 것이다.** $\alpha = 0.0005$ 로 seasonal
손실이 매우 작게 눌려 있어, 공유 trunk 는 거의 전적으로 `L_MoCo`(추세, **시간축**)가 만든다.
그런데 ③ 의 차수 분리기는 그 특질을 **각도축으로 재표본화**해서 쓴다 — **시간축 전용으로
변형된 특질을 다른 축으로 비트는 것**이 충돌의 원인이다.

전용 trunk 를 주면 그 변형을 피한다. 장비 관점으로 말하면, **"제어 루프를 보는 눈" 과
"롤러를 보는 눈" 을 따로 두는 것**이다.

### 측정된 효과

차수를 고정한 상태로 공유 ↔ 분리만 바꾼 짝 비교 (같은 fold×seed, $n=25$):

| | F1(ns) | AUPRC(ns) | AUROC | 보존률 |
|---|---|---|---|---|
| 분리 − 공유 | **+0.0029** (18/7) p=.006 | **+0.0052** p=.002 | **+0.0032** (20/5) p=.001 | −0.0001 |

**용량 때문이 아니다** — 공유 trunk 를 폭 2배(`hidden_dims` 64→116, trunk 파라미터 1.99배)로
키우면 F1 −0.0018 (**p=0.49**, 무효)이고, 같은 용량에서 분리가 공유를 이긴다
(**+0.0047, 20/5, p=0.0002**). 공유 구조에서는 용량을 늘리는 것이 오히려 약간 해롭다.

**gradient 간섭 때문도 아니다** — seasonal 손실이 trunk 로 가는 gradient 를 완전히 차단해도
(stop-gradient, 단위 검증에서 7.05 → 0.000000) 결과가 바뀌지 않는다 (**p=0.88**).
문제는 gradient 의 방향이 아니라 **trunk 가 어디로 최적화되는가**다.

> **차수가 없으면 분리도 무효다** — 원본 Hz `sfd` 는 추세와 **같은 시간축**을 쓰므로 공유해도
> 무리가 없다. 차수 없는 분리는 +0.0018 (**p=0.13**). 그래서 ②는 ③과 짝으로만 의미가 있다.

### 코드

```python
class DualCoST(CoST):
    # 원본 fit 루프·옵티마이저·학습률 스케줄을 그대로 쓰고 **추론만** 바꾼다.
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        x = x.to(self.device, non_blocking=True)
        out_t, _ = self.net(x[:, -WINDOW_T:])     # trend : 전용 trunk
        _, out_s = self.net_s(x)                  # seasonal : 전용 trunk
        out = torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)   # 320차원 (원본과 같은 형태)
        return rearrange(out.cpu(), 'b d -> b () d')

    def save(self, fn):                           # 체크포인트에 두 trunk 를 담는다
        torch.save({'net': self.net.state_dict(), 'net_s': self.net_s.state_dict()}, fn)
```

```python
class DualCoSTModel(CoSTModel):
    # encoder_q / encoder_k 는 trend trunk 쌍이므로 MoCo 기구(모멘텀 갱신·큐)가 그대로 돈다.
    def __init__(self, *a, encoder_s, window_t, dim, **kw):
        super().__init__(*a, dim=dim, **kw)
        self.encoder_s = encoder_s                # seasonal 전용 trunk
        self.window_t = window_t

    def forward(self, x_q, x_k):
        # ---- trend 항 : 전용 trunk. 원본과 같은 연산 ----
        x_q_t, x_k_t = x_q[:, -self.window_t:], x_k[:, -self.window_t:]
        rand_idx = np.random.randint(0, x_q_t.shape[1])
        q_t, _ = self.encoder_q(x_q_t)
        q_t = Fnn.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_t, _ = self.encoder_k(x_k_t)
            k_t = Fnn.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)
        loss = self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        # ---- seasonal 항 : **별도 trunk**. 원본과 같은 연산 ----
        _, q_s = self.encoder_s(x_q)
        _, k_s = self.encoder_s(x_k)
        q_s, k_s = Fnn.normalize(q_s, dim=-1), Fnn.normalize(k_s, dim=-1)
        q_s_amp, q_s_phase = self.convert_coeff(fft.rfft(q_s, dim=1))
        k_s_amp, k_s_phase = self.convert_coeff(fft.rfft(k_s, dim=1))
        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        return loss + self.alpha * (seasonal_loss / 2)
```

---

## ③ 차수 추적 (order tracking) — 제안 방법의 본체

### 정의 — 차수란 무엇인가

**차수(order) = 회전 주파수의 배수로 센 주파수.** 차수 1 = 1회전당 1회, 차수 2 = 1회전당 2회.
주파수를 Hz 가 아니라 "회전 몇 번당 몇 주기" 로 세는 것이고, 회전기계 진동 분석의 표준 개념이다.

롤러 둘레 $C = 783.5$ mm 이고 웹 속도 $V$ 에서 롤러 회전 주파수는

$$f_{\rm rot} = \frac{V}{C}\ \ \text{[Hz]}$$

| $V$ (mm/s) | $f_{\rm rot}$ (Hz) | 1회전 주기 | 1회전 샘플 수 | **창 144 에서의 FFT bin** |
|---|---|---|---|---|
| 50 | 0.0638 | 15.7 s | 143.8 | **1** |
| 100 | 0.1276 | 7.8 s | 71.9 | **2** |
| 200 | 0.2553 | 3.9 s | 35.9 | **4** |

### R2R 에서 왜 효과적인가 — **같은 결함이 속도에 따라 다른 bin 으로 옮겨 다닌다**

R2R 설비에서 주기 외란의 원인은 **롤러 편심, 베어링 결함, 접합부(seam), 표면 불균일**처럼
**회전체에 붙어 있는 물리적 결함**이다. 이런 결함은 "**1회전당 1회**" 로 고정이다 — 속도를 바꿔도
회전당 한 번 나타난다. 그런데 Hz 로 보면 속도에 비례해 움직인다: $V=50$ 에서 0.064 Hz,
$V=200$ 에서 0.255 Hz 로 **4배**다.

**CoST 의 `sfd` 는 FFT bin 마다 독립된 복소 가중치를 학습한다.**

```python
self.weight = nn.Parameter(torch.empty((num_freqs, in_channels, out_channels), dtype=torch.cfloat))
...
output = torch.einsum('bti,tio->bto', input[:, start:end], self.weight)   # t = 주파수 bin 인덱스
```

즉 모델에게 **"bin 번호" 가 의미를 갖는다.** 그래서 저속에서 bin 1 의 가중치가 배운 "롤러 외란"
이 고속에서는 bin 1 에 없다 — **가중치가 엉뚱한 성분에 걸린다.**

**R2R 운영 관점에서 이것이 중요한 이유**: 제품·레시피가 바뀌면 웹 속도가 바뀐다. 한 속도에서
학습한 모델을 다른 속도에 그대로 쓰려면 주기 외란의 표현이 **속도에 불변**이어야 한다.
차수 축으로 재색인하면 같은 결함이 **전 속도에서 항상 차수 1** 에 놓인다.

이것이 이 연구의 최종 목표(저속 데이터만 학습 → 미학습 고속에 적용)와 직결된다.

### 수식 — FFT **전에** 시간축을 비튼다

핵심은 순서다. **스펙트럼을 사후에 변환하는 것이 아니라, 신호를 각도축으로 재표본화한 뒤 FFT 한다**
(회전기계 분야의 computed order tracking 과 같다).

창 길이 $W$, 속도 $V$ 의 trunk 출력 $x \in \mathbb{R}^{B\times W\times 320}$ 에 대해:

**1단계 — 뒤쪽 $N_{\rm rot}$ 회전분을 자른다** (창의 뒤쪽이므로 미래를 보지 않는다)

$$w(V) = \Big\lceil N_{\rm rot}\cdot \frac{C}{V\,\Delta t} \Big\rceil,\qquad
\tilde x = x[\,W-w(V):\,W\,]$$

**2단계 — 고정 길이 $L_{\rm ORD}$ 로 선형 재표본화** (각도축으로 보냄)

$$\hat x[j] = \mathrm{interp}_{\rm linear}\big(\tilde x\big)\Big[\tfrac{j}{L_{\rm ORD}-1}\big(w(V)-1\big)\Big],
\qquad j = 0,\dots,L_{\rm ORD}-1$$

**3단계 — 원본 연산 그대로**

$$X = \mathrm{rfft}(\hat x),\qquad Y[k] = W_k X[k] + b_k,\qquad h_S = \mathrm{irfft}(Y)$$

재표본화 뒤에는 모든 속도의 구간이 "$N_{\rm rot}$ 회전 $= L_{\rm ORD}$ 샘플" 이므로

$$\boxed{\text{bin } k \;=\; \text{차수} \frac{k}{N_{\rm rot}}} \qquad\text{(속도와 무관)}$$

**차수 해상도** $= 1/N_{\rm rot}$ 차수/bin. final v1 은 $N_{\rm rot}=1$, $L_{\rm ORD}=36$ 이라
**1.00 차수/bin** 이고 정수 차수를 분해한다.

$L_{\rm ORD}$ 는 **가장 빠른 속도 기준**으로 잡는다: $L_{\rm ORD} = N_{\rm rot}\cdot
\mathrm{round}\big(C/(V_{\max}\Delta t)\big) = 36$. 더 크게 잡으면 고속 구간을 **업샘플**하게
되고, "윗 bin 이 보간으로 채워졌다" 는 사실 자체가 속도 단서가 되어 모델이 우회할 수 있다.

속도별로 자르는 길이 ($N_{\rm rot}=1$):

| $V$ (mm/s) | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| $w(V)$ 샘플 | 144 | 96 | 72 | 58 | 48 | 42 | 36 |
| → 재표본화 후 | 36 | 36 | 36 | 36 | 36 | 36 | 36 |

### 전제의 측정 — 롤러 외란이 실제로 거기 있다

| $V$ | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| $f_{\rm rot}$ 예측 ($V/783.5$) | .0638 | .0957 | .1276 | .1595 | .1914 | .2234 | .2553 |
| **실측 스펙트럼 피크** | .0627 | .0986 | .1254 | .1613 | .1971 | .2240 | .2598 |
| 피크 두드러짐 (중앙값 대비 배) | 935 | 1046 | 1260 | 1329 | 623 | 933 | 777 |

**오차 0.006 Hz 이내, 피크가 중앙값 파워의 600–1300배.** 차수 축으로 옮기면 피크 bin 이
속도별로 흩어져 있던 것이 **하나로 모인다.**

### 측정된 효과

| | F1(ns) | AUPRC(ns) | 보존률 | AUROC |
|---|---|---|---|---|
| 차수 − baseline (공유 trunk) | **+0.0124** (19/6) p=.006 | **+0.0191** (20/5) p=.003 | **+0.0033** (**25/0**) | −0.0039 (n.s.) |

**모든 fold 에서 보존률이 개선**되고(25/25), 가장 어려운 fold 에서 가장 크게 좋아진다
(fold 1: F1 0.724→0.765).

속도별로 보면 **baseline 이 가장 약한 속도에서 가장 크게 듣는다.**

| $V$ | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|
| baseline F1(ns) | .9417 | .9868 | .9755 | **.9052** | .9269 | .9542 | .9693 |
| 차수의 F1 변화 | +0.0114 | +0.0018 | +0.0178 | **+0.0345** | +0.0188 | +0.0044 | **−0.0076** |

> **정직하게 적어둘 한계** — "속도 불변성" 으로 서술하면 안 된다. 속도 불변이면 전 속도에
> 고르게 들어야 하는데 실제로는 $V=125$ 에 몰리고 $V=200$ 은 손해다. 원인 후보: order window 의
> **시간 범위가 속도에 반비례**해서 ($V=50$ 15.7 s → $V=200$ 3.9 s) 고속에서는 과도구간의
> 29% 만 덮는다. 그리고 ②에서 본 대로 **과도구간 자체는 시간 고정**이라, 차수 축으로 전부
> 재색인하면 방해물(롤러 외란)은 고정되는 대신 **표적(과도구간)이 bin 사이를 움직인다.**
> Hz 분기를 되살려 더하는 하이브리드를 시험했으나 order 를 희석해 더 나빠졌다 (−0.0075, 3/22).

### 코드

```python
class OrderBandedFourierLayer(BandedFourierLayer):
    # trunk 출력의 뒤쪽 N_ROT 회전분을 잘라 고정 길이 L_ORD 로 선형 재표본화한 뒤,
    # **원본과 똑같은 연산**(rfft -> bin별 복소 가중 -> irfft)을 수행한다.
    # 인과성 : 창의 뒤쪽을 쓰므로 미래 시점을 보지 않는다.
    def forward(self, input):                                   # input: B x T x in_channels
        v = self.speeds                                         # (B,) — 인코더가 넣어 준다
        b, t, _ = input.shape
        out = input.new_zeros(b, self.length, self.out_channels)
        for s in torch.unique(v):                               # 속도마다 창 길이가 다르므로 묶어서 처리
            m = v == s
            w = min(order_window(float(s)), t)
            seg = input[m, t - w:]                              # ① 뒤에서 N_ROT 회전분
            seg = Fnn.interpolate(seg.transpose(1, 2), size=self.length,
                                  mode='linear', align_corners=True).transpose(1, 2)   # ② 각도축
            out[m] = super().forward(seg)                       # ③ 원본 연산 그대로
        return out


class OrderCoSTEncoder(CoSTEncoder):
    # 입력 (B, T, 2) 에서 속도 채널을 분리해 seasonal 분기에 전달하고,
    # 채널 0(장력 오차 e) 만 원본 forward 에 넘긴다 -> trunk 는 e 1채널만 본다.
    def __init__(self, *a, l_ord, **kw):
        super().__init__(*a, **kw)
        self.sfd = nn.ModuleList([OrderBandedFourierLayer(
            self.output_dims, self.component_dims, 0, 1, length=l_ord)])   # Hz sfd 를 교체

    def forward(self, x, **kw):
        # 패딩 구간은 두 채널 모두 NaN 이므로 시간축 최댓값으로 속도를 읽는다 (V > 0 이라 안전).
        self.sfd[0].speeds = torch.nan_to_num(x[:, :, 1], nan=0.0).amax(dim=1)
        return super().forward(x[:, :, :1].contiguous(), **kw)
```

`order_window(v)` 와 `rot_samples(v)`:

```python
def rot_samples(v):
    # 속도 v 에서 회전체 1바퀴에 해당하는 샘플 수. f_rot = v / C 의 역수를 샘플로 환산.
    return (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / v

def order_window(v):
    # 속도 v 에서 N_ROT 회전을 담는 창 길이 (샘플). N_ROT=1 이면 V=50 -> 144, V=200 -> 36
    return int(np.ceil(N_ROT * rot_samples(v)))
```

> **알려진 구현 한계** — `F.interpolate(mode='linear')` 는 앤티앨리어싱을 하지 않는다.
> $V=50$ 에서 144→36 의 4배 다운샘플이라 차수 18~72 가 0~18 로 접힌다. 롤러 외란(차수 1)이
> 중앙값의 600~1300배라 상대적 영향은 작을 것이고 실제로 저속 AUROC 가 개선되므로 해롭다는
> 증거는 없으나, 아직 재지 않았다 (추후 할일).

---

## ④ seasonal 분기 동결 (frozen seasonal)

### 정의

seasonal 전용 trunk 와 차수 `sfd` **전체를 무작위 초기값에 고정**한다.

$$\theta_S \;=\; \theta_S^{(0)} \quad (\text{학습하지 않음}),\qquad
\mathcal{L} = \mathcal{L}^{\rm MoCo}(h_T;\theta_T) + \alpha\cdot\tfrac12\mathcal{L}_{\rm season}(h_S;\theta_S^{(0)})$$

seasonal 손실 항은 계산되지만 학습 가능한 파라미터가 없어 **최적화에 대해 상수**가 된다.
그래서 실질적으로 trend 분기만 학습되고, 표현의 절반(160/320)은 **고정된 무작위 변환**에서 나온다.

$$h_S \;=\; \mathrm{irfft}\Big(W^{(0)}\cdot\mathrm{rfft}\big(\mathrm{resample}_{\rm order}(g_{\theta_S^{(0)}}(e))\big)\Big)$$

### R2R 에서 왜 효과적인가 — **주기 외란은 장비의 고정된 물리 속성이다**

롤러의 주기 외란은 **학습해서 알아내야 하는 패턴이 아니다.** 롤러 둘레 $C$ 와 속도 $V$ 로
"어디에 나타날지" 가 **이미 결정되어 있고**(차수 1), ③ 의 각도축 재표본화가 그것을
**결정론적으로** 그 자리에 놓는다. 남은 일은 "그 자리에 에너지가 얼마나 있는지" 를 재는 것이고,
**고정된 무작위 투영으로도 충분하다** (random feature).

반대로 **학습시키면 해롭다**는 것이 측정됐다. $\alpha = 0.0005$ 로 seasonal 손실이 매우 약하게
눌려 있어서, 전용 trunk 를 학습시키면 그 약한 신호가 trunk 를 조금씩 끌어당겨 **어설픈 중간
상태**가 된다 — 무작위도 아니고 제대로 최적화된 것도 아닌 상태다.

분리된 seasonal trunk 의 **데이터 기인 이동량이 0.00022** 로 trend trunk(0.00232)의 1/10 이었다
(같은 seed 에서 fold 를 바꿀 때의 상대 L2 거리). 그 "조금씩" 의 크기가 바로 어설픈 중간 상태의
정도이고, **아예 무작위로 두는 것이 더 깨끗하다.**

**산업적 의미 두 가지**

1. **학습 용량이 줄어든다.** gradient 를 받는 파라미터가 **13.69 M 으로 baseline(17.44 M)보다
   21% 적다.** 성능이 가장 좋은 변형이 **가장 적게 배운다** — "파라미터를 늘려서 좋아진 것"
   이라는 반론이 원천적으로 성립하지 않고, 현장 배포 시 학습 비용도 작다.
2. **설계가 단순해진다.** 주기 외란 쪽은 학습 대상이 아니라 **장비 사양($C$)에서 유도되는
   고정 변환**이다. 롤러를 교체하면 $C$ 만 바꾸면 되고 재학습이 필요 없다.

### 측정된 효과

동결 ↔ 학습만 바꾼 짝 비교 (같은 fold×seed, $n=25$):

| | F1(ns) | AUPRC(ns) | AUROC | 보존률 |
|---|---|---|---|---|
| 동결 − 학습 | **+0.0018** (18/7) p=.011 | **+0.0033** (19/6) p=.007 | **+0.0032** (**24/1**) **p<.0001** | −0.0003 (0/22) p<.0001 |

예측은 "얼려도 같을 것" 이었는데 **유의하게 더 좋다.** AUROC 24/25 개선이 가장 선명하다.

유일한 역방향 신호는 보존률 −0.0003 으로, 크기는 무시할 수준이지만 **0/22 (동점 제외 전부 악화)**
라 방향이 체계적이다 — 논문 표에 함께 적는다.

> **차수가 없으면 동결도 무효다** — 분리 + 동결 + **차수 없음** 은 baseline 과 같다
> (+0.0001, 13/12, **p=1.00**). 즉 ③ 이 ②·④ 를 작동시키는 전제다. 세 요소가 모두 필요하다.

### 코드

```python
if FREEZE:        # seasonal trunk 를 초기값에 동결한다.
    for p in m.net_s.parameters():       # trunk + 차수 sfd + (쓰이지 않는 tfd) 전부
        p.requires_grad = False
```

원본 `fit` 의 옵티마이저가 `requires_grad` 로 걸러내므로 **학습 루프를 고치지 않아도** 된다.

```python
optimizer = torch.optim.SGD([p for p in self.cost.parameters() if p.requires_grad],
                            lr=self.lr, momentum=0.9, weight_decay=1e-4)
```

> **동결 범위를 정확히** — `m.net_s` 는 seasonal **인코더 전체**다. 따라서
> trunk(637,248) + 차수 `sfd`(975,840) + 쓰이지 않는 `tfd`(13,057,280) 가 모두 고정된다.
> 각도축 재표본화도, 차수 bin별 복소 가중치도 전부 초기값이다.

---

## 5. 네 모듈의 조립 — 전체 구조

```
 입력 (B, T, 2)        ch0 = e = T1 - T_set      ch1 = V (웹 속도)
   │
   │  ① 증강: jitter 를 ch0 에만 (scale·shift 제거)         [학습 시]
   │
   ├──────────────── trend 분기 ────────────────┐   ② trunk 분리
   │  Ch0CoSTEncoder (ch1 제거)                 │
   │  trunk_T (학습)  →  tfd  →  h_T (B,T,160)  │   ← L_MoCo 가 학습
   │                                            │
   └──────────────── seasonal 분기 ─────────────┤
      OrderCoSTEncoder (ch1 에서 V 를 읽음)      │   ④ **전체 동결**
      trunk_S (고정)  →  차수 sfd (고정)         │   ③ 각도축 재표본화 후 FFT
                      →  h_S (B,36,160)         │
                                                │
   표현 = concat( h_T[:,-1], h_S[:,-1] ) = 320 ─┘
   → StandardScaler + GMM(2, diag) → |e 평균| 작은 군집 = steady
```

**바꾸지 않은 것**: 창 144 · 표현 320 · 커널 · $\alpha$ · lr · 배치 · 손실 구조 ·
평가 후단(GMM) · 분할 · 지표 · **학습량(600 iteration, 데이터 통과 3.17회)**.
그래서 baseline 과의 차이가 위 네 모듈로만 설명된다.

**원본 저장소는 한 줄도 수정하지 않는다** — 전부 런타임 서브클래스 shim 이다
(`salesforce/CoST`, commit `7bb5271c`).

**관련 문서**: 실험 설정·지표 수식은 `proposed_final_v1_experimental_settings.md`,
ablation 효과는 `proposed_final_v1_ablation_effect.md`.
