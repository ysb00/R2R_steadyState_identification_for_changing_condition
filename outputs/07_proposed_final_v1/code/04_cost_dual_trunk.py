# =============================================================================
# 04_proposed/04_cost_dual_trunk.py
#
# ★★★ 이 스크립트가 **Proposed (final v1)** 을 만든다 (연구자 결정 2026-10-29) ★★★
#
#   실행      : python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze
#   방법 이름  : CoST-dual-T144-S144-frz
#   구성      : baseline(CoST-W144-noscaleshift) + trunk 분리 + seasonal 분기 동결 + 차수 정규화
#               (창은 trend·seasonal **둘 다 144**. 창 분리는 기각됐다 — 아래 참조)
#   결과      : outputs/03_baselines/12_proposed_cost_dual_trunk/CoST-dual-T144-S144-frz_*
#               사본과 논문 표는 outputs/07_proposed_final_v1/ (code/04_proposed/06_final_v1_tables.py)
#   A 실험    : F1(ns) .9303 / AUROC .9807 / 보존률 .9932
#               baseline 대비 F1 +0.0171 (22/3, Wilcoxon p<0.0001)
#               **gradient 를 받는 파라미터는 baseline 보다 21% 적다** (13.69M vs 17.44M)
#   현재까지의 최종판이고 추후 검증으로 개선해 나간다. 해석은
#   docs/idea/proposed_ideas_already_implemented.md 의 "전체 결과 요약" 과 §4.
#
#   같은 스크립트의 다른 조합 (ablation·기각된 변형)
#     nrot=1                 -> CoST-dual-T144-S144        분리 + 차수 (동결 없음)   +0.0153
#     nrot=1 noorder         -> CoST-dual-T144-S144-noord   분리만                   +0.0018 (p=.13)
#     nrot=1 noorder freeze  -> CoST-dual-T144-S144-noord-frz 분리 + 동결            (ablation)
#     nrot=2 (기본)           -> CoST-dual-T144-S288        분리 + **창 분리**        −0.0332 기각
#
# ---------------------------------------------------------------------------
# 원래의 설계 동기 (아이디어 05) — 창 분리 가설. **기각됐다.**
#
# 가설 : CoST 는 표현을 trend(시간 영역) + seasonal(주파수 영역) 으로 분해하면서
#   **두 영역에 같은 창(수용 영역)을 강제**한다. 그런데 두 영역이 원하는 창은 반대다.
#     - trend    : 창이 길면 나쁘다. 창 288 = 31.4 s 가 transient 7~18 s 보다 길어 경계가 흐려진다
#                  (실측: Prec(ns) 0.877 -> 0.464, 진입 MAE 5.1 -> 12.8 s)
#     - seasonal : 창이 길면 좋다. 아이디어 01 이 축을 차수에 묶은 뒤로는 창이 차수 해상도를
#                  직접 정한다 — 창 144 -> 1.00 차수/bin, 창 288 -> 0.50 차수/bin
#   시간 국소화와 주파수 해상도를 한 창으로 동시에 올릴 수 없다는 것은 불확정성 원리다.
#   다중해상도 분석이 쓰는 해법(영역마다 다른 창)을 CoST 의 두 분기에 적용한다.
#
# 측정된 상충 (A 실험, docs/idea/01_*.md §10) — 한 창에서 둘이 맞붙는다
#     창 144 차수없음 F1(ns) 0.9132 | 창 288 차수없음 0.6223   <- 창 손해 -0.291 (trend 기인)
#     창 144 차수     F1(ns) 0.9256 | 창 288 차수     0.7975   <- 차수 효과 +0.175 (seasonal 기인)
#   차수 효과가 창 288 에서 14배 크지만 창 손해가 더 커서 묻혔다. 이 스크립트가 둘을 분리한다.
#
# 구현의 핵심 — **CoST 의 두 손실 항이 이미 분기별로 갈라져 있다**
#     L = L_MoCo(trend 만)  +  alpha * L_season(seasonal 만)
#   두 분기를 묶는 것은 공유 trunk 하나뿐이다. 그래서 trunk 를 가르면 손실 구조를 안 건드린다.
#     trunk_T (창 144) -> tfd      -> trend_160   <- L_MoCo 가 학습
#     trunk_S (창 288) -> 차수 sfd -> season_160  <- alpha * L_season 이 학습
#     표현 = concat = 320  (GMM 후단이 baseline 과 동일)
#
# **baseline(CoST-W144-noscaleshift)에서의 변형이다** (연구자 지시). 바꾼 것은 trunk 분리뿐이고
#   trend 분기의 창은 baseline 과 같은 144 다. 추론 시 trend 가 보는 입력이 baseline 과
#   **바이트 단위로 같다**는 것을 Step 2 에서 확인해 출력한다 (288 창의 뒤쪽 144 = baseline 의 144 창).
#   증강(jitter only)·표현 차원(320)·커널·alpha·lr·배치·분할·지표·후단 전부 baseline 과 같다.
#
# 차수 변환(shim 2·3)은 docs/idea/01_order_normalized_seasonal.md 의 것을 그대로 쓴다.
#
# 변형 2종 — 'nrot=' 하나로 제안과 대조군이 모두 나온다
#     nrot=2 (기본) : trunk_S 창 288, L_ORD 72, 해상도 0.50 차수/bin  -> CoST-dual-T144-S288 (제안)
#     nrot=1        : trunk_S 창 144, L_ORD 36, 해상도 1.00 차수/bin  -> CoST-dual-T144-S144 (대조군)
#   대조군의 역할 : 파라미터가 2배인데 **창 분리는 없는** 설정이다. 이것과 비교해야
#   "좋아진 것이 창 분리 때문인가 용량 2배 때문인가" 가 분리된다.
#
# 학습량 (hyperparameters.md §11 문제 9) — **baseline 과 동등하게 맞춘다** (연구자 지시 2026-10-29:
#   "언제나 baseline 과 동등한 비교가 되어야 한다"). max_train_length = 창_S 라 창 288 에서는
#   run 1300 이 4조각으로 쪼개져 1 epoch = 84 iteration (창 144 는 9조각 189 iteration) 이고,
#   같은 600 iteration 이면 데이터 통과가 3.17 -> 7.14 로 2.25배가 된다. 그래서 통과 횟수가
#   같아지는 n_iters = 600 x 4/9 = 267 을 유도해서 쓴다. 자세한 유도는 아래 BASE_N_ITERS 주석.
#   fold 마다 data_passes 와 data_passes_baseline 을 나란히 기록해 동등성을 감사할 수 있게 둔다.
#
# 원본 저장소는 한 줄도 수정하지 않는다 — 런타임 서브클래스 shim 이다.
#
# 실행 : python code/04_proposed/04_cost_dual_trunk.py                      (A, 제안 nrot=2)
#        python code/04_proposed/04_cost_dual_trunk.py tension nrot=1       (A, 대조군)
#        python code/04_proposed/04_cost_dual_trunk.py tension iters=600    (진단용 — iteration 을 맞춘 변형)
#        python code/04_proposed/04_cost_dual_trunk.py tension amb          (ambiguous 포함, 추론만)
# =============================================================================
import os, sys, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
import torch.fft as fft
from torch import nn
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, '03_baselines'))
sys.path.insert(0, os.path.join(ROOT, '03_baselines', 'external', 'CoST'))
import evaluation as ev
import splits
import cost as cost_module                                              # shim 대상 (원본 미수정)
from cost import CoST, CoSTModel                                        # 외부 저장소 원본
from models.encoder import BandedFourierLayer, CoSTEncoder              # 외부 저장소 원본
from utils import init_dl_program                                       # 외부 저장소 원본의 시드 초기화

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
_opt = sys.argv[2:]
_nrot = [t for t in _opt if t.startswith('nrot=')]
_iters = [t for t in _opt if t.startswith('iters=')]
# --- 메커니즘 실험용 토큰 2종 (2026-10-29). **기본은 둘 다 꺼져 있다** ---
# 'freeze'  : seasonal trunk 를 초기값에 **동결**한다 (requires_grad=False).
#             근거 — 분리된 seasonal trunk 의 데이터 기인 이동량이 0.00022 로 trend trunk(0.00232)의
#             1/10 이다. 얼려도 같은 성능이면 "seasonal 분리기는 trend 과제에 오염되지 않은 trunk 만
#             필요하고 학습조차 필요 없다" 가 된다.
# 'noorder' : seasonal trunk 의 sfd 를 **원본 BandedFourierLayer** 로 둔다 (차수 재색인 없음).
#             근거 — 지금 분리 효과는 차수가 있는 상태에서만 측정됐다. 차수 없이도 이로우면
#             주장이 "CoST 의 공유 backbone 이 해롭다" 로 넓어지고, 아니면 "차수 재색인이
#             전용 trunk 를 요구한다" 로 좁아진다. 비교 상대는 CoST-W144-noscaleshift 다.
FREEZE = 'freeze' in _opt
NOORDER = 'noorder' in _opt
_rest = [t for t in _opt if not t.startswith('nrot=') and not t.startswith('iters=')
         and t not in ('freeze', 'noorder')]
INFER = splits.set_amb_mode(_rest)          # 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만

# =============================================================================
# hyperparameter
# =============================================================================
SEED, N_SEEDS = 0, 5

# --- trend 분기 : baseline 과 같은 창. **고정값이다** ---
WINDOW_T = 144                  # = CoST-W144-noscaleshift 의 창. 이 값을 바꾸면 baseline 변형이 아니다

# --- seasonal 분기 : 차수 정규화 (docs/idea/01_order_normalized_seasonal.md §4 와 동일) ---
ROLL_CIRCUMFERENCE_MM = 783.5   # 둘레. 205 run 평균 (docs/periodic_disturbance.md)
N_ROT = int(_nrot[0].split('=')[1]) if _nrot else 2
V_MIN, V_MAX = 50, 200          # 조건 격자의 속도 양 끝
WINDOW_S = int(np.ceil(N_ROT * (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / V_MIN))     # N_ROT=2 -> 288
L_ORD = int(N_ROT * round((1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / V_MAX))          # N_ROT=2 ->  72
# L_ORD 를 가장 빠른 속도 기준으로 잡는 이유는 아이디어 01 §6 과 같다 — 더 크게 잡으면 고속 창을
# 업샘플하게 되고, "윗 bin 이 보간으로 채워졌다" 는 사실 자체가 속도 단서가 된다.
assert WINDOW_S >= WINDOW_T, f'seasonal 창({WINDOW_S})이 trend 창({WINDOW_T})보다 짧으면 설계가 깨진다'

# --- 학습량 : **baseline 과 동등하게 맞춘다** (연구자 지시 2026-10-29) ---
# 창이 길면 run 이 적은 조각으로 쪼개져 1 epoch 의 iteration 수가 줄고, 같은 iteration 수라면
# 같은 데이터를 더 여러 번 보게 된다 (hyperparameters.md §11 문제 9). 실측:
#   창 144 : 1300 // 144 = 9 조각 -> 189 iter/epoch -> 600 iter 에서 데이터 통과 3.17회
#   창 288 : 1300 // 288 = 4 조각 ->  84 iter/epoch -> 600 iter 면 통과 7.14회 (2.25배. 불공정)
# 그래서 **데이터 통과 횟수가 baseline 과 같아지는** iteration 수를 유도해서 쓴다.
#   통과 = n_iters x 배치 / (run수 x 조각수)  이므로  n_iters / 조각수  를 맞추면 된다
#   -> n_iters = BASE_N_ITERS x (조각수_S / 조각수_T) = 600 x 4/9 = 267
# run 수·배치와 무관하게 맞으므로 fold 마다 다시 계산할 필요가 없다. 실제값은 fold 마다
# data_passes 로 기록하고 baseline 값(data_passes_baseline)과 나란히 남긴다.
#
# 대가 : 갱신 횟수가 600 -> 267 로 줄어 제안 방법이 **덜 학습된다.** 즉 이 선택은
#   제안 방법에 불리한 쪽으로 보수적이다. 통과 횟수를 맞추는 것이 동등 비교의 기준이므로 그렇게 둔다.
#   'iters=600' 으로 지정하면 "iteration 을 맞춘" 진단용 변형을 따로 돌릴 수 있다 (이름에 -it600 이 붙는다).
BASE_N_ITERS = 600        # baseline(CoST-W144-noscaleshift) 실측. 저장소 규칙
                          # (train_data.size > 100000 -> 600)의 결과이고 fold_metrics 에 600 으로 기록돼 있다
N_ITERS_FIXED = int(_iters[0].split('=')[1]) if _iters else None   # 수동 지정 (진단용)
N_ITERS = None            # Step 1 뒤에서 run 길이를 보고 유도한다

# --- 나머지는 04_cost.py / 06_cost_ablation.py 와 완전히 동일 ---
DEVICE, BATCH_SIZE, LR, REPR_DIMS = 'cuda', 8, 0.001, 320
HIDDEN = 64               # trunk 폭 (저장소 기본값). trunk 하나당 파라미터 637,248
# 용량 기록 : 이원 trunk 는 trend·seasonal 이 각자 trunk 를 가지므로 학습되는 파라미터가
#   14.67M -> 15.31M (+4.4%) 로 늘어난다. 나머지 16.8M(trend trunk 의 sfd, seasonal trunk 의 tfd)은
#   forward 에서 계산만 되고 버려져 gradient 가 0 이다. 용량 대조군은
#   `01_cost_order.py tension nrot=1 hidden=116` (공유 trunk 1,266,552 = 1.988배) 이다.
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005

IN_PATH = 'outputs/merged_data.csv'
SUFFIX = '' if N_ITERS_FIXED is None else f'-it{N_ITERS_FIXED}'   # 수동 지정 변형은 이름으로 분리
NOORD_SUF = '-noord' if NOORDER else ''
FRZ_SUF = '-frz' if FREEZE else ''
METHOD = f'CoST-dual-T{WINDOW_T}-S{WINDOW_S}{NOORD_SUF}{FRZ_SUF}{SUFFIX}'

# 출력 폴더는 **방법 이름으로 갈린다** (연구자 지시 2026-10-29) — 한 스크립트가 여러 변형을 만들어
# 결과가 한 폴더에 섞이면 읽기 어렵다. 확정된 제안방법만 `(final v1)` 로 분리한다.
#   nrot=1 freeze          -> CoST-dual-T144-S144-frz        -> (final v1)
#   그 외 전부(noorder / noorder freeze / 없음 / nrot=2 / iters=…) -> (ablation)
FINAL_V1_METHOD = 'CoST-dual-T144-S144-frz'
STAGE = ('12_proposed_cost_dual_trunk (final v1)' if METHOD == FINAL_V1_METHOD
         else '12_proposed_cost_dual_trunk (ablation)')
OUT_DIR = splits.resolve(SPLIT, STAGE)
CKPT_DIR = splits.ckpt_dir(SPLIT, STAGE)

T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS


# =============================================================================
# 함수 — 차수 변환 (아이디어 01 과 동일)
# =============================================================================
def rot_samples(v):
    # 역할 : 속도 v 에서 회전체 1바퀴에 해당하는 샘플 수. f_rot = v / C 의 역수를 샘플로 환산한 것.
    return (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / v


def order_window(v):
    # 역할 : 속도 v 에서 N_ROT 회전을 담는 창 길이 (샘플). N_ROT=2 면 V=50 -> 288, V=200 -> 72
    return int(np.ceil(N_ROT * rot_samples(v)))


# =============================================================================
# shim 1 — 증강 : scale·shift 제거(baseline 과 동일) + 속도 채널은 건드리지 않는다
#          아이디어 01·03 과 글자까지 같다.
# =============================================================================
class JitterCh0Dataset(cost_module.PretrainDataset):
    # 역할 : baseline(CoST-W144-noscaleshift)과 같은 'jitter only' 증강을 쓰되, 입력 2번째 채널에
    #        실어 보낸 속도 값이 잡음에 오염되지 않게 **채널 0 에만** 적용한다.
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])
        return out


# =============================================================================
# shim 2 — BandedFourierLayer 를 차수 축에서 동작시킨다 (아이디어 01 의 본체, 그대로 사용)
# =============================================================================
class OrderBandedFourierLayer(BandedFourierLayer):
    # 역할 : trunk 출력의 뒤쪽 N_ROT 회전분을 잘라 고정 길이 L_ORD 로 선형 재표본화한 뒤,
    #        원본과 똑같은 연산(rfft -> bin별 복소 가중 -> irfft)을 수행한다.
    #        그 결과 bin k 가 '차수 k/N_ROT' 라는 속도 무관 물리량에 묶인다.
    # 인과성 : 창의 뒤쪽을 쓰므로 미래 시점을 보지 않는다.
    def forward(self, input):                                   # input: B x T x in_channels
        v = self.speeds                                         # (B,) — shim 3 이 넣어 준다
        b, t, _ = input.shape
        out = input.new_zeros(b, self.length, self.out_channels)
        for s in torch.unique(v):                               # 속도마다 창 길이가 다르므로 묶어서 처리
            m = v == s
            w = min(order_window(float(s)), t)
            seg = input[m, t - w:]                                              # 뒤에서 N_ROT 회전분
            seg = Fnn.interpolate(seg.transpose(1, 2), size=self.length,
                                  mode='linear', align_corners=True).transpose(1, 2)
            out[m] = super().forward(seg)                                       # 원본 연산 그대로
        return out


# =============================================================================
# shim 3 — seasonal trunk : 속도 채널을 떼어 내고 채널 0 만 원본 forward 에 넘긴다 (아이디어 01)
# =============================================================================
class OrderCoSTEncoder(CoSTEncoder):
    # 역할 : 입력 (B, T, 2) 에서 속도 채널을 분리해 seasonal 분기에 전달하고,
    #        채널 0(장력 오차 e) 만 원본 forward 에 넘긴다.
    def __init__(self, *a, l_ord, **kw):
        super().__init__(*a, **kw)
        self.sfd = nn.ModuleList([OrderBandedFourierLayer(
            self.output_dims, self.component_dims, 0, 1, length=l_ord)])

    def forward(self, x, **kw):
        # 패딩 구간은 두 채널 모두 NaN 이므로 시간축 최댓값으로 속도를 읽는다 (V > 0 이라 안전).
        self.sfd[0].speeds = torch.nan_to_num(x[:, :, 1], nan=0.0).amax(dim=1)
        return super().forward(x[:, :, :1].contiguous(), **kw)


# =============================================================================
# shim 4 — trend trunk : 속도 채널만 떼어 낸다 (아이디어 03 의 Ch0 인코더와 같다)
# =============================================================================
class Ch0CoSTEncoder(CoSTEncoder):
    # 역할 : 입력 (B, T, 2) 에서 채널 0(오차 e) 만 원본 forward 에 넘긴다.
    #        따라서 input_fc·conv trunk·trend 분기는 baseline 과 같은 입력을 본다.
    def forward(self, x, tcn_output=False, mask='all_true'):
        return super().forward(x[..., :1].contiguous(), tcn_output, mask)


# =============================================================================
# shim 5 — 손실 : trend 는 창_T, seasonal 은 창_S 에서 계산한다  ★ 아이디어의 본체
# =============================================================================
class DualCoSTModel(CoSTModel):
    # 역할 : 원본 forward 의 두 항을 **서로 다른 trunk·서로 다른 창**에서 계산한다.
    #        encoder_q / encoder_k 는 trend trunk 쌍이므로 MoCo 기구(모멘텀 갱신·큐)가 그대로 돈다.
    #        seasonal 은 별도 trunk(encoder_s)가 창_S 전체를 본다.
    # 시점 정렬 : 두 분기 모두 **같은 시점에서 끝난다** (창_S 의 뒤쪽 창_T 를 trend 가 본다).
    #        최종 표현이 [:, -1] 만 쓰므로 이 정렬이 유지되어야 한다.
    def __init__(self, *a, encoder_s, window_t, dim, **kw):
        super().__init__(*a, dim=dim, **kw)
        self.encoder_s = encoder_s              # 학습된다 (requires_grad=True, 옵티마이저가 집는다)
        self.window_t = window_t

    def forward(self, x_q, x_k):
        # ---- trend 항 : 창_S 의 뒤쪽 창_T 만 쓴다. 원본과 같은 연산 ----
        x_q_t, x_k_t = x_q[:, -self.window_t:], x_k[:, -self.window_t:]
        rand_idx = np.random.randint(0, x_q_t.shape[1])
        q_t, _ = self.encoder_q(x_q_t)
        if q_t is not None:
            q_t = Fnn.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_t, _ = self.encoder_k(x_k_t)
            if k_t is not None:
                k_t = Fnn.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)
        loss = 0
        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)

        # ---- seasonal 항 : 별도 trunk 가 창_S 전체를 본다. 원본과 같은 연산 ----
        _, q_s = self.encoder_s(x_q)
        q_s = Fnn.normalize(q_s, dim=-1)
        _, k_s = self.encoder_s(x_k)
        k_s = Fnn.normalize(k_s, dim=-1)
        q_s_freq, k_s_freq = fft.rfft(q_s, dim=1), fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)
        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        loss += (self.alpha * (seasonal_loss / 2))
        return loss


# =============================================================================
# shim 6 — 래퍼 : 추론에서 두 trunk 를 합치고, 체크포인트에 둘 다 담는다
# =============================================================================
class DualCoST(CoST):
    # 역할 : 원본 CoST 의 fit 루프·옵티마이저·학습률 스케줄을 그대로 쓰고, **추론만** 바꾼다.
    #        _eval_with_pooling 이 원본에서 cat([out_t[:, -1], out_s[:, -1]]) 를 만드는 지점이라
    #        여기 한 군데만 바꾸면 창이 다른 두 trunk 를 합칠 수 있다.
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        x = x.to(self.device, non_blocking=True)
        out_t, _ = self.net(x[:, -WINDOW_T:])           # trend : 뒤쪽 창_T (= baseline 의 창)
        _, out_s = self.net_s(x)                        # seasonal : 창_S 전체
        out = torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)
        return rearrange(out.cpu(), 'b d -> b () d')

    def encode(self, *a, **kw):
        # 역할 : 원본 encode 는 self.net 만 eval 로 돌린다. seasonal trunk 도 eval 이어야 한다
        #        — CoSTEncoder 가 season 출력에 Dropout(0.1) 을 걸기 때문이다.
        org = self.net_s.training
        self.net_s.eval()
        try:
            return super().encode(*a, **kw)
        finally:
            self.net_s.train(org)

    def save(self, fn):
        torch.save({'net': self.net.state_dict(), 'net_s': self.net_s.state_dict()}, fn)

    def load(self, fn):
        sd = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(sd['net'])
        self.net_s.load_state_dict(sd['net_s'])


# =============================================================================
# Step 1~2. 데이터 로드 + 저장소 입력 형식 변환 (채널 2개: e, V — V 는 trunk 진입 전에 제거된다)
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
lengths = np.array([len(by_run[r]) for r in run_ids])
speeds = np.array([by_run[r][VS].iloc[0] for r in run_ids], dtype=float)

x = np.full((len(run_ids), lengths.max(), 2), np.nan)
for i, r in enumerate(run_ids):
    x[i, :lengths[i], 0] = by_run[r].e.to_numpy()
    x[i, :lengths[i], 1] = speeds[i]                   # 패딩 구간은 두 채널 모두 NaN 유지
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 1] runs = {len(run_ids)}, 입력 {x.shape} ({time.time() - t0:.1f} s)')
print(f'[Step 2] trend 창 {WINDOW_T} ({WINDOW_T * ev.DT:.1f} s, baseline 과 동일) | '
      f'seasonal 창 {WINDOW_S} ({WINDOW_S * ev.DT:.1f} s) = V{V_MIN} 의 {N_ROT} 회전 | '
      f'차수영역 길이 {L_ORD} | 차수 해상도 {1 / N_ROT:.2f} 차수/bin')
print(f'[Step 2] 속도별 seasonal 창: ' +
      ', '.join(f'V{int(v)}:{order_window(v)}' for v in sorted(set(speeds))))
print(f'[Step 2] seasonal trunk: ' +
      ('원본 BandedFourierLayer (차수 재색인 없음)' if NOORDER else f'차수 sfd (L_ORD={L_ORD})') +
      (', **초기값 동결**' if FREEZE else ', 학습함'))

# --- 학습량을 baseline 에 맞춘다 (위 BASE_N_ITERS 주석 참조) ---
SEC_T, SEC_S = x.shape[1] // WINDOW_T, x.shape[1] // WINDOW_S     # 1300 -> 9 조각 / 4 조각
N_ITERS = N_ITERS_FIXED if N_ITERS_FIXED is not None else int(round(BASE_N_ITERS * SEC_S / SEC_T))
PASS_BASE, PASS_OURS = BASE_N_ITERS / SEC_T, N_ITERS / SEC_S      # 통과 횟수에 비례하는 양
print(f'[Step 2] 학습량 맞춤: 조각 {SEC_T}(창 {WINDOW_T}) vs {SEC_S}(창 {WINDOW_S}) -> '
      f'n_iters {BASE_N_ITERS} -> {N_ITERS}'
      + ('  [수동 지정]' if N_ITERS_FIXED is not None else '  [baseline 통과 횟수에 맞춘 유도값]'))
print(f'[Step 2] 데이터 통과 비례량: baseline {PASS_BASE:.2f} vs 제안 {PASS_OURS:.2f} '
      f'(오차 {abs(PASS_OURS / PASS_BASE - 1):.2%})')


def trend_input_matches_baseline(series, n_check=400):
    # 역할 : **baseline 변형임을 확인하는 검사**. 추론 시 trend 가 보는 입력(창_S 창의 뒤쪽 창_T)이
    #        baseline(창_T 단독)이 보는 입력과 같은지 시점마다 비교한다.
    #        두 경우 모두 run 시작부에서는 왼쪽을 NaN 으로 채우므로 그 구간이 특히 중요하다.
    n = len(series)
    for t in np.linspace(0, n - 1, n_check).astype(int):
        lo_s, lo_t = t - WINDOW_S + 1, t - WINDOW_T + 1
        w_s = np.full(WINDOW_S, np.nan)
        w_s[max(0, -lo_s):] = series[max(lo_s, 0):t + 1]
        w_t = np.full(WINDOW_T, np.nan)
        w_t[max(0, -lo_t):] = series[max(lo_t, 0):t + 1]
        if not np.array_equal(w_s[-WINDOW_T:], w_t, equal_nan=True):
            return False, int(t)
    return True, -1


_ok, _bad_t = trend_input_matches_baseline(x[0, :lengths[0], 0])
print(f'[Step 2] baseline 변형 검사, trend 입력이 창 {WINDOW_T} 단독과 동일: {_ok}'
      + ('' if _ok else f' (시점 {_bad_t} 에서 불일치)'))
assert _ok, 'trend 분기가 baseline 과 다른 입력을 본다 (설계 위반)'


def unpad(z):
    # 역할 : (n_run, t_max, d) 에서 패딩을 떼고 시점 순서로 이어 붙인다 (e_flat, id_flat 과 같은 순서)
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


def new_model():
    # 역할 : trunk 두 개를 가진 모델을 만든다. 나머지 설정은 04_cost.py 와 완전히 같다.
    #        max_train_length = 창_S 다 — seasonal 이 창_S 를 보아야 하므로 학습 조각이 그 길이여야 한다.
    #        trend 는 그 조각의 뒤쪽 창_T 만 본다 (shim 5).
    m = DualCoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW_S,
                 output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)
    m.net = Ch0CoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                           length=WINDOW_T, hidden_dims=HIDDEN, depth=10).to(DEVICE)
    if NOORDER:       # 차수 재색인 없음 — 원본 BandedFourierLayer 를 그대로 쓰는 seasonal trunk
        m.net_s = Ch0CoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                                 length=WINDOW_S, hidden_dims=HIDDEN, depth=10).to(DEVICE)
    else:
        m.net_s = OrderCoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                                   length=WINDOW_S, hidden_dims=HIDDEN, depth=10,
                                   l_ord=L_ORD).to(DEVICE)
    if FREEZE:        # seasonal trunk 를 초기값에 동결한다. fit 의 옵티마이저가 requires_grad 로 걸러낸다
        for p in m.net_s.parameters():
            p.requires_grad = False
    m.cost = DualCoSTModel(m.net, copy.deepcopy(m.net), kernels=KERNELS,
                           dim=m.net.component_dims, alpha=ALPHA, K=256, device=DEVICE,
                           encoder_s=m.net_s, window_t=WINDOW_T).to(DEVICE)
    return m


def encode_all(model):
    # 역할 : 전체 run 을 인과 이동창으로 시점별 인코딩한다. 창을 정확히 창_S 로 맞추고
    #        (sliding_padding = 창_S - 1), trend 는 그 뒤쪽 창_T 를 본다 (shim 6).
    return unpad(model.encode(x, mode='forecasting', casual=True, sliding_length=1,
                              sliding_padding=WINDOW_S - 1, batch_size=len(run_ids)))


# =============================================================================
# Step 3. fold × seed 로 학습(또는 체크포인트 로드)하고 비교군과 동일한 후단으로 채점한다
# =============================================================================
if not INFER:
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

if os.path.exists(f'{OUT_DIR}/{METHOD}_fold_metrics.csv'):
    print(f'[Step 3] {METHOD}: 이미 결과가 있어 건너뛴다', flush=True)
    sys.exit(0)

run_m, fold_m, loss_m = [], [], []
for f in sp.folds:
    is_train_run, is_val_run = sp.run_role(f)
    is_train_pt, is_val_pt = sp.point_role(f, lengths)
    for seed in range(SEED, SEED + N_SEEDS):
        t0 = time.time()
        init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)
        iter_loss, epoch_iters = [], []
        model = new_model()
        model.after_iter_callback = lambda _m, l: iter_loss.append(l)
        model.after_epoch_callback = lambda m, _l: epoch_iters.append(m.n_iters)
        ck = f'{CKPT_DIR}/{METHOD}_fold{f}_seed{seed}.pt'
        if INFER:                                   # 재학습 없이 저장된 인코더만 불러온다
            assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습할 것)'
            model.load(ck)
            t_fit = time.time() - t0
        else:
            cost_module.PretrainDataset = JitterCh0Dataset   # 증강 shim 은 학습 직전에만 건다
            model.fit(x[is_train_run], n_iters=N_ITERS, verbose=False)
            t_fit = time.time() - t0
            model.save(f'{OUT_DIR}/checkpoints/{METHOD}_fold{f}_seed{seed}.pt')

        z = encode_all(model)
        gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
        pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
        r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
        del z
        fr['window'], fr['window_t'], fr['window_s'] = WINDOW_T, WINDOW_T, WINDOW_S
        fr['n_rot'], fr['l_ord'] = N_ROT, L_ORD
        fr['total_sec'] = time.time() - t0
        fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
        if not INFER:                               # 학습량 기록은 학습했을 때만 의미가 있다
            fr['fit_sec'], fr['n_epochs'], fr['n_iters'] = t_fit, model.n_epochs, model.n_iters
            fr['iters_per_epoch'] = epoch_iters[0] if epoch_iters else np.nan
            fr['data_passes'] = model.n_iters / epoch_iters[0] if epoch_iters else np.nan
            # baseline(창 144, 600 iter)이 같은 fold 에서 가졌을 통과 횟수. 동등성을 감사할 수 있게 남긴다
            _ipe_base = (int(is_train_run.sum()) * SEC_T) // BATCH_SIZE
            fr['data_passes_baseline'] = BASE_N_ITERS / _ipe_base if _ipe_base else np.nan
            fr['samples_seen_seasonal'] = model.n_iters * BATCH_SIZE * WINDOW_S
            fr['samples_seen_trend'] = model.n_iters * BATCH_SIZE * WINDOW_T
            fr['final_loss'] = float(np.mean(iter_loss[-20:])) if iter_loss else np.nan
        run_m.append(r), fold_m.append(fr)
        ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)
        ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
        if not INFER:
            lo = pd.DataFrame({'fold': f, 'seed': seed,
                               'iter': np.arange(1, len(iter_loss) + 1), 'loss': iter_loss})
            loss_m.append(lo)
            ev.append_progress(OUT_DIR, METHOD, 'loss', lo)
        loss_txt = (f'손실 {iter_loss[0]:.3f} -> {np.mean(iter_loss[-20:]):.3f}'
                    if iter_loss else '학습 생략 (체크포인트 재사용)')
        print(f'[Step 3] {METHOD} fold {f} seed {seed}: {time.time() - t0:.0f} s, {loss_txt}, '
              f'steady 판정 비율 {pred.mean():.3f}, '
              f'오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
if not INFER:
    ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, METHOD,
                        'CoST contrastive loss (dual trunk)')
ev.save_results(OUT_DIR, METHOD, run_m, fold_m,
                {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS,
                 'WINDOW': WINDOW_T, 'N_ITERS': N_ITERS, 'DEVICE': DEVICE,
                 'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'REPR_DIMS': REPR_DIMS,
                 'KERNELS': KERNELS, 'ALPHA': ALPHA,
                 'idea': 'hyperparameters.md 13.10 P5 (아이디어 01 이원 trunk)',
                 'design': 'trunk 2개 — trend 는 창 %d, seasonal 은 창 %d' % (WINDOW_T, WINDOW_S),
                 'WINDOW_T': WINDOW_T, 'WINDOW_S': WINDOW_S, 'HIDDEN_DIMS': HIDDEN,
                 'trunk_sharing': 'separate (trend trunk / seasonal trunk 각자 %d)' % HIDDEN,
                 'trunk_params_each': 637248,
                 'trainable_params_note': '학습되는 파라미터 15.31M (공유 trunk 판 14.67M 대비 +4.4%). '
                                          '죽은 분기(trend trunk 의 sfd, seasonal trunk 의 tfd) 16.8M 은 '
                                          'gradient 0. 용량 대조군은 CoST-W144-order-h116',
                 'role': ('제안 — 창 분리 (trend 144 / seasonal 288)' if WINDOW_S != WINDOW_T else
                          '대조군 — trunk 2개(파라미터 2배)이지만 창 분리 없음'),
                 'max_train_length': WINDOW_S,
                 'BASE_N_ITERS': BASE_N_ITERS,
                 'n_iters_rule': ('수동 지정 (iters=)' if N_ITERS_FIXED is not None else
                                  'baseline 의 데이터 통과 횟수에 맞춘 유도값 '
                                  f'= {BASE_N_ITERS} x {SEC_S}/{SEC_T}'),
                 'sections_trend_window': SEC_T, 'sections_seasonal_window': SEC_S,
                 'trend_input_equals_baseline': bool(_ok),
                 'seasonal_trunk_frozen': FREEZE,
                 'seasonal_axis': 'Hz (원본 BandedFourierLayer, 차수 재색인 없음)' if NOORDER
                                  else 'rotation order (not Hz)',
                 'N_ROT': N_ROT, 'L_ORD': L_ORD,
                 'order_resolution_per_bin': 1.0 / N_ROT,
                 'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
                 'order_window_by_speed': {int(v): order_window(v) for v in sorted(set(speeds))},
                 'interp': 'linear (F.interpolate, align_corners=True)',
                 'loss_split': 'L_MoCo <- trend trunk(창 %d) / alpha*L_season <- seasonal trunk(창 %d)'
                               % (WINDOW_T, WINDOW_S),
                 'augmentation': 'jitter only (scale, shift removed via runtime shim)',
                 'uses_setting_value': 'Web Speed — seasonal 창 길이 결정에만. 표현 입력 아님. '
                                       '선례: MovStd-Slope-speed-rule1b (01_stat_ssd.py:163)',
                 'baseline': 'CoST-W144-noscaleshift (trend 창이 같고 입력도 동일)',
                 'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
for metric, label in [('contamination_rate', 'Contamination rate'),
                      ('retention_rate', 'Retention rate')]:
    ev.plot_condition_heatmap(run_m, metric, OUT_DIR, METHOD, label)
print(f'[done] {METHOD} -> {OUT_DIR}', flush=True)
