# =============================================================================
# 04_proposed/01_cost_order.py : 제안 아이디어 01 — 회전 차수 축 seasonal 분기
#
# 아이디어 : CoST 의 seasonal 분기에서 **주파수 축을 Hz 대신 '회전 차수' 로 재색인**한다.
#   steady 구간에는 지름 약 249 mm(둘레 783.5 mm) 고정 회전체의 1회전 외란이 있고 그
#   주파수가 속도에 비례한다 (f_rot = V / C). 창 144 에서 이 외란이 놓이는 bin 이
#   V=50 의 1 에서 V=200 의 4 로 옮겨 다니는데, `BandedFourierLayer` 는 **bin 마다 독립
#   가중치**를 학습하므로 저속에서 배운 가중치가 고속에서 엉뚱한 성분에 걸린다.
#   계절 대조손실(L_amp/L_phase)도 bin 단위로 짝을 맞추므로 같은 문제를 겪는다.
#   -> FFT 직전에 시간축을 '회전 N_ROT 바퀴 = L_ORD 샘플' 로 비틀면 bin 이 차수가 되어
#      **전 속도에서 같은 물리량**을 가리킨다. 설계·수식·사전진단은 docs/idea/01_*.md
#
# 변경 범위 : seasonal 분기 **하나**. trend 분기·trunk·증강·평가 후단은 baseline 과 동일하다.
#   과도구간은 회전 수가 아니라 시간(7~11 s)으로 정의되므로 trend 는 원 시간축을 유지한다.
#
# 원본 저장소는 한 줄도 수정하지 않는다 — 전부 런타임 서브클래스 shim 이다
#   (docs/worklog.md 2026-10-23 결정 3 "shim 우선").
#
# 실행하는 방법 2종
#   CoST-W288-noscaleshift : **창 대조군.** 창만 144 -> 288, 차수 변환 없음.
#                            이게 없으면 이득이 '창 때문' 인지 '차수 때문' 인지 가를 수 없다
#   CoST-W288-order        : **제안 방법**
#   (기존 baseline 인 CoST-W144-noscaleshift 는 06_cost_ablation.py 에 이미 있다)
#
# 속도 설정값 V 의 사용 : 창 길이를 정하는 데만 쓰고 표현에 특징으로 넣지 않는다.
#   같은 용법의 선례가 있다 — MovStd-Slope-speed-rule1b 가 w_rot = ceil((1/Δt)·C/V) 로
#   창을 정한다 (01_stat_ssd.py:163). research_plan.md 14행이 장비 설정값 사용을 허용한다.
#   **논문에는 명시해야 한다** — 비교군 23종 중 같은 정보를 쓰는 것은 2종뿐이다.
#
# 입력 : outputs/merged_data.csv      (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        splits.py 가 분할에 맞는 fold 파일을 읽는다 (A: folds.csv / C: folds_c.csv)
# 출력 : <분할 폴더>/09_proposed_cost_order_normalized_seasonal/  run/fold/summary/config csv + heatmap + 손실곡선
# 실행 : python code/04_proposed/01_cost_order.py            (A 실험, 기본값)
#        python code/04_proposed/01_cost_order.py c          (C 실험)
#        python code/04_proposed/01_cost_order.py tension amb  (A, ambiguous 포함 — 추론만)
#        python code/04_proposed/01_cost_order.py c amb        (C, ambiguous 포함 — 추론만)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, '03_baselines'))
sys.path.insert(0, os.path.join(ROOT, '03_baselines', 'external', 'CoST'))
import evaluation as ev
import splits
import cost as cost_module                                              # shim 대상 (원본 미수정)
from cost import CoST                                                   # 외부 저장소 원본
from models.encoder import BandedFourierLayer, CoSTEncoder              # 외부 저장소 원본
from utils import init_dl_program                                       # 외부 저장소 원본의 시드 초기화

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
# 'nrot=<정수>' 로 회전 수를 고를 수 있다 (기본 2). 창 길이와 방법 이름이 따라 바뀌므로
# 서로 다른 N_ROT 결과가 같은 폴더에서 섞이지 않는다 (CoST-W288-order vs CoST-W144-order).
# 'hidden=<정수>' 로 trunk 폭을 바꿀 수 있다 (기본 64 = 저장소 기본값). **기본값은 꺼져 있다.**
# 용도 : P5 이원 trunk 의 용량 대조군 (hyperparameters.md 13.10 P9, 2026-10-29).
#   이원 trunk 는 trend 와 seasonal 이 각자 trunk(637k)를 갖는다. 학습되는 파라미터가
#   14.67M -> 15.31M (+4.4%) 로 늘어나므로, "분리해서 좋아졌나 용량이 늘어 좋아졌나" 를 갈라야 한다.
#   hidden=116 이면 **공유 trunk 한 개가 1,266,552 (1.988배)** 로 같은 용량이 되고 공유는 유지된다.
# 'sg' (stop-gradient) : trunk 는 공유하되 **seasonal 손실의 gradient 가 trunk 로 흐르지 않게** 막는다.
#   기본은 꺼져 있다. 근거 (hyperparameters.md 13.10 P10, 2026-10-29) —
#   체크포인트 측정에서 공유 trunk 의 데이터 기인 이동량(0.00477)이 분리된 두 trunk 의 합(0.00254)보다
#   1.9배 컸다. alpha=0.0005 로 손실 값은 작아도 d(L_season)/d(trunk) 는 작지 않다는 뜻이고,
#   두 손실이 trunk 를 서로 다른 방향으로 밀고 있다는 그림에 맞는다.
#   stop-gradient 는 **파라미터를 전혀 늘리지 않고** 그 교차 압력만 제거한다 — 용량 교란이 원천적으로 없다.
#   분리(CoST-dual-T144-S144)와 같은 성능이면 원인이 '용량'도 '구조'도 아니라 **gradient 간섭** 하나로 좁혀진다.
#   sfd 는 detach 뒤에 있으므로 **sfd 자신은 그대로 학습한다** (trunk 만 seasonal gradient 를 안 받는다).
# 'hybrid' : seasonal 분기에 **Hz sfd 를 되살려 order sfd 와 나란히 두고 합산**한다 (B2 안).
#   기본은 꺼져 있다. 근거 (2026-10-29 측정) —
#   롤러 교란은 **각도 고정**이지만(f_rot = V/783.5, 실측 피크와 0.006 Hz 이내)
#   검출 대상인 transient 는 **시간 고정에 훨씬 가깝다** (지속시간 9.4~17.8 s. 각도 고정 가설은
#   V=200 에서 예측 4.45 s vs 실측 13.67 s 로 기각). 즉 order 축으로 전부 재색인하면 방해물은
#   고정되는 대신 **표적이 bin 사이를 움직인다.** 두 축을 함께 두어 그 손실을 메우는지 본다.
#   합성 = **element-wise 합** (B2). order 출력 (B, L_ORD, 160) 을 길이축으로 (B, WINDOW, 160) 으로
#   되돌린 뒤 Hz 출력과 더한다. 채널 160·표현 320 이 유지되어 GMM 후단이 그대로다.
#   되돌리기는 길이축만 바꾸고 채널 160 은 손대지 않으며, align_corners=True 라 **마지막 시점 값이
#   불변**이다 (표현은 season[:, -1] 만 쓰므로 되돌리기 방식이 표현을 바꾸지 않는다).
_rest = [t for t in sys.argv[2:]
         if not t.startswith('nrot=') and not t.startswith('hidden=')
         and t not in ('sg', 'hybrid')]
_nrot = [t for t in sys.argv[2:] if t.startswith('nrot=')]
_hid = [t for t in sys.argv[2:] if t.startswith('hidden=')]
STOPGRAD = 'sg' in sys.argv[2:]
HYBRID = 'hybrid' in sys.argv[2:]
INFER = splits.set_amb_mode(_rest)          # 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만

# =============================================================================
# hyperparameter
# =============================================================================
SEED, N_SEEDS = 0, 5

# --- 차수 정규화 (docs/idea/01_order_normalized_seasonal.md §4) ---
ROLL_CIRCUMFERENCE_MM = 783.5   # 둘레. 205 run 평균, 지름 249.4 mm (docs/periodic_disturbance.md)
N_ROT = int(_nrot[0].split('=')[1]) if _nrot else 2
# seasonal 분기가 담을 회전 수. 창 길이를 결정한다 — V=50 의 1회전이 143.8 샘플이므로
#   N_ROT=1 -> 창 144 (baseline 과 같다. 창 대조군 불필요)
#   N_ROT=2 -> 창 288 (V=50 의 2회전. 창 대조군 필요)
# 사전진단은 N_ROT=1 을 기각했지만(고속 AUPRC 0.249) 그 진단은 **스펙트럼 단독**을 잰 것이다.
# 실제 CoST 는 trend 분기 160차원이 함께 있어 seasonal 이 약해도 받쳐 준다. 그리고 A 실험에서
# 창 288 이 trend 를 크게 해치는 것이 확인됐다(F1(ns) 0.913 -> 0.622). 그래서 N_ROT=1 도 돌려본다.
V_MIN, V_MAX = 50, 200          # 조건 격자의 속도 양 끝
WINDOW = int(np.ceil(N_ROT * (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / V_MIN))      # = 288 (V=50 의 2회전)
L_ORD = int(N_ROT * round((1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / V_MAX))         # =  72 (V=200 의 2회전)
# L_ORD 를 **가장 빠른 속도 기준**으로 잡는 것이 핵심이다. 더 크게 잡으면 고속 창을 업샘플하게 되고,
# 그러면 "윗 bin 이 보간으로 채워졌다" 는 사실 자체가 속도 단서가 된다 (사전진단에서 확인: §6)

# --- 나머지는 04_cost.py / 06_cost_ablation.py 와 완전히 동일 ---
N_ITERS = None if SPLIT == 'tension' else 600
DEVICE, BATCH_SIZE, LR, REPR_DIMS = 'cuda', 8, 0.001, 320
HIDDEN = int(_hid[0].split('=')[1]) if _hid else 64   # trunk 폭. 64 = 저장소 기본값
# 창을 바꾸지 않으므로 학습량(조각 9개, 189 iter/epoch, 600 iter, 통과 3.17회)은 baseline 과 동등하다
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005

IN_PATH = 'outputs/merged_data.csv'
STAGE = '09_proposed_cost_order_normalized_seasonal'   # 비교군 단계 폴더(01~08)와 같은 명명 규칙
OUT_DIR = splits.resolve(SPLIT, STAGE)
CKPT_DIR = splits.ckpt_dir(SPLIT, STAGE)
HSUF = '' if HIDDEN == 64 else f'-h{HIDDEN}'        # 기본 폭이 아니면 이름으로 분리한다
SGSUF = '-sg' if STOPGRAD else ''
HYSUF = '-hz' if HYBRID else ''                     # Hz 분기를 되살려 합산한 판 (B2)
BASE_METHOD = f'CoST-W{WINDOW}-noscaleshift{HSUF}{SGSUF}'  # 창 대조군 (hidden!=64 면 폭 대조군 겸)
ORDER_METHOD = f'CoST-W{WINDOW}-order{HSUF}{SGSUF}{HYSUF}'  # 제안 방법

T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS


# =============================================================================
# 함수 — 차수 변환
# =============================================================================
def rot_samples(v):
    # 역할 : 속도 v 에서 회전체 1바퀴에 해당하는 샘플 수. f_rot = v / C 의 역수를 샘플로 환산한 것.
    return (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / v


def order_window(v):
    # 역할 : 속도 v 에서 N_ROT 회전을 담는 창 길이 (샘플). V=50 -> 288, V=200 -> 72
    return int(np.ceil(N_ROT * rot_samples(v)))


# =============================================================================
# shim 1 — 증강 : scale·shift 제거(baseline 과 동일) + **속도 채널은 건드리지 않는다**
# =============================================================================
class JitterCh0Dataset(cost_module.PretrainDataset):
    # 역할 : baseline(CoST-W144-noscaleshift)과 같은 'jitter only' 증강을 쓰되, 입력 2번째 채널에
    #        실어 보낸 속도 값이 잡음에 오염되지 않게 **채널 0 에만** 적용한다.
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])
        return out


# =============================================================================
# shim 2 — BandedFourierLayer 를 차수 축에서 동작시킨다  ★ 아이디어의 본체
# =============================================================================
class OrderBandedFourierLayer(BandedFourierLayer):
    # 역할 : trunk 출력의 **뒤쪽 N_ROT 회전분**을 잘라 고정 길이 L_ORD 로 선형 재표본화한 뒤,
    #        원본과 똑같은 연산(rfft -> bin별 복소 가중 -> irfft)을 수행한다.
    #        그 결과 bin k 가 '차수 k/N_ROT' 라는 **속도 무관 물리량**에 묶인다.
    # 인과성 : 창의 **뒤쪽**을 쓰므로 미래 시점을 보지 않는다 (baseline 의 casual=True 와 같은 규약).
    # 되돌리지 않는 이유 : 최종 표현이 season[:, -1] (마지막 시점) 만 쓰는데, 차수 영역의 마지막
    #        샘플이 곧 원 시간축의 마지막 샘플이다. 덤으로 L_amp/L_phase 도 차수 영역에서 계산된다.
    # 출력 : (B, L_ORD, out_channels) — 입력 길이 T 와 다르다. trend 는 (B, T, ·) 그대로이고,
    #        둘 다 [:, -1] 만 쓰이므로 concat 에 문제가 없다.
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
# shim 3 — 인코더 : 속도 채널을 떼어 내고 나머지는 원본 forward 에 그대로 넘긴다
# =============================================================================
class OrderCoSTEncoder(CoSTEncoder):
    # 역할 : 입력 (B, T, 2) 에서 속도 채널을 분리해 seasonal 분기에 전달하고,
    #        **채널 0(장력 오차 e) 만** 원본 forward 에 넘긴다.
    #        따라서 input_fc·conv trunk·trend 분기는 baseline 과 비트 단위로 같은 입력을 본다.
    def __init__(self, *a, l_ord, **kw):
        super().__init__(*a, **kw)
        self.sfd = nn.ModuleList([OrderBandedFourierLayer(
            self.output_dims, self.component_dims, 0, 1, length=l_ord)])

    def forward(self, x, **kw):
        # 패딩 구간은 두 채널 모두 NaN 이므로 시간축 최댓값으로 속도를 읽는다 (V > 0 이라 안전).
        self.sfd[0].speeds = torch.nan_to_num(x[:, :, 1], nan=0.0).amax(dim=1)
        return super().forward(x[:, :, :1].contiguous(), **kw)


# =============================================================================
# shim 3b — 하이브리드 인코더 : Hz sfd 와 order sfd 를 나란히 두고 합산한다 (B2)
# =============================================================================
class HybridCoSTEncoder(CoSTEncoder):
    # 역할 : 원본이 만든 **Hz sfd(창 WINDOW, 73 bin)를 그대로 쓰고**, order sfd(L_ORD, 19 bin)를
    #        추가해 둘의 출력을 **더한다**. 현재의 order 단독 판은 Hz sfd 를 버리는데, 여기서는 살린다.
    # 왜 더하나 : 두 출력 모두 (B, ·, 160) 이고 표현은 season[:, -1] 만 쓴다. 같은 시점의
    #        160차원 특질 둘을 더하는 것이라 의미가 성립한다.
    # 길이 : Hz 는 (B, WINDOW, 160), order 는 (B, L_ORD, 160) 이라 **order 를 길이축으로 되돌려** 맞춘다.
    #        align_corners=True 라 마지막 시점 값은 변하지 않는다 (= 표현은 되돌리기에 무관).
    #        되돌리기를 **균일 보간**으로 하는 이유 : order 영역 시퀀스는 이미 "1회전 = L_ORD 샘플" 로
    #        속도 정규화돼 있고, 균일 보간은 "1회전 = WINDOW 샘플" 로 그 성질을 유지한다.
    #        (원래 잘라낸 길이 w 로만 되돌리고 앞을 패딩하면 속도별 패딩 길이가 손실에 새 단서를 준다)
    # 주의 : 더한 시퀀스는 손실의 rfft(q_s, dim=1) 에 들어간다. 인덱스 j 의 의미가 두 분기에서
    #        다르므로(시간 j vs 회전 위상 j/WINDOW) 손실이 보는 시퀀스는 두 축의 혼합이다.
    #        표현(마지막 시점)에는 이 문제가 없다. 채널 concat(B1)은 이 혼합이 없는 대안이다.
    def __init__(self, *a, l_ord, **kw):
        super().__init__(*a, **kw)              # 원본이 Hz sfd 를 만든다 — 버리지 않는다
        self.sfd_ord = OrderBandedFourierLayer(
            self.output_dims, self.component_dims, 0, 1, length=l_ord)

    def add_order(self, _module, args, output):
        # 역할 : Hz sfd 의 forward hook. 같은 trunk 출력(args[0])으로 order 분기를 계산해 더한다.
        ord_out = self.sfd_ord(args[0])                                  # (B, L_ORD, 160)
        ord_up = Fnn.interpolate(ord_out.transpose(1, 2), size=output.shape[1],
                                 mode='linear', align_corners=True).transpose(1, 2)
        return output + ord_up

    def forward(self, x, **kw):
        self.sfd_ord.speeds = torch.nan_to_num(x[:, :, 1], nan=0.0).amax(dim=1)
        return super().forward(x[:, :, :1].contiguous(), **kw)


# =============================================================================
# Step 1~2. 데이터 로드 + 저장소 입력 형식 변환
#           제안 방법은 채널 2개 (e, V) — V 는 인코더 진입 전에 제거된다
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

x1 = np.full((len(run_ids), lengths.max(), 1), np.nan)          # 대조군용 (e 1채널)
x2 = np.full((len(run_ids), lengths.max(), 2), np.nan)          # 제안용   (e, V)
for i, r in enumerate(run_ids):
    x1[i, :lengths[i], 0] = by_run[r].e.to_numpy()
    x2[i, :lengths[i], 0] = by_run[r].e.to_numpy()
    x2[i, :lengths[i], 1] = speeds[i]                           # 패딩 구간은 두 채널 모두 NaN 유지
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 1] runs = {len(run_ids)}, 입력 {x1.shape} / {x2.shape} ({time.time() - t0:.1f} s)')
print(f'[Step 2] 창 {WINDOW} samples ({WINDOW * ev.DT:.1f} s) = V{V_MIN} 의 {N_ROT} 회전 | '
      f'차수영역 길이 {L_ORD} = V{V_MAX} 의 {N_ROT} 회전 | 차수 해상도 {1/N_ROT:.2f} 차수/bin')
print('[Step 2] 속도별 seasonal 창: ' +
      ', '.join(f'V{int(v)}:{order_window(v)}' for v in sorted(set(speeds))))


def unpad(z):
    # 역할 : (n_run, t_max, d) 에서 패딩을 떼고 시점 순서로 이어 붙인다 (e_flat, id_flat 과 같은 순서)
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


def _detach_sfd_input(_module, args):
    # 역할 : sfd 진입 직전에 trunk 출력을 detach 한다 (stop-gradient).
    #        forward pre-hook 이 튜플을 반환하면 원본 입력을 대체한다 — 원본 forward 를 복사하지 않아도 된다.
    #        seasonal 손실의 gradient 는 sfd 자신까지만 흐르고 trunk 에는 닿지 않는다.
    return (args[0].detach(),)


def new_model(order):
    # 역할 : 대조군/제안 모델을 만든다. order=True 면 인코더를 차수 축 버전으로 바꿔 끼운다.
    #        나머지 설정은 04_cost.py 와 완전히 같다.
    m = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
             output_dims=REPR_DIMS, hidden_dims=HIDDEN, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)
    if order:
        # CoST.__init__ 가 만든 net/cost 를 차수 축 인코더로 교체한다 (원본 미수정)
        import copy
        from cost import CoSTModel
        cls = HybridCoSTEncoder if HYBRID else OrderCoSTEncoder
        m.net = cls(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                    length=WINDOW, hidden_dims=HIDDEN, depth=10, l_ord=L_ORD).to(DEVICE)
        m.cost = CoSTModel(m.net, copy.deepcopy(m.net), kernels=KERNELS,
                           dim=m.net.component_dims, alpha=ALPHA, K=256, device=DEVICE).to(DEVICE)
        if HYBRID:
            # hook 은 deepcopy **뒤에** 질의 인코더에만 건다. 키 인코더의 seasonal 출력은
            # 원본에서 쓰이지 않으므로(`_, k_s = self.encoder_q(x_k)`) 걸 필요가 없다.
            m.net.sfd[0].register_forward_hook(m.net.add_order)
    if STOPGRAD:
        # 질의 인코더에만 건다. 키 인코더는 no_grad 안에서 쓰이므로 detach 여부가 무관하다.
        m.net.sfd[0].register_forward_pre_hook(_detach_sfd_input)
    return m


def encode_all(model, x):
    # 역할 : 전체 run 을 인과 이동창으로 시점별 인코딩한다. 창 길이를 정확히 WINDOW 로 맞춘다
    #        (sliding_padding = WINDOW - 1). baseline 과 같은 규약이다.
    return unpad(model.encode(x, mode='forecasting', casual=True, sliding_length=1,
                              sliding_padding=WINDOW - 1, batch_size=len(run_ids)))


def evaluate(z, method, f, seed, is_train_pt, is_val_pt, extra):
    # 역할 : 표현 z 를 **비교군과 동일한** 트랙 A 후단에 넣어 채점한다.
    gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
    pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
    r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, method, f, seed)
    for k, v in {**extra, 'steady_cluster': sidx,
                 'e_mean_c0': emeans[0], 'e_mean_c1': emeans[1]}.items():
        fr[k] = v
    return r, fr, pred


# =============================================================================
# Step 3. 두 방법을 각각 fold × seed 로 학습(또는 체크포인트 로드)하고 채점한다
# =============================================================================
if not INFER:
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

for method, use_order in [(BASE_METHOD, False), (ORDER_METHOD, True)]:
    if os.path.exists(f'{OUT_DIR}/{method}_fold_metrics.csv'):
        print(f'[Step 3] {method}: 이미 결과가 있어 건너뛴다', flush=True)
        continue
    x = x2 if use_order else x1
    run_m, fold_m, loss_m = [], [], []
    for f in sp.folds:
        is_train_run, is_val_run = sp.run_role(f)
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)
            iter_loss, epoch_iters = [], []
            model = new_model(use_order)
            model.after_iter_callback = lambda _m, l: iter_loss.append(l)
            model.after_epoch_callback = lambda m, _l: epoch_iters.append(m.n_iters)
            ck = f'{CKPT_DIR}/{method}_fold{f}_seed{seed}.pt'
            if INFER:                                   # 재학습 없이 저장된 인코더만 불러온다
                assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습할 것)'
                model.load(ck)
                t_fit = time.time() - t0
            else:
                # 증강 shim 은 학습 직전에만 건다 (추론 모드에서는 불필요)
                cost_module.PretrainDataset = JitterCh0Dataset
                model.fit(x[is_train_run], n_iters=N_ITERS, verbose=False)
                t_fit = time.time() - t0
                model.save(f'{OUT_DIR}/checkpoints/{method}_fold{f}_seed{seed}.pt')

            z = encode_all(model, x)
            extra = {'window': WINDOW, 'total_sec': time.time() - t0}
            if use_order:
                extra.update({'n_rot': N_ROT, 'l_ord': L_ORD,
                              'roll_circumference_mm': ROLL_CIRCUMFERENCE_MM})
            if not INFER:                               # 학습량 기록은 학습했을 때만 의미가 있다
                extra.update({'fit_sec': t_fit, 'n_epochs': model.n_epochs, 'n_iters': model.n_iters,
                              'iters_per_epoch': epoch_iters[0] if epoch_iters else np.nan,
                              'data_passes': model.n_iters / epoch_iters[0] if epoch_iters else np.nan,
                              'samples_seen': model.n_iters * BATCH_SIZE * WINDOW,
                              'final_loss': float(np.mean(iter_loss[-20:])) if iter_loss else np.nan})
            r, fr, pred = evaluate(z, method, f, seed, is_train_pt, is_val_pt, extra)
            del z
            run_m.append(r), fold_m.append(fr)
            ev.append_progress(OUT_DIR, method, 'progress_fold', fr)
            ev.append_progress(OUT_DIR, method, 'progress_run', r)
            if not INFER:
                lo = pd.DataFrame({'fold': f, 'seed': seed,
                                   'iter': np.arange(1, len(iter_loss) + 1), 'loss': iter_loss})
                loss_m.append(lo)
                ev.append_progress(OUT_DIR, method, 'loss', lo)
            loss_txt = (f'손실 {iter_loss[0]:.3f} -> {np.mean(iter_loss[-20:]):.3f}'
                        if iter_loss else '학습 생략 (체크포인트 재사용)')
            print(f'[Step 3] {method} fold {f} seed {seed}: {time.time() - t0:.0f} s, {loss_txt}, '
                  f'steady 판정 비율 {pred.mean():.3f}, '
                  f'오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                  f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

    # ---- 방법 하나가 끝날 때마다 산출물을 저장한다 ----
    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    if not INFER:
        ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, method,
                            'CoST contrastive loss')
    cfg = {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOW': WINDOW,
           'N_ITERS': N_ITERS, 'DEVICE': DEVICE, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
           'REPR_DIMS': REPR_DIMS, 'KERNELS': KERNELS, 'ALPHA': ALPHA,
           'HIDDEN_DIMS': HIDDEN, 'stop_gradient_to_trunk': STOPGRAD,
           'stop_gradient_note': ('seasonal 손실의 gradient 가 trunk 로 흐르지 않게 막았다 '
                                  '(sfd 는 그대로 학습한다). 파라미터 증가 0'
                                  if STOPGRAD else '없음 (원본대로 두 손실이 trunk 를 함께 만든다)'),
           'trunk_params': 'hidden_dims %d (64 = 저장소 기본값). P5 이원 trunk 의 용량 대조군으로 '
                           'hidden=116 을 쓰면 공유 trunk 가 1,266,552 = 1.988배가 된다' % HIDDEN,
           'augmentation': 'jitter only (scale, shift removed via runtime shim)',
           'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'}
    if use_order:
        cfg.update({'idea': 'docs/idea/01_order_normalized_seasonal.md',
                    'hybrid_hz_plus_order': HYBRID,
                    'hybrid_note': ('B2 — Hz sfd(창 %d, 73 bin)와 order sfd(%d, 19 bin)를 나란히 두고 '
                                    'order 를 길이축으로 되돌려 element-wise 합. 채널 160·표현 320 유지'
                                    % (WINDOW, L_ORD)) if HYBRID else '없음 (order sfd 단독, Hz 는 버린다)',
                    'seasonal_axis': ('rotation order + Hz (두 sfd 합산)' if HYBRID
                                      else 'rotation order (not Hz)'),
                    'N_ROT': N_ROT, 'L_ORD': L_ORD,
                    'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
                    'order_window_by_speed': {int(v): order_window(v) for v in sorted(set(speeds))},
                    'interp': 'linear (F.interpolate, align_corners=True)',
                    'uses_setting_value': 'Web Speed — seasonal 창 길이 결정에만. 표현 입력 아님. '
                                          '선례: MovStd-Slope-speed-rule1b (01_stat_ssd.py:163)'})
    else:
        cfg['role'] = f'창 대조군 — 창만 144 -> {WINDOW}, 차수 변환 없음'
    ev.save_results(OUT_DIR, method, run_m, fold_m, cfg)
    for metric, label in [('contamination_rate', 'Contamination rate'),
                          ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[done] {method}', flush=True)

print(f'[done] 제안 아이디어 01 완료 -> {OUT_DIR}')
