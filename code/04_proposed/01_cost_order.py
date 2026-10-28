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
_rest = [t for t in sys.argv[2:] if not t.startswith('nrot=')]
_nrot = [t for t in sys.argv[2:] if t.startswith('nrot=')]
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
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005

IN_PATH = 'outputs/merged_data.csv'
STAGE = '09_proposed_cost_order_normalized_seasonal'   # 비교군 단계 폴더(01~08)와 같은 명명 규칙
OUT_DIR = splits.resolve(SPLIT, STAGE)
CKPT_DIR = splits.ckpt_dir(SPLIT, STAGE)
BASE_METHOD = f'CoST-W{WINDOW}-noscaleshift'        # 창 대조군
ORDER_METHOD = f'CoST-W{WINDOW}-order'              # 제안 방법

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


def new_model(order):
    # 역할 : 대조군/제안 모델을 만든다. order=True 면 인코더를 차수 축 버전으로 바꿔 끼운다.
    #        나머지 설정은 04_cost.py 와 완전히 같다.
    m = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
             output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)
    if order:
        # CoST.__init__ 가 만든 net/cost 를 차수 축 인코더로 교체한다 (원본 미수정)
        import copy
        from cost import CoSTModel
        m.net = OrderCoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                                 length=WINDOW, hidden_dims=64, depth=10, l_ord=L_ORD).to(DEVICE)
        m.cost = CoSTModel(m.net, copy.deepcopy(m.net), kernels=KERNELS,
                           dim=m.net.component_dims, alpha=ALPHA, K=256, device=DEVICE).to(DEVICE)
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
           'augmentation': 'jitter only (scale, shift removed via runtime shim)',
           'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'}
    if use_order:
        cfg.update({'idea': 'docs/idea/01_order_normalized_seasonal.md',
                    'seasonal_axis': 'rotation order (not Hz)',
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
