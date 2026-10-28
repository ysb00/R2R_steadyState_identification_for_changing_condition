# =============================================================================
# 04_proposed/03_cost_saux.py : 제안 아이디어 03 — 설정값 방향을 예측하는 보조 손실
#
# 아이디어 : trend 표현에서 **선형 head 하나**로 "설정값 접근 비율" s̄ 를 예측하게 하고,
#   그 MSE 를 CoST 손실에 더한다.  L = L_MoCo + alpha*L_season + **beta*L_aux**
#   예측하려면 표현이 방향 정보를 담을 수밖에 없다.
#
# 왜 아이디어 02 가 아니라 이것인가 — 02 는 **같은 양**을 FiLM 게이트로 구조에 주입했는데
#   효과가 0 이었다 (F1(ns) +0.0009, FiLM |w| 0.00133 = 영 초기화에서 거의 안 움직임).
#   CoST 손실이 "같은 run 의 두 증강본을 가깝게" 만 요구하는데 s̄ 는 두 증강본에서 같은 값이라
#   인스턴스 판별에 도움이 0 이고, FiLM 을 움직이면 오히려 손실이 나빠진다. 그래서 학습이 버렸다.
#   **구조를 주는 것으로는 부족하고 손실이 그 정보를 요구해야 한다** — 03 은 손실에 넣는다.
#   재료 자체는 강하다: s̄ 단독 AUROC 0.912 (진폭 기준선 0.789, 진폭 통제 후 0.86~0.97).
#   설계·수식·사전진단은 docs/idea/03_direction_aux_loss.md
#
# head 를 **선형으로 최소화**하는 것이 핵심이다 (연구자 지시) — MLP 를 쓰면 head 가 혼자 일을 해버려
#   표현이 안 바뀐다. 선형이면 표현이 직접 선형 분리 가능해져야 한다.
#
# 읽기 시점 2종을 비교한다 (연구자 지시)
#   point : MoCo 가 쓰는 무작위 시점 1개만 제약 (약한 압력)
#   mean  : **시점별 오차를 전 시점에서 평균** (강한 압력, 시간 해상도 유지)
#           "창 평균끼리 MSE" 가 아니다 — 선형 head 에서는 그게 평균만 맞추는 약한 제약이 된다
#
# 변경 범위 : 손실 항 하나. trunk·trend·seasonal 분기·증강·창 길이·표현 차원(320)·평가 후단 전부 동일.
#   **창이 baseline 과 같은 144** 라 창 대조군이 필요 없고 CoST-W144-noscaleshift 와 직접 비교된다.
#   인코더 forward 추가 호출도 없다 (화살표 판별 방식은 반전 입력으로 +1회가 필요하다).
#
# 원본 저장소는 한 줄도 수정하지 않는다 — 런타임 서브클래스 shim 이다.
#
# 실행 : python code/04_proposed/03_cost_saux.py                   (A, beta=12 기본)
#        python code/04_proposed/03_cost_saux.py c                 (C)
#        python code/04_proposed/03_cost_saux.py tension beta=30   (민감도)
#        python code/04_proposed/03_cost_saux.py tension amb       (ambiguous 포함 — 추론만)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.fft as fft
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, '03_baselines'))
sys.path.insert(0, os.path.join(ROOT, '03_baselines', 'external', 'CoST'))
import evaluation as ev
import splits
import cost as cost_module                                              # shim 대상 (원본 미수정)
from cost import CoST, CoSTModel                                        # 외부 저장소 원본
from models.encoder import CoSTEncoder                                  # 외부 저장소 원본
from utils import init_dl_program                                       # 외부 저장소 원본의 시드 초기화

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
_rest = [t for t in sys.argv[2:] if not t.startswith('beta=')]
_beta = [t for t in sys.argv[2:] if t.startswith('beta=')]
INFER = splits.set_amb_mode(_rest)          # 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만

# =============================================================================
# hyperparameter
# =============================================================================
SEED, N_SEEDS = 0, 5

# --- 보조 손실 (docs/idea/03_direction_aux_loss.md §2.3) ---
BETA_DEFAULT = 12.0
BETA = float(_beta[0].split('=')[1]) if _beta else BETA_DEFAULT
# beta 는 손실 항의 가중치다:  L = L_MoCo + alpha*L_season + beta*L_aux
# 실측 크기에서 유도했다 — L_MoCo 가 0.045 -> 3.25 (MoCo 큐가 차면서 70배 증가),
# L_aux 초기값 ~= Var(s̄) = 0.026. 두 항을 같은 크기로 두면 beta~=125, 기존의 10% 면 beta~=12.
# beta=1 은 전체의 0.8% 라 아이디어 02 와 같은 결과(무시됨)가 난다.
# CoST 저자가 보조 항(seasonal)을 alpha=0.0005 로 매우 작게 눌러 둔 설계와도 10% 쪽이 덜 충돌한다.
# **주의**: L_MoCo 가 학습 중 70배 변하므로 고정 beta 에서는 초반에 보조가 지배하고 후반에 묻힌다.

READOUTS = ['point', 'mean']    # 비교할 읽기 시점 2종

# --- s̄ 정의 (아이디어 02 와 동일) ---
ROLL_CIRCUMFERENCE_MM = 783.5   # 둘레. 205 run 평균 (docs/periodic_disturbance.md)
WINDOW = 144                    # **baseline 과 동일**

# --- 나머지는 04_cost.py 와 완전히 동일 ---
N_ITERS = None if SPLIT == 'tension' else 600
DEVICE, BATCH_SIZE, LR, REPR_DIMS = 'cuda', 8, 0.001, 320
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005

IN_PATH = 'outputs/merged_data.csv'
STAGE = '11_proposed_cost_direction_aux_loss'
OUT_DIR = splits.resolve(SPLIT, STAGE)
CKPT_DIR = splits.ckpt_dir(SPLIT, STAGE)
SUFFIX = '' if BETA == BETA_DEFAULT else f'-b{BETA:g}'    # 기본값이 아니면 이름에 붙여 섞이지 않게

T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS


# =============================================================================
# 함수 — s̄ (설정값 접근 비율)
# =============================================================================
def rot_samples(v):
    # 역할 : 속도 v 에서 회전체 1바퀴에 해당하는 샘플 수 (평활 길이). V=50 -> 144, V=200 -> 36
    return (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / v


def direction_ratio(e, v, w=WINDOW):
    # 역할 : 시점마다 "창 [t-w+1, t] 안에서 설정값에 접근한 시점의 비율" 을 계산한다.
    #        r = -e*ė > 0 이면 |e| 가 줄어드는 중 = 접근. sag 회복과 overshoot 회복이 같은 부호로 묶인다.
    #        창이 전부 과거이므로 **창 안에서 중심 평활을 써도 시점 t 기준으로 인과**다.
    #        인과 이동평균은 지연 P/2 때문에 고속(P=36)에서 반주기와 겹쳐 부호가 뒤집힌다
    #        (실측: V=200 에서 AUROC 0.912 -> 0.599).
    # 출력 : (n,) [0, 1]. 0.5 = 방향 없음, 작을수록 설정값에 접근 중 (= transient)
    n = len(e)
    p = max(3, int(round(rot_samples(v))))                 # 평활 길이 = 1회전
    X = np.zeros((n, w))
    for i in range(n):
        seg = e[max(0, i - w + 1):i + 1]
        X[i, w - len(seg):] = seg
    pad = p // 2
    k = np.ones(p) / p
    Xs = np.apply_along_axis(lambda r: np.convolve(r, k, mode='valid')[:w], 1,
                             np.pad(X, ((0, 0), (pad, pad)), mode='edge'))
    de = np.gradient(Xs, axis=1) / ev.DT
    r = -Xs * de
    nz = np.abs(r) > 1e-14
    return ((r > 0) & nz).sum(1) / np.maximum(nz.sum(1), 1)


# =============================================================================
# shim 1 — 증강 : scale·shift 제거(baseline 과 동일) + **s̄ 채널은 건드리지 않는다**
# =============================================================================
class JitterCh0Dataset(cost_module.PretrainDataset):
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])
        return out


# =============================================================================
# shim 2 — 인코더 : s̄ 채널을 떼어 내고 나머지는 원본 그대로
# =============================================================================
class Ch0CoSTEncoder(CoSTEncoder):
    # 역할 : 입력 (B, T, 2) 에서 채널 0(오차 e) 만 원본 forward 에 넘긴다.
    #        trunk·trend·seasonal 은 baseline 과 비트 단위로 같은 입력을 본다.
    #        s̄ 는 손실 계산에만 쓰이며 CoSTModel 이 x_q 에서 직접 읽는다.
    def forward(self, x, tcn_output=False, mask='all_true'):
        return super().forward(x[..., :1].contiguous(), tcn_output, mask)


# =============================================================================
# shim 3 — 손실 : 보조 항 추가  ★ 아이디어의 본체
# =============================================================================
class SAuxCoSTModel(CoSTModel):
    # 역할 : 원본 forward 를 그대로 수행하고 **끝에 보조 항 하나**를 더한다.
    #        trend 원시 표현(정규화·head 통과 전)에서 선형 head 로 s̄ 를 예측하는 MSE 다.
    def __init__(self, *a, beta, readout, dim, **kw):
        super().__init__(*a, dim=dim, **kw)
        self.beta, self.readout = beta, readout
        self.aux_head = nn.Linear(dim, 1)         # 160 -> 1. 선형 최소 구조
        self.last_aux = float('nan')

    def forward(self, x_q, x_k):
        # ---- 여기부터 원본 CoSTModel.forward 와 동일 (t_raw 만 따로 잡아 둔다) ----
        rand_idx = np.random.randint(0, x_q.shape[1])
        q_t, q_s = self.encoder_q(x_q)
        t_raw = q_t                                           # 보조 손실용 (정규화 전 trend)
        if q_t is not None:
            q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_t, k_s = self.encoder_k(x_k)
            if k_t is not None:
                k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)
        loss = 0
        loss += self.compute_loss(q_t, k_t, self.queue.clone().detach())
        self._dequeue_and_enqueue(k_t)
        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(x_k)
        k_s = F.normalize(k_s, dim=-1)
        q_s_freq, k_s_freq = fft.rfft(q_s, dim=1), fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)
        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        loss += (self.alpha * (seasonal_loss / 2))
        # ---- 여기까지 원본. 아래가 추가분 ----
        s = x_q[..., 1]                                       # (B, T) 목표값 s̄
        if self.readout == 'point':
            pred, tgt = self.aux_head(t_raw[:, rand_idx]).squeeze(-1), s[:, rand_idx]
        else:                                                 # 'mean' — 시점별 오차를 전 시점 평균
            pred, tgt = self.aux_head(t_raw).squeeze(-1), s
        m = torch.isfinite(tgt)                               # 패딩(NaN) 제외
        aux = F.mse_loss(pred[m], tgt[m]) if m.any() else t_raw.new_zeros(())
        self.last_aux = float(aux.detach())
        return loss + self.beta * aux


# =============================================================================
# Step 1~2. 데이터 로드 + 저장소 입력 형식 변환 (채널 2개: e, s̄)
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
    e = by_run[r].e.to_numpy()
    x[i, :lengths[i], 0] = e
    x[i, :lengths[i], 1] = direction_ratio(e, speeds[i])
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
s_flat = np.concatenate([x[i, :lengths[i], 1] for i in range(len(run_ids))])
print(f'[Step 1] runs = {len(run_ids)}, 입력 {x.shape} ({time.time() - t0:.1f} s)')
print(f'[Step 2] 창 {WINDOW} (baseline 동일) | 평활 = 1회전 P(V): ' +
      ', '.join(f'V{int(v)}:{int(round(rot_samples(v)))}' for v in sorted(set(speeds))))
print(f'[Step 2] s_bar 평균 {s_flat.mean():.3f}  분산 {s_flat.var():.5f}  -> L_aux 초기값 기대치')
print(f'[Step 2] beta = {BETA:g}  (L_MoCo 수렴값 약 3.25 기준 보조 항 비중 '
      f'{BETA * s_flat.var() / 3.25:.1%})')


def unpad(z):
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


def new_model(readout):
    # 역할 : 04_cost.py 와 같은 설정의 CoST 를 만들고 인코더·손실을 바꿔 끼운다 (원본 미수정).
    import copy
    m = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
             output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)
    m.net = Ch0CoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                           length=WINDOW, hidden_dims=64, depth=10).to(DEVICE)
    m.cost = SAuxCoSTModel(m.net, copy.deepcopy(m.net), kernels=KERNELS,
                           dim=m.net.component_dims, alpha=ALPHA, K=256, device=DEVICE,
                           beta=BETA, readout=readout).to(DEVICE)
    return m


# =============================================================================
# Step 3. 읽기 시점 2종 x fold x seed
# =============================================================================
if not INFER:
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

for readout in READOUTS:
    method = f'CoST-W{WINDOW}-saux-{readout}{SUFFIX}'
    if os.path.exists(f'{OUT_DIR}/{method}_fold_metrics.csv'):
        print(f'[Step 3] {method}: 이미 결과가 있어 건너뛴다', flush=True)
        continue
    run_m, fold_m, loss_m = [], [], []
    for f in sp.folds:
        is_train_run, is_val_run = sp.run_role(f)
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        var_s = float(np.nanvar(s_flat[is_train_pt]))          # aux_r2 의 기준 분산
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)
            iter_loss, aux_log, epoch_iters = [], [], []
            model = new_model(readout)
            model.after_iter_callback = lambda m, l: (iter_loss.append(l),
                                                      aux_log.append(m.cost.last_aux))
            model.after_epoch_callback = lambda m, _l: epoch_iters.append(m.n_iters)
            ck = f'{CKPT_DIR}/{method}_fold{f}_seed{seed}.pt'
            if INFER:
                assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습할 것)'
                model.load(ck)
                t_fit = time.time() - t0
            else:
                cost_module.PretrainDataset = JitterCh0Dataset
                model.fit(x[is_train_run], n_iters=N_ITERS, verbose=False)
                t_fit = time.time() - t0
                model.save(f'{OUT_DIR}/checkpoints/{method}_fold{f}_seed{seed}.pt')

            z = unpad(model.encode(x, mode='forecasting', casual=True, sliding_length=1,
                                   sliding_padding=WINDOW - 1, batch_size=len(run_ids)))
            gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
            pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
            r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, method, f, seed)
            del z
            fr['window'], fr['total_sec'] = WINDOW, time.time() - t0
            fr['beta'], fr['readout'] = BETA, readout
            fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
            if not INFER:
                fr['fit_sec'], fr['n_epochs'], fr['n_iters'] = t_fit, model.n_epochs, model.n_iters
                fr['iters_per_epoch'] = epoch_iters[0] if epoch_iters else np.nan
                fr['data_passes'] = model.n_iters / epoch_iters[0] if epoch_iters else np.nan
                fr['final_loss'] = float(np.mean(iter_loss[-20:])) if iter_loss else np.nan
                # **보조 과제가 실제로 학습됐는지의 직접 증거** (아이디어 02 의 film_weight 와 같은 역할)
                fr['aux_loss_first'] = float(np.mean(aux_log[:20])) if aux_log else np.nan
                fr['aux_loss_final'] = float(np.mean(aux_log[-20:])) if aux_log else np.nan
                fr['aux_var_s'] = var_s
                fr['aux_r2'] = 1.0 - fr['aux_loss_final'] / var_s if aux_log and var_s > 0 else np.nan
            run_m.append(r), fold_m.append(fr)
            ev.append_progress(OUT_DIR, method, 'progress_fold', fr)
            ev.append_progress(OUT_DIR, method, 'progress_run', r)
            if not INFER:
                lo = pd.DataFrame({'fold': f, 'seed': seed,
                                   'iter': np.arange(1, len(iter_loss) + 1),
                                   'loss': iter_loss, 'aux': aux_log})
                loss_m.append(lo)
                ev.append_progress(OUT_DIR, method, 'loss', lo)
            txt = (f'손실 {iter_loss[0]:.3f} -> {np.mean(iter_loss[-20:]):.3f}, '
                   f'aux {np.mean(aux_log[:20]):.4f} -> {np.mean(aux_log[-20:]):.4f} '
                   f'(R2 {fr.aux_r2.iloc[0]:+.3f})' if iter_loss else '학습 생략 (체크포인트 재사용)')
            print(f'[Step 3] {method} fold {f} seed {seed}: {time.time() - t0:.0f} s, {txt}, '
                  f'오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                  f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    if not INFER:
        ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, method,
                            'CoST contrastive + direction aux loss')
    ev.save_results(OUT_DIR, method, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS,
                     'WINDOW': WINDOW, 'N_ITERS': N_ITERS, 'DEVICE': DEVICE,
                     'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'REPR_DIMS': REPR_DIMS,
                     'KERNELS': KERNELS, 'ALPHA': ALPHA,
                     'idea': 'docs/idea/03_direction_aux_loss.md',
                     'aux_loss': 'beta * MSE(Linear(trend_160), s_bar)',
                     'BETA': BETA, 'readout': readout,
                     'readout_meaning': {'point': 'MoCo 가 쓰는 무작위 시점 1개',
                                         'mean': '시점별 오차를 전 시점에서 평균'}[readout],
                     'aux_head': f'nn.Linear({REPR_DIMS // 2}, 1) — 선형 최소 구조',
                     'aux_target': 's_bar = 창 내 설정값 접근 비율, 평활 1회전 P(V)',
                     'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
                     'augmentation': 'jitter only (scale, shift removed via runtime shim)',
                     'uses_setting_value': 'Web Speed — 평활 길이 결정에만. 표현 입력 아님',
                     'baseline': 'CoST-W144-noscaleshift (창이 같아 창 대조군 불필요)',
                     'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
    for metric, label in [('contamination_rate', 'Contamination rate'),
                          ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[done] {method}', flush=True)

print(f'[done] 제안 아이디어 03 완료 -> {OUT_DIR}')
