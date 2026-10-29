# =============================================================================
# 04_proposed/05_cost_offset_aux.py : 제안 아이디어 06 — 잔차 오프셋 크기를 예측하는 보조 손실
#
# 아이디어 : trend 표현에서 **선형 head 하나**로 "창 평균 잔차의 크기" m̄ = |mean(e)| 를
#   예측하게 하고, 그 MSE 를 CoST 손실에 더한다.
#     L = L_MoCo + alpha*L_season + **beta*L_aux**,  L_aux = || w^T h_T - z(m̄) ||^2
#   아이디어 03 과 **틀은 같고 타깃만 다르다** — 03 은 "가까워지는 중인가"(방향)를 물었고
#   이것은 "얼마나 떨어져 있나"(크기)를 묻는다.
#
# 왜 이 타깃인가 (2026-10-29 저속 진단, 연구자 제안)
#   저속에서 상태를 가르는 정보는 요동이 아니라 **오프셋**이다. 창 144 가 V=50 에서 정확히
#   1회전이라 창 내 요동 통계가 회전 위상에 지배되기 때문이다. 실측 (전체 모집단, 라벨 0/1):
#       V=50  : movstd AUROC 0.890 / |slope| 0.842 / **|창평균 e| 0.999**
#       V=200 : movstd AUROC 0.996 / |slope| 0.990 /   |창평균 e| 0.995
#       전체   : movstd AUROC 0.953                  / **|창평균 e| 0.984** (CoST 0.978 보다 높다)
#   V=50 의 부호 있는 창평균은 steady 5% 분위 -0.0519 > ns 95% 분위 -0.0963 으로
#   **90% 구간이 겹치지 않는다.** 그런데 CoST 의 V=50 AUROC 는 0.956 으로 전 속도 중 최하다.
#   scale/shift 증강 제거가 저속에서만 크게 들은 것(V=50 AUROC +0.0707, 고속 ±0.005)도
#   "증강이 파괴하던 것이 오프셋"이라는 같은 이야기다. 그것을 손실로 **요구**해 본다.
#
# 절댓값을 쓰는 이유 : 부호 있는 창평균은 고속에서 AUROC 0.929 로 떨어진다 (overshoot 이
#   양의 e 를 만들어 한쪽 부호만 보면 놓친다). 절댓값은 전 속도에서 0.98 이상이다.
#
# 아이디어 03 과의 차이, 그리고 왜 더 될 만한가
#   - 재료가 강하다 : V=50 에서 0.999 vs s̄ 의 0.931
#   - **타깃이 단순하다** : 이동평균(선형) + 절댓값. s̄ 는 미분·부호·카운트를 거친 비선형 양이다.
#     선형 head 가 맞춰야 하는 대상이 단순하면 압력이 표현으로 전가되기 쉽다.
#     (03 의 흡수 실패 — aux_r2 0.09 — 가 이 지점의 문제였다)
#   - 창을 바꾸지 않으므로 학습량이 baseline 과 **자동으로 동등**하다 (600 iteration 그대로)
#
# beta : **타깃을 표준화해서** 쓴다 (연구자 승인 2026-10-29). 원단위 m̄ 의 분산은 0.001675 로
#   s̄(0.02386)의 1/14 이라, 항 크기 10% 를 맞추려면 beta ~= 194 가 필요한데 03 에서 beta=30 이
#   이미 발산했다(10 iteration 에 L_aux 1.8e4 -> NaN). 표준화하면 분산이 1 로 고정되어
#   beta = 3.25/1.0/10 = **0.33** 이 되고, 유도가 타깃 단위에 무관해진다.
#   표준화는 **학습 fold 시점으로만** 적합한다 (누출 차단).
#
# 인과성 : 창 평균은 `min_periods=1` 의 **인과 이동평균**이다. 비교군 MovStd-Slope 는 앞 w-1 개를
#   첫 유효값으로 채우는데(01_stat_ssd.py:127) 그건 k<143 에서 미래를 쓴다. 아이디어 03 이
#   인과성을 명시적으로 지킨 것과 일관되게 **완전 인과**로 간다.
#   (참고: 채움 관례를 쓰면 V=50 AUROC 가 0.983 -> 0.999 로 더 좋아지지만 쓰지 않는다)
#
# 변경 범위 : 손실 항 하나. trunk·trend·seasonal 분기·증강·창 길이(144)·표현 차원(320)·
#   평가 후단 전부 baseline(CoST-W144-noscaleshift)과 같다. 창이 같아 창 대조군이 필요 없다.
#
# 원본 저장소는 한 줄도 수정하지 않는다 — 런타임 서브클래스 shim 이다.
#
# 실행 : python code/04_proposed/05_cost_offset_aux.py                   (A, beta=0.33 기본)
#        python code/04_proposed/05_cost_offset_aux.py tension beta=1    (민감도)
#        python code/04_proposed/05_cost_offset_aux.py tension amb       (ambiguous 포함, 추론만)
# =============================================================================
import os, sys, time, copy
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

# --- 보조 손실 ---
BETA_DEFAULT = 0.33        # 표준화 타깃(분산 1) 기준, L_MoCo 수렴값 3.25 의 10%
BETA = float(_beta[0].split('=')[1]) if _beta else BETA_DEFAULT
READOUTS = ['point', 'mean']    # 비교할 읽기 시점 2종 (아이디어 03 과 동일)

WINDOW = 144               # **baseline 과 동일**. 창을 바꾸지 않으므로 학습량도 동등하다

# --- 나머지는 04_cost.py / 06_cost_ablation.py 와 완전히 동일 ---
N_ITERS = None if SPLIT == 'tension' else 600
DEVICE, BATCH_SIZE, LR, REPR_DIMS = 'cuda', 8, 0.001, 320
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005

IN_PATH = 'outputs/merged_data.csv'
STAGE = '13_proposed_cost_offset_aux_loss'
OUT_DIR = splits.resolve(SPLIT, STAGE)
CKPT_DIR = splits.ckpt_dir(SPLIT, STAGE)
SUFFIX = '' if BETA == BETA_DEFAULT else f'-b{BETA:g}'    # 기본값이 아니면 이름에 붙여 섞이지 않게

T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS


# =============================================================================
# 함수 — m̄ (창 평균 잔차의 크기)
# =============================================================================
def offset_mag(e, w=WINDOW):
    # 역할 : 시점마다 "창 [t-w+1, t] 의 평균 잔차" 의 절댓값. 창이 전부 과거이므로 완전 인과다.
    #        min_periods=1 이라 run 앞부분은 가진 만큼으로 평균한다 (미래를 쓰지 않는다).
    # 출력 : (n,) >= 0. 작을수록 설정값 근처 (= steady), 클수록 벗어나 있다 (= transient)
    return pd.Series(e).rolling(w, min_periods=1).mean().abs().to_numpy()


# =============================================================================
# shim 1 — 증강 : scale·shift 제거(baseline 과 동일) + **m̄ 채널은 건드리지 않는다**
# =============================================================================
class JitterCh0Dataset(cost_module.PretrainDataset):
    # 역할 : baseline 과 같은 'jitter only' 증강을 쓰되, 2번째 채널에 실은 m̄ 가
    #        잡음에 오염되지 않게 **채널 0 에만** 적용한다.
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])
        return out


# =============================================================================
# shim 2 — 인코더 : m̄ 채널을 떼어 내고 채널 0 만 원본 forward 에 넘긴다
# =============================================================================
class Ch0CoSTEncoder(CoSTEncoder):
    # 역할 : 입력 (B, T, 2) 에서 채널 0(오차 e) 만 원본 forward 에 넘긴다.
    #        trunk·trend·seasonal 은 baseline 과 비트 단위로 같은 입력을 본다.
    #        m̄ 는 손실 계산에만 쓰이며 CoSTModel 이 x_q 에서 직접 읽는다.
    def forward(self, x, tcn_output=False, mask='all_true'):
        return super().forward(x[..., :1].contiguous(), tcn_output, mask)


# =============================================================================
# shim 3 — 손실 : 보조 항 추가  ★ 아이디어의 본체
# =============================================================================
class OAuxCoSTModel(CoSTModel):
    # 역할 : 원본 forward 를 그대로 수행하고 **끝에 보조 항 하나**를 더한다.
    #        trend 원시 표현(정규화·head 통과 전)에서 선형 head 로 표준화된 m̄ 를 예측하는 MSE 다.
    #        표준화 통계(mu, sd)는 **학습 fold 에서만** 계산해 fold 마다 넣어 준다 (누출 차단).
    def __init__(self, *a, beta, readout, dim, **kw):
        super().__init__(*a, dim=dim, **kw)
        self.beta, self.readout = beta, readout
        self.aux_head = nn.Linear(dim, 1)         # 160 -> 1. 선형 최소 구조
        self.mu, self.sd = 0.0, 1.0               # set_target_stats() 가 fold 마다 채운다
        self.last_aux = float('nan')

    def set_target_stats(self, mu, sd):
        # 역할 : 학습 fold 시점의 m̄ 평균·표준편차를 받아 타깃 표준화에 쓴다.
        self.mu, self.sd = float(mu), float(sd) if sd > 0 else 1.0

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
        s = (x_q[..., 1] - self.mu) / self.sd                 # (B, T) 표준화된 목표값 m̄
        if self.readout == 'point':
            pred, tgt = self.aux_head(t_raw[:, rand_idx]).squeeze(-1), s[:, rand_idx]
        else:                                                 # 'mean' — 시점별 오차를 전 시점 평균
            pred, tgt = self.aux_head(t_raw).squeeze(-1), s
        m = torch.isfinite(tgt)                               # 패딩(NaN) 제외
        aux = F.mse_loss(pred[m], tgt[m]) if m.any() else t_raw.new_zeros(())
        self.last_aux = float(aux.detach())
        return loss + self.beta * aux


# =============================================================================
# Step 1~2. 데이터 로드 + 저장소 입력 형식 변환 (채널 2개: e, m̄)
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
    x[i, :lengths[i], 1] = offset_mag(e)
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
m_flat = np.concatenate([x[i, :lengths[i], 1] for i in range(len(run_ids))])
print(f'[Step 1] runs = {len(run_ids)}, 입력 {x.shape} ({time.time() - t0:.1f} s)')
print(f'[Step 2] 창 {WINDOW} (baseline 동일) | m_bar = |인과 이동평균(e, {WINDOW})|')
print(f'[Step 2] m_bar 평균 {m_flat.mean():.5f}  분산 {m_flat.var():.6f}  '
      f'5~95% [{np.percentile(m_flat, 5):.5f}, {np.percentile(m_flat, 95):.5f}]')
print(f'[Step 2] beta = {BETA:g}  (타깃을 학습 fold 통계로 표준화하므로 분산 1 기준, '
      f'L_MoCo 수렴값 3.25 대비 보조 항 비중 {BETA / 3.25:.1%})')


def unpad(z):
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


def new_model(readout):
    # 역할 : 04_cost.py 와 같은 설정의 CoST 를 만들고 인코더·손실을 바꿔 끼운다 (원본 미수정).
    m = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
             output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)
    m.net = Ch0CoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                           length=WINDOW, hidden_dims=64, depth=10).to(DEVICE)
    m.cost = OAuxCoSTModel(m.net, copy.deepcopy(m.net), kernels=KERNELS,
                           dim=m.net.component_dims, alpha=ALPHA, K=256, device=DEVICE,
                           beta=BETA, readout=readout).to(DEVICE)
    return m


# =============================================================================
# Step 3. 읽기 시점 2종 x fold x seed
# =============================================================================
if not INFER:
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)

for readout in READOUTS:
    method = f'CoST-W{WINDOW}-oaux-{readout}{SUFFIX}'
    if os.path.exists(f'{OUT_DIR}/{method}_fold_metrics.csv'):
        print(f'[Step 3] {method}: 이미 결과가 있어 건너뛴다', flush=True)
        continue
    run_m, fold_m, loss_m = [], [], []
    for f in sp.folds:
        is_train_run, is_val_run = sp.run_role(f)
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        # **학습 fold 시점으로만** 표준화 통계를 만든다 (누출 차단)
        mu = float(np.nanmean(m_flat[is_train_pt]))
        sd = float(np.nanstd(m_flat[is_train_pt]))
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)
            iter_loss, aux_log, epoch_iters = [], [], []
            model = new_model(readout)
            model.cost.set_target_stats(mu, sd)
            model.after_iter_callback = lambda m_, l: (iter_loss.append(l),
                                                       aux_log.append(m_.cost.last_aux))
            model.after_epoch_callback = lambda m_, _l: epoch_iters.append(m_.n_iters)
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
            fr['aux_target_mu'], fr['aux_target_sd'] = mu, sd
            fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
            if not INFER:
                fr['fit_sec'], fr['n_epochs'], fr['n_iters'] = t_fit, model.n_epochs, model.n_iters
                fr['iters_per_epoch'] = epoch_iters[0] if epoch_iters else np.nan
                fr['data_passes'] = model.n_iters / epoch_iters[0] if epoch_iters else np.nan
                fr['final_loss'] = float(np.mean(iter_loss[-20:])) if iter_loss else np.nan
                # **보조 과제가 실제로 학습됐는지의 직접 증거** (아이디어 03 의 aux_r2 와 같은 역할)
                fr['aux_loss_first'] = float(np.mean(aux_log[:20])) if aux_log else np.nan
                fr['aux_loss_final'] = float(np.mean(aux_log[-20:])) if aux_log else np.nan
                fr['aux_var_target'] = 1.0            # 표준화했으므로 학습 fold 기준 분산은 1
                fr['aux_r2'] = 1.0 - fr['aux_loss_final'] if aux_log else np.nan
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
                            'CoST contrastive + offset aux loss')
    ev.save_results(OUT_DIR, method, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS,
                     'WINDOW': WINDOW, 'N_ITERS': N_ITERS, 'DEVICE': DEVICE,
                     'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'REPR_DIMS': REPR_DIMS,
                     'KERNELS': KERNELS, 'ALPHA': ALPHA,
                     'idea': 'hyperparameters.md 13.10 P8 (오프셋 크기 보조 손실)',
                     'motivation': '저속 진단 2026-10-29 — V=50 에서 |창평균 e| AUROC 0.999 vs '
                                   'movstd 0.890. CoST 의 V=50 AUROC 0.956 이 전 속도 최하',
                     'aux_loss': 'beta * MSE(Linear(trend_160), z(|mean(e,144)|))',
                     'aux_target': 'm_bar = |인과 이동평균(e, 144)|, min_periods=1 (완전 인과)',
                     'aux_target_standardized': '학습 fold 시점의 평균·표준편차로만 표준화 (누출 차단)',
                     'BETA': BETA, 'beta_rule': '표준화 타깃(분산 1) 기준 L_MoCo 수렴값 3.25 의 10%',
                     'readout': readout,
                     'readout_meaning': {'point': 'MoCo 가 쓰는 무작위 시점 1개',
                                         'mean': '시점별 오차를 전 시점에서 평균'}[readout],
                     'aux_head': f'nn.Linear({REPR_DIMS // 2}, 1), 선형 최소 구조',
                     'augmentation': 'jitter only (scale, shift removed via runtime shim)',
                     'uses_setting_value': '없음. m_bar 는 e 만으로 계산된다 (속도 설정값 불필요)',
                     'baseline': 'CoST-W144-noscaleshift (창이 같아 창 대조군 불필요, 학습량도 동등)',
                     'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
    for metric, label in [('contamination_rate', 'Contamination rate'),
                          ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[done] {method}', flush=True)

print(f'[done] 오프셋 보조 손실 완료 -> {OUT_DIR}', flush=True)
