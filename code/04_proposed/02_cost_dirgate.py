# =============================================================================
# 04_proposed/02_cost_dirgate.py : 제안 아이디어 02 — 설정값 방향 게이트 (trend branch)
#
# 아이디어 : "지금 설정값에 다가가는 중인가" 를 시점마다 계산해 CoST 의 **trend 표현을 조건화**한다.
#   transient 는 설정값으로 돌아가는 운동이고 steady 는 설정값 주변의 방향 없는 요동이다.
#   그런데 추세의 **부호**로는 둘을 가를 수 없다 — 아래로 처진 뒤 회복하면 추세가 +, 위로
#   overshoot 된 뒤 회복하면 - 다 (실측: 184 run 중 처짐 181 / overshoot 48). 그래서 방향을
#   **설정값 기준 접근/이탈** 로 정의한다:  r = -e·ė > 0 이면 |e| 가 줄어드는 중 = 접근.
#   이러면 sag 회복과 overshoot 회복이 같은 부호로 묶인다.
#
# **평활이 필수다 — 이 아이디어의 핵심 발견.**
#   샘플 단위 r 은 신호가 없다 (transient 의 접근 비율 0.428, steady 0.502 -> AUROC 0.743).
#   이 데이터는 lag-1 자기상관이 0.922 이고 회전 외란이 겹쳐 있어 샘플 단위 미분을 요동이 지배한다.
#   **1회전 길이 P(V) 로 평활**하면 회전 외란이 지워지고 회복 방향만 남는다 -> AUROC 0.912.
#   P(V) 는 유도값이라 임의값이 늘지 않는다 (아이디어 01 과 같은 둘레 상수 783.5 에서 나온다).
#   설계·수식·사전진단은 docs/idea/02_setpoint_direction_gate.md
#
# 변경 범위 : trend 분기 출력에 FiLM 게이트 **한 층**. 영 초기화라 학습 시작 시점에는
#   baseline 과 **완전히 같다**. trunk·seasonal·증강·손실·창 길이·평가 후단 전부 동일하다.
#
# **창이 baseline 과 같은 144 다** -> 창 대조군이 필요 없고 CoST-W144-noscaleshift 와 직접 비교된다
#   (아이디어 01 은 창 288 이 강제되어 대조군이 필요했다).
#
# 원본 저장소는 한 줄도 수정하지 않는다 — 런타임 서브클래스 shim 이다.
#
# 입력 : outputs/merged_data.csv      (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
# 출력 : <분할 폴더>/10_proposed_cost_setpoint_direction_gate/
# 실행 : python code/04_proposed/02_cost_dirgate.py            (A 실험, 기본값)
#        python code/04_proposed/02_cost_dirgate.py c          (C 실험)
#        python code/04_proposed/02_cost_dirgate.py tension amb  (A, ambiguous 포함 — 추론만)
#        python code/04_proposed/02_cost_dirgate.py c amb        (C, ambiguous 포함 — 추론만)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
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
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만

# =============================================================================
# hyperparameter
# =============================================================================
SEED, N_SEEDS = 0, 5

# --- 방향 게이트 (docs/idea/02_setpoint_direction_gate.md §4) ---
ROLL_CIRCUMFERENCE_MM = 783.5   # 둘레. 205 run 평균, 지름 249.4 mm (docs/periodic_disturbance.md)
WINDOW = 144                    # **baseline 과 동일**. 규칙 1b 유도값이자 V=50 의 1회전
GATE_NEUTRAL = 0.5              # 게이트 중립값 (= 접근과 이탈이 반반). 패딩 구간에 쓴다

# --- 나머지는 04_cost.py / 06_cost_ablation.py 와 완전히 동일 ---
N_ITERS = None if SPLIT == 'tension' else 600
DEVICE, BATCH_SIZE, LR, REPR_DIMS = 'cuda', 8, 0.001, 320
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005

IN_PATH = 'outputs/merged_data.csv'
STAGE = '10_proposed_cost_setpoint_direction_gate'
OUT_DIR = splits.resolve(SPLIT, STAGE)
CKPT_DIR = splits.ckpt_dir(SPLIT, STAGE)
METHOD = f'CoST-W{WINDOW}-dirgate'

T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS


# =============================================================================
# 함수 — 방향 게이트 계산
# =============================================================================
def rot_samples(v):
    # 역할 : 속도 v 에서 회전체 1바퀴에 해당하는 샘플 수 (평활 길이). V=50 -> 144, V=200 -> 36
    return (1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / v


def direction_gate(e, v, w=WINDOW):
    # 역할 : 시점마다 "창 [t-w+1, t] 안에서 설정값에 접근한 시점의 비율" 을 계산한다.
    #        창이 전부 과거이므로 **창 안에서 중심 평활을 써도 시점 t 기준으로 인과**다.
    #        인과 이동평균을 쓰면 지연 P/2 가 생겨 고속(P=36)에서 반주기와 겹쳐 부호가 뒤집힌다
    #        (실측: V=200 에서 AUROC 0.912 -> 0.599). 그래서 창 내 중심 평활을 쓴다.
    # 입력 : e (n,) run 1개의 오차 신호 / v 그 run 의 속도 설정값 / w 집계 창
    # 출력 : (n,) 게이트 값 [0, 1]. 0.5 = 방향 없음, 작을수록 설정값에 접근 중
    n = len(e)
    p = max(3, int(round(rot_samples(v))))                 # 평활 길이 = 1회전
    X = np.zeros((n, w))                                   # 창 행렬 (앞은 0 패딩 — 파이프라인 규약)
    for i in range(n):
        seg = e[max(0, i - w + 1):i + 1]
        X[i, w - len(seg):] = seg
    pad = p // 2
    k = np.ones(p) / p
    Xs = np.apply_along_axis(lambda r: np.convolve(r, k, mode='valid')[:w], 1,
                             np.pad(X, ((0, 0), (pad, pad)), mode='edge'))
    de = np.gradient(Xs, axis=1) / ev.DT
    r = -Xs * de                                           # > 0 이면 |e| 가 줄어드는 중 = 접근
    nz = np.abs(r) > 1e-14
    return ((r > 0) & nz).sum(1) / np.maximum(nz.sum(1), 1)


# =============================================================================
# shim 1 — 증강 : scale·shift 제거(baseline 과 동일) + **게이트 채널은 건드리지 않는다**
# =============================================================================
class JitterCh0Dataset(cost_module.PretrainDataset):
    # 역할 : baseline 과 같은 'jitter only' 증강을 쓰되, 채널 1 에 실어 보낸 게이트 값이
    #        잡음에 오염되지 않게 **채널 0 에만** 적용한다.
    def transform(self, x):
        out = x.clone()
        out[..., :1] = self.jitter(x[..., :1])
        return out


# =============================================================================
# shim 2 — 인코더 : trend 표현에만 FiLM 게이트  ★ 아이디어의 본체
# =============================================================================
class DirGateCoSTEncoder(CoSTEncoder):
    # 역할 : 입력 (B, T, 2) 에서 게이트 채널을 분리하고, **채널 0(오차 e) 만** 원본 forward 에 넘긴다.
    #        따라서 input_fc·conv trunk·seasonal 분기는 baseline 과 같은 입력을 본다.
    #        원본이 돌려준 trend 에만 FiLM 변조를 건다:  h' = h * (1 + gamma(s)) + beta(s)
    # 영 초기화 : gamma = beta = 0 에서 출발하므로 **학습 시작 시점에 baseline 과 완전히 동일**하다.
    #        게이트가 쓸모없으면 학습이 0 근처에 머물고, 쓸모 있으면 그쪽으로 움직인다.
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.film = nn.Linear(1, 2 * self.component_dims)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, tcn_output=False, mask='all_true'):
        s = torch.nan_to_num(x[..., 1:2], nan=GATE_NEUTRAL)      # 패딩은 중립값
        out = super().forward(x[..., :1].contiguous(), tcn_output, mask)
        if tcn_output:
            return out
        trend, season = out
        gamma, beta = self.film(s).chunk(2, dim=-1)
        return trend * (1 + gamma) + beta, season


# =============================================================================
# Step 1~2. 데이터 로드 + 저장소 입력 형식 변환 (채널 2개: e, 게이트)
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
    x[i, :lengths[i], 1] = direction_gate(e, speeds[i])         # 시점별 게이트 (한 번만 계산)
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
gate = np.concatenate([x[i, :lengths[i], 1] for i in range(len(run_ids))])
print(f'[Step 1] runs = {len(run_ids)}, 입력 {x.shape} ({time.time() - t0:.1f} s)')
print(f'[Step 2] 창 {WINDOW} (baseline 과 동일) | 평활 = 1회전 P(V): ' +
      ', '.join(f'V{int(v)}:{int(round(rot_samples(v)))}' for v in sorted(set(speeds))))
print(f'[Step 2] 게이트 분포: 평균 {gate.mean():.3f}, 5~95% [{np.percentile(gate, 5):.3f}, '
      f'{np.percentile(gate, 95):.3f}]  (0.5 = 방향 없음)')


def unpad(z):
    # 역할 : (n_run, t_max, d) 에서 패딩을 떼고 시점 순서로 이어 붙인다 (e_flat, id_flat 과 같은 순서)
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


def new_model():
    # 역할 : 04_cost.py 와 같은 설정의 CoST 를 만들고 인코더만 게이트 버전으로 바꿔 끼운다.
    import copy
    m = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
             output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)
    m.net = DirGateCoSTEncoder(input_dims=1, output_dims=REPR_DIMS, kernels=KERNELS,
                               length=WINDOW, hidden_dims=64, depth=10).to(DEVICE)
    m.cost = CoSTModel(m.net, copy.deepcopy(m.net), kernels=KERNELS,
                       dim=m.net.component_dims, alpha=ALPHA, K=256, device=DEVICE).to(DEVICE)
    return m


# =============================================================================
# Step 3. fold × seed 별로 학습(또는 체크포인트 로드)하고 비교군과 동일한 후단으로 채점한다
# =============================================================================
if not INFER:
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
if os.path.exists(f'{OUT_DIR}/{METHOD}_fold_metrics.csv'):
    print(f'[Step 3] {METHOD}: 이미 결과가 있어 건너뛴다 (다시 돌리려면 해당 파일들을 먼저 지울 것)')
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
        if INFER:                                       # 재학습 없이 저장된 인코더만 불러온다
            assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습할 것)'
            model.load(ck)
            t_fit = time.time() - t0
        else:
            cost_module.PretrainDataset = JitterCh0Dataset      # 증강 shim (학습 직전에만)
            model.fit(x[is_train_run], n_iters=N_ITERS, verbose=False)
            t_fit = time.time() - t0
            model.save(f'{OUT_DIR}/checkpoints/{METHOD}_fold{f}_seed{seed}.pt')

        z = unpad(model.encode(x, mode='forecasting', casual=True, sliding_length=1,
                               sliding_padding=WINDOW - 1, batch_size=len(run_ids)))
        gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
        pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
        r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
        del z
        fr['window'], fr['total_sec'] = WINDOW, time.time() - t0
        fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
        if not INFER:                                   # 학습량 기록은 학습했을 때만 의미가 있다
            fr['fit_sec'], fr['n_epochs'], fr['n_iters'] = t_fit, model.n_epochs, model.n_iters
            fr['iters_per_epoch'] = epoch_iters[0] if epoch_iters else np.nan
            fr['data_passes'] = model.n_iters / epoch_iters[0] if epoch_iters else np.nan
            fr['samples_seen'] = model.n_iters * BATCH_SIZE * WINDOW
            fr['final_loss'] = float(np.mean(iter_loss[-20:])) if iter_loss else np.nan
            # FiLM 이 영 초기화에서 얼마나 움직였는지 — 게이트가 실제로 쓰였는지의 직접 증거
            fr['film_weight_absmean'] = float(model.net.film.weight.abs().mean())
            fr['film_bias_absmean'] = float(model.net.film.bias.abs().mean())
        run_m.append(r), fold_m.append(fr)
        ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)
        ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
        if not INFER:
            lo = pd.DataFrame({'fold': f, 'seed': seed,
                               'iter': np.arange(1, len(iter_loss) + 1), 'loss': iter_loss})
            loss_m.append(lo)
            ev.append_progress(OUT_DIR, METHOD, 'loss', lo)
        loss_txt = (f'손실 {iter_loss[0]:.3f} -> {np.mean(iter_loss[-20:]):.3f}, '
                    f'FiLM |w| {fr.film_weight_absmean.iloc[0]:.4f}'
                    if iter_loss else '학습 생략 (체크포인트 재사용)')
        print(f'[Step 3] {METHOD} fold {f} seed {seed}: {time.time() - t0:.0f} s, {loss_txt}, '
              f'steady 판정 비율 {pred.mean():.3f}, '
              f'오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

# =============================================================================
# Step 4. 산출물 저장
# =============================================================================
run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
if not INFER:
    ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, METHOD, 'CoST contrastive loss')
ev.save_results(OUT_DIR, METHOD, run_m, fold_m,
                {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOW': WINDOW,
                 'N_ITERS': N_ITERS, 'DEVICE': DEVICE, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
                 'REPR_DIMS': REPR_DIMS, 'KERNELS': KERNELS, 'ALPHA': ALPHA,
                 'idea': 'docs/idea/02_setpoint_direction_gate.md',
                 'gate': 'FiLM on trend: h*(1+gamma(s))+beta(s), s = 창 내 설정값 접근 비율',
                 'gate_smoothing': '1회전 P(V) = (1/DT)*C/V — 유도값 (회전 외란 제거)',
                 'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
                 'smoothing_by_speed': {int(v): int(round(rot_samples(v))) for v in sorted(set(speeds))},
                 'film_init': 'zeros — 학습 시작 시점에 baseline 과 동일',
                 'augmentation': 'jitter only (scale, shift removed via runtime shim)',
                 'uses_setting_value': 'Web Speed — 평활 길이 결정에만. 표현 입력 아님. '
                                       '선례: MovStd-Slope-speed-rule1b (01_stat_ssd.py:163)',
                 'baseline': 'CoST-W144-noscaleshift (창이 같아 창 대조군 불필요)',
                 'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
for metric, label in [('contamination_rate', 'Contamination rate'),
                      ('retention_rate', 'Retention rate')]:
    ev.plot_condition_heatmap(run_m, metric, OUT_DIR, METHOD, label)
print(f'[done] {METHOD} -> {OUT_DIR}')
