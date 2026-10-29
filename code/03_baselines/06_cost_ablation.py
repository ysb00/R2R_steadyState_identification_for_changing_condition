# =============================================================================
# 03_baselines/06_cost_ablation.py : CoST ablation 2종
#
# 왜 하는가 : CoST 가 현재 비교군 1위다(창 144 에서 AUPRC 0.832). "왜 잘하는가" 를 분해하면
#   제안 방법 설계의 근거가 된다. 두 가지를 본다.
#
#   A. 표현 성분 분해 — 추세(trend) / 계절(season) / 결합
#      CoST 의 표현은 `cat([trend[:, -1], season[:, -1]])` 로 앞 160차원이 추세, 뒤 160차원이
#      계절이다(`cost.py:334`, component_dims = output_dims // 2). 어느 성분이 steady/non-steady
#      정보를 담는지 본다. **재학습이 필요 없다** — 저장된 체크포인트로 한 번 encode 한 뒤
#      슬라이스만 달리해 같은 트랙 A 후단에 넣는다.
#      관련: research_plan.md 3.1 D3 (신호 분해 구조)
#
#   B. scaling/offset 증강 제거 — 재학습
#      CoST 의 증강은 `jitter(shift(scale(x)))`, sigma = 0.5 다 (`cost.py:46-53`).
#        scale  : x * (randn * 0.5 + 1)     곱셈 스케일링 (약 2.3% 확률로 부호가 뒤집힌다)
#        shift  : x + randn * 0.5           덧셈 오프셋
#        jitter : x + randn(x.shape) * 0.5  시점별 잡음
#      본 데이터의 steady 구간 e 표준편차가 0.041 kg 이므로 **구분 정보보다 12배 큰 교란**이다.
#      research_plan.md 3.1 D2("scaling 은 오차 진폭(구분 정보)을 변경")가 지목한 지점이고,
#      scale·shift 를 끄고 재학습해 D2 를 직접 검증한다.
#
#      **jitter 는 남긴다.** CoST 는 두 view 가 증강으로만 달라지므로(crop offset 은 두 view 에
#      동일하게 적용된다) 증강을 전부 끄면 양성 쌍이 완전히 같아져 대조 과제가 자명해진다.
#
# 원본 수정 : 없다. B 는 `cost.PretrainDataset` 을 런타임에 서브클래스로 바꿔치기하는 shim 이다
#   (docs/worklog.md 2026-10-23 결정 3 "shim 우선"). 저장소 파일은 건드리지 않는다.
#
# 창 길이 : **144 고정** (docs/periodic_disturbance.md §7 규칙 1b). 본문 표와 같은 값이라
#   기존 CoST-W144 결과와 직접 비교된다. 창 길이 민감도는 이미 CoST-W{144,160,256,306,512} 에 있다.
#
# 입력 : outputs/merged_data.csv / splits.py 가 분할에 맞는 fold 파일을 읽는다
#        <분할 폴더>/04_cost/checkpoints/CoST-W144_fold*_seed*.pt   (Ablation A 는 학습 생략)
#        **같은 분할로 04_cost.py 를 먼저 돌려야 한다** — C 를 돌리려면 C 체크포인트가 필요하다
# 출력 : A -> outputs/03_baselines/06_cost_ablation/ , C -> outputs/04_speed_extrap/06_cost_ablation/
#        변형별 run/fold/summary/config csv + heatmap
#        + 기존 CoST-W144 와 비교하는 그림 2종
# 실험 : 인자로 고른다 — 'tension' (A: 장력 5-fold) / 'c' (C: 저속 학습 -> 고속 test, 8 fold)
# 실행 : python code/03_baselines/06_cost_ablation.py        (A 실험, 기본값)
#        python code/03_baselines/06_cost_ablation.py c      (C 실험)
# =============================================================================
import os, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'CoST'))
import evaluation as ev
import splits
import cost as cost_module                                              # shim 대상 (원본 미수정)
from cost import CoST                                                   # 외부 저장소 원본
from utils import init_dl_program                                       # 외부 저장소 원본의 시드 초기화


SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만
# Ablation A 는 원래부터 재학습이 없다. Ablation B 도 추론 모드에서는 기존 noscaleshift 체크포인트를
# 불러 쓴다 — 즉 **같은 분할로 06_cost_ablation.py 를 먼저 돌려 두어야** 한다.
# =============================================================================
# hyperparameter — 창 길이와 증강을 뺀 나머지는 04_cost.py 와 완전히 동일하다
# =============================================================================
SEED, N_SEEDS = 0, 5
WINDOW = 144                    # 규칙 1b 유도값. 본문 표와 같은 창으로 고정한다
N_ITERS = None if SPLIT == 'tension' else 600
# A 실험 : 저장소 기본값 None. 내부 규칙(학습 데이터 원소 수 > 100,000 이면 600, 아니면 200)이 적용되고,
#          학습 168 run x 1300 = 218,400 이라 **항상 600** 이 선택됐다.
# C 실험 : **600 으로 고정한다** (연구자 결정 2026-10-26). fold 마다 학습 run 수가 60~120 으로 다른데
#          자동 규칙을 두면 cv_S1(60 run, 78,000) 만 200 이 되어 혼자 1/3 만 학습한다. 그러면 cv_S1 의
#          성능이 '외삽이 어려워서' 인지 '덜 학습해서' 인지 분리되지 않는다. 600 은 A 에서 저장소 규칙이
#          실제로 고른 값이므로 임의값을 새로 들인 것이 아니다.
#          **As-published 에서 벗어난 유일한 지점** (docs/hyperparameters.md 13.0)
DEVICE, BATCH_SIZE, LR, REPR_DIMS = 'cuda', 8, 0.001, 320
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]
ALPHA = 0.0005
COMPONENT_DIMS = REPR_DIMS // 2  # = 160. 앞 160 = 추세, 뒤 160 = 계절

IN_PATH = 'outputs/merged_data.csv'

COST_DIR = splits.resolve(SPLIT, '04_cost (ablation)')       # Step 5 가 읽는 CoST-W144 **결과** 폴더
COST_CKPT = splits.ckpt_dir(SPLIT, '04_cost (ablation)')     # CoST-W144 **체크포인트** 출처. 같은 분할이어야 한다
CKPT_DIR = splits.ckpt_dir(SPLIT, '06_cost_ablation (ablation)')   # noscaleshift 체크포인트 출처 (추론 모드)
OUT_DIR = splits.resolve(SPLIT, '06_cost_ablation (ablation)')
BASE_METHOD = f'CoST-W{WINDOW}'                                         # 비교 기준 (이미 실행됨)

T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS


class NoScaleShiftDataset(cost_module.PretrainDataset):
    # 역할 : scaling(곱셈)·offset(덧셈) 증강을 빼고 jitter 만 남긴다. 저장소 원본을 수정하지 않는
    #        런타임 shim 이다. CoST.fit 이 모듈 전역 PretrainDataset 를 참조하므로 교체가 적용된다.
    def transform(self, x):
        return self.jitter(x)


# =============================================================================
# Step 1~2. 데이터 로드 + 저장소 입력 형식 변환 (04_cost.py 와 동일)
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
lengths = np.array([len(by_run[r]) for r in run_ids])
x = np.full((len(run_ids), lengths.max(), 1), np.nan)
for i, r in enumerate(run_ids):
    x[i, :lengths[i], 0] = by_run[r].e.to_numpy()
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 1] runs = {len(run_ids)}, 입력 {x.shape} ({time.time() - t0:.1f} s)')


def unpad(z):
    # 역할 : (n_run, t_max, d) 에서 패딩을 떼고 시점 순서로 이어 붙인다 (e_flat, id_flat 과 같은 순서)
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


def new_model():
    # 역할 : 04_cost.py 와 동일한 설정의 CoST 를 만든다 (학습·추론 공용)
    return CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
                output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE)


def encode_all(model):
    # 역할 : 전체 run 을 이동창(causal)으로 시점별 인코딩한다. 창 길이를 정확히 WINDOW 로 맞춘다.
    return unpad(model.encode(x, mode='forecasting', casual=True, sliding_length=1,
                              sliding_padding=WINDOW - 1, batch_size=len(run_ids)))


def evaluate(z, method, f, seed, is_train_pt, is_val_pt, extra):
    # 역할 : 표현 z 를 트랙 A 후단에 넣어 채점한다 (모든 비교군과 동일한 후단).
    # 입력 : z (n_point, d) 시점별 표현 / method 이름 / f fold / seed / is_val_pt 검증 시점 마스크
    #        is_train_pt 학습 시점 마스크 — C 실험에서는 ~is_val_pt 와 다르다 (미사용 run 이 있다)
    #        extra fold_metrics 에 덧붙일 dict
    # 출력 : (run_metrics, fold_row)
    gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
    pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
    r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, method, f, seed)
    for k, v in {**extra, 'steady_cluster': sidx, 'e_mean_c0': emeans[0], 'e_mean_c1': emeans[1]}.items():
        fr[k] = v
    return r, fr


# =============================================================================
# Step 3. Ablation A — 표현 성분 분해 (재학습 없음, 체크포인트 재사용)
# =============================================================================
SLICES = {'trend': slice(0, COMPONENT_DIMS),                # 앞 160차원
          'season': slice(COMPONENT_DIMS, REPR_DIMS)}       # 뒤 160차원
res = {f'{BASE_METHOD}-{k}': {'run': [], 'fold': []} for k in SLICES}
todo_a = [m for m in res if not os.path.exists(f'{OUT_DIR}/{m}_fold_metrics.csv')]
if todo_a:
    for f in sp.folds:
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            ckpt = f'{COST_CKPT}/{BASE_METHOD}_fold{f}_seed{seed}.pt'
            assert os.path.exists(ckpt), f'체크포인트가 없다: {ckpt} (04_cost.py 를 먼저 실행할 것)'
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)
            model = new_model()
            model.load(ckpt)                                            # 학습 생략 — 가중치만 불러온다
            z = encode_all(model)                                       # encode 는 한 번만
            for name, sl in SLICES.items():
                method = f'{BASE_METHOD}-{name}'
                if method not in todo_a:
                    continue
                r, fr = evaluate(z[:, sl], method, f, seed, is_train_pt, is_val_pt,
                                 {'window': WINDOW, 'component': name, 'repr_dims': sl.stop - sl.start,
                                  'retrained': False, 'total_sec': time.time() - t0})
                res[method]['run'].append(r), res[method]['fold'].append(fr)
                ev.append_progress(OUT_DIR, method, 'progress_fold', fr)
            print(f'[Step 3] fold {f} seed {seed}: encode {time.time() - t0:.0f} s, '
                  f'trend 보존률 {res[f"{BASE_METHOD}-trend"]["fold"][-1].pooled_retention_rate.iloc[0]:.3f}, '
                  f'season 보존률 {res[f"{BASE_METHOD}-season"]["fold"][-1].pooled_retention_rate.iloc[0]:.3f}',
                  flush=True)
    for method in todo_a:
        run_m = pd.concat(res[method]['run'], ignore_index=True)
        fold_m = pd.concat(res[method]['fold'], ignore_index=True)
        ev.save_results(OUT_DIR, method, run_m, fold_m,
                        {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOW': WINDOW,
                         'ablation': 'component', 'component': method.split('-')[-1],
                         'repr_dims': COMPONENT_DIMS, 'retrained': False,
                         'checkpoint_from': BASE_METHOD,
                         'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
        for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
            ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
        print(f'[done] {method}')
else:
    print('[Step 3] Ablation A: 이미 결과가 있어 건너뛴다')

# =============================================================================
# Step 4. Ablation B — scaling/offset 증강 제거 후 재학습
# =============================================================================
method_b = f'{BASE_METHOD}-noscaleshift'
if not os.path.exists(f'{OUT_DIR}/{method_b}_fold_metrics.csv'):
    if not INFER:                                                       # 추론 모드는 학습하지 않으므로 shim 도 불필요
        cost_module.PretrainDataset = NoScaleShiftDataset               # shim 적용 (원본 미수정)
        print(f'[Step 4] shim 적용: PretrainDataset.transform = jitter only (scale·shift 제거)')
        os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
    run_m, fold_m, loss_m = [], [], []
    for f in sp.folds:
        is_train_run, is_val_run = sp.run_role(f)
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)
            iter_loss, epoch_iters = [], []
            model = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=WINDOW,
                         output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE,
                         after_iter_callback=lambda _m, l: iter_loss.append(l),
                         after_epoch_callback=lambda m, _l: epoch_iters.append(m.n_iters))
            if INFER:                                       # 재학습 없이 저장된 인코더만 불러온다
                ck = f'{CKPT_DIR}/{method_b}_fold{f}_seed{seed}.pt'
                assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습을 돌릴 것)'
                model.load(ck)
                z = encode_all(model)
                r, fr = evaluate(z, method_b, f, seed, is_train_pt, is_val_pt,
                                 {'window': WINDOW, 'ablation': 'no scale/shift aug', 'retrained': True,
                                  'total_sec': time.time() - t0})
                run_m.append(r), fold_m.append(fr)
                ev.append_progress(OUT_DIR, method_b, 'progress_fold', fr)
                print(f'[Step 4] fold {f} seed {seed}: 체크포인트 재사용 (학습 생략), '
                      f'전체 {time.time() - t0:.0f} s, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                      f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)
                continue
            loss_log = model.fit(x[is_train_run], n_iters=N_ITERS, verbose=False)
            t_fit = time.time() - t0
            model.save(f'{OUT_DIR}/checkpoints/{method_b}_fold{f}_seed{seed}.pt')
            z = encode_all(model)
            r, fr = evaluate(z, method_b, f, seed, is_train_pt, is_val_pt,
                             {'window': WINDOW, 'ablation': 'no scale/shift aug', 'retrained': True,
                              'fit_sec': t_fit, 'total_sec': time.time() - t0,
                              'n_epochs': model.n_epochs, 'n_iters': model.n_iters,
                              'iters_per_epoch': epoch_iters[0] if epoch_iters else np.nan,
                              'final_loss': float(np.mean(iter_loss[-20:])) if iter_loss else np.nan})
            lo = pd.DataFrame({'fold': f, 'seed': seed,
                               'iter': np.arange(1, len(iter_loss) + 1), 'loss': iter_loss})
            run_m.append(r), fold_m.append(fr), loss_m.append(lo)
            ev.append_progress(OUT_DIR, method_b, 'progress_fold', fr)
            ev.append_progress(OUT_DIR, method_b, 'loss', lo)
            print(f'[Step 4] fold {f} seed {seed}: 학습 {t_fit:.0f} s, 손실 {iter_loss[0]:.3f} -> '
                  f'{np.mean(iter_loss[-20:]):.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                  f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)
    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    if not INFER:                                           # 추론 모드는 학습하지 않아 손실 곡선이 없다
        ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, method_b, 'CoST contrastive loss')
    ev.save_results(OUT_DIR, method_b, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOW': WINDOW,
                     'N_ITERS': N_ITERS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'REPR_DIMS': REPR_DIMS,
                     'KERNELS': KERNELS, 'ALPHA': ALPHA, 'ablation': 'no scale/shift aug',
                     'augmentation': 'jitter only (scale, shift removed via runtime shim)',
                     'retrained': True, 'repo': 'salesforce/CoST',
                     'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
    for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method_b, label)
    print(f'[done] {method_b}')
else:
    print('[Step 4] Ablation B: 이미 결과가 있어 건너뛴다')

# =============================================================================
# Step 5. 기존 CoST-W144 와 비교하는 그림
# =============================================================================
VARIANTS = [(f'{BASE_METHOD}', 'Full (baseline)', '#d55e00', COST_DIR),
            (f'{BASE_METHOD}-trend', 'Trend only', '#0072b2', OUT_DIR),
            (f'{BASE_METHOD}-season', 'Season only', '#009e73', OUT_DIR),
            (f'{BASE_METHOD}-noscaleshift', 'No scale/shift aug', '#cc79a7', OUT_DIR)]
rows = []
for m, lbl, col, d in VARIANTS:
    fm = ev.derive_precision_recall(pd.read_csv(f'{d}/{m}_fold_metrics.csv'))
    mean = fm.mean(numeric_only=True)
    rows.append({'method': m, 'label': lbl, 'color': col,
                 # 두 클래스를 대칭으로 기록한다. steady 쪽을 빠뜨리면 연구의 최우선 지표인
                 # 오염률(= 1 - precision_steady)이 그림에서 사라진다 (outputs_guide.md §2.3).
                 'Precision (st)': mean.pooled_precision_steady,        # = 1 - 오염률
                 'Recall (st)': mean.pooled_recall_steady,              # = 보존률
                 'F1 (st)': mean.pooled_f1_steady,
                 'Precision (ns)': mean.pooled_precision_nonsteady,
                 'Recall (ns)': mean.pooled_recall_nonsteady,
                 'F1 (ns)': mean.pooled_f1_nonsteady,
                 'AUROC': mean.pooled_auroc, 'AUPRC (ns)': mean.pooled_auprc,
                 'contamination': mean.pooled_contamination_rate,
                 'entry_bias': mean.runmean_entry_err_s, 'entry_mae': mean.runmae_entry_err_s})
cmp = pd.DataFrame(rows)
cmp.drop(columns='color').to_csv(f'{OUT_DIR}/ablation_comparison.csv', index=False)

# steady 쪽 2개 + non-steady 쪽 3개 + 순위 2개. 모두 '높을수록 좋다' 로 방향을 통일했다
# (오염률은 '낮을수록 좋다' 라 같은 축에 섞으면 오독을 부른다 — Precision (st) = 1 - 오염률 로 넣는다)
METRICS = ['Precision (st)', 'Recall (st)', 'F1 (st)',
           'Precision (ns)', 'Recall (ns)', 'F1 (ns)', 'AUROC', 'AUPRC (ns)']
BASE_RATE = 0.0813                                                      # non-steady 비율

# 점 그림(dot plot). 값이 0.8~1.0 에 몰려 막대로는 차이가 안 보이는데 막대는 0 에서 시작해야
# 하므로 축을 좁힐 수 없다. 점은 길이로 크기를 나타내지 않으므로 축을 좁혀도 왜곡이 아니다.
# 가로축 = 지표, 세로축 = 점수.
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH, 3.2), layout='constrained')
xpos = np.arange(len(METRICS))
off = np.linspace(-0.26, 0.26, len(cmp))
for xx in xpos:
    ax.axvline(xx, color='#e1e0d9', linewidth=0.8, zorder=0)
for i, r in cmp.iterrows():
    ax.plot(xpos + off[i], [r[m] for m in METRICS], linestyle='none', marker='o', markersize=6,
            color=r['color'], markeredgecolor='#ffffff', markeredgewidth=0.6, label=r['label'], zorder=3)
ax.set_xticks(xpos, [m.replace(' (', '\n(') for m in METRICS], fontsize=ev.FONT_SIZE - 1)
ax.set_ylim(float(cmp[METRICS].min().min()) - 0.02, 1.005)
ax.set_xlim(-0.6, len(METRICS) - 0.4)
ax.set_ylabel('Score')
ax.grid(axis='x', visible=False)
ax.legend(ncol=4, frameon=False, fontsize=ev.FONT_SIZE - 1,
          loc='upper center', bbox_to_anchor=(0.5, -0.20))
ax.set_title('(st) = steady positive,  (ns) = non-steady positive;  higher is better for all.  '
             f'AUPRC (ns) baseline = {BASE_RATE:.3f}',
             fontsize=ev.FONT_SIZE - 2, color='#52514e', loc='left')
for ext in ('png', 'pdf'):
    pth = f'{OUT_DIR}/ablation_metrics.{ext}'
    assert not os.path.exists(pth), f'이미 그림 파일이 있다 (덮어쓰기 금지): {pth}'
    fig.savefig(pth, dpi=ev.DPI)
plt.close(fig)
print(f'[save] {OUT_DIR}/ablation_metrics.png / .pdf')

# 진입 시점 오차는 부호가 있고 0 이 의미를 가지므로 막대가 맞다 (0 기준선이 자연스럽다).
fig, ax = plt.subplots(figsize=(ev.PAGE_WIDTH * 0.68, 2.7), layout='constrained')
ax.axhline(0, color='#0b0b0b', linewidth=0.8)
xx = np.arange(len(cmp))
ax.bar(xx - 0.19, cmp.entry_bias, width=0.36, color=cmp.color.tolist())
ax.bar(xx + 0.19, cmp.entry_mae, width=0.36, color=cmp.color.tolist(), alpha=0.45)
ax.set_xticks(xx, [l.replace(' ', '\n') for l in cmp.label], fontsize=ev.FONT_SIZE - 2)
ax.set_ylabel('Steady-entry error (s)')
ax.text(0.02, 0.97, 'left = bias (negative: declares steady early),  right = MAE (lighter)',
        transform=ax.transAxes, va='top', fontsize=ev.FONT_SIZE - 2, color='#52514e')
for ext in ('png', 'pdf'):
    pth = f'{OUT_DIR}/ablation_entry_error.{ext}'
    assert not os.path.exists(pth), f'이미 그림 파일이 있다 (덮어쓰기 금지): {pth}'
    fig.savefig(pth, dpi=ev.DPI)
plt.close(fig)
print(f'[save] {OUT_DIR}/ablation_entry_error.png / .pdf')

print('\n' + cmp.drop(columns='color').set_index('label').round(3).to_string())
print('[done] CoST ablation 완료')
