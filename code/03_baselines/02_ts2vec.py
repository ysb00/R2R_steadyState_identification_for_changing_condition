# =============================================================================
# 03_baselines/02_ts2vec.py : TS2Vec 비교군 어댑터
#
# 방법 : TS2Vec (Yue et al., AAAI 2022) — 시점 단위·다중 해상도 대조학습.
#        계층적 대조 손실로 같은 시계열의 겹치는 두 crop 을 양성 쌍으로 쓴다.
#        시점마다 표현을 내놓으므로 추가 변환 없이 그대로 트랙 A 후단에 넣는다.
#
# 외부 저장소 : https://github.com/yuezhihan/ts2vec
#   clone 위치 : code/03_baselines/external/ts2vec  (git 추적 제외)
#   커밋       : b0088e14a99706c05451316dc6db8d3da9351163 (2023-06-06)
#   원본은 한 줄도 수정하지 않는다. 본 파일이 어댑터(데이터 변환 + 호출 + 평가)다.
#   본 환경(torch 2.11+cu128 / numpy 2.4.4)에서 수정 없이 동작하는 것을 확인했다.
#   (numpy 에서 사라진 np.bool / np.float 은 data_dropout 과 datautils.py 에만 있고 이 경로에서 쓰지 않는다)
#
# hyperparameter : 저장소 기본값 그대로 쓴다 (train.py 의 argparse 기본값).
#   라벨로 조정하지 않는다 — docs/research_experiments_plan.md 3절
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        outputs/03_baselines/folds.csv (고정된 fold 배정)
# 출력 : outputs/03_baselines/02_ts2vec/  run/fold/summary/config csv + 조건별 heatmap
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행
# 실행 : python code/03_baselines/02_ts2vec.py   (repo 루트에서)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'ts2vec'))
import evaluation as ev
from ts2vec import TS2Vec                                              # 외부 저장소 원본
from utils import init_dl_program                                      # 외부 저장소 원본의 시드 초기화

# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값)
N_SEEDS = 5                     # seed 반복 횟수

DEVICE = 'cuda'                 # 저장소 기본값 (train.py --gpu 0)
BATCH_SIZE = 8                  # 저장소 기본값 (train.py --batch-size)
LR = 0.001                      # 저장소 기본값 (train.py --lr)
REPR_DIMS = 320                 # 저장소 기본값 (train.py --repr-dims)
MAX_TRAIN_LENGTH = 3000         # 저장소 기본값 (train.py --max-train-length)
N_ITERS = None                  # 저장소 기본값. None 이면 TS2Vec 내부 규칙으로 정해진다
                                # (학습 데이터 원소 수 > 100,000 이면 600, 아니면 200). 본 데이터는 600 이 된다

IN_PATH = 'outputs/merged_data.csv'
FOLD_PATH = 'outputs/03_baselines/folds.csv'
OUT_DIR = 'outputs/03_baselines/02_ts2vec'
METHOD = 'TS2Vec'

T1 = 'Film Tension #1 (kg)'     # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS

# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 배정
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
folds = pd.read_csv(FOLD_PATH)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = folds.run_id.tolist()                                        # run 순서는 folds.csv 를 따른다
print(f'[Step 1] rows = {len(df)}, runs = {len(run_ids)} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 저장소 입력 형식으로 변환 : (n_instance, n_timestamps, n_features)
#         run 마다 길이가 1275~1300 으로 다르므로 최대 길이에 맞춰 NaN 으로 채운다
#         (TS2Vec 은 결측을 NaN 으로 받도록 설계되어 있다)
# =============================================================================
t0 = time.time()
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
lengths = np.array([len(by_run[r]) for r in run_ids])
t_max = lengths.max()
x = np.full((len(run_ids), t_max, 1), np.nan)
for i, r in enumerate(run_ids):
    x[i, :lengths[i], 0] = by_run[r].e.to_numpy()
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])     # 패딩을 제거한 시점 순서 (표현과 같은 순서)
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
row_fold = folds.set_index('run_id').loc[run_ids, 'fold_tension'].to_numpy()
pt_fold = np.repeat(row_fold, lengths)                                 # 시점마다의 fold 번호
print(f'[Step 2] 입력 배열 {x.shape}, run 길이 {lengths.min()}~{lengths.max()}, '
      f'패딩 비율 {np.isnan(x).mean():.3f} ({time.time() - t0:.1f} s)')


def unpad(z):
    # 역할 : (n_run, t_max, d) 표현에서 패딩 구간을 떼어내고 시점 순서로 이어 붙인다.
    # 입력 : z (n_run, t_max, d) encode 결과 / 전역 lengths 를 쓴다
    # 출력 : (sum(lengths), d) — e_flat, id_flat 과 같은 순서
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


# =============================================================================
# Step 3. fold × seed 별로 학습 fold 에서 인코더를 자기지도 학습하고,
#         학습 fold 표현으로 트랙 A 후단(GMM)을 적합한 뒤 검증 fold 에 적용한다
# =============================================================================
#         중간 결과(fold×seed 지표, 학습 손실, 인코더 가중치)는 하나 끝날 때마다 바로 저장한다
run_m, fold_m, loss_m = [], [], []
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
for f in sorted(folds.fold_tension.unique()):
    is_val_run = row_fold == f
    is_val_pt = pt_fold == f
    for seed in range(SEED, SEED + N_SEEDS):
        t0 = time.time()
        init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)   # 저장소 원본의 시드 초기화
        model = TS2Vec(input_dims=1, output_dims=REPR_DIMS, device=DEVICE,
                       lr=LR, batch_size=BATCH_SIZE, max_train_length=MAX_TRAIN_LENGTH)
        loss_log = model.fit(x[~is_val_run], n_iters=N_ITERS, verbose=False)   # 학습 fold 신호만, 라벨 없음
        t_fit = time.time() - t0
        model.save(f'{OUT_DIR}/checkpoints/{METHOD}_fold{f}_seed{seed}.pt')    # 학습된 인코더 (재현·재분석용)

        z = unpad(model.encode(x))                                     # 전체 run 을 인코딩 후 fold 로 나눈다
        gmm, sidx, emeans = ev.fit_steady_gmm(z[~is_val_pt], e_flat[~is_val_pt], seed)
        pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
        r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
        fr['fit_sec'], fr['total_sec'] = t_fit, time.time() - t0
        fr['n_epochs'], fr['n_iters'], fr['final_loss'] = model.n_epochs, model.n_iters, loss_log[-1]
        fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
        lo = pd.DataFrame({'fold': f, 'seed': seed, 'epoch': np.arange(len(loss_log)), 'loss': loss_log})
        run_m.append(r), fold_m.append(fr), loss_m.append(lo)

        ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)       # 중간 저장 (끊겨도 여기까지 남는다)
        ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
        ev.append_progress(OUT_DIR, METHOD, 'loss', lo)
        print(f'[Step 3] fold {f} seed {seed}: 학습 {t_fit:.0f} s, 전체 {time.time() - t0:.0f} s, '
              f'epoch {model.n_epochs} 손실 {loss_log[0]:.3f} -> {loss_log[-1]:.3f}, '
              f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
              f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

# =============================================================================
# Step 4. 결과 저장 + 조건별 heatmap
# =============================================================================
run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, METHOD, 'Hierarchical contrastive loss')
ev.save_results(OUT_DIR, METHOD, run_m, fold_m,
                {**ev.EVAL_CONFIG, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'DEVICE': DEVICE, 'BATCH_SIZE': BATCH_SIZE,
                 'LR': LR, 'REPR_DIMS': REPR_DIMS, 'MAX_TRAIN_LENGTH': MAX_TRAIN_LENGTH, 'N_ITERS': N_ITERS,
                 'repo': 'yuezhihan/ts2vec', 'commit': 'b0088e14a99706c05451316dc6db8d3da9351163'})
for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
    ev.plot_condition_heatmap(run_m, metric, OUT_DIR, METHOD, label)
print('[done] TS2Vec 평가 완료')
