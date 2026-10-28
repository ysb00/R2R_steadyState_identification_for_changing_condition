# =============================================================================
# 03_baselines/05_softclt.py : SoftCLT 비교군 어댑터
#
# 방법 : SoftCLT (Lee et al., ICLR 2024) — TS2Vec 의 hard 양성/음성 라벨을 연속 가중치로 바꾼다.
#        인스턴스 축은 run 간 DTW 거리로, 시간 축은 시점 간 시간 차로 가중치를 만든다.
#        거짓 음성(실제로 같은 상태인데 음성으로 밀어내는 쌍) 완화를 노린 설계다.
#
# 외부 저장소 : https://github.com/seunghan96/softclt   (softclt_ts2vec 하위 폴더)
#   clone 위치 : code/03_baselines/external/softclt  (git 추적 제외)
#   커밋       : 14c638979b129075d7a1111e9f529b9a275ea394
#   원본은 한 줄도 수정하지 않는다. 본 파일이 어댑터(데이터 변환 + 호출 + 평가)다.
#   추가 설치 : tslearn(DTW), bottleneck, tensorboard — 저장소가 import 하는 의존성. 원본 미수정
#
# tau 를 저장소 기본값으로 쓸 수 없다 (docs/worklog.md 2026-10-23 §6)
#   `train.py` 기본값은 tau_inst=0, tau_temp=0 이고 코드가 `soft_instance = (tau_inst > 0)` 이므로
#   기본값에서는 soft 요소가 전부 꺼져 **평범한 TS2Vec 과 같아진다**. 반대로 tau 가 너무 크면
#   모든 가중치가 0 이 되어 다시 hard 음성 = TS2Vec 이 된다. 중간 구간을 써야 한다.
#   그리고 tau 가 바뀌면 목적함수 자체가 바뀌므로 **손실값으로 tau 를 고를 수 없다**.
#
# 저자 표의 tau_inst 격자(0.1~40)는 본 데이터에서 전부 퇴화한다 — 실측 확인
#   SoftCLT 는 DTW 거리를 MinMax 로 [0,1] 에 정규화한다. 그런데 본 데이터에는 다른 모든 run 과
#   현격히 먼 run 2개(`T3_R2_V200`, `T4_R6_V150`)가 있어 최대값이 중앙값의 19.8배까지 올라가고,
#   나머지 99% 가 [0, 0.05] 로 짓눌린다 (정규화 거리 중앙값 0.021~0.028).
#   그 결과 앵커당 유효 양성 수 n_eff(가중치 합)가 tau_inst=5 에서 75.5 / tau_inst=40 에서 58~64 다
#   (학습 167개 중). 즉 음성이 사실상 사라진다. 비퇴화가 되려면 tau_inst 가 158~232 여야 한다.
#   측정 근거는 docs/worklog.md 2026-10-23 §6.
#
# 그래서 변형 2개를 돌린다 — 두 변형 모두 라벨을 쓰지 않는다.
#   (1) SoftCLT-ref : 저자 표(UCR 128개) 중앙값 tau_inst=5.0, tau_temp=1.0 을 그대로.
#       퇴화 상태 그대로의 결과이고, "UCR 에서 튜닝된 값이 본 데이터로 전이되지 않는다" 를 보여준다.
#   (2) SoftCLT-fs  : 두 tau 를 모두 '목표를 고정하고 파라미터를 유도' 하는 방식으로 정한다.
#       tau_temp : 시간 가중치가 2/(1+exp(tau_temp*|dt|)) 이므로 절반이 되는 반경 r 에 대해
#                  tau_temp = ln3 / r. r = 32 샘플(3.5 s) = MovStd-Slope 창 64 의 절반 -> 0.034.
#                  통계 기준선과 시간 규모를 맞춰 "창 길이가 달라서" 라는 교란을 없앤다.
#       tau_inst : n_eff 중앙값을 N_EFF_TARGET 으로 맞추는 값을 fold 마다 수치해로 구한다.
#                  목표값의 근거는 조건 격자다 — 조건당 run 1개이고 각 축 ±1 이 인접 조건이므로
#                  (장력 × 리와인더 × 속도) 인접 run 은 최대 6개, 학습 fold(장력 4수준)에서 4~6개다.
#                  MinMax 가 fold 마다 다시 적합되므로 tau 값 자체도 fold 마다 다르다 —
#                  고정하는 것은 tau 가 아니라 '이웃으로 보는 run 수' 다.
#   (3) SoftCLT-fs-rule1b : (2) 에서 tau_temp 만 창 규칙 1b 로 바꾼 것 (docs/hyperparameters.md §13.2 B5).
#       (2) 의 반경 32 는 MovStd-Slope 의 임의 창 64 에서 왔고, 그 64 가 규칙 1b 의 144 로 교체되면서
#       근거가 사라졌다. 반경 규약('창의 절반')은 그대로 두고 창만 144 로 바꾼다 -> r = 72 샘플(7.8 s),
#       tau_temp = ln3 / 72 = 0.0153. 한 번에 하나만 바꾼다는 원칙(research_experiments_plan.md §8.4)에 따라
#       tau_inst 유도 방식과 나머지 설정은 (2) 와 완전히 동일하다.
#       주의 — 반경은 규칙 1b 가 정해주는 양이 아니다. 규칙 1b 는 '1주기를 담는 창 길이' 를 주는데
#       tau_temp 반경은 '두 시점을 같은 상태로 볼 시간 거리' 로 다른 양이다. '창의 절반' 은 본 연구의 규약이다.
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        splits.py 가 분할에 맞는 fold 파일을 읽는다 (A: folds.csv / C: folds_c.csv)
# 출력 : A -> outputs/03_baselines/05_softclt/ , C -> outputs/04_speed_extrap/05_softclt/
#        변형마다 run/fold/summary/config csv + heatmap + 손실곡선
#        outputs/03_baselines/05_softclt/dtw_dist.npy  run 간 DTW 거리 행렬
#        (210x210 전체를 한 번 계산해 둔 것이고 run 쌍 거리는 분할과 무관하므로 C 도 이 캐시를 쓴다)
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행
# 실험 : 인자로 고른다 — 'tension' (A: 장력 5-fold) / 'c' (C: 저속 학습 -> 고속 test, 8 fold)
# 실행 : python code/03_baselines/05_softclt.py        (A 실험, 기본값)
#        python code/03_baselines/05_softclt.py c      (C 실험)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'softclt', 'softclt_ts2vec'))
import evaluation as ev
import splits
from soft_ts2vec import TS2Vec as SoftTS2Vec                            # 외부 저장소 원본
from utils import init_dl_program                                       # 외부 저장소 원본의 시드 초기화
from utils_distance_matrix import densify                               # 외부 저장소 원본의 soft label 변환


SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만
# 추론 모드에서도 tau_inst 수치해는 그대로 구한다 — 체크포인트를 만든 설정을 config 에 남기기 위해서다
# (DTW 행렬은 캐시되어 있어 비용이 작고, tau 는 학습에만 쓰이므로 결과를 바꾸지 않는다).
# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값)
N_SEEDS = 5                     # seed 반복 횟수

# 변형 2개. tau_inst 는 값(tau_inst) 또는 목표 n_eff(n_eff_target) 중 하나로 지정한다
N_EFF_TARGET = 5.0              # 인접 조건 run 수(장력·리와인더·속도 각 축 ±1 -> 학습 fold 에서 4~6개)의 중앙
VARIANTS = [
    {'name': 'SoftCLT-fs',  'tau_inst': None, 'n_eff_target': N_EFF_TARGET, 'tau_temp': 0.034},
    {'name': 'SoftCLT-ref', 'tau_inst': 5.0,  'n_eff_target': None,         'tau_temp': 1.0},
    # 규칙 1b 반영 : 반경 r = W_RULE(144)/2 = 72 샘플 -> tau_temp = ln3/72 = 0.0153 (헤더 (3) 참조)
    {'name': 'SoftCLT-fs-rule1b', 'tau_inst': None, 'n_eff_target': N_EFF_TARGET, 'tau_temp': 0.0153},
]
N_ITERS = None if SPLIT == 'tension' else 600
# A 실험 : 저장소 기본값 None. 내부 규칙(학습 데이터 원소 수 > 100,000 이면 600, 아니면 200)이 적용되고,
#          학습 168 run x 1300 = 218,400 이라 **항상 600** 이 선택됐다.
# C 실험 : **600 으로 고정한다** (연구자 결정 2026-10-26). fold 마다 학습 run 수가 60~120 으로 다른데
#          자동 규칙을 두면 cv_S1(60 run, 78,000) 만 200 이 되어 혼자 1/3 만 학습한다. 그러면 cv_S1 의
#          성능이 '외삽이 어려워서' 인지 '덜 학습해서' 인지 분리되지 않는다. 600 은 A 에서 저장소 규칙이
#          실제로 고른 값이므로 임의값을 새로 들인 것이 아니다.
#          **As-published 에서 벗어난 유일한 지점** (docs/hyperparameters.md 13.0)
                                # (학습 데이터 원소 수 > 100,000 이면 600, 아니면 200). 본 데이터는 600 이 된다
TAU_INST_BRACKET = (1.0, 1e5)   # n_eff 목표를 맞추는 tau_inst 수치해의 탐색 구간

ALPHA = 0.5                     # 저장소 기본값 (train.py --alpha). soft label 의 최대 가중치를 정한다
LAMBDA = 0.5                    # 저장소 기본값 (train.py --lambda_). 인스턴스 손실과 시간 손실의 비중
DIST_TYPE = 'DTW'               # 저장소 기본값 (train.py --dist_type)
DEVICE = 'cuda'                 # 저장소 기본값 (train.py --gpu 0)
BATCH_SIZE = 8                  # 저장소 기본값 (train.py --batch-size)
LR = 0.001                      # 저장소 기본값 (train.py --lr)
REPR_DIMS = 320                 # 저장소 기본값 (train.py --repr-dims)
MAX_TRAIN_LENGTH = 3000         # 저장소 기본값 (train.py --max-train-length). run 길이 1300 < 3000 이므로 분할되지 않는다

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = splits.resolve(SPLIT, '05_softclt')
CKPT_DIR = splits.ckpt_dir(SPLIT, '05_softclt')   # 인코더를 읽는 곳 (추론 모드에서도 기존 실험 것을 쓴다)
DTW_PATH = 'outputs/03_baselines/05_softclt/dtw_dist.npy'   # run 쌍 거리는 분할과 무관하므로 A 캐시를 재사용한다

T1 = 'Film Tension #1 (kg)'     # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS

# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 배정
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids                                                    # run 순서는 분할 파일을 따른다
print(f'[Step 1] rows = {len(df)}, runs = {len(run_ids)} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 저장소 입력 형식으로 변환 : (n_instance, n_timestamps, n_features)
#         run 마다 길이가 1275~1300 으로 다르므로 최대 길이에 맞춰 NaN 으로 채운다
# =============================================================================
t0 = time.time()
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
lengths = np.array([len(by_run[r]) for r in run_ids])
x = np.full((len(run_ids), lengths.max(), 1), np.nan)
for i, r in enumerate(run_ids):
    x[i, :lengths[i], 0] = by_run[r].e.to_numpy()
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])      # 패딩을 제거한 시점 순서 (표현과 같은 순서)
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 2] 입력 배열 {x.shape}, run 길이 {lengths.min()}~{lengths.max()}, '
      f'패딩 비율 {np.isnan(x).mean():.3f} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 3. run 간 DTW 거리 행렬 (한 번 계산해 파일로 두고 재사용)
#         거리는 e 신호만으로 계산한다 (라벨 없음). fold 와 무관한 양이라 210x210 을 한 번만 만든다.
#         fold 별 soft label 은 아래에서 **학습 fold 부분 행렬만** 떼어 정규화한다
#         (정규화에 검증 fold 가 섞이지 않게 한다)
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(DTW_PATH):
    dtw_full = np.load(DTW_PATH)
    print(f'[Step 3] DTW 거리 행렬 재사용 {dtw_full.shape} <- {DTW_PATH}')
else:
    from tslearn.metrics import dtw                                     # 저장소가 쓰는 것과 같은 구현
    t0 = time.time()
    series = [by_run[r].e.to_numpy().reshape(-1, 1) for r in run_ids]
    n = len(series)
    dtw_full = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(series[i], series[j])
            dtw_full[i, j] = dtw_full[j, i] = d
        if (i + 1) % 20 == 0:
            print(f'          DTW {i + 1}/{n} run ({time.time() - t0:.0f} s)', flush=True)
    np.save(DTW_PATH, dtw_full)
    print(f'[Step 3] DTW 거리 행렬 저장 {dtw_full.shape} -> {DTW_PATH} ({time.time() - t0:.0f} s)')


def norm_dist(train_idx):
    # 역할 : 학습 fold 의 DTW 거리 부분행렬을 저장소와 같은 절차로 정규화한다.
    #        저장소 절차(utils_distance_matrix.save_sim_mat):
    #          대각을 비대각 최솟값으로 치환 -> MinMax 로 [0,1] 정규화
    # 입력 : train_idx (n_tr,) 학습 fold run 의 인덱스
    # 출력 : (n_tr, n_tr) 정규화된 거리 행렬
    d = dtw_full[np.ix_(train_idx, train_idx)].copy()
    m = ~np.eye(len(d), dtype=bool)
    d[np.diag_indices_from(d)] = d[m].min()                             # 저장소와 동일: 대각 = 비대각 최솟값
    return (d - d.min()) / (d.max() - d.min())


def n_eff_of(d, tau_inst):
    # 역할 : 앵커당 유효 양성 수(비대각 가중치의 합)의 중앙값. soft label 이 퇴화했는지 보는 양이다.
    #        0 에 가까우면 모든 쌍이 hard 음성(= TS2Vec), 학습 run 수에 가까우면 음성이 사라진다.
    # 입력 : d (n_tr, n_tr) 정규화 거리 / tau_inst soft label 날카로움
    # 출력 : n_eff 중앙값 (float)
    sl = densify(-d, tau_inst, ALPHA)
    off = sl[~np.eye(len(sl), dtype=bool)].reshape(len(sl), -1)
    return float(np.median(off.sum(1)))


def soft_labels_for(d, tau_inst):
    # 역할 : 정규화 거리에서 soft label 행렬을 만든다.
    #        train.py 는 sim = 1 - d 를 저장한 뒤 다시 d = 1 - sim 으로 되돌려 densify 에 넣으므로
    #        결과는 densify(-d_norm, tau, alpha) 와 같다. 여기서는 그 결과를 바로 만든다.
    # 입력 : d (n_tr, n_tr) 정규화 거리 / tau_inst soft label 날카로움
    # 출력 : (n_tr, n_tr) soft label 행렬 (대각 = 1.0)
    return densify(-d, tau_inst, ALPHA)                                 # 저장소 원본 함수


# =============================================================================
# Step 4. 변형 × fold × seed 별로 학습 fold 에서 인코더를 자기지도 학습하고,
#         학습 fold 표현으로 트랙 A 후단(GMM)을 적합한 뒤 검증 fold 에 적용한다
# =============================================================================
#         중간 결과(fold×seed 지표, 학습 손실, 인코더 가중치)는 하나 끝날 때마다 바로 저장한다
if not INFER:                                                       # 추론 모드는 체크포인트를 쓰지 않는다
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
for v in VARIANTS:
    method, tau_temp = v['name'], v['tau_temp']
    if os.path.exists(f'{OUT_DIR}/{method}_fold_metrics.csv'):          # 이미 돌린 변형은 건너뛴다
        print(f'[Step 4] {method}: 이미 결과가 있어 건너뛴다 (다시 돌리려면 해당 파일들을 먼저 지울 것)', flush=True)
        continue
    run_m, fold_m, loss_m = [], [], []
    for f in sp.folds:
        # C 실험에서는 '학습 = 검증 아닌 전부' 가 아니다 (고속 run 은 CV fold 에서 미사용)
        is_train_run, is_val_run = sp.run_role(f)
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        train_idx = np.flatnonzero(is_train_run)        # DTW 부분행렬·soft label 도 학습 run 만
        d_norm = norm_dist(train_idx)
        if v['tau_inst'] is not None:
            tau_inst = v['tau_inst']
        else:                                                           # n_eff 목표를 맞추는 tau 를 수치해로 구한다
            tau_inst = float(brentq(lambda t: n_eff_of(d_norm, t) - v['n_eff_target'],
                                    *TAU_INST_BRACKET, xtol=0.5))
        sl = soft_labels_for(d_norm, tau_inst)
        n_eff = sl[~np.eye(len(sl), dtype=bool)].reshape(len(sl), -1).sum(1)   # 앵커당 유효 양성 수
        print(f'[Step 4] {method} fold {f}: 정규화 거리 중앙값 {np.median(d_norm[~np.eye(len(d_norm), dtype=bool)]):.4f}, '
              f'tau_inst = {tau_inst:.1f}, n_eff 중앙값 {np.median(n_eff):.2f} '
              f'(학습 run {len(sl) - 1}개 중)', flush=True)
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)    # 저장소 원본의 시드 초기화
            # 손실은 iteration 단위로 받는다 (epoch 당 iteration 수가 저장소마다 달라 epoch 축은
            # 방법 간 비교가 불가능하다 — TS2Vec·SoftCLT 는 21, CoST 는 창 길이에 따라 42~168).
            # after_iter_callback 은 저장소가 제공하는 확장점이므로 원본 수정이 아니다.
            iter_loss, epoch_iters = [], []
            model = SoftTS2Vec(input_dims=1, output_dims=REPR_DIMS, device=DEVICE,
                               lr=LR, batch_size=BATCH_SIZE, max_train_length=MAX_TRAIN_LENGTH,
                               lambda_=LAMBDA, tau_temp=tau_temp,
                               soft_instance=True, soft_temporal=True,
                               after_iter_callback=lambda _m, l: iter_loss.append(l),
                               after_epoch_callback=lambda m, _l: epoch_iters.append(m.n_iters))
            if INFER:                                                   # 재학습 없이 저장된 인코더만 불러온다
                ck = f'{CKPT_DIR}/{method}_fold{f}_seed{seed}.pt'
                assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습을 돌릴 것)'
                model.load(ck)
                loss_log = []
                t_fit = time.time() - t0
            else:
                loss_log = model.fit(x[is_train_run], 0, 0, 0, sl,       # 학습 fold 신호 + soft label, 라벨 없음
                                     run_dir=f'{OUT_DIR}/_result_tmp', n_iters=N_ITERS, verbose=False)
                t_fit = time.time() - t0
                model.save(f'{OUT_DIR}/checkpoints/{method}_fold{f}_seed{seed}.pt')   # 학습된 인코더 (재현·재분석용)

            if INFER:
                # 아래 줄(학습 경로)은 `model.encode(x)` 를 **리스트 컴프리헨션 안에서** 부르기 때문에
                # run 1개를 꺼낼 때마다 210 run 전체를 다시 인코딩한다 — 210번 중복이다.
                # 결과는 같고 속도만 다르므로 추론 경로에서는 encode 를 1번만 부른다.
                # 실측(SoftCLT-fs fold1 seed0): 74.9 s -> 0.9 s, 두 배열 **완전 일치**(최대 차이 0.0).
                # 학습 경로는 손대지 않는다 — 기존 A·C 결과의 재현 경로를 그대로 보존하기 위함이다.
                z = np.concatenate([zz[:n] for zz, n in zip(model.encode(x), lengths)])
            else:
                z = np.concatenate([model.encode(x)[i, :lengths[i]] for i in range(len(lengths))])
            gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
            pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
            r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, method, f, seed)
            fr['tau_inst'], fr['tau_temp'] = tau_inst, tau_temp
            fr['n_eff_median'], fr['n_eff_min'], fr['n_eff_max'] = np.median(n_eff), n_eff.min(), n_eff.max()
            fr['total_sec'] = time.time() - t0
            if INFER:                                       # 학습량 기록은 학습했을 때만 의미가 있다
                fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
                run_m.append(r), fold_m.append(fr)
                ev.append_progress(OUT_DIR, method, 'progress_fold', fr)
                ev.append_progress(OUT_DIR, method, 'progress_run', r)
                print(f'[Step 4] {method} fold {f} seed {seed}: 체크포인트 재사용 (학습 생략), '
                      f'전체 {time.time() - t0:.0f} s, n_eff 중앙값 {np.median(n_eff):.2f}, '
                      f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
                      f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                      f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)
                continue
            fr['fit_sec'] = t_fit
            fr['n_epochs'], fr['n_iters'] = model.n_epochs, model.n_iters
            # 1 epoch 의 iteration 수. n_iters // n_epochs 는 마지막 epoch 이 끊겨 참값보다 크게 나온다.
            # 첫 epoch 끝의 누적 iter 가 참값이다. (SoftCLT 는 run 1300 < 3000 이라 분할 없음 -> 21)
            fr['iters_per_epoch'] = epoch_iters[0] if epoch_iters else np.nan
            fr['data_passes'] = model.n_iters / epoch_iters[0] if epoch_iters else np.nan
            fr['samples_seen'] = model.n_iters * BATCH_SIZE * min(MAX_TRAIN_LENGTH, int(lengths.max()))
            fr['epoch_mean_loss_last'] = loss_log[-1] if len(loss_log) else np.nan   # 저장소가 주는 epoch 평균
            fr['final_loss'] = float(np.mean(iter_loss[-20:])) if iter_loss else np.nan   # 마지막 20 iter 평균
            fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
            lo = pd.DataFrame({'fold': f, 'seed': seed,
                               'iter': np.arange(1, len(iter_loss) + 1), 'loss': iter_loss})
            run_m.append(r), fold_m.append(fr), loss_m.append(lo)

            ev.append_progress(OUT_DIR, method, 'progress_fold', fr)    # 중간 저장 (끊겨도 여기까지 남는다)
            ev.append_progress(OUT_DIR, method, 'progress_run', r)
            ev.append_progress(OUT_DIR, method, 'loss', lo)
            loss_txt = f'{loss_log[0]:.4f} -> {loss_log[-1]:.4f}' if len(loss_log) else '기록 없음'
            print(f'[Step 4] {method} fold {f} seed {seed}: 학습 {t_fit:.0f} s, 전체 {time.time() - t0:.0f} s, '
                  f'epoch {model.n_epochs} 손실 {loss_txt}, n_eff 중앙값 {np.median(n_eff):.2f}, '
                  f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
                  f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                  f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

    # ---- 변형 하나가 끝날 때마다 최종 산출물을 저장한다 ----
    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    if not INFER:                                           # 추론 모드는 학습하지 않아 손실 곡선이 없다
        ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, method,
                            'Soft hierarchical contrastive loss')
    ev.save_results(OUT_DIR, method, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'N_ITERS': N_ITERS,
                     'TAU_INST_SPEC': v['tau_inst'] if v['tau_inst'] is not None
                     else f"n_eff 중앙값 = {v['n_eff_target']} 이 되도록 fold 마다 수치해",
                     'TAU_INST_BY_FOLD': dict(zip(fold_m.fold, fold_m.tau_inst.round(1))),
                     'TAU_TEMP': tau_temp, 'ALPHA': ALPHA, 'LAMBDA': LAMBDA,
                     'DIST_TYPE': DIST_TYPE, 'DEVICE': DEVICE, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
                     'REPR_DIMS': REPR_DIMS, 'MAX_TRAIN_LENGTH': MAX_TRAIN_LENGTH,
                     'repo': 'seunghan96/softclt (softclt_ts2vec)',
                     'commit': '14c638979b129075d7a1111e9f529b9a275ea394'})
    for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[done] {method} 평가 완료', flush=True)

print('[done] SoftCLT 전체 완료')
