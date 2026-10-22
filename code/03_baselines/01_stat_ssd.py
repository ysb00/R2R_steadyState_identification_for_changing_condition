# =============================================================================
# 03_baselines/01_stat_ssd.py : 통계 기반 steady state 판정 기준선 2종 (비교군)
#
# 방법 1. R-test  (Cao & Rhinehart, J. Process Control 5(6):363-374, 1995)
#   분산비 R = (2-λ1)·ν²_f / δ²_f 를 EWMA 로 온라인 갱신한다.
#   ν² 는 '데이터 - 직전 필터값' 제곱, δ² 는 '연속 데이터 차' 제곱의 EWMA 이고,
#   정상상태면 두 분산 추정이 같아 R ≈ 1, 과도상태면 필터값이 뒤처져 R >> 1 이 된다.
#   학습하지 않고 원 논문 기본 필터값·임계값을 그대로 쓴다.
#   식·기본값 출처: Rhinehart, "Automated Steady and Transient State Identification
#   in Noisy Processes" (r3eda 튜토리얼, 2019-05-08) pp.12-16.
#
# 방법 2. 이동창 표준편차 + 기울기  (원 논문 없는 단순 기준선)
#   뒤쪽(causal) 이동창 안에서 e 의 표준편차와 선형추세 기울기를 구해 2차원 특징으로 쓴다.
#   "조용하고 흐르지 않으면 정상상태"라는 정의를 그대로 옮긴 것이다.
#   임계값을 손으로 정하지 않기 위해, 이 2차원을 표현으로 보고 다른 비교군과 동일한
#   트랙 A 후단(학습 fold 2-군집 GMM -> |e 평균| 작은 군집 = steady)에 그대로 넣는다.
#   창 길이는 고정(fixed)과 속도별 조정(speed-scaled) 두 가지를 각각 돌려 비교한다.
#   속도별 조정은 steady 구간 주기 성분의 주파수가 속도에 비례한다는 관찰에 따라
#   W(V) = WINDOW * SPEED_REF / V 로 둔다 (= 필름 이동 거리 기준 창과 같은 형태).
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        outputs/03_baselines/folds.csv (고정된 fold 배정)
# 출력 : outputs/03_baselines/01_stat_ssd/  방법별 run/fold/summary/config csv + 조건별 heatmap
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행한다. 보조 실험(속도별 7-fold)은 1차 결과 확인 후.
# 실행 : python code/03_baselines/01_stat_ssd.py   (repo 루트에서)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev

# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값)
N_SEEDS = 5                     # GMM 후단을 쓰는 방법의 seed 반복 횟수 (R-test 는 결정론적이라 1회)

LAMBDA1 = 0.2                   # R-test 필터 상수 (원 논문 권장값)
LAMBDA2 = 0.1                   # R-test 필터 상수 (원 논문 권장값)
LAMBDA3 = 0.1                   # R-test 필터 상수 (원 논문 권장값)
R_CRIT_TS = 2.5                 # R > 이 값이면 steady 기각 -> non-steady (원 논문 기본 임계값)
R_CRIT_SS = 0.9                 # R < 이 값이면 steady 수용. 두 값 사이면 직전 판정을 유지 (원 논문 기본 임계값)
RTEST_DECIMATE = 10             # R-test 용 다운샘플링 배수. 원 논문은 "정상상태 잡음에 자기상관이 없도록
                                # 샘플링 간격을 고른다"고 전제하는데 본 데이터는 9 Hz 에서 lag1 자기상관 0.85 라
                                # 전제가 깨진다. lag10 에서 0.16 까지 떨어지는 것을 근거로 10 을 쓴다. 임의의 값

WINDOW = 64                     # 이동창 길이 (samples, 약 7.0 s). 임의의 값
SPEED_REF = 125                 # 속도별 창 조정의 기준 속도 (mm/s). 조건 격자 중앙값. 임의의 값

IN_PATH = 'outputs/merged_data.csv'
FOLD_PATH = 'outputs/03_baselines/folds.csv'
OUT_DIR = 'outputs/03_baselines/01_stat_ssd'

T1 = 'Film Tension #1 (kg)'     # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS

# =============================================================================
# 함수
# =============================================================================


def rtest_statistic(x):
    # 역할 : run 1개의 신호에 대해 R-statistic 시계열을 계산한다 (원 논문 식 (1)~(4)).
    #        필터값은 모두 0 으로 초기화한다 (원 논문 지시). 초기 약 35 샘플은 초기화 영향이 남는다.
    # 입력 : x (n,) run 1개의 신호 (여기서는 e)
    # 출력 : R (n,) 시점별 R 통계량. δ²_f 가 0 이 되는 시점(= 연속 변화가 전혀 없는 구간)은
    #        R 이 무한대가 되므로 그 run 의 유한 최댓값으로 바꾼다 (원 논문도 무잡음 구간에서 깨진다고 명시).
    n = len(x)
    nu2, delta2 = np.empty(n), np.empty(n)
    xf, nu2f, delta2f, x_old = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        nu2f = LAMBDA2 * (x[i] - xf) ** 2 + (1 - LAMBDA2) * nu2f      # 직전 필터값 xf 를 쓴다 (자기상관 편향 방지)
        xf = LAMBDA1 * x[i] + (1 - LAMBDA1) * xf
        delta2f = LAMBDA3 * (x[i] - x_old) ** 2 + (1 - LAMBDA3) * delta2f
        x_old = x[i]
        nu2[i], delta2[i] = nu2f, delta2f
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (2 - LAMBDA1) * nu2 / delta2
    bad = ~np.isfinite(r)
    r[bad] = np.nanmax(r[~bad]) if (~bad).any() else 0.0
    return r


def rtest_decide(r):
    # 역할 : R 시계열을 두 임계값으로 판정한다. 두 값 사이에서는 직전 판정을 유지한다 (원 논문 규칙).
    # 입력 : r (n,) R 통계량
    # 출력 : pred (n,) 1 = steady, 0 = non-steady. 초기 상태는 non-steady (필터 초기화 구간의 높은 R 과 일치)
    pred = np.empty(len(r), dtype=int)
    state = 0
    for i, v in enumerate(r):
        if v > R_CRIT_TS:
            state = 0
        elif v < R_CRIT_SS:
            state = 1
        pred[i] = state
    return pred


def rolling_std_slope(e, w):
    # 역할 : 길이 w 의 뒤쪽(causal) 이동창에서 표준편차와 선형추세 기울기를 계산한다.
    # 입력 : e (n,) run 1개의 오차 신호 / w 창 길이 (samples)
    # 출력 : (std (n,) kg, slope (n,) kg/s). 창이 아직 다 차지 않은 앞 w-1 개 시점은
    #        첫 완전창의 값으로 채운다 (창 길이가 달라도 채점 대상 시점 수가 같도록).
    std = np.array(pd.Series(e).rolling(w).std())
    j = np.arange(w) - (w - 1) / 2.0
    kern = j / (j ** 2).sum() / ev.DT                                  # 최소제곱 기울기 가중치, kg/sample -> kg/s
    slope = np.empty(len(e))
    slope[w - 1:] = np.convolve(e, kern[::-1], mode='valid')
    std[:w - 1], slope[:w - 1] = std[w - 1], slope[w - 1]
    return std, slope


# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 배정
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
folds = pd.read_csv(FOLD_PATH)
fold_of = dict(zip(folds.run_id, folds.fold_tension))
df['fold'] = df.run_id.map(fold_of)
print(f'[Step 1] rows = {len(df)}, runs = {df.run_id.nunique()}, folds = {sorted(df.fold.unique())} '
      f'({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. run별 특징 계산 (R-statistic, 고정창 std/기울기, 속도별창 std/기울기)
# =============================================================================
t0 = time.time()
parts = []
for run_id, g in df.groupby('run_id', sort=False):
    g = g.sort_values('k').copy()
    e = g.e.to_numpy()
    w_speed = int(round(WINDOW * SPEED_REF / g[VS].iloc[0]))           # 속도가 빠를수록 짧은 창
    g['R'] = rtest_statistic(e)
    g['pred_rtest'] = rtest_decide(g.R.to_numpy())
    # 다운샘플링판 R-test : 간격을 늘린 신호로 R 을 계산·판정하고, 판정을 원래 시점으로 되돌린다.
    # 되돌릴 때는 직전 다운샘플 시점의 판정을 그대로 끌고 온다 (causal, forward-fill).
    r_dec = rtest_statistic(e[::RTEST_DECIMATE])
    back = np.arange(len(e)) // RTEST_DECIMATE
    g['R_dec'] = r_dec[back]
    g['pred_rtest_dec'] = rtest_decide(r_dec)[back]
    g['std_fixed'], g['slope_fixed'] = rolling_std_slope(e, WINDOW)
    g['std_speed'], g['slope_speed'] = rolling_std_slope(e, w_speed)
    g['w_speed'] = w_speed
    parts.append(g)
df = pd.concat(parts, ignore_index=True)
print(f'[Step 2] 특징 계산 완료. 속도별 창 길이 = '
      f'{df.groupby(VS).w_speed.first().to_dict()} samples ({time.time() - t0:.1f} s)')
for name, col, dec in [('원 샘플링(9 Hz)', 'R', 1), (f'1/{RTEST_DECIMATE} 다운샘플링', 'R_dec', RTEST_DECIMATE)]:
    # R-test 전제 확인용: 다운샘플링 후 lag1 자기상관이 실제로 얼마나 줄었는지 run 평균으로 본다
    ac = df.groupby('run_id').e.apply(lambda s, d=dec: np.corrcoef(s.to_numpy()[::d][:-1], s.to_numpy()[::d][1:])[0, 1])
    print(f'[Step 2] {name}: lag1 자기상관 run평균 {ac.mean():.2f}, R 중앙값 {df[col].median():.2f}, '
          f'steady 임계값({R_CRIT_SS}) 미만 {(df[col] < R_CRIT_SS).mean():.3f}, '
          f'TS 임계값({R_CRIT_TS}) 초과 {(df[col] > R_CRIT_TS).mean():.3f}')

ID_COLS = ['run_id', 'k', TS, RS, VS]

# =============================================================================
# Step 3. 방법 1 — R-test : 학습 없이 원 논문 임계값으로 판정하고 검증 fold 에서 채점
#         non-steady 점수는 R 통계량 자체 (클수록 과도상태)
# =============================================================================
for method, pcol, scol, dec in [('R-test', 'pred_rtest', 'R', 1),
                                ('R-test-decimated', 'pred_rtest_dec', 'R_dec', RTEST_DECIMATE)]:
    t0 = time.time()
    run_m, fold_m = [], []
    for f in sorted(df.fold.unique()):
        va = df[df.fold == f]
        r, fr = ev.score_fold(va[ID_COLS], va[pcol].to_numpy(), va[scol].to_numpy(), method, f, SEED)
        run_m.append(r), fold_m.append(fr)
        print(f'[Step 3] {method} fold {f}: steady 판정 비율 {va[pcol].mean():.3f}, '
              f'오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, 보존률 {fr.pooled_retention_rate.iloc[0]:.3f}')
    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    ev.save_results(OUT_DIR, method, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SEED': SEED, 'LAMBDA1': LAMBDA1, 'LAMBDA2': LAMBDA2, 'LAMBDA3': LAMBDA3,
                     'R_CRIT_TS': R_CRIT_TS, 'R_CRIT_SS': R_CRIT_SS, 'decimate': dec})
    for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[Step 3] {method} 완료 ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 4. 방법 2 — 이동창 std + 기울기 : (std, |기울기|) 2차원을 표현으로 보고 트랙 A 후단 적용
#         창 길이 고정(fixed) / 속도별 조정(speed-scaled) 두 변형을 각각 돌린다
# =============================================================================
for variant, cols in [('fixed', ['std_fixed', 'slope_fixed']), ('speed', ['std_speed', 'slope_speed'])]:
    t0 = time.time()
    method = f'MovStd-Slope-{variant}'
    z = np.column_stack([df[cols[0]].to_numpy(), np.abs(df[cols[1]].to_numpy())])   # 기울기는 크기만 쓴다
    run_m, fold_m = [], []
    for f in sorted(df.fold.unique()):
        is_val = (df.fold == f).to_numpy()
        for seed in range(SEED, SEED + N_SEEDS):
            model, sidx, emeans = ev.fit_steady_gmm(z[~is_val], df.e.to_numpy()[~is_val], seed)
            pred, score = ev.predict_steady(model, sidx, z[is_val])
            r, fr = ev.score_fold(df.loc[is_val, ID_COLS], pred, score, method, f, seed)
            run_m.append(r), fold_m.append(fr)
        print(f'[Step 4] {method} fold {f}: 군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
              f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}')
    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    ev.save_results(OUT_DIR, method, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOW': WINDOW,
                     'SPEED_REF': SPEED_REF, 'window_mode': variant})
    for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[Step 4] {method} 완료 ({time.time() - t0:.1f} s)')

print('[done] 통계 기준선 2종 (창 길이 2변형 포함) 평가 완료')
