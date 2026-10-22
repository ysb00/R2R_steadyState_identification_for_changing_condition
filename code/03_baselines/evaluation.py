# =============================================================================
# 03_baselines/evaluation.py : 트랙 A 공통 평가 모듈 (제안 모델 · 전 비교군 공용)
#
# 역할 : 시점별 표현(representation)을 받아 라벨 없이 steady/non-steady 를 판정하고,
#        `Trigger 4` 로 채점하여 run별 · fold별 지표와 조건별 heatmap 을 만든다.
#        모든 방법이 이 파일 하나를 import 해서 쓴다 (공정성: 동일 후단 · 동일 채점).
#
# 트랙 A 파이프라인 (docs/research_experiments_plan.md 3절 · 4.3절)
#   학습 fold : 표현 -> 2-군집 GMM 학습 -> |e 평균| 이 작은 군집을 steady 로 지정
#   검증 fold : 학습된 GMM 을 그대로 적용해 판정 -> `Trigger 4` 로 채점 (ambiguous 제외)
#
# [라벨 누출 차단 — 코드 구조]
#   `Trigger 4` 를 읽는 코드는 이 파일의 _labels() 한 곳뿐이다.
#   비교군 · 제안 모델 스크립트는 merged_data.csv 에서 `Trigger 4` 를 usecols 에 넣지 않으며,
#   라벨 배열을 변수로 들고 있지 않는다. 채점은 전부 이 파일 안에서 일어난다.
#
# 제공 함수
#   fit_steady_gmm        학습 fold 표현으로 GMM 학습 + steady 군집 지정
#   predict_steady        검증 fold 판정 (pred, non-steady 사후확률)
#   steady_entry_k        판정 시계열에서 steady 진입 시점 읽기
#   score_fold            run별 지표 + fold 단위 집계 (pooled / run 평균 둘 다)
#   save_results          결과 csv 저장 (덮어쓰기 금지)
#   plot_condition_heatmap  조건별 성능 heatmap (PNG 300 dpi + PDF)
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# =============================================================================
# hyperparameter (평가 쪽 설정 — 모든 방법에 동일하게 적용된다)
# =============================================================================
DT = 0.109                      # 샘플 간격 (s). 원본 `Date` 기준 평균의 근사값 (docs/data_dictionary.md)
ENTRY_MIN_CONSEC = 10           # steady 진입 판정: 이만큼 연속으로 steady 로 판정된 첫 시점 = 진입. 임의의 값
N_GMM_COMPONENTS = 2            # 2-군집 GMM (계획 고정값)
GMM_COVARIANCE_TYPE = 'diag'    # GMM 공분산 형태. full 은 고차원 표현(예: TS2Vec 320차원)에서 공분산이
                                # 특이행렬이 되어 적합에 실패한다. diag 는 2차원부터 수백 차원까지 안정적이고
                                # 모든 비교군이 같은 후단을 쓰게 해준다 (공정성). 대신 차원 간 상관구조는 버린다

MERGED_PATH = 'outputs/merged_data.csv'
LABEL_COL = 'Trigger 4'         # 원본 컬럼명 (변경하지 않음). 이 파일 안에서만 읽는다
TS = 'Film-Tension Setting Value (kg)'
RS = 'ReWinder-Tension Setting Value (kg)'
VS = 'Web Speed setting value (mm/sec)'

# 평가 쪽 hyperparameter 기록용 (각 방법 스크립트가 자기 설정과 합쳐 config csv 로 저장한다)
EVAL_CONFIG = {'DT': DT, 'ENTRY_MIN_CONSEC': ENTRY_MIN_CONSEC,
               'N_GMM_COMPONENTS': N_GMM_COMPONENTS, 'GMM_COVARIANCE_TYPE': GMM_COVARIANCE_TYPE}

# =============================================================================
# 플롯 스타일 (논문 스타일 — 모든 그림이 공유한다)
# =============================================================================
FONT_SIZE = 9                   # 모든 그림의 기본 글꼴 크기 (논문 본문 기준). 임의의 값
DPI = 300                       # PNG 해상도

mpl.rcParams.update({'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                     'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
                     'axes.grid': True, 'grid.linewidth': 0.5, 'grid.color': '#e1e0d9',
                     'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8, 'savefig.bbox': 'tight'})

# 크기(magnitude) 표현용 sequential colormap — 단일 색상(blue) 밝음->어두움
SEQ_CMAP = LinearSegmentedColormap.from_list('seq_blue', [
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'])

# 방법별 고정 스타일 (color, marker, linestyle).
# 검증된 categorical 색은 5계열까지이므로 색은 '계열' 단위로 고정하고,
# 같은 계열 안의 방법은 marker · linestyle 로 구분한다 -> 방법마다 고정된 식별자 1조합.
METHOD_STYLE = {
    'R-test':       ('#2a78d6', 'o', '-'),      # 통계 SSD
    'MovStd-Slope': ('#2a78d6', 's', '--'),     # 통계 SSD
    'TS2Vec':       ('#eb6834', 'o', '-'),      # 대조학습
    'SoftCLT':      ('#eb6834', 's', '--'),     # 대조학습
    'TNC':          ('#eb6834', '^', '-.'),     # 대조학습
    'TS-TCC':       ('#eb6834', 'D', ':'),      # 대조학습
    'CoST':         ('#eb6834', 'v', (0, (3, 1, 1, 1))),  # 대조학습
    'SimMTM':       ('#1baf7a', 'o', '-'),      # 생성/복원
    'MOMENT':       ('#eda100', 'o', '-'),      # 파운데이션 모델
    'Proposed':     ('#e87ba4', 'o', '-'),      # 제안 방법
}

# =============================================================================
# 함수
# =============================================================================
_LABEL_CACHE = None


def _labels():
    # 역할 : `Trigger 4` 라벨 표를 읽어 캐시한다. 라벨을 읽는 유일한 지점 (라벨 누출 차단)
    # 입력 : 없음 (MERGED_PATH 를 읽는다)
    # 출력 : DataFrame [run_id, k, `Trigger 4`]
    global _LABEL_CACHE
    if _LABEL_CACHE is None:
        _LABEL_CACHE = pd.read_csv(MERGED_PATH, usecols=['run_id', 'k', LABEL_COL])
    return _LABEL_CACHE


def fit_steady_gmm(z_train, e_train, seed):
    # 역할 : 학습 fold 의 시점별 표현으로 2-군집 GMM 을 학습하고, 어느 군집이 steady 인지 지정한다.
    #        지정 규칙: 군집별 e 평균의 절댓값이 더 작은 쪽 = steady (e 평균이 0 에 가까운 쪽).
    #        표현의 축마다 단위·분산이 다르면 분산이 큰 축이 군집을 지배하므로, GMM 앞에 표준화를 둔다.
    #        표준화도 GMM 도 학습 fold 로만 적합하며, 라벨을 쓰지 않는다.
    # 입력 : z_train (n_tr, d) 학습 fold 시점별 표현 / e_train (n_tr,) 같은 시점의 e / seed 난수 시드
    # 출력 : (model 학습된 표준화+GMM, steady_idx steady 로 지정된 군집 번호, e_means 군집별 e 평균)
    model = make_pipeline(StandardScaler(),
                          GaussianMixture(n_components=N_GMM_COMPONENTS, covariance_type=GMM_COVARIANCE_TYPE,
                                          random_state=seed)).fit(z_train)
    c = model.predict(z_train)
    e_means = np.array([e_train[c == j].mean() if (c == j).any() else np.nan for j in range(N_GMM_COMPONENTS)])
    steady_idx = int(np.nanargmin(np.abs(e_means)))
    return model, steady_idx, e_means


def predict_steady(model, steady_idx, z):
    # 역할 : 학습된 표준화+GMM 을 그대로 적용해 시점별 steady 여부와 non-steady 점수를 만든다.
    # 입력 : model 학습된 모델 / steady_idx steady 군집 번호 / z (n, d) 시점별 표현
    # 출력 : (pred (n,) 1 = steady, score (n,) non-steady 사후확률 = 1 - steady 군집 사후확률)
    p_steady = model.predict_proba(z)[:, steady_idx]
    return (p_steady >= 0.5).astype(int), 1.0 - p_steady


def steady_entry_k(pred, min_consec=ENTRY_MIN_CONSEC):
    # 역할 : 판정 시계열에서 steady 진입 시점을 읽는다. 연속 min_consec 개가 steady 로 판정된 구간의 첫 index.
    #        일시적 깜빡임(판정 1개)에 끌려가지 않도록 연속 조건을 둔다.
    # 입력 : pred (n,) k 순서로 정렬된 판정 (1 = steady) / min_consec 연속 유지 개수
    # 출력 : 진입 index (int), 조건을 만족하는 구간이 없으면 -1
    if len(pred) < min_consec:
        return -1
    run_sum = pd.Series(pred).rolling(min_consec).sum().to_numpy()   # index i 는 [i-min_consec+1, i] 구간의 합
    hit = np.flatnonzero(run_sum == min_consec)
    return int(hit[0] - min_consec + 1) if len(hit) else -1


def score_fold(df_val, pred, score, method, fold, seed):
    # 역할 : 검증 fold 의 판정을 `Trigger 4` 로 채점한다. ambiguous(2) 는 채점에서 제외한다.
    #        지표는 pooled(fold 전체 시점을 한 덩어리로) 와 run 평균(run별 계산 후 평균) 두 가지로 모두 기록한다.
    # 입력 : df_val  검증 fold 의 시점 표 [run_id, k, TS, RS, VS] (라벨 없음, pred/score 와 같은 순서)
    #        pred (n,) 판정 (1 = steady) / score (n,) non-steady 점수
    #        method 방법 이름 / fold fold 번호 / seed 난수 시드
    # 출력 : (run_metrics run별 1행 DataFrame, fold_row fold×seed 1행 DataFrame)
    d = df_val.reset_index(drop=True).copy()
    d['pred'] = np.asarray(pred)
    d['score'] = np.asarray(score)
    d = d.merge(_labels(), on=['run_id', 'k'], how='left')           # 라벨 결합 (채점 블록)
    d = d.sort_values(['run_id', 'k']).reset_index(drop=True)

    # --- run별 지표 ---
    rows = []
    for run_id, g in d.groupby('run_id', sort=True):
        s = g[g[LABEL_COL] != 2]                                     # ambiguous 제외 (채점 대상만)
        y_ns = (s[LABEL_COL] == 0).to_numpy()                        # non-steady = positive
        p_st = (s['pred'] == 1).to_numpy()
        entry_pred = steady_entry_k(g['pred'].to_numpy())            # 진입 시점은 ambiguous 포함 전체 시계열에서 읽는다
        true_steady = g.loc[g[LABEL_COL] == 1, 'k']
        entry_true = int(true_steady.min()) if len(true_steady) else -1
        rows.append({
            'method': method, 'fold': fold, 'seed': seed, 'run_id': run_id,
            TS: g[TS].iloc[0], RS: g[RS].iloc[0], VS: g[VS].iloc[0],
            'n': len(g), 'n_scored': len(s), 'n_nonsteady': int(y_ns.sum()),
            'contamination_rate': float((p_st & y_ns).sum() / p_st.sum()) if p_st.sum() else np.nan,
            'retention_rate': float((p_st & ~y_ns).sum() / (~y_ns).sum()) if (~y_ns).sum() else np.nan,
            'f1_nonsteady': f1_score(y_ns, ~p_st, zero_division=0) if y_ns.any() else np.nan,
            'auroc': roc_auc_score(y_ns, s['score']) if 0 < y_ns.sum() < len(y_ns) else np.nan,
            'auprc': average_precision_score(y_ns, s['score']) if y_ns.any() else np.nan,
            'entry_pred_k': entry_pred, 'entry_true_k': entry_true,
            'entry_err_s': (entry_pred - entry_true) * DT if entry_pred >= 0 else np.nan,
        })
    run_metrics = pd.DataFrame(rows)

    # --- fold 단위 집계 : pooled (시점 전체) + run 평균 ---
    sd = d[d[LABEL_COL] != 2]
    y_ns, p_st = (sd[LABEL_COL] == 0).to_numpy(), (sd['pred'] == 1).to_numpy()
    fold_row = {
        'method': method, 'fold': fold, 'seed': seed,
        'n_runs': run_metrics.run_id.nunique(), 'n_scored': len(sd), 'n_nonsteady': int(y_ns.sum()),
        'pooled_contamination_rate': float((p_st & y_ns).sum() / p_st.sum()) if p_st.sum() else np.nan,
        'pooled_retention_rate': float((p_st & ~y_ns).sum() / (~y_ns).sum()) if (~y_ns).sum() else np.nan,
        'pooled_f1_nonsteady': f1_score(y_ns, ~p_st, zero_division=0),
        'pooled_auroc': roc_auc_score(y_ns, sd['score']) if 0 < y_ns.sum() < len(y_ns) else np.nan,
        'pooled_auprc': average_precision_score(y_ns, sd['score']) if y_ns.any() else np.nan,
    }
    for c in ['contamination_rate', 'retention_rate', 'f1_nonsteady', 'auroc', 'auprc']:
        fold_row[f'runmean_{c}'] = run_metrics[c].mean()              # 정의되지 않은 run 은 자동 제외 (NaN)
        fold_row[f'runstd_{c}'] = run_metrics[c].std()
        fold_row[f'n_runs_{c}'] = int(run_metrics[c].notna().sum())
    fold_row['runmean_entry_err_s'] = run_metrics.entry_err_s.mean()  # 부호 포함 (bias: + 는 늦게 진입)
    fold_row['runmae_entry_err_s'] = run_metrics.entry_err_s.abs().mean()
    fold_row['runstd_entry_err_s'] = run_metrics.entry_err_s.std()
    fold_row['n_runs_no_entry'] = int((run_metrics.entry_pred_k < 0).sum())   # steady 를 전혀 판정하지 못한 run
    return run_metrics, pd.DataFrame([fold_row])


def save_results(out_dir, prefix, run_metrics, fold_metrics, config):
    # 역할 : run별 · fold별 지표와 실행에 사용한 설정값을 csv 로 저장한다. 같은 이름의 결과는 덮어쓰지 않는다.
    #        seed 평균 ± 표준편차 요약표도 함께 만든다.
    # 입력 : out_dir 결과 폴더 / prefix 파일 이름 앞부분 (보통 방법 이름)
    #        run_metrics, fold_metrics score_fold 결과를 concat 한 것 / config 설정값 dict
    # 출력 : 없음 (csv 3~4개 저장)
    os.makedirs(out_dir, exist_ok=True)
    paths = {'run': f'{out_dir}/{prefix}_run_metrics.csv', 'fold': f'{out_dir}/{prefix}_fold_metrics.csv',
             'summary': f'{out_dir}/{prefix}_summary.csv', 'config': f'{out_dir}/{prefix}_config.csv'}
    exist = [p for p in paths.values() if os.path.exists(p)]
    assert not exist, f'이미 결과 파일이 있다 (덮어쓰기 금지). 다시 실행하려면 먼저 지울 것: {exist}'

    run_metrics.to_csv(paths['run'], index=False)
    fold_metrics.to_csv(paths['fold'], index=False)
    num = fold_metrics.select_dtypes('number').columns.drop('seed')
    summary = fold_metrics.groupby('fold')[num].agg(['mean', 'std'])  # seed 반복에 대한 평균 ± 표준편차
    summary.loc['all_folds'] = fold_metrics[num].agg(['mean', 'std']).T.stack()
    summary.to_csv(paths['summary'])
    pd.DataFrame({'name': list(config), 'value': [str(v) for v in config.values()]}).to_csv(paths['config'], index=False)
    for k, p in paths.items():
        print(f'[save] {k:7s} -> {p}')


PAGE_WIDTH = 6.5                # 논문 지면 폭 (in). 그림 폭을 여기에 맞춘다. 임의의 값
HEATMAP_NCOLS = 3               # heatmap subplot 을 한 줄에 몇 개 놓을지. 임의의 값


def plot_condition_heatmap(run_metrics, metric, out_dir, prefix, label, row_col=RS, col_col=VS, fold_setting=TS):
    # 역할 : 검증 fold 안의 운전조건 격자에 run별 성능을 깔아 heatmap 으로 그린다 (fold 하나 = subplot 하나).
    #        seed 반복이 있으면 run별로 평균한 값을 쓴다.
    # 입력 : run_metrics save 대상 run별 지표 / metric 그릴 지표 컬럼명 / out_dir 저장 폴더 / prefix 파일 이름 앞부분
    #        label colorbar 에 쓸 영문 지표 이름(단위 포함) / row_col, col_col heatmap 축이 될 설정값 컬럼
    #        fold_setting subplot 제목에 쓸 fold 구분 설정값 컬럼
    # 출력 : 없음 (PNG 300 dpi + PDF 저장)
    m = run_metrics.groupby(['fold', row_col, col_col, fold_setting], as_index=False)[metric].mean()
    folds = sorted(m.fold.unique())
    vmin, vmax = np.nanmin(m[metric]), np.nanmax(m[metric])
    ncols = min(HEATMAP_NCOLS, len(folds))
    nrows = int(np.ceil(len(folds) / ncols))
    n_row_cells, n_col_cells = m[row_col].nunique(), m[col_col].nunique()
    cell = (PAGE_WIDTH / ncols - 0.55) / n_col_cells                  # 셀 한 칸의 폭 (in) — 축 라벨 자리를 뺀 나머지
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout='constrained',
                             figsize=(PAGE_WIDTH, nrows * (cell * n_row_cells + 0.95) + 0.3))
    for ax, f in zip(axes.ravel(), folds):
        g = m[m.fold == f]
        grid = g.pivot(index=row_col, columns=col_col, values=metric)
        im = ax.imshow(grid.to_numpy(), cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(grid.shape[1]), [f'{int(v)}' for v in grid.columns], fontsize=FONT_SIZE - 2)
        ax.set_yticks(range(grid.shape[0]), [f'{int(v)}' for v in grid.index], fontsize=FONT_SIZE - 2)
        ax.set_title(f'Fold {int(f)}: film tension {int(g[fold_setting].iloc[0])} kg')
        ax.grid(False)
        for i in range(grid.shape[0]):                                # 셀 값 직접 표기 (대비 보완 · 수치 확인용)
            for j in range(grid.shape[1]):
                v = grid.to_numpy()[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}'.lstrip('0'), ha='center', va='center', fontsize=FONT_SIZE - 3,
                            color='#ffffff' if v > (vmin + vmax) / 2 else '#0b0b0b')
    for ax in axes.ravel()[len(folds):]:                              # 남는 칸은 비운다
        ax.axis('off')
    fig.supxlabel('Web speed setting (mm/s)', fontsize=FONT_SIZE)     # 축 이름은 그림 전체에 하나씩만
    fig.supylabel('ReWinder tension setting (kg)', fontsize=FONT_SIZE)
    fig.colorbar(im, ax=axes, label=label, fraction=0.03, pad=0.02)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        p = f'{out_dir}/{prefix}_heatmap_{metric}.{ext}'
        assert not os.path.exists(p), f'이미 그림 파일이 있다 (덮어쓰기 금지): {p}'
        fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f'[save] heatmap -> {out_dir}/{prefix}_heatmap_{metric}.png / .pdf')
