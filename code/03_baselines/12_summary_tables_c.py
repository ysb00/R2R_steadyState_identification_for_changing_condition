# =============================================================================
# 03_baselines/12_summary_tables_c.py : C 실험(저속 학습 -> 고속 테스트) 결과를 집계한다
#
# 목적 : `09_summary_tables.py` 는 A 실험 전용이다. A 는 fold 5개가 전부 같은 질문
#        ("새 장력 설정값에 일반화되는가")을 물어서 25회를 한 덩어리로 평균해도 됐다.
#        **C 는 fold 묶음마다 묻는 질문이 다르다** (hyperparameters.md §13.0 F6):
#          cv_T1~T5   : 저속 안에서 **새 장력** 에 되는가      (A 와 같은 질문)
#          cv_S1      : 학습{50,75}   -> 검증{100}   1수준 속도 외삽
#          cv_S2      : 학습{50,75,100} -> 검증{125} 1수준 속도 외삽 (학습량 증가)
#          final_test : 학습 저속 4수준 -> 검증 고속 3수준. **최종 목표. 한 번만 쓴다**
#        그래서 **네 단위를 가로질러 평균내지 않는다.** 표의 열로 분리해 싣는다.
#
# 출력 (outputs/04_speed_extrap/)
#   ① `_all_methods_by_fold_seed.csv`  방법 x fold x seed — 평균으로 누르기 전 원자료
#   ② `_all_methods_by_fold.csv`       방법 x fold (seed 평균)
#   ③ `_all_methods_by_group.csv`      방법 x 묶음(cv_T / cv_S / final_test). **묶음 안에서만 평균**
#   ④ `_paper_table_c.csv`             방법당 1행. 지표마다 cv_T / cv_S1 / cv_S2 / final_test 4열
#   ⑤ `_final_test_by_speed.csv`       방법 x 속도(150/175/200) — 최종 test 를 속도별로 쪼갠 것
#   ⑥ `_final_test_by_speed.png|pdf`   ⑤ 의 속도별 성능 곡선 (본문 표 후보 8종만)
#
# ⑤ 의 집계 2종 — 왜 섞여 있는가 (연구자 결정 2026-10-27)
#   `*_fold_metrics.csv` 의 `pooled_` 는 fold 전체를 한 덩어리로 계산한 값이라 final_test 는
#   속도 3수준이 이미 뭉쳐 있다. 속도별로 쪼개려면 `*_run_metrics.csv` 에서 다시 집계해야 한다.
#     `pooled_`  개수 기반 지표 — run 별 혼동행렬을 복원해 속도 안에서 **합산**한다.
#                본문 표·cv_* 와 **같은 의미**다. 복원식은 evaluation.derive_precision_recall 과 같다
#     `runmean_` AUROC·AUPRC·진입 시점 — run 별 값을 **평균**한다. 점수 분포를 저장하지 않아
#                pooled 로는 복원할 수 없다 (§13.4 A7 이 확인한 것과 같은 제약)
#   복원이 맞는지는 Step 2 에서 fold 전체로 다시 합산해 저장된 `pooled_` 와 대조한다.
#
# 입력 : outputs/04_speed_extrap/*/*_fold_metrics.csv, *_run_metrics.csv (+ *_config.csv)
#        outputs/03_baselines/_all_methods_full_table.csv  (방법 분류 — A 집계의 것을 그대로 쓴다)
# 실행 : python code/03_baselines/12_summary_tables_c.py   (repo 루트에서)
# =============================================================================
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev
import splits

OUT_DIR = 'outputs/04_speed_extrap'
A_TABLE = 'outputs/03_baselines/_all_methods_full_table.csv'   # 방법 분류(category·table_role)의 출처
GROUP_ORDER = ['cv_T', 'cv_S', 'final_test']                   # 묶음을 싣는 순서
PAPER_UNITS = ['cv_T', 'cv_S1', 'cv_S2', 'final_test']         # ④ 의 열 단위 — cv_S 는 두 fold 를 나눠 싣는다
                                                               # (학습량이 달라 평균내면 둘이 섞인다)
CURVE_METRICS = [('pooled_contamination_rate', 'Contamination rate'),
                 ('pooled_retention_rate', 'Retention rate'),
                 ('runmean_auprc', 'AUPRC (non-steady), run mean')]

# --- 방법 분류를 A 집계 결과에서 읽는다 (같은 23종이므로 정의를 중복해 두지 않는다) ---
a_tbl = pd.read_csv(A_TABLE, usecols=['method', 'category', 'table_role'])
CATEGORY = dict(zip(a_tbl.method, a_tbl.category))
ROLE = dict(zip(a_tbl.method, a_tbl.table_role))


def confusion(run_m):
    # 역할 : run 별 지표에서 혼동행렬 4칸을 복원한다 (steady = positive).
    #        속도 안에서 이것을 합산하면 pooled 지표가 된다. 식은 evaluation.derive_precision_recall 과 같다:
    #          n_steady = n_scored - n_nonsteady,  TP = 보존률 x n_steady,  FN = n_steady - TP
    #          f1_nonsteady 를 FP 에 대해 풀어  FP = [2·n_ns·(1 - f1) - f1·FN] / (2 - f1),  TN = n_ns - FP
    #        오염률이 아니라 f1 로 푸는 이유: TP = 0 이고 FP > 0 인 run 에서 오염률이 1 이 되어 식이 깨진다.
    #        non-steady 시점이 없는 run(210 중 26개)은 f1 이 NaN 이지만 FP = TN = 0 으로 정해진다.
    # 입력 : run_m  run별 지표 DataFrame [n_scored, n_nonsteady, retention_rate, f1_nonsteady]
    # 출력 : 같은 길이의 DataFrame [tp, fp, fn, tn] (float)
    n_ns = run_m.n_nonsteady.to_numpy(float)
    n_s = run_m.n_scored.to_numpy(float) - n_ns
    tp = run_m.retention_rate.to_numpy(float) * n_s
    fn = n_s - tp
    f1 = run_m.f1_nonsteady.to_numpy(float)
    fp = np.where(n_ns > 0, (2 * n_ns * (1 - f1) - f1 * fn) / (2 - f1), 0.0)
    return pd.DataFrame({'tp': tp, 'fp': fp, 'fn': fn, 'tn': n_ns - fp}, index=run_m.index)


def pooled_from_counts(c):
    # 역할 : 합산된 혼동행렬 4칸에서 pooled 지표를 만든다. 이름·정의는 evaluation.score_fold 와 같다.
    # 입력 : c  tp/fp/fn/tn 을 가진 Series (합산된 값)
    # 출력 : 지표 dict ({'pooled_...': 값})
    tp, fp, fn, tn = c.tp, c.fp, c.fn, c.tn
    pred_st, true_st, true_ns = tp + fp, tp + fn, fp + tn
    return {'pooled_contamination_rate': fp / pred_st if pred_st else np.nan,
            'pooled_retention_rate': tp / true_st if true_st else np.nan,
            'pooled_precision_steady': tp / pred_st if pred_st else np.nan,
            'pooled_recall_steady': tp / true_st if true_st else np.nan,
            'pooled_f1_steady': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            'pooled_precision_nonsteady': tn / (tn + fn) if (tn + fn) else np.nan,
            'pooled_recall_nonsteady': tn / true_ns if true_ns else np.nan,
            'pooled_f1_nonsteady': 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else np.nan,
            'n_scored': tp + fp + fn + tn, 'n_nonsteady': true_ns}


def style_of(method):
    # 역할 : 방법 이름(변형 포함, 예 'CoST-W144')에 맞는 고정 스타일을 찾는다.
    #        evaluation.METHOD_STYLE 의 키 중 이름 앞부분이 가장 길게 일치하는 것을 쓴다.
    # 입력 : method 방법 이름
    # 출력 : (color, marker, linestyle)
    keys = [k for k in ev.METHOD_STYLE if method.startswith(k)]
    assert keys, f'METHOD_STYLE 에 등록되지 않은 방법: {method}'
    return ev.METHOD_STYLE[max(keys, key=len)]


# =============================================================================
# Step 1. fold_metrics 를 읽어 ① fold x seed  ② fold  ③ 묶음 으로 집계
# =============================================================================
by_fold_seed, by_fold, by_group = [], [], []
for path in sorted(glob.glob(f'{OUT_DIR}/*/*_fold_metrics.csv')):
    method = os.path.basename(path).replace('_fold_metrics.csv', '')
    stage = os.path.basename(os.path.dirname(path))
    d = pd.read_csv(path)
    d['group'] = [splits.fold_group(f) for f in d.fold]
    meta = {'stage': stage, 'category': CATEGORY.get(method, 'unregistered'),
            'table_role': ROLE.get(method, 'unregistered')}
    num = [c for c in d.select_dtypes(include=[np.number]).columns if c != 'seed']

    fs = d.copy()                                                   # --- ① fold x seed 원자료 ---
    for k, v in reversed(list(meta.items())):
        fs.insert(0, k, v)
    by_fold_seed.append(fs)

    fb = d.groupby(['group', 'fold'], sort=False)[num].mean().reset_index()   # --- ② fold 별 (seed 평균) ---
    fb.insert(0, 'method', method)
    for k, v in reversed(list(meta.items())):
        fb.insert(0, k, v)
    by_fold.append(fb)

    for g, dg in d.groupby('group', sort=False):                    # --- ③ 묶음별. 묶음을 가로질러 평균하지 않는다 ---
        fold_mean = dg.groupby('fold')[num].mean()                  # fold 평균 (seed 를 평균한 값)
        fold_std = dg.groupby('fold')[num].std()                    # fold 내 seed 표준편차
        row = {**meta, 'method': method, 'group': g, 'n_rows': len(dg),
               'n_folds': dg.fold.nunique(), 'n_seeds': dg.seed.nunique(),
               # 퇴화 = GMM 이 'steady vs non-steady' 가 아니라 '이상치 소수 vs 나머지' 로 쪼갠 경우
               # (AUROC 가 정확히 0.5). hyperparameters.md §11 문제 11
               'n_degenerate': int((dg.pooled_auroc.round(6) == 0.5).sum())}
        for c in num:
            row[f'{c}__mean'] = dg[c].mean()
            row[f'{c}__std'] = dg[c].std()
            row[f'{c}__foldstd'] = fold_mean[c].std()               # 묶음 안 fold 간 차이 (final_test 는 fold 1개라 NaN)
            row[f'{c}__seedstd'] = fold_std[c].mean()               # 학습 운
        by_group.append(row)

    cfg_path = path.replace('_fold_metrics.csv', '_config.csv')
    if os.path.exists(cfg_path):
        for _, x in pd.read_csv(cfg_path).iterrows():
            by_group[-1][f'cfg_{x["name"]}'] = x['value']           # 설정값은 묶음 표에만 붙인다

fold_seed_tbl = pd.concat(by_fold_seed, ignore_index=True)
fold_tbl = pd.concat(by_fold, ignore_index=True)
group_tbl = pd.DataFrame(by_group)
group_tbl['group'] = pd.Categorical(group_tbl.group, GROUP_ORDER, ordered=True)
group_tbl = group_tbl.sort_values(['method', 'group'])
front = ['stage', 'category', 'table_role', 'method', 'group', 'n_rows', 'n_folds', 'n_seeds', 'n_degenerate']
group_tbl = group_tbl[front + [c for c in group_tbl.columns if c not in front]]

# =============================================================================
# Step 2. final_test 를 속도별로 쪼갠다 (run_metrics 재집계)
#         pooled_  : run 별 혼동행렬을 복원해 속도 안에서 합산
#         runmean_ : run 별 값의 평균 (AUROC·AUPRC·진입 시점 — pooled 복원 불가)
#         검증     : 속도로 쪼개지 않고 fold 전체로 합산해 저장된 pooled_ 와 일치하는지 본다
# =============================================================================
RUNMEAN_COLS = ['auroc', 'auprc', 'auprc_steady', 'entry_err_s']
by_speed, recon_err = [], []
for path in sorted(glob.glob(f'{OUT_DIR}/*/*_run_metrics.csv')):
    method = os.path.basename(path).replace('_run_metrics.csv', '')
    r = pd.read_csv(path)
    r = r[r.fold == 'final_test']
    if not len(r):
        continue
    r = pd.concat([r, confusion(r)], axis=1)

    # --- 검증 : 속도를 묶어 fold 전체로 합산한 값이 저장된 pooled_ 와 같은가 ---
    stored = pd.read_csv(path.replace('_run_metrics.csv', '_fold_metrics.csv'))
    stored = stored[stored.fold == 'final_test']
    for seed, rs in r.groupby('seed'):
        got = pooled_from_counts(rs[['tp', 'fp', 'fn', 'tn']].sum())
        ref = stored[stored.seed == seed].iloc[0]
        for c in ['pooled_contamination_rate', 'pooled_retention_rate', 'pooled_f1_steady',
                  'pooled_f1_nonsteady', 'pooled_precision_nonsteady', 'pooled_recall_nonsteady']:
            if pd.notna(ref[c]) and pd.notna(got[c]):
                recon_err.append(abs(got[c] - ref[c]))

    # --- 속도별 집계 (seed 평균 ± 표준편차) ---
    for v, rv in r.groupby(ev.VS):
        per_seed = []
        for seed, rs in rv.groupby('seed'):
            row = pooled_from_counts(rs[['tp', 'fp', 'fn', 'tn']].sum())
            row['seed'] = seed
            row['n_runs'] = rs.run_id.nunique()
            for c in RUNMEAN_COLS:
                row[f'runmean_{c}'] = rs[c].mean()
            row['runmae_entry_err_s'] = rs.entry_err_s.abs().mean()
            row['n_runs_no_entry'] = int((rs.entry_pred_k < 0).sum())
            per_seed.append(row)
        ps = pd.DataFrame(per_seed)
        num = [c for c in ps.columns if c != 'seed']
        out = {'category': CATEGORY.get(method, 'unregistered'), 'table_role': ROLE.get(method, 'unregistered'),
               'method': method, ev.VS: int(v), 'n_seeds': ps.seed.nunique()}
        for c in num:
            out[f'{c}__mean'] = ps[c].mean()
            out[f'{c}__std'] = ps[c].std()
        by_speed.append(out)

speed_tbl = pd.DataFrame(by_speed).sort_values(['method', ev.VS])
assert recon_err, 'final_test run_metrics 를 찾지 못했다 — F5 가 끝났는지 확인할 것'
print(f'[검증] 혼동행렬 복원 오차 (pooled 재합산 vs 저장값): 최대 {max(recon_err):.2e}, {len(recon_err)}건')
assert max(recon_err) < 1e-6, '혼동행렬 복원이 저장된 pooled 값과 다르다 — 복원식을 확인할 것'

# =============================================================================
# Step 3. 논문 표 — 지표마다 cv_T / cv_S1 / cv_S2 / final_test 4열
# =============================================================================
PAPER_COLS = [                                                      # (표에 쓸 이름, 원 컬럼, 소수 자리)
    ('Contamination',   'pooled_contamination_rate', 3),
    ('Retention',       'pooled_retention_rate',     3),
    ('F1 (ns)',         'pooled_f1_nonsteady',       3),
    ('AUROC',           'pooled_auroc',              3),
    ('AUPRC (ns)',      'pooled_auprc',              3),
    ('Entry MAE (s)',   'runmae_entry_err_s',        1),
]


def fmt_pm(mean, std, nd):
    # 역할 : "0.832±0.147" 형태로 합친다. 표준편차가 없으면(fold 1개 등) 평균만 돌려준다.
    # 입력 : mean, std 값 / nd 소수 자리 수
    # 출력 : 문자열 ('—' 이면 값이 없음)
    if pd.isna(mean):
        return '—'
    if pd.isna(std):
        return f'{mean:.{nd}f}'
    return f'{mean:.{nd}f}±{std:.{nd}f}'


paper = []
for method, g in fold_seed_tbl.groupby('method', sort=False):
    row = {'method': method, 'category': CATEGORY.get(method, 'unregistered'),
           'role': ROLE.get(method, 'unregistered'),
           'degenerate': int((g.pooled_auroc.round(6) == 0.5).sum())}
    for unit in PAPER_UNITS:                                        # cv_T 는 fold 5개 묶음, 나머지는 fold 1개
        sub = g[g.group == unit] if unit in GROUP_ORDER else g[g.fold == unit]
        row[f'{unit} n'] = len(sub)
        for name, col, nd in PAPER_COLS:
            row[f'{name} | {unit}'] = fmt_pm(sub[col].mean(), sub[col].std(), nd) if len(sub) else '—'
    paper.append(row)
paper = pd.DataFrame(paper)
paper = paper.sort_values(['role', 'AUPRC (ns) | final_test'], ascending=[True, False])

# =============================================================================
# Step 4. 저장
# =============================================================================
paths = {'fold_seed': f'{OUT_DIR}/_all_methods_by_fold_seed.csv',
         'fold': f'{OUT_DIR}/_all_methods_by_fold.csv',
         'group': f'{OUT_DIR}/_all_methods_by_group.csv',
         'paper': f'{OUT_DIR}/_paper_table_c.csv',
         'speed': f'{OUT_DIR}/_final_test_by_speed.csv'}
fold_seed_tbl.to_csv(paths['fold_seed'], index=False)
fold_tbl.to_csv(paths['fold'], index=False)
group_tbl.to_csv(paths['group'], index=False)
paper.to_csv(paths['paper'], index=False)
speed_tbl.to_csv(paths['speed'], index=False)
for k, p in paths.items():
    s = pd.read_csv(p)
    print(f'[save] {p}  {s.shape[0]} 행 x {s.shape[1]} 컬럼')

unreg = sorted(set(group_tbl.loc[group_tbl.category == 'unregistered', 'method']))
if unreg:
    print(f'[주의] A 집계 표에 없는 방법: {unreg}')

# =============================================================================
# Step 5. 최종 test 속도별 성능 곡선 (본문 표 후보 8종만 — 23종을 한 그림에 두면 읽을 수 없다)
# =============================================================================
main = speed_tbl[speed_tbl.table_role == 'main']
fig, axes = plt.subplots(1, len(CURVE_METRICS), figsize=(ev.PAGE_WIDTH, 2.6), layout='constrained')
for ax, (col, ylabel) in zip(axes, CURVE_METRICS):
    for method, g in main.groupby('method'):
        color, marker, ls = style_of(method)
        g = g.sort_values(ev.VS)
        ax.errorbar(g[ev.VS], g[f'{col}__mean'], yerr=g[f'{col}__std'], label=method,
                    color=color, marker=marker, linestyle=ls, linewidth=1.4, markersize=4,
                    capsize=2, elinewidth=0.8)
    ax.set_xticks(sorted(main[ev.VS].unique()))
    ax.set_ylabel(ylabel)
fig.supxlabel('Web speed setting (mm/s) — unseen high speed, final test', fontsize=ev.FONT_SIZE)
handles, labels = axes[0].get_legend_handles_labels()
# 범례는 그림 위에 둔다 — 아래에 두면 constrained layout 에서 supxlabel 과 같은 자리에 겹친다
fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False)
for ext in ['png', 'pdf']:
    p = f'{OUT_DIR}/_final_test_by_speed.{ext}'
    assert not os.path.exists(p), f'이미 그림 파일이 있다 (덮어쓰기 금지): {p}'
    fig.savefig(p, dpi=ev.DPI)
plt.close(fig)
print(f'[save] {OUT_DIR}/_final_test_by_speed.png / .pdf  (본문 표 후보 {main.method.nunique()}종)')

# =============================================================================
# Step 6. 전 비율 지표가 [0, 1] 범위인지 검증 (복원식 오용·집계 실수 탐지)
# =============================================================================
RATIO = ('precision', 'recall', 'retention', 'contamination', 'f1', 'auroc', 'auprc')
bad = []
for t, name in [(group_tbl, 'group'), (speed_tbl, 'speed')]:
    for c in t.columns:
        if not any(k in c for k in RATIO) or c.startswith(('cfg_', 'n_runs_')) \
                or any(k in c for k in ('__std', '__foldstd', '__seedstd', 'runstd_')):
            continue
        s = pd.to_numeric(t[c], errors='coerce').dropna()
        if len(s) and (s.max() > 1.0001 or s.min() < -0.0001):
            bad.append((name, c, round(s.min(), 4), round(s.max(), 4)))
print(f'[검증] 비율 지표 범위 이탈: {bad if bad else "없음"}')
assert not bad, '비율 지표가 [0,1] 을 벗어났다 — 집계식을 확인할 것'
print(f'[done] 방법 {group_tbl.method.nunique()}종, 묶음 {GROUP_ORDER}, '
      f'속도별 표 {speed_tbl.method.nunique()}종 x {speed_tbl[ev.VS].nunique()}수준')
