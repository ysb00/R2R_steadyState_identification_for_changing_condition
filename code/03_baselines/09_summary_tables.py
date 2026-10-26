# =============================================================================
# 03_baselines/09_summary_tables.py : 전 비교군 결과를 3단계 집계 표로 묶는다
#
# 목적 : 방법별로 흩어져 있는 `*_fold_metrics.csv` 를 한곳에 모아, 어느 행·열을 논문에
#        남길지 판단할 수 있게 한다. **지표를 고르거나 버리지 않고 전부 담는다.**
#
# 출력 3단계 (해상도가 다르다 — 위에서 아래로 갈수록 자세하다)
#   ① outputs/03_baselines/_all_methods_full_table.csv       방법당 1행
#        지표마다 4개 컬럼: `__mean` / `__std` / `__foldstd` / `__seedstd`
#   ② outputs/03_baselines/_all_methods_by_fold.csv          방법 x fold
#        fold 별 값. `research_experiments_plan.md` 4.1 이 "fold별 값을 함께 보고" 하라고 요구한 것
#   ③ outputs/03_baselines/_all_methods_by_fold_seed.csv     방법 x fold x seed
#        평균으로 누르기 전 원자료
#   (run 단위 원자료는 각 방법의 `*_run_metrics.csv` 에 그대로 있다. 23종을 합치면 24,150행이라
#    여기서는 묶지 않는다.)
#
# 표준편차 4종의 뜻 — 분산을 분해해야 "무엇 때문에 흔들리는가" 가 보인다
#   `__mean`    : fold x seed 25회 전체 평균
#   `__std`     : 25회 전체 표준편차.  **논문 표의 "±" 에 쓸 값**
#   `__foldstd` : fold 평균들의 표준편차 → **조건 난이도 차이** (fold 1·5 는 외삽, 2·3·4 는 내삽)
#   `__seedstd` : fold 내 seed 표준편차의 fold 평균 → **학습 운**
#   실측: CoST·MovStd·R-test·MOMENT 는 `__seedstd` ≈ 0 (재현성 높음),
#         SoftCLT·TS-TCC·SimMTM·TS2Vec 은 `__seedstd` ≥ `__foldstd` (학습 운이 지배)
#
# 주의 — `runmean_` 에는 결측 도출식을 쓰지 않는다
#   `evaluation.derive_precision_recall` 은 혼동행렬 항등식이라 **pooled_ 에만 유효**하다.
#   `runmean_` 에 쓰면 fold 전체 `n_scored`/`n_nonsteady` 와 run별 비율의 평균을 섞게 되어
#   항등식이 깨진다 (recall > 1 이 나온다). 그래서 pooled_ 만 도출하고 runmean_ 은 저장된 값만 쓴다.
#
# 입력 : outputs/03_baselines/*/*_fold_metrics.csv  (+ 같은 이름의 `_config.csv`)
# 실행 : python code/03_baselines/09_summary_tables.py   (repo 루트에서)
# =============================================================================
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluation as ev

OUT_DIR = 'outputs/03_baselines'

# --- 방법 분류 (그림·표 묶음용). 새 방법을 추가하면 여기에 등록한다 ---
CATEGORY = {
    'R-test': 'statistical', 'R-test-decimated': 'statistical',
    'R-test-calibrated': 'statistical', 'R-test-decimated-calibrated': 'statistical',
    'MovStd-Slope-fixed': 'statistical', 'MovStd-Slope-speed': 'statistical',
    'MovStd-Slope-fixed-rule1b': 'statistical', 'MovStd-Slope-speed-rule1b': 'statistical',
    'TS2Vec': 'contrastive', 'SoftCLT-fs': 'contrastive', 'SoftCLT-fs-rule1b': 'contrastive',
    'SoftCLT-ref': 'contrastive', 'CoST-W144': 'contrastive', 'CoST-W160': 'contrastive',
    'CoST-W256': 'contrastive', 'CoST-W306': 'contrastive', 'CoST-W512': 'contrastive',
    'TS-TCC': 'contrastive', 'SimMTM': 'generative', 'MOMENT': 'foundation',
    'CoST-W144-trend': 'ablation', 'CoST-W144-season': 'ablation', 'CoST-W144-noscaleshift': 'ablation',
}
# 본문 표 후보 — 방법별 대표 1종 (근거는 docs/baseline_cases.md §10)
MAIN = {'CoST-W144', 'MovStd-Slope-fixed', 'SimMTM', 'R-test-decimated-calibrated',
        'TS2Vec', 'TS-TCC', 'SoftCLT-fs-rule1b', 'MOMENT'}


def role_of(method):
    # 역할 : 표에서의 쓰임을 정한다. ablation 은 별도 절, main 은 본문 표, 나머지는 민감도.
    if CATEGORY.get(method) == 'ablation':
        return 'ablation'
    return 'main' if method in MAIN else 'sensitivity'


# =============================================================================
# Step 1. 방법별 fold_metrics 를 읽어 3단계로 집계
# =============================================================================
per_method, by_fold, by_fold_seed = [], [], []
for path in sorted(glob.glob(f'{OUT_DIR}/*/*_fold_metrics.csv')):
    method = os.path.basename(path).replace('_fold_metrics.csv', '')
    stage = os.path.basename(os.path.dirname(path))
    d = ev.derive_precision_recall(pd.read_csv(path))               # pooled_ 결측만 복원 (위 주의 참조)
    meta = {'stage': stage, 'category': CATEGORY.get(method, 'unregistered'),
            'table_role': role_of(method)}
    num = [c for c in d.select_dtypes(include=[np.number]).columns if c not in ('fold', 'seed')]

    # --- ③ fold x seed 원자료 ---
    fs = d.copy()
    for k, v in reversed(list(meta.items())):
        fs.insert(0, k, v)
    by_fold_seed.append(fs)

    # --- ② fold 별 (seed 평균) ---
    fb = d.groupby('fold')[num].mean().reset_index()
    fb.insert(0, 'method', method)
    for k, v in reversed(list(meta.items())):
        fb.insert(0, k, v)
    by_fold.append(fb)

    # --- ① 방법당 1행 : 지표마다 mean / std / foldstd / seedstd ---
    fold_mean = d.groupby('fold')[num].mean()                       # fold 평균 (seed 를 평균한 값)
    fold_std = d.groupby('fold')[num].std()                         # fold 내 seed 표준편차
    row = {**meta, 'method': method, 'n_rows': len(d),
           'n_folds': d.fold.nunique(), 'n_seeds': d.seed.nunique(),
           # 퇴화 = GMM 이 'steady vs non-steady' 가 아니라 '이상치 소수 vs 나머지' 로 쪼갠 경우.
           # AUROC 가 정확히 0.5 이고 보존률이 정확히 1.0 이 된다 (hyperparameters.md §11 문제 11)
           'n_degenerate': int((d.pooled_auroc.round(6) == 0.5).sum())}
    for c in num:
        row[f'{c}__mean'] = d[c].mean()
        row[f'{c}__std'] = d[c].std()
        row[f'{c}__foldstd'] = fold_mean[c].std()
        row[f'{c}__seedstd'] = fold_std[c].mean()
    cfg_path = path.replace('_fold_metrics.csv', '_config.csv')
    if os.path.exists(cfg_path):                                    # 실행 설정값을 cfg_ 접두사로 붙인다
        for _, x in pd.read_csv(cfg_path).iterrows():
            row[f'cfg_{x["name"]}'] = x['value']
    per_method.append(row)

# =============================================================================
# Step 2. 저장
# =============================================================================
t = pd.DataFrame(per_method)
front = ['stage', 'category', 'table_role', 'method', 'n_rows', 'n_folds', 'n_seeds', 'n_degenerate']
t = t[front + [c for c in t.columns if c not in front]]
paths = {'full': f'{OUT_DIR}/_all_methods_full_table.csv',
         'fold': f'{OUT_DIR}/_all_methods_by_fold.csv',
         'fold_seed': f'{OUT_DIR}/_all_methods_by_fold_seed.csv'}
t.to_csv(paths['full'], index=False)
pd.concat(by_fold, ignore_index=True).to_csv(paths['fold'], index=False)
pd.concat(by_fold_seed, ignore_index=True).to_csv(paths['fold_seed'], index=False)

unreg = sorted(t.loc[t.category == 'unregistered', 'method'])
if unreg:
    print(f'[주의] CATEGORY 에 등록되지 않은 방법: {unreg}')
for k, p in paths.items():
    s = pd.read_csv(p)
    print(f'[save] {p}  {s.shape[0]} 행 x {s.shape[1]} 컬럼')

# =============================================================================
# Step 4. 논문 표용 — "평균±표준편차" 문자열로 합친 표
#         ①의 `__mean`/`__std` 가 두 열로 나뉘어 있어 수동 결합이 필요한 것을 없앤다.
#         `research_experiments_plan.md` 4.1 이 요구한 **fold별 값**도 한 열에 담는다.
# =============================================================================
#   비율 지표는 소수 3자리, 초 단위는 1자리, 개수는 정수로 맞춘다 (표에 그대로 붙일 수 있게).
PAPER_COLS = [                                                      # (표에 쓸 이름, 원 컬럼, 소수 자리)
    ('Contamination',   'pooled_contamination_rate', 3),
    ('Retention',       'pooled_retention_rate',     3),
    ('Precision (ns)',  'pooled_precision_nonsteady', 3),
    ('Recall (ns)',     'pooled_recall_nonsteady',   3),
    ('F1 (ns)',         'pooled_f1_nonsteady',       3),
    ('F1 (st)',         'pooled_f1_steady',          3),
    ('AUROC',           'pooled_auroc',              3),
    ('AUPRC (ns)',      'pooled_auprc',              3),
    ('AUPRC (st)',      'pooled_auprc_steady',       3),
    ('Entry bias (s)',  'runmean_entry_err_s',       1),
    ('Entry MAE (s)',   'runmae_entry_err_s',        1),
    ('No-entry runs',   'n_runs_no_entry',           1),
]
FOLD_COLS = ['pooled_auprc', 'pooled_auroc', 'pooled_retention_rate']   # fold별 값을 실을 지표


def fmt_pm(mean, std, nd):
    # 역할 : "0.832±0.147" 형태로 합친다. 표준편차가 없으면(seed 1회 등) 평균만 돌려준다.
    # 입력 : mean, std 값 / nd 소수 자리 수
    # 출력 : 문자열 ('—' 이면 값이 없음)
    if pd.isna(mean):
        return '—'
    if pd.isna(std):
        return f'{mean:.{nd}f}'
    return f'{mean:.{nd}f}±{std:.{nd}f}'


fold_tbl = pd.concat(by_fold, ignore_index=True)
paper = []
for _, r in t.iterrows():
    row = {'method': r.method, 'category': r.category, 'role': r.table_role,
           'n_fold x n_seed': f'{int(r.n_folds)}x{int(r.n_seeds)}',
           'degenerate': int(r.n_degenerate)}
    for name, col, nd in PAPER_COLS:
        row[name] = fmt_pm(r.get(f'{col}__mean'), r.get(f'{col}__std'), nd)
    # fold별 값 — 'fold1/fold2/.../fold5' 로 한 칸에 담는다
    sub = fold_tbl[fold_tbl.method == r.method].sort_values('fold')
    for col in FOLD_COLS:
        vals = sub[col].tolist() if col in sub else []
        row[f'{col} by fold'] = '/'.join('—' if pd.isna(v) else f'{v:.3f}' for v in vals) or '—'
    paper.append(row)

paper = pd.DataFrame(paper).sort_values(['role', 'AUPRC (ns)'], ascending=[True, False])
paper.to_csv(f'{OUT_DIR}/_paper_table.csv', index=False)
print(f'[save] {OUT_DIR}/_paper_table.csv  {paper.shape[0]} 행 x {paper.shape[1]} 컬럼 '
      f'(평균±표준편차 문자열 + fold별 값)')

# =============================================================================
# Step 3. 전 비율 지표가 [0, 1] 범위인지 검증 (도출식 오용·집계 실수 탐지)
# =============================================================================
RATIO = ('precision', 'recall', 'retention', 'contamination', 'f1', 'auroc', 'auprc')
bad = []
for c in t.columns:
    if not any(k in c for k in RATIO) or c.startswith(('cfg_', 'n_runs_')) or '__std' in c \
            or '__foldstd' in c or '__seedstd' in c or c.startswith('runstd_'):
        continue
    s = pd.to_numeric(t[c], errors='coerce').dropna()
    if len(s) and (s.max() > 1.0001 or s.min() < -0.0001):
        bad.append((c, round(s.min(), 4), round(s.max(), 4)))
print(f'[검증] 비율 지표 범위 이탈: {bad if bad else "없음"}')
assert not bad, '비율 지표가 [0,1] 을 벗어났다 — 집계식을 확인할 것'
print(f'[done] 방법 {len(t)}종, 분류 {dict(t.category.value_counts())}')
