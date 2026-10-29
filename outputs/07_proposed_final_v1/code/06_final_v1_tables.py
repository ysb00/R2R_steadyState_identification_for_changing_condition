# =============================================================================
# 04_proposed/06_final_v1_tables.py : Proposed (final v1) 지정 + 논문 비교표 2종 생성
#
# **Proposed (final v1)** (연구자 결정 2026-10-29) =
#   baseline `CoST-W144-noscaleshift`
#     + trunk 분리 (trend trunk / seasonal trunk, 창은 **둘 다 144** — 창 분리 아님)
#     + seasonal 분기 **전체 동결** (trunk + 차수 sfd 가 무작위 초기값)
#     + **차수 정규화** (order tracking, N_rot=1, L_ORD=36)
#   = 방법 이름 `CoST-dual-T144-S144-frz`
#   = 실행 `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze`
#   A 실험 F1(ns) .9303 (baseline 대비 +0.0171, 22/3, Wilcoxon p<0.0001)
#   **현재까지의 최종판이고 추후 검증으로 개선해 나간다.**
#
# 이 스크립트가 하는 일
#   1. 표 (1) 모델별 성능 비교  -> table1_model_comparison_final_v1.csv
#   2. 표 (2) 제안 방법 ablation -> table2_ablation_study_final_v1.csv
#   3. final v1 산출물 사본 + README 를 outputs/07_proposed_final_v1/ 에 모은다
#
# 표의 수치는 **A 실험 5 fold x 5 seed = 25 회의 pooled 지표 평균과 표준편차**다.
# 모든 변형의 학습량이 동등하다 (data_passes 3.17).
#
# 실행 : python code/04_proposed/06_final_v1_tables.py
# =============================================================================
import os, shutil, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B = os.path.join(ROOT, 'outputs', '03_baselines')
OUT = os.path.join(ROOT, 'outputs', '07_proposed_final_v1')

FINAL_V1 = ('12_proposed_cost_dual_trunk (final v1)', 'CoST-dual-T144-S144-frz')
FINAL_V1_CMD = 'python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze'

# --- 표에 넣는 지표 (fold_metrics 의 pooled 지표 전부 + 진입 오차) ---
METRICS = [
    ('pooled_retention_rate', 'retention_rate'),                # 보존률 = recall(steady)
    ('pooled_contamination_rate', 'contamination_rate'),         # 오염률 = 1 - precision(steady)
    ('pooled_precision_steady', 'precision_steady'),
    ('pooled_recall_steady', 'recall_steady'),
    ('pooled_f1_steady', 'f1_steady'),
    ('pooled_auprc_steady', 'auprc_steady'),
    ('pooled_precision_nonsteady', 'precision_nonsteady'),
    ('pooled_recall_nonsteady', 'recall_nonsteady'),
    ('pooled_f1_nonsteady', 'f1_nonsteady'),
    ('pooled_auprc', 'auprc_nonsteady'),
    ('pooled_auroc', 'auroc'),
    ('runmae_entry_err_s', 'entry_mae_s'),                       # run별 진입 오차의 절댓값 평균
    ('runmean_entry_err_s', 'entry_bias_s'),                     # 부호 있는 평균 (음수 = 이르게 판정)
    ('n_runs_no_entry', 'n_runs_no_entry'),
]

# --- 표 (1) 모델별 성능 비교 (행 순서 = 연구자 지정) ---
TABLE1 = [
    ('Statistics-based', '01_stat_ssd', 'MovStd-Slope-fixed'),
    ('TS-TCC (2021)', '07_tstcc', 'TS-TCC'),
    ('TS2Vec (2022)', '02_ts2vec', 'TS2Vec'),
    ('CoST (2022)', '04_cost (ablation)', 'CoST-W144'),
    ('SimMTM (2023)', '08_simmtm', 'SimMTM'),
    ('SoftCLT (2024)', '05_softclt', 'SoftCLT-ref'),
    ('MOMENT (2024)', '03_moment', 'MOMENT'),
    ('Proposed method', *FINAL_V1),
]

# --- 표 (2) ablation. 'sep' trunk 분리 / 'frz' seasonal 동결 / 'ord' 차수 정규화 ---
# 동결은 **분리된 seasonal trunk 를 얼리는 것**이므로 분리가 전제다.
# 따라서 "동결만" / "동결 + 차수(분리 없이)" 조합은 구조적으로 존재할 수 없다.
TABLE2 = [
    ('CoST', '04_cost (ablation)', 'CoST-W144', '', '', ''),
    ('CoST (no scale/shift)', '06_cost_ablation (ablation)', 'CoST-W144-noscaleshift', '', '', ''),
    ('+ trunk separation', '12_proposed_cost_dual_trunk (ablation)', 'CoST-dual-T144-S144-noord', 'O', '', ''),
    ('+ order tracking', '09_proposed_cost_order_normalized_seasonal (ablation)', 'CoST-W144-order', '', '', 'O'),
    ('+ trunk separation + freeze', '12_proposed_cost_dual_trunk (ablation)',
     'CoST-dual-T144-S144-noord-frz', 'O', 'O', ''),
    ('+ trunk separation + order tracking', '12_proposed_cost_dual_trunk (ablation)',
     'CoST-dual-T144-S144', 'O', '', 'O'),
    ('Proposed method', *FINAL_V1, 'O', 'O', 'O'),
]
ABLATION_BASE = 'CoST-W144-noscaleshift'      # ablation 의 기준 (= baseline)

# --- 사본 폴더 구성용 매핑 (표 내용과 무관. 코드 파일 ↔ 결과 폴더를 1:1 로 보이게 한다) ---
# `04_cost_dual_trunk.py` 하나가 플래그로 **4행**을 만든다 (noorder / noorder freeze / 없음 / freeze).
# 그래서 표는 7행인데 코드는 4개다 — 같은 코드 경로에서 플래그만 달라지므로 ablation 비교가 깨끗하다.
ABL_SUFFIX = ' (ablation)'                    # 코드 파일 이름과 결과 폴더 이름에 공통으로 붙인다
ABL_CODE_MAP = [
    ('03_baselines/04_cost.py',
     [('04_cost (ablation)', 'CoST-W144')]),
    ('03_baselines/06_cost_ablation.py',
     [('06_cost_ablation (ablation)', 'CoST-W144-noscaleshift')]),
    ('04_proposed/01_cost_order.py',
     [('09_proposed_cost_order_normalized_seasonal (ablation)', 'CoST-W144-order')]),
    ('04_proposed/04_cost_dual_trunk.py',
     [('12_proposed_cost_dual_trunk (ablation)', 'CoST-dual-T144-S144-noord'),
      ('12_proposed_cost_dual_trunk (ablation)', 'CoST-dual-T144-S144-noord-frz'),
      ('12_proposed_cost_dual_trunk (ablation)', 'CoST-dual-T144-S144'),
      ('12_proposed_cost_dual_trunk (final v1)', 'CoST-dual-T144-S144-frz')]),
]


# --- 오래된 실행의 누락 열 복원 ---------------------------------------------
# `evaluation.py` 가 precision/recall/f1 열을 나중에 추가해서, 먼저 돌린 방법들
# (MovStd-Slope-fixed, TS2Vec, MOMENT 등)의 fold_metrics 에 그 열이 없다.
# **혼동행렬 항등식으로 정확히 복원된다** — 열이 있는 CoST-W144 에서 검증했고 오차가 1e-15
# (부동소수점 수준)이며, 복원한 precision·recall 로 다시 만든 F1 이 기록된 F1 과 일치한다.
#   N = n_scored, P_ns = n_nonsteady, P_st = N - P_ns
#   TP_st = retention x P_st,  precision_st = 1 - contamination
#   예측 steady 수 = TP_st / precision_st,  FP_st = 그 수 - TP_st
#   TP_ns = P_ns - FP_st,  precision_ns = TP_ns / (N - 예측steady수),  recall_ns = TP_ns / P_ns
# `auprc_steady` 는 점수가 필요해 복원할 수 없다 — 없는 방법은 공란으로 둔다.
DERIVABLE = ['pooled_precision_steady', 'pooled_recall_steady', 'pooled_f1_steady',
             'pooled_precision_nonsteady', 'pooled_recall_nonsteady', 'pooled_f1_nonsteady']


def derive_missing(d):
    # 역할 : 누락된 혼동행렬 기반 열을 복원한다. 반환 = 복원한 열 이름 목록
    need = [c for c in DERIVABLE if c not in d.columns]
    if not need:
        return []
    N, p_ns = d.n_scored, d.n_nonsteady
    p_st = N - p_ns
    prec_st = 1.0 - d.pooled_contamination_rate
    tp_st = d.pooled_retention_rate * p_st
    pred_st = tp_st / prec_st
    tp_ns = p_ns - (pred_st - tp_st)
    prec_ns, rec_ns = tp_ns / (N - pred_st), tp_ns / p_ns
    vals = {'pooled_precision_steady': prec_st, 'pooled_recall_steady': d.pooled_retention_rate,
            'pooled_f1_steady': 2 * prec_st * d.pooled_retention_rate / (prec_st + d.pooled_retention_rate),
            'pooled_precision_nonsteady': prec_ns, 'pooled_recall_nonsteady': rec_ns,
            'pooled_f1_nonsteady': 2 * prec_ns * rec_ns / (prec_ns + rec_ns)}
    for c in need:
        d[c] = vals[c]
    return [c.replace('pooled_', '') for c in need]


def load(stage, method):
    p = os.path.join(B, stage, f'{method}_fold_metrics.csv')
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def row(display, stage, method, extra=None, base=None):
    # 역할 : 한 방법의 25회 pooled 지표를 평균·표준편차로 요약한다.
    #        base 가 있으면 같은 fold x seed 로 짝지은 차이와 Wilcoxon p 를 F1(ns) 에 대해 덧붙인다.
    d = load(stage, method)
    if d is None:
        print(f'[skip] {display}: {stage}/{method}_fold_metrics.csv 없음', flush=True)
        return None
    derived = derive_missing(d)
    r = {'model': display, 'source_method': method, 'source_folder': stage, 'n_runs': len(d)}
    if extra:
        r.update(extra)
    miss = []
    for col, name in METRICS:
        if col in d.columns:
            r[f'{name}_mean'], r[f'{name}_std'] = d[col].mean(), d[col].std()
        else:
            r[f'{name}_mean'], r[f'{name}_std'] = np.nan, np.nan
            miss.append(name)
    r['derived_from_confusion_matrix'] = ';'.join(derived)
    r['unavailable_metrics'] = ';'.join(miss)
    if base is not None and method != ABLATION_BASE:
        j = d.merge(base, on=['fold', 'seed'], suffixes=('', '_b'))
        diff = j.pooled_f1_nonsteady - j.pooled_f1_nonsteady_b
        r['delta_f1_nonsteady_vs_base'] = diff.mean()
        r['win_loss_vs_base'] = f'{(diff > 0).sum()}/{(diff < 0).sum()}'
        r['wilcoxon_p_vs_base'] = wilcoxon(j.pooled_f1_nonsteady, j.pooled_f1_nonsteady_b).pvalue
    elif base is not None:
        r['delta_f1_nonsteady_vs_base'], r['win_loss_vs_base'], r['wilcoxon_p_vs_base'] = 0.0, '-', np.nan
    return r


os.makedirs(OUT, exist_ok=True)

# =============================================================================
# 표 (1)
# =============================================================================
t1 = pd.DataFrame([x for x in (row(*s) for s in TABLE1) if x is not None])
t1.to_csv(os.path.join(OUT, 'table1_model_comparison_final_v1.csv'), index=False)
print('[save] table1_model_comparison_final_v1.csv  (%d행)' % len(t1))

# =============================================================================
# 표 (2)
# =============================================================================
base = load('06_cost_ablation (ablation)', ABLATION_BASE)
rows2 = []
for disp, stage, method, sep, frz, ordr in TABLE2:
    rows2.append(row(disp, stage, method,
                     extra={'trunk_separation': sep, 'seasonal_frozen': frz, 'order_tracking': ordr},
                     base=base))
t2 = pd.DataFrame([x for x in rows2 if x is not None])
t2.to_csv(os.path.join(OUT, 'table2_ablation_study_final_v1.csv'), index=False)
print('[save] table2_ablation_study_final_v1.csv  (%d행)' % len(t2))

# =============================================================================
# 산출물 사본 (연구자 지시 2026-10-29)
#   `baselines_A/`            : **Proposed (final v1) 결과물만.** 비교군(표 1)의 다른 7종은
#                               정본이 `outputs/03_baselines/<단계>/` 에 있으므로 여기 두지 않는다
#   `baselines_A_ablations/`  : **코드 파일별 하위 폴더**로 나눈다. 폴더 이름을
#                               `code_ablations/` 의 파일 이름에서 `.py` 만 뗀 것으로 두어
#                               코드 ↔ 결과가 눈으로 1:1 로 짝지어진다
#                                 `04_cost_dual_trunk (ablation).py`
#                                 <-> `04_cost_dual_trunk (ablation)/`
#   여기 결과는 **모두 A 실험**이고, 방법 이름이 파일 접두사라 한 폴더에서 섞이지 않는다.
#   `_progress_*` 는 제외한다 — 실행 중 append 하는 복구용 로그이고 run/fold metrics 와 중복이다.
# =============================================================================
A_DIR = os.path.join(OUT, 'baselines_A')              # Proposed (final v1) 하나
ABL_DIR = os.path.join(OUT, 'baselines_A_ablations')  # 표 (2) ablation, 코드별 하위 폴더


def copy_method(stage, method, dst_dir):
    src_dir = os.path.join(B, stage)
    if not os.path.isdir(src_dir):
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src_dir)):
        if not f.startswith(method + '_') or '_progress_' in f:
            continue
        s = os.path.join(src_dir, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst_dir, f))
            n += 1
    return n


# 사본 폴더는 매 실행마다 비우고 다시 채운다 (파생물이라 안전하고, 구성이 바뀌어도 잔여물이 없다)
for d in (A_DIR, ABL_DIR):
    if os.path.isdir(d):
        shutil.rmtree(d)

# --- baselines_A/ : Proposed (final v1) 하나만 ---
n = copy_method(*FINAL_V1, A_DIR)
print('[copy] baselines_A/ %-34s %2d개  <- ★ Proposed (final v1) 전용' % (FINAL_V1[1], n))

# --- baselines_A_ablations/<코드 이름 (ablation)>/ : 그 코드가 만든 행의 결과물 ---
total = 0
for code_rel, methods in ABL_CODE_MAP:
    sub = os.path.splitext(os.path.basename(code_rel))[0] + ABL_SUFFIX
    for stage, method in methods:
        n = copy_method(stage, method, os.path.join(ABL_DIR, sub))
        total += n
        tag = '  <- ★ Proposed (final v1)' if method == FINAL_V1[1] else ''
        print('[copy] baselines_A_ablations/%-30s %-32s %2d개%s%s'
              % (sub + '/', method, n, '  <- 없음!' if n == 0 else '', tag))
print('[copy] baselines_A_ablations/ 합계 %d개 파일, 폴더 %d개'
      % (total, len(ABL_CODE_MAP)))

# =============================================================================
# 코드 사본 — final v1 을 재현하는 데 필요한 파이썬 코드 (연구자 지시 2026-10-29)
#   외부 저장소(code/03_baselines/external/CoST)는 추적 대상이 아니라 복사하지 않는다.
#   쓴 커밋 hash 는 각 방법의 `_config.csv` 의 'repo'/'commit' 열에 기록돼 있다.
# =============================================================================
#   `code/`            : final v1 본체 + 공통 평가 후단 + 표 생성기
#   `code_ablations/`   : ablation 7행을 만든 어댑터. 파일 이름에 ` (ablation)` 을 붙인다
#                         (연구자 지시 2026-10-29). **참조용 사본이라 그 자리에서 실행하지 않는다** —
#                         괄호·공백이 있어 import 되지 않으므로 실행은 정본 `code/` 에서 한다.
CODE_FINAL = [
    ('04_proposed/04_cost_dual_trunk.py', '★ final v1 본체 — `nrot=1 freeze` 조합이 final v1'),
    ('03_baselines/evaluation.py', '공통 평가 후단 (GMM·채점·그림). 라벨을 읽는 유일한 지점'),
    ('03_baselines/splits.py', '분할 해석 (A/C, ambiguous 모드)'),
    ('04_proposed/06_final_v1_tables.py', '표 2종과 이 사본들을 만든 스크립트'),
]
CODE_ABL = [
    ('03_baselines/04_cost.py', '`CoST` 행'),
    ('03_baselines/06_cost_ablation.py', '`CoST (no scale/shift)` 행 (= ablation 기준)'),
    ('04_proposed/01_cost_order.py', '`+ order tracking` 행'),
    ('04_proposed/04_cost_dual_trunk.py',
     '`+ trunk separation` / `+ trunk separation + freeze` / `+ trunk separation + order tracking` / `Proposed method` 행'),
]
ABL_CMDS = [
    ('CoST', 'python code/03_baselines/04_cost.py'),
    ('CoST (no scale/shift)', 'python code/03_baselines/06_cost_ablation.py'),
    ('+ trunk separation', 'python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 noorder'),
    ('+ order tracking', 'python code/04_proposed/01_cost_order.py tension nrot=1'),
    ('+ trunk separation + freeze',
     'python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 noorder freeze'),
    ('+ trunk separation + order tracking',
     'python code/04_proposed/04_cost_dual_trunk.py tension nrot=1'),
    ('Proposed method', FINAL_V1_CMD),
]


def copy_code(specs, dst_dir, title, intro, suffix=''):
    os.makedirs(dst_dir, exist_ok=True)
    lines = ['# %s' % title, '', intro, '',
             '**정본은 저장소의 `code/` 아래이고 여기는 사본**이므로 수정은 정본에서 한다.', '',
             '| 파일 | 역할 |', '|---|---|']
    for rel, why in specs:
        src = os.path.join(ROOT, 'code', rel)
        if not os.path.exists(src):
            print('[skip] 코드 없음: %s' % rel)
            continue
        stem, ext = os.path.splitext(os.path.basename(rel))
        base = f'{stem}{suffix}{ext}'
        shutil.copy2(src, os.path.join(dst_dir, base))
        lines.append('| `%s` | %s (정본 `code/%s`) |' % (base, why, rel))
        print('[copy] %-22s %-44s <- code/%s' % (os.path.basename(dst_dir) + '/', base, rel))
    return lines


lines = copy_code(CODE_FINAL, os.path.join(OUT, 'code'),
                  '`07_proposed_final_v1/code/` — Proposed (final v1) 코드 사본',
                  'final v1 과 표 2종을 재현하는 데 필요한 파이썬 코드다.')
lines += ['', '**외부 저장소(`code/03_baselines/external/CoST`)는 복사하지 않았다** — 추적 대상이 아니다.',
          '쓴 커밋 hash 는 각 방법의 `_config.csv` 의 `repo` / `commit` 열에 있다.', '',
          '### final v1 재현', '', '```', FINAL_V1_CMD,
          'python code/04_proposed/06_final_v1_tables.py', '```']
with open(os.path.join(OUT, 'code', 'README.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('[save] code/README.md')

lines = copy_code(CODE_ABL, os.path.join(OUT, 'code_ablations'),
                  '`07_proposed_final_v1/code_ablations/` — ablation (표 2) 코드 사본',
                  '표 (2) 의 7행을 만든 어댑터다. 파일 이름에 ` (ablation)` 을 붙였다.\n'
                  '괄호·공백 때문에 python import 가 되지 않으니 **실행은 정본 `code/` 에서 한다.**',
                  suffix=ABL_SUFFIX)
lines += ['', '### 행별 재현 명령', '', '| 표 (2) 의 행 | 명령 |', '|---|---|']
for disp, cmd in ABL_CMDS:
    lines.append('| %s | `%s` |' % (disp, cmd))
lines += ['', '같은 명령을 다시 돌리면 `{방법}_fold_metrics.csv` 가 있으면 **건너뛴다**',
          '(재실행하려면 먼저 지운다).']
with open(os.path.join(OUT, 'code_ablations', 'README.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('[save] code_ablations/README.md')

# =============================================================================
# 화면 출력 (검토용)
# =============================================================================
SHOW = ['retention_rate_mean', 'contamination_rate_mean', 'precision_nonsteady_mean',
        'recall_nonsteady_mean', 'f1_nonsteady_mean', 'auprc_nonsteady_mean', 'auroc_mean',
        'entry_mae_s_mean']
HDR = ['보존률', '오염률', 'Prec(ns)', 'Rec(ns)', 'F1(ns)', 'AUPRC(ns)', 'AUROC', '진입MAE']
for name, t in [('표 (1) 모델별 성능 비교', t1), ('표 (2) 제안 방법 ablation', t2)]:
    print()
    print('=== %s ===' % name)
    print('%-38s' % '' + ''.join('%10s' % h for h in HDR))
    for _, r in t.iterrows():
        print('%-38s' % r.model + ''.join('%10.4f' % r[c] if pd.notna(r[c]) else '%10s' % '-'
                                          for c in SHOW))
print()
print('[done] -> %s' % OUT)
