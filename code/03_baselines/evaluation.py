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
import splits                     # fold 묶음 이름 (A 는 묶음 하나, C 는 cv_T* / cv_S* / final_test)

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

# 설정값의 영문 이름 — heatmap 의 축 이름(그림 전체)과 subplot 제목(고정된 설정값, 이름 + 단위)에 쓴다
_AXIS_LABEL = {TS: 'Film tension setting (kg)', RS: 'ReWinder tension setting (kg)',
               VS: 'Web speed setting (mm/s)'}
_SETTING_LABEL = {TS: ('film tension', 'kg'), RS: ('rewinder tension', 'kg'), VS: ('web speed', 'mm/s')}

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

# 방법별 고정 스타일 (color, marker, linestyle). 방법마다 고유 색 + 고유 marker 다.
#
# 색 선정: Okabe-Ito 8색(색각이상 대응 표준) 중 흰 종이 대비가 1.29 밖에 안 되는 yellow 를 빼고,
# 나머지 5색은 OKLab 거리 + Machado-Oliveira-Fernandes(2009) 색각이상 시뮬레이션으로 탐색해
# '전체 66쌍의 min(protan, deutan) 거리 최솟값' 을 최대화했다. 측정값(전체 쌍, OKLab dE x100):
#   색각이상 7.6 / 정상시각 15.6 / tritan 8.5,  정상시각 15 미달 쌍 0개
# 색각이상 7.6 은 '6-8 구간 = 보조 식별자 필수' 에 해당하므로 marker 와 linestyle 을 함께 고정한다.
# 명도를 색상과 별개의 식별 채널로 쓰려고(명도 차이는 색각이상에 불변) 어두운 색 4개를 포함했고,
# 방법 변형 2개짜리(R-test, MovStd-Slope)는 같은 색상의 명암 쌍으로 묶었다.
METHOD_STYLE = {
    'R-test':             ('#56b4e9', 'v', '-'),    # 통계 SSD — 원 샘플링
    'R-test-decimated':   ('#0072b2', '^', '--'),   # 통계 SSD — 다운샘플링 (R-test 와 같은 색상 계열)
    'MovStd-Slope-fixed': ('#009e73', 's', '-'),    # 통계 SSD — 고정 창
    'MovStd-Slope-speed': ('#005a00', 'D', '--'),   # 통계 SSD — 속도별 창 (위와 같은 색상 계열)
    'TS2Vec':             ('#e69f00', 'p', '-'),    # 대조학습
    'SoftCLT':            ('#5a1b00', 'P', '--'),   # 대조학습 (TS2Vec 파생이라 같은 색상 계열)
    'TNC':                ('#cc79a7', 'X', '-'),    # 대조학습
    'TS-TCC':             ('#873f51', '*', '--'),   # 대조학습
    'CoST':               ('#d55e00', '<', '-.'),   # 대조학습
    'SimMTM':             ('#091b7e', '>', '-.'),   # 생성/복원
    'MOMENT':             ('#662aa2', 'h', ':'),    # 파운데이션 모델
    'Proposed':           ('#000000', 'o', '-'),    # 제안 방법
}

# =============================================================================
# 함수
# =============================================================================
_LABEL_CACHE = None
_LABEL_CACHE_AMB = None           # ambiguous 포함 모드용 (2026-10-28 추가). 모드마다 따로 캐시한다


def _nearest_label(v):
    # 역할 : ambiguous(2) 를 **시간상 가장 가까운 비-ambiguous 시점의 라벨**로 바꾼다
    #        (ambiguous 포함 모드 전용. 2026-10-28 연구자 결정).
    #        이 규칙 하나가 "transient 에 가까우면 transient, steady 에 가까우면 steady" 를 모두 덮는다:
    #          0 ... [2 2 2 2 2 2] ... 1   -> 앞 절반 0 / 뒤 절반 1  (구간 179개, 정확히 1/2 분할)
    #          run 시작 [2 2 2 2] 0 ...    -> 전부 0   (구간 84개. 한쪽에만 이웃이 있으면 그 라벨)
    #          run 시작 [2 2 2 2] 1 ...    -> 전부 1   (구간 20개. transient 가 아예 없는 run)
    #          그 밖 (0-0 / 1-1 / 1-0 이웃) -> 같은 규칙으로 자동 처리  (구간 5개)
    #        **동점 처리** — 길이가 홀수인 구간(288개 중 133개)은 한가운데 1개 시점이 양쪽에서
    #        거리가 같다. 이때는 **transient(0)** 로 보낸다. 오염률(= steady 로 판정한 시점 중 실제
    #        non-steady)이 본 연구의 최우선 지표라, 모호한 시점을 steady 쪽에 얹지 않는 쪽을 택한다.
    # 입력 : v (n,) k 순서로 정렬된 한 run 의 라벨 (0/1/2)
    # 출력 : (n,) 0/1 만 남은 라벨. 입력에 2 가 없으면 그대로다
    idx = np.flatnonzero(v != 2)
    if len(idx) == 0 or len(idx) == len(v):
        return v.copy()                                  # 전부 ambiguous 이거나 ambiguous 가 없는 run
    pos = np.arange(len(v))
    j = np.searchsorted(idx, pos)                        # pos 이상인 첫 비-ambiguous 의 idx 내 위치
    jl, jr = np.clip(j - 1, 0, len(idx) - 1), np.clip(j, 0, len(idx) - 1)
    dl = np.where(j == 0, np.inf, pos - idx[jl])         # 왼쪽 이웃까지 거리 (없으면 inf)
    dr = np.where(j >= len(idx), np.inf, idx[jr] - pos)  # 오른쪽 이웃까지 거리 (없으면 inf)
    ll, rl = v[idx[jl]], v[idx[jr]]
    out = np.where(dl < dr, ll, np.where(dr < dl, rl, np.minimum(ll, rl)))   # 동점 -> 작은 쪽 = 0
    return out.astype(v.dtype)


def _labels():
    # 역할 : `Trigger 4` 라벨 표를 읽어 캐시한다. 라벨을 읽는 유일한 지점 (라벨 누출 차단)
    #        ambiguous 포함 모드(splits.AMB_INCLUDED)에서는 2 를 _nearest_label 로 흡수한 표를 준다.
    #        **읽는 지점이 여기 하나뿐이므로, 모드 전환도 여기 한 줄로 끝난다** — 채점·그림·진단이
    #        전부 이 함수를 거치므로 라벨 해석이 갈라지지 않는다.
    # 입력 : 없음 (MERGED_PATH 와 splits.AMB_INCLUDED 를 읽는다)
    # 출력 : DataFrame [run_id, k, `Trigger 4`]
    global _LABEL_CACHE, _LABEL_CACHE_AMB
    if not splits.AMB_INCLUDED:
        if _LABEL_CACHE is None:
            _LABEL_CACHE = pd.read_csv(MERGED_PATH, usecols=['run_id', 'k', LABEL_COL])
        return _LABEL_CACHE
    if _LABEL_CACHE_AMB is None:
        d = pd.read_csv(MERGED_PATH, usecols=['run_id', 'k', LABEL_COL]).sort_values(['run_id', 'k'])
        d[LABEL_COL] = np.concatenate([_nearest_label(g.to_numpy())
                                       for _, g in d.groupby('run_id', sort=False)[LABEL_COL]])
        assert not (d[LABEL_COL] == 2).any(), 'ambiguous 흡수 후에도 라벨 2 가 남았다'
        _LABEL_CACHE_AMB = d.reset_index(drop=True)
    return _LABEL_CACHE_AMB


def label_mode():
    # 역할 : 지금 쓰고 있는 라벨 해석을 한 줄 문자열로 돌려준다 (config csv 기록용).
    # 입력 : 없음
    # 출력 : 설명 문자열
    return ('ambiguous(2) 를 시간상 가장 가까운 라벨로 흡수해 2클래스로 채점 (동점은 transient). '
            '재학습 없이 저장된 체크포인트로 추론만 함' if splits.AMB_INCLUDED
            else 'ambiguous(2) 를 채점에서 제외 (기본값)')


def label_series(df_rk):
    # 역할 : [run_id, k] 짝에 대응하는 **채점용 라벨**을 돌려준다 (현재 라벨 모드 반영).
    #        그림·진단 스크립트가 merged_data.csv 에서 `Trigger 4` 를 직접 읽지 않게 하는 통로다
    #        — 라벨 해석의 단일 출처를 유지하기 위함이다 (헤더의 라벨 누출 차단 구조).
    # 입력 : df_rk  [run_id, k] 를 가진 DataFrame (순서 보존)
    # 출력 : (len(df_rk),) 라벨 배열
    return df_rk[['run_id', 'k']].merge(_labels(), on=['run_id', 'k'], how='left')[LABEL_COL].to_numpy()


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
    #        (ambiguous 포함 모드에서는 _labels() 가 2 를 미리 0/1 로 흡수해 주므로, 아래의
    #         `!= 2` 필터가 저절로 무해해지고 전 시점이 채점 대상이 된다 — 코드는 그대로다)
    #        지표는 pooled(fold 전체 시점을 한 덩어리로) 와 run 평균(run별 계산 후 평균) 두 가지로 모두 기록한다.
    #
    #        positive 클래스 규약 — 지표마다 다르므로 이름에 클래스를 명시한다:
    #          오염률 = 1 - precision_steady   (steady 로 판정한 시점 중 실제 non-steady 비율)
    #          보존률 = recall_steady          (실제 steady 시점 중 steady 로 판정한 비율)
    #          precision_steady / recall_steady       : steady = positive
    #          precision_nonsteady / recall_nonsteady : non-steady = positive
    #          f1_steady, auprc_steady                : steady = positive
    #          f1_nonsteady, auprc                    : non-steady = positive
    #          auroc                                  : **클래스 대칭** — positive 를 바꾸고 점수를
    #                                                   뒤집으면 같은 값이므로 하나만 기록한다
    #        AUPRC 는 비대칭이다. 기준선이 positive 클래스의 유병률이라
    #        (ns 0.0813 / st 0.9187) 두 값을 같은 잣대로 비교하면 안 된다.
    #        오염률·보존률은 도메인 이름이라 유지하고, 같은 양을 precision/recall 이름으로도 함께 남긴다.
    #        논문 표에는 중복해서 싣지 않는다.
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
            # precision/recall 을 두 클래스 모두에 대해 명시적으로 기록한다 (어느 쪽이 positive 인지 혼동 방지).
            # precision_steady = 1 - contamination_rate, recall_steady = retention_rate 로 같은 양이다.
            'precision_steady': float((p_st & ~y_ns).sum() / p_st.sum()) if p_st.sum() else np.nan,
            'recall_steady': float((p_st & ~y_ns).sum() / (~y_ns).sum()) if (~y_ns).sum() else np.nan,
            'f1_steady': f1_score(~y_ns, p_st, zero_division=0) if (~y_ns).any() else np.nan,
            'precision_nonsteady': float((~p_st & y_ns).sum() / (~p_st).sum()) if (~p_st).sum() else np.nan,
            'recall_nonsteady': float((~p_st & y_ns).sum() / y_ns.sum()) if y_ns.sum() else np.nan,
            'f1_nonsteady': f1_score(y_ns, ~p_st, zero_division=0) if y_ns.any() else np.nan,
            # AUPRC 는 **비대칭**이다 — 기준선이 positive 클래스의 유병률이라 두 클래스 값이 전혀 다르다
            # (ns 기준선 0.081, st 기준선 0.919). 그래서 양쪽을 모두 기록한다.
            # AUROC 는 대칭이므로(positive 를 바꾸고 점수를 뒤집으면 같은 값) 하나만 기록한다.
            'auprc_steady': average_precision_score(~y_ns, 1.0 - s['score'])
            if (~y_ns).any() and y_ns.any() else np.nan,
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
        'pooled_precision_steady': float((p_st & ~y_ns).sum() / p_st.sum()) if p_st.sum() else np.nan,
        'pooled_recall_steady': float((p_st & ~y_ns).sum() / (~y_ns).sum()) if (~y_ns).sum() else np.nan,
        'pooled_f1_steady': f1_score(~y_ns, p_st, zero_division=0),
        'pooled_precision_nonsteady': float((~p_st & y_ns).sum() / (~p_st).sum()) if (~p_st).sum() else np.nan,
        'pooled_recall_nonsteady': float((~p_st & y_ns).sum() / y_ns.sum()) if y_ns.sum() else np.nan,
        'pooled_f1_nonsteady': f1_score(y_ns, ~p_st, zero_division=0),
        'pooled_auprc_steady': average_precision_score(~y_ns, 1.0 - sd['score']) if y_ns.any() else np.nan,
        'pooled_auroc': roc_auc_score(y_ns, sd['score']) if 0 < y_ns.sum() < len(y_ns) else np.nan,
        'pooled_auprc': average_precision_score(y_ns, sd['score']) if y_ns.any() else np.nan,
    }
    for c in ['contamination_rate', 'retention_rate',
              'precision_steady', 'recall_steady', 'f1_steady', 'auprc_steady',
              'precision_nonsteady', 'recall_nonsteady', 'f1_nonsteady', 'auroc', 'auprc']:
        fold_row[f'runmean_{c}'] = run_metrics[c].mean()              # 정의되지 않은 run 은 자동 제외 (NaN)
        fold_row[f'runstd_{c}'] = run_metrics[c].std()
        fold_row[f'n_runs_{c}'] = int(run_metrics[c].notna().sum())
    fold_row['runmean_entry_err_s'] = run_metrics.entry_err_s.mean()  # 부호 포함 (bias: + 는 늦게 진입)
    fold_row['runmae_entry_err_s'] = run_metrics.entry_err_s.abs().mean()
    fold_row['runstd_entry_err_s'] = run_metrics.entry_err_s.std()
    fold_row['n_runs_no_entry'] = int((run_metrics.entry_pred_k < 0).sum())   # steady 를 전혀 판정하지 못한 run
    return run_metrics, pd.DataFrame([fold_row])


def derive_precision_recall(metrics, prefix='pooled_'):
    # 역할 : precision/recall 컬럼이 없는 과거 결과(2026-10-23 이전 실행)에 그 값을 채워 넣는다.
    #        혼동행렬 4칸이 저장된 컬럼만으로 정확히 복원된다 — 무작위 혼동행렬 10만 건으로 검증했다.
    #          n_steady = n_scored - n_nonsteady,  TP = 보존률 x n_steady,  FN = n_steady - TP
    #          f1 = 2·TN / (2·TN + FN + FP) 를 FP 에 대해 풀면
    #          FP = [2·n_nonsteady·(1 - f1) - f1·FN] / (2 - f1),   TN = n_nonsteady - FP
    #        이미 컬럼이 있으면(새 실행) 그대로 돌려준다.
    # 입력 : metrics  fold_metrics (prefix='pooled_') 또는 run_metrics (prefix='')
    #        prefix   지표 컬럼 이름 앞부분
    # 출력 : {prefix}precision_steady / recall_steady / precision_nonsteady / recall_nonsteady 가 채워진 사본
    d = metrics.copy()
    if f'{prefix}f1_steady' in d.columns:                               # 이미 다 있으면 그대로
        return d
    ret, f1 = d[f'{prefix}retention_rate'], d[f'{prefix}f1_nonsteady']
    n_s = d.n_scored - d.n_nonsteady
    tp = ret * n_s
    fn = n_s - tp
    fp = (2 * d.n_nonsteady * (1 - f1) - f1 * fn) / (2 - f1)
    tn = d.n_nonsteady - fp
    for col, val in [(f'{prefix}precision_steady', 1 - d[f'{prefix}contamination_rate']),
                     (f'{prefix}recall_steady', ret),
                     (f'{prefix}f1_steady', 2 * tp / (2 * tp + fp + fn)),
                     (f'{prefix}precision_nonsteady', tn / (tn + fn)),
                     (f'{prefix}recall_nonsteady', tn / d.n_nonsteady)]:
        if col not in d.columns:                                        # 저장된 값이 있으면 보존
            d[col] = val
    # auprc_steady 는 혼동행렬만으로 복원할 수 없다 (점수 분포가 필요하다).
    # 과거 결과에는 없으며, 채우려면 재실행해야 한다.
    return d


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
    # 라벨 해석은 **항상** 기록한다 (어댑터가 넘기지 않아도). 같은 방법·같은 체크포인트라도
    # ambiguous 처리가 다르면 다른 결과이므로, config 만 보고 구분할 수 있어야 한다
    config = {**config, 'LABEL_MODE': label_mode()}
    pd.DataFrame({'name': list(config), 'value': [str(v) for v in config.values()]}).to_csv(paths['config'], index=False)
    for k, p in paths.items():
        print(f'[save] {k:7s} -> {p}')


PAGE_WIDTH = 6.5                # 논문 지면 폭 (in). 그림 폭을 여기에 맞춘다. 임의의 값
HEATMAP_NCOLS = 3               # heatmap subplot 을 한 줄에 몇 개 놓을지. 임의의 값

# heatmap 색 범위 — **방법·실험을 가로질러 고정한다** (사용자 결정, 2026-10-28)
#   이전에는 그림마다 그 방법의 min~max 로 범위를 잡아서, 같은 색이 방법마다 다른 값을 뜻했다.
#   (예: 보존률 0.95 가 어떤 방법에서는 가장 어두운 색, 다른 방법에서는 중간 색)
#   범위를 고정하면 **A / C / ambiguous 포함 A / ambiguous 포함 C 의 모든 그림에서 색이 같은 뜻**이 된다.
#   범위를 벗어나는 셀은 색이 포화되지만 **셀 안에 수치를 직접 찍으므로 값은 잃지 않는다.**
#   포화가 일어나는 것은 colorbar 양끝 화살표(extend)로 표시한다.
#
#   범위 선정 (사용자 결정):
#     보존률  0.0-1.0  비율의 전 범위다 — **포화가 없다.** 잘라내는 값이 하나도 없다
#     오염률  0.0-0.3  셀의 **99%** 를 담는다 (중앙값 0.000, 90분위 0.04-0.05).
#                      0-1 로 두면 거의 모든 칸이 같은 색이 되어 방법 간 비교가 불가능해진다.
#                      넘는 1% 는 R-test 2종과 MOMENT 에 몰려 있다 (A 40/4,519 · C 58/5,799)
HEATMAP_RANGE = {'contamination_rate': (0.0, 0.3),      # 낮을수록 좋다
                 'retention_rate':     (0.0, 1.0)}      # 높을수록 좋다
# 등록되지 않은 지표는 예전처럼 그 그림의 데이터 범위를 쓴다 (고정값을 임의로 만들지 않는다)


def append_progress(out_dir, prefix, name, frame):
    # 역할 : fold×seed 가 하나 끝날 때마다 중간 결과를 이어 붙여 저장한다 (덮어쓰지 않고 append).
    #        장시간 실행이 중간에 끊겨도 여기까지의 결과가 남는다. 최종 결과는 save_results 가 따로 쓴다.
    # 입력 : out_dir 결과 폴더 / prefix 파일 이름 앞부분 / name 중간 파일 종류 ('progress', 'loss' 등)
    #        frame 이어 붙일 DataFrame
    # 출력 : 없음 (csv 1개에 append)
    os.makedirs(out_dir, exist_ok=True)
    p = f'{out_dir}/{prefix}_{name}.csv'
    frame.to_csv(p, mode='a', header=not os.path.exists(p), index=False)


def plot_loss_curves(loss_df, out_dir, prefix, ylabel='Training loss', x_col='iter', xlabel=None):
    # 역할 : fold×seed 별 자기지도 학습 손실 곡선을 그린다 (fold 하나 = subplot, seed 는 옅은 선 + 평균 진한 선).
    #        학습이 제대로 수렴했는지, seed 간 흔들림이 얼마나 큰지 확인용.
    # 입력 : loss_df [fold, seed, x_col, loss] / out_dir 저장 폴더 / prefix 파일 이름 앞부분 / ylabel y축 이름
    #        x_col 가로축 컬럼 이름. **기본값은 'iter' 다** — 'epoch' 은 저장소마다 iteration 수가 달라
    #        (TS2Vec·SoftCLT 21, CoST 는 창 길이에 따라 42~168) 방법 간·설정 간 비교가 불가능하다.
    #        iteration 단위 손실은 저장소가 제공하는 after_iter_callback 훅으로 받는다.
    #        epoch 단위 손실만 얻을 수 있는 저장소(TS-TCC, SimMTM)는 x_col='epoch' 로 쓰고 캡션에 명시한다.
    #        xlabel 가로축 이름 (None 이면 x_col 로 자동)
    # 출력 : 없음 (PNG 300 dpi + PDF 저장)
    assert x_col in loss_df.columns, f'loss_df 에 {x_col} 컬럼이 없다: {list(loss_df.columns)}'
    folds = sorted(loss_df.fold.unique())
    ncols = min(HEATMAP_NCOLS, len(folds))
    nrows = int(np.ceil(len(folds) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout='constrained', sharex=True, sharey=True,
                             figsize=(PAGE_WIDTH, nrows * 1.7 + 0.5))
    for ax, f in zip(axes.ravel(), folds):
        g = loss_df[loss_df.fold == f]
        for _, s in g.groupby('seed'):
            ax.plot(s[x_col], s.loss, color='#2a78d6', linewidth=1.0, alpha=0.35)
        m = g.groupby(x_col).loss.mean()
        ax.plot(m.index, m.to_numpy(), color='#2a78d6', linewidth=2.0)       # seed 평균
        ax.set_title(f'Fold {f}' if str(f).isdigit() else f'{f}')    # A 는 fold 번호, C 는 fold 이름
    for ax in axes.ravel()[len(folds):]:
        ax.axis('off')
    fig.supxlabel(xlabel or ('Iteration' if x_col == 'iter' else x_col.capitalize()), fontsize=FONT_SIZE)
    fig.supylabel(ylabel, fontsize=FONT_SIZE)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        p = f'{out_dir}/{prefix}_loss.{ext}'
        assert not os.path.exists(p), f'이미 그림 파일이 있다 (덮어쓰기 금지): {p}'
        fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f'[save] loss curve -> {out_dir}/{prefix}_loss.png / .pdf')


def _heatmap_panels(m, metric):
    # 역할 : fold 별 검증 조건 구성을 보고 heatmap panel(= subplot 하나)의 축과 제목을 정한다.
    #        A 실험은 fold 안에서 장력이 고정이라 (ReWinder x 속도) 격자가 유일했지만,
    #        C 실험은 fold 마다 고정되는 설정값이 다르다 — 그래서 '고정된 설정값은 제목으로 빼고
    #        나머지 두 축으로 그린다' 는 규칙 하나로 통일한다. ReWinder 는 항상 행으로 둔다.
    #          cv_T*      : 장력 고정  -> (ReWinder x 속도),  제목 = 장력
    #          cv_S*      : 속도 고정  -> (ReWinder x 장력),  제목 = 속도
    #          final_test : 고정 없음  -> 속도로 쪼개 panel 3개, 각 (ReWinder x 장력), 제목 = 속도
    # 입력 : m run별 지표를 [fold, TS, RS, VS] 로 평균낸 표 / metric 그릴 지표 컬럼명
    # 출력 : panel dict 리스트 [{group, fold, head, sub, col_col, data}]. group 이 같은 panel 들이 그림 하나가 된다.
    #        제목은 head(fold 이름) + sub(고정된 설정값)로 나눠 두고, 그릴 때 합친다 — 묶음 안에 fold 가
    #        하나뿐이면(final_test) head 가 모든 subplot 에 반복돼 제목이 겹치므로 그때는 sub 만 쓴다
    panels = []
    for f in m.fold.unique():
        g = m[m.fold == f]
        assert g[RS].nunique() > 1, f'fold {f}: ReWinder 가 고정이다 — 행 축으로 쓸 수 없다'
        fixed = [c for c in [TS, VS] if g[c].nunique() == 1]
        head = f'Fold {f}' if str(f).isdigit() else f'{f}'            # A 는 fold 번호, C 는 fold 이름
        base = {'group': splits.fold_group(f), 'fold': f, 'head': head}
        if len(fixed) == 1:                                           # 장력 또는 속도 하나가 고정 -> panel 1개
            c = fixed[0]
            nm, unit = _SETTING_LABEL[c]
            panels.append({**base, 'col_col': VS if c == TS else TS,
                           'sub': f'{nm} {int(g[c].iloc[0])} {unit}', 'data': g})
        else:                                                         # 고정 없음 (final_test) -> 속도별로 쪼갠다
            nm, unit = _SETTING_LABEL[VS]
            for v in sorted(g[VS].unique()):
                panels.append({**base, 'col_col': TS,
                               'sub': f'{nm} {int(v)} {unit}', 'data': g[g[VS] == v]})
    return panels


def plot_condition_heatmap(run_metrics, metric, out_dir, prefix, label):
    # 역할 : 검증 fold 안의 운전조건 격자에 run별 성능을 깔아 heatmap 으로 그린다 (panel 하나 = subplot 하나).
    #        seed 반복이 있으면 run별로 평균한 값을 쓴다. 축 결정은 _heatmap_panels 가 한다.
    #        C 실험은 fold 묶음(cv_T* / cv_S* / final_test)이 묻는 질문이 달라 그림 파일을 분리 저장한다.
    #        색 범위(vmin, vmax)는 방법 전체에서 하나로 공유한다 (panel 간 비교 가능하게).
    # 입력 : run_metrics save 대상 run별 지표 / metric 그릴 지표 컬럼명 / out_dir 저장 폴더 / prefix 파일 이름 앞부분
    #        label colorbar 에 쓸 영문 지표 이름(단위 포함)
    # 출력 : 없음 (PNG 300 dpi + PDF 저장)
    m = run_metrics.groupby(['fold', TS, RS, VS], as_index=False)[metric].mean()
    if metric in HEATMAP_RANGE:                                       # 방법·실험을 가로질러 고정 (위 주석)
        vmin, vmax = HEATMAP_RANGE[metric]
        below = int((m[metric] < vmin).sum())
        above = int((m[metric] > vmax).sum())
        cbar_extend = ('both' if below and above else
                       'min' if below else 'max' if above else 'neither')
    else:                                                             # 등록되지 않은 지표는 예전 동작
        vmin, vmax = np.nanmin(m[metric]), np.nanmax(m[metric])
        below = above = 0
        cbar_extend = 'neither'
    panels = _heatmap_panels(m, metric)
    for group in dict.fromkeys(p['group'] for p in panels):           # 묶음별로 그림 하나 (등장 순서 유지)
        ps = [p for p in panels if p['group'] == group]
        col_col = ps[0]['col_col']
        assert all(p['col_col'] == col_col for p in ps), f'묶음 {group!r} 안에서 heatmap 축이 다르다'
        ncols = min(HEATMAP_NCOLS, len(ps))
        nrows = int(np.ceil(len(ps) / ncols))
        n_row_cells = m[RS].nunique()
        n_col_cells = max(p['data'][col_col].nunique() for p in ps)
        cell = (PAGE_WIDTH / ncols - 0.55) / n_col_cells              # 셀 한 칸의 폭 (in) — 축 라벨 자리를 뺀 나머지
        one_fold = len({p['fold'] for p in ps}) == 1                  # 묶음에 fold 가 하나면 제목에서 fold 이름을 뺀다
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout='constrained',
                                 figsize=(PAGE_WIDTH, nrows * (cell * n_row_cells + 0.95) + 0.3))
        for ax, p in zip(axes.ravel(), ps):
            grid = p['data'].pivot(index=RS, columns=col_col, values=metric)
            im = ax.imshow(grid.to_numpy(), cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_xticks(range(grid.shape[1]), [f'{int(v)}' for v in grid.columns], fontsize=FONT_SIZE - 2)
            ax.set_yticks(range(grid.shape[0]), [f'{int(v)}' for v in grid.index], fontsize=FONT_SIZE - 2)
            ax.set_title(p['sub'] if one_fold else f'{p["head"]}: {p["sub"]}')
            ax.grid(False)
            for i in range(grid.shape[0]):                            # 셀 값 직접 표기 (대비 보완 · 수치 확인용)
                for j in range(grid.shape[1]):
                    v = grid.to_numpy()[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f'{v:.2f}'.lstrip('0'), ha='center', va='center', fontsize=FONT_SIZE - 3,
                                color='#ffffff' if v > (vmin + vmax) / 2 else '#0b0b0b')
        for ax in axes.ravel()[len(ps):]:                             # 남는 칸은 비운다
            ax.axis('off')
        fig.supxlabel(_AXIS_LABEL[col_col], fontsize=FONT_SIZE)       # 축 이름은 그림 전체에 하나씩만
        fig.supylabel(_AXIS_LABEL[RS], fontsize=FONT_SIZE)
        # 색 범위가 고정이라 포화가 생길 수 있다 -> colorbar 끝에 화살표로 표시한다 (값은 셀에 찍혀 있다)
        fig.colorbar(im, ax=axes, label=label, fraction=0.03, pad=0.02, extend=cbar_extend)
        os.makedirs(out_dir, exist_ok=True)
        stem = f'{out_dir}/{prefix}_heatmap_{metric}' + (f'_{group}' if group else '')
        for ext in ['png', 'pdf']:
            path = f'{stem}.{ext}'
            assert not os.path.exists(path), f'이미 그림 파일이 있다 (덮어쓰기 금지): {path}'
            fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f'[save] heatmap -> {stem}.png / .pdf')
