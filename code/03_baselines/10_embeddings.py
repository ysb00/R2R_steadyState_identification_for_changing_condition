# =============================================================================
# 03_baselines/10_embeddings.py : 방법별 임베딩을 재생성하고 2D PCA 로 저장한다
#
# 목적 : "표현학습인데 임베딩 그림이 없다" 를 메운다. 같은 트랙 A 후단을 쓰는데
#        **각 방법이 어떤 공간을 만들어 주는지** 를 본다.
#
# 왜 재생성인가 : 임베딩은 저장하지 않았고(체크포인트만 저장) 용량이 크다
#        (TS-TCC 2,560차원 x 27만 시점 = 2.8 GB). 체크포인트 300개가 보존돼 있으므로
#        **학습 없이 encode 만** 다시 한다.
#
# 고정 조건 : **seed 0**, fold 는 분할마다 다르다 (연구자 결정)
#        A('tension') : **fold 1** — 장력 3 kg 검증 = 외삽이고 전 방법에서 가장 어려운 fold 다
#                       (`_all_methods_by_fold.csv`). 퇴화 사례(§11 문제 11)를 보려면
#                       fold 2 seed 2 를 따로 돌려야 한다 — FOLD/SEED 상수 참조.
#        C('c')       : **final_test** — 저속 4수준(120 run) 학습 -> 미학습 고속 3수준(90 run) 검증.
#                       최종 목표와 같은 조건이라 '학습 조건 vs 미학습 조건' 그림이
#                       그대로 **저속 학습 vs 미학습 고속**이 된다 (2026-10-27 연구자 결정).
#
# 투영 : **PCA** (연구자 결정). UMAP/t-SNE 는 하이퍼파라미터가 새 임의값이 되고 군집이
#        인공물일 수 있다. 그리고 본 후단이 **축정렬 가우시안**(diag GMM)을 가정하므로
#        표준화 후 선형 투영인 PCA 가 "후단이 보는 공간" 을 가장 정직하게 보여준다.
#        **표준화·PCA 는 학습 fold 로만 적합**한다 (GMM 과 같은 규약).
#
# 그림이 오해를 만들지 않게 세 숫자를 함께 저장한다
#   `ev2`       : PC1+PC2 가 담은 분산 비율 — 투영이 원 공간을 얼마나 대표하나
#   `auprc_full`: 원 차원에서의 AUPRC (= A 실험 값)
#   `auprc_2d`  : **2D 에서 같은 diag GMM 을 다시 적합해 측정한 AUPRC**
#   -> auprc_2d 가 auprc_full 과 비슷하면 "그림이 성능을 대표한다", 크게 낮으면
#      **"그림만 보고 판단하면 안 된다"** 를 수치로 말할 수 있다.
#
# 차원이 1~2 인 방법은 투영하지 않는다
#   `R-test-calibrated` 는 1차원(PC2 가 없다), `MovStd-Slope-fixed` 는 2차원이라 PCA 가
#   거의 무손실이다. 원 축을 그대로 저장하고 그림에서 따로 다룬다 (plot 스크립트 참조).
#
# 입력 : outputs/merged_data.csv, splits.py 가 읽는 fold 파일, <분할 폴더>/*/checkpoints/*.pt
# 출력 : <분할 폴더>/_embeddings_fold{FOLD}_seed{SEED}.npz
#        (A -> outputs/03_baselines/ , C -> outputs/04_speed_extrap/)
# 실험 : 인자로 고른다 — 'tension' (A) / 'c' (C). 어댑터 8개와 같은 규약이다
# 실행 : python code/03_baselines/10_embeddings.py        (A 실험, 기본값)
#        python code/03_baselines/10_embeddings.py c      (C 실험)
#        C 는 **F6(12_summary_tables_c.py) 이후**에 돌린다 — 정식 AUPRC 를 그 표에서 읽는다
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluation as ev
import splits

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'                # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 를 흡수한 라벨 + 별도 폴더.
# 이 스크립트는 원래부터 학습하지 않는다(체크포인트로 encode 만). 바뀌는 것은 색칠·진단에 쓰는
# **라벨 해석**과 **저장 위치**뿐이고, 체크포인트는 기존 실험 것을 그대로 읽는다.

# =============================================================================
# hyperparameter
# =============================================================================
FOLD = 1 if SPLIT == 'tension' else 'final_test'    # [사용자] A: 장력 3 kg 검증(외삽) / C: 미학습 고속 검증
SEED = 0                        # [사용자] seed 0
STRIDE = 10                     # 시점 추출 간격. **[임의]** — 그림·PCA 용 서브샘플.
                                # 27만 시점을 다 쓰면 TS-TCC 2,560차원에서 2.8 GB 다.
                                # 10 이면 2.7만 시점(276 MB)이고 2D PCA·산점도에 충분하다.
                                # 지표(auprc_2d)도 이 부분집합에서 재므로 원 값과 미세하게 다를 수 있다.
WINDOW = 144                    # 규칙 1b (docs/periodic_disturbance.md §7)
ENCODE_BATCH = 512              # **[임의]** — 결과에 영향 없음
DEVICE = 'cuda'

IN_PATH = 'outputs/merged_data.csv'
OUT_ROOT = splits.out_root(SPLIT)
OUT = f'{OUT_ROOT}/_embeddings_fold{FOLD}_seed{SEED}.npz'
T1 = 'Film Tension #1 (kg)'
TS, RS, VS = ev.TS, ev.RS, ev.VS

# 그릴 방법 — 본문 8종에서 R-test 원 논문판(GMM 미사용)을 빼고, 퇴화 2종을 포함
# (method, 단계폴더, 표현 종류)
METHODS = [
    ('R-test-calibrated',   '01_stat_ssd', 'rtest'),      # 1차원
    ('MovStd-Slope-fixed',  '01_stat_ssd', 'movstd'),      # 2차원
    ('TS2Vec',              '02_ts2vec',   'ts2vec'),      # 320
    ('MOMENT',              '03_moment',   'moment'),      # 512
    ('CoST-W144',           '04_cost (ablation)',     'cost'),        # 320
    ('SoftCLT-fs-rule1b',   '05_softclt',  'softclt'),     # 320
    ('TS-TCC',              '07_tstcc',    'tstcc'),       # 2,560
    ('SimMTM',              '08_simmtm',   'simmtm'),      # 1,024
]

# =============================================================================
# Step 1. 데이터 로드 + 서브샘플 인덱스
#         `Trigger 4` 는 **그림의 색칠용으로만** 읽는다. 표현 생성에는 쓰지 않는다.
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS, 'Trigger 4'])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
e_by_run = {r: by_run[r].e.to_numpy() for r in run_ids}
lengths = np.array([len(e_by_run[r]) for r in run_ids])
# run 역할을 분할에서 그대로 받는다. C 의 CV fold 는 '학습도 검증도 아닌' run 이 있으므로
# '학습 = 검증 아닌 전부' 로 두면 안 된다 (어댑터 8개와 같은 규약).
tr_run, va_run = sp.run_role(FOLD)
role_of_run = np.where(tr_run, 'train', np.where(va_run, 'val', 'unused'))

# 시점 메타 (서브샘플 후)
meta = []
for r, role in zip(run_ids, role_of_run):
    g = by_run[r]
    idx = np.arange(0, len(g), STRIDE)
    meta.append(pd.DataFrame({'run_id': r, 'k': g.k.to_numpy()[idx], 'role': role,
                              'label': g['Trigger 4'].to_numpy()[idx],
                              'e': g.e.to_numpy()[idx],
                              'tension': g[TS].to_numpy()[idx], 'rewinder': g[RS].to_numpy()[idx],
                              'speed': g[VS].to_numpy()[idx], 'pos': idx}))
meta = pd.concat(meta, ignore_index=True)
# 라벨 해석은 evaluation 한 곳에서만 한다 — ambiguous 포함 모드면 2 가 흡수된 라벨이 돌아온다.
# 기본 모드에서는 위에서 읽은 `Trigger 4` 와 같은 값이므로 기존 결과가 바뀌지 않는다.
meta['label'] = ev.label_series(meta)
is_val = (meta.role == 'val').to_numpy()
is_train = (meta.role == 'train').to_numpy()
print(f'[Step 1] 분할 {SPLIT} fold {FOLD} seed {SEED} | 서브샘플 {len(meta)} 시점 (stride {STRIDE}) '
      f'| 학습 {int(is_train.sum())} / 검증 {int(is_val.sum())} / 미사용 {int((~is_train & ~is_val).sum())} '
      f'| 라벨 분포 {dict(meta.label.value_counts().sort_index())} ({time.time()-t0:.1f}s)')


# =============================================================================
# 함수 — 방법별 표현 생성 (서브샘플 시점만)
# =============================================================================
def causal_windows(e, pos, window):
    # 역할 : 시점 pos 마다 causal 창 [pos-window+1, pos] 를 만든다. 창이 안 찬 앞부분은
    #        첫 완전창으로 대체한다 (전 어댑터와 동일 규약 — 채점 시점 수를 맞추기 위한 것).
    # 출력 : (len(pos), window)
    start = np.clip(pos - window + 1, 0, None)
    start = np.where(pos < window - 1, 0, start)
    return e[start[:, None] + np.arange(window)]


def encode_sliding(fwd, e, pos, window, batch, dtype=np.float32):
    # 역할 : causal 창을 배치로 모델에 넣고 시점별 표현을 얻는다.
    # 입력 : fwd  (torch (B,1,W) -> numpy (B,d)) 콜백
    out = []
    X = causal_windows(e, pos, window)
    for b in range(0, len(X), batch):
        xb = torch.from_numpy(X[b:b + batch]).float().unsqueeze(1).to(DEVICE)
        out.append(fwd(xb))
    return np.concatenate(out).astype(dtype)


# 외부 저장소들이 `models` · `config_files` · `utils` 같은 **같은 이름의 모듈**을 각자 갖고 있어
# 한 프로세스에서 여러 개를 import 하면 먼저 올라온 것이 이긴다 (sys.path / sys.modules 오염).
# 그래서 방법마다 외부 경로를 초기화하고 충돌 모듈을 내린다.
_EXT_NAMES = ('models', 'config_files', 'utils', 'dataloader', 'trainer', 'experiments',
              'ts2vec', 'soft_ts2vec', 'cost', 'model', 'loss', 'layers', 'momentfm')


def isolate(*subpath):
    # 역할 : 외부 저장소 하나만 import 가능한 상태로 sys.path / sys.modules 를 되돌린다.
    # 입력 : subpath  external 하위 경로 조각들
    for name in [k for k in list(sys.modules) if k.split('.')[0] in _EXT_NAMES]:
        del sys.modules[name]
    sys.path[:] = [q for q in sys.path if 'external' not in q]
    sys.path.insert(0, os.path.join(HERE, 'external', *subpath))


def build(kind, method, stage):
    # 역할 : 방법별로 서브샘플 시점의 표현을 만든다. 학습은 하지 않고 체크포인트만 쓴다.
    # 출력 : (n_sub, d) float32
    ck = f'{splits.ckpt_dir(SPLIT, stage)}/{method}_fold{FOLD}_seed{SEED}.pt'   # 항상 기존 실험 체크포인트

    if kind == 'rtest':                                     # 1차원 — R 통계량 (학습 없음)
        sys.path.insert(0, HERE)
        import importlib.util as iu
        # 01_stat_ssd.py 는 모듈 이름이 숫자로 시작해 import 불가 -> 함수만 복제하지 않고
        # 저장된 설정과 같은 상수로 여기서 직접 계산한다 (원 스크립트와 같은 식).
        L1, L2, L3, DEC = 0.2, 0.1, 0.1, 10
        def rstat(x):
            n = len(x); nu2 = np.empty(n); d2 = np.empty(n)
            xf = nu2f = d2f = xo = 0.0
            for i in range(n):
                nu2f = L2 * (x[i] - xf) ** 2 + (1 - L2) * nu2f
                xf = L1 * x[i] + (1 - L1) * xf
                d2f = L3 * (x[i] - xo) ** 2 + (1 - L3) * d2f
                xo = x[i]; nu2[i], d2[i] = nu2f, d2f
            with np.errstate(divide='ignore', invalid='ignore'):
                r = (2 - L1) * nu2 / d2
            bad = ~np.isfinite(r)
            r[bad] = np.nanmax(r[~bad]) if (~bad).any() else 0.0
            return r
        zs = []
        for r in run_ids:
            e = e_by_run[r]
            rd = rstat(e[::DEC])                            # R-test-calibrated 는 다운샘플판이 아니다
            zs.append(rstat(e)[np.arange(0, len(e), STRIDE)])
        return np.concatenate(zs).astype(np.float32)[:, None]

    if kind == 'movstd':                                    # 2차원 — (std, |기울기|), 학습 없음
        zs = []
        w = 64                                              # MovStd-Slope-fixed 의 창 (임의값 64)
        j = np.arange(w) - (w - 1) / 2.0
        kern = j / (j ** 2).sum() / ev.DT
        for r in run_ids:
            e = e_by_run[r]
            std = np.array(pd.Series(e).rolling(w).std())
            slope = np.empty(len(e)); slope[w - 1:] = np.convolve(e, kern[::-1], mode='valid')
            std[:w - 1], slope[:w - 1] = std[w - 1], slope[w - 1]
            idx = np.arange(0, len(e), STRIDE)
            zs.append(np.column_stack([std[idx], np.abs(slope[idx])]))
        return np.concatenate(zs).astype(np.float32)

    if kind in ('ts2vec', 'softclt'):                       # run 전체를 한 번에 — 창 없음
        isolate('ts2vec') if kind == 'ts2vec' else isolate('softclt', 'softclt_ts2vec')
        mod = 'ts2vec' if kind == 'ts2vec' else 'soft_ts2vec'
        TSV = __import__(mod).TS2Vec
        m = TSV(input_dims=1, output_dims=320, device=DEVICE, max_train_length=3000)
        m.load(ck)
        x = np.full((len(run_ids), lengths.max(), 1), np.nan)
        for i, r in enumerate(run_ids):
            x[i, :lengths[i], 0] = e_by_run[r]
        z = m.encode(x)
        return np.concatenate([z[i, np.arange(0, lengths[i], STRIDE)] for i in range(len(run_ids))]).astype(np.float32)

    if kind == 'cost':                                      # 이동창, 창 마지막 시점만 출력
        isolate('CoST')
        from cost import CoST
        m = CoST(input_dims=1, kernels=[1, 2, 4, 8, 16, 32, 64, 128], alpha=0.0005,
                 max_train_length=WINDOW, device=DEVICE, output_dims=320)
        m.load(ck)
        zs = []
        for r in run_ids:
            e = e_by_run[r]; pos = np.arange(0, len(e), STRIDE)
            X = causal_windows(e, pos, WINDOW)[:, :, None]
            zs.append(m.encode(X, mode='forecasting', casual=True,
                               sliding_length=1, sliding_padding=WINDOW - 1,
                               batch_size=ENCODE_BATCH)[:, -1, :])
        return np.concatenate(zs).astype(np.float32)

    if kind == 'moment':                                    # 512 구조 강제, 학습 없음
        isolate('moment')                                   # 03_moment.py 와 같은 경로
        from momentfm import MOMENTPipeline
        mm = MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-small',
                                            model_kwargs={'task_name': 'embedding'})
        mm.init(); mm = mm.to(DEVICE).eval()
        zs = []
        for r in run_ids:
            e = e_by_run[r]; pos = np.arange(0, len(e), STRIDE)
            X = causal_windows(e, pos, 512)
            out = []
            with torch.no_grad():
                for b in range(0, len(X), 64):
                    xb = torch.from_numpy(X[b:b + 64]).float().unsqueeze(1).to(DEVICE)
                    out.append(mm(x_enc=xb).embeddings.cpu().numpy())
            zs.append(np.concatenate(out))
        return np.concatenate(zs).astype(np.float32)

    if kind == 'tstcc':                                     # 평탄화 특징맵 2,560
        isolate('TS-TCC')
        from config_files.Epilepsy_Configs import Config
        from models.model import base_Model
        c = Config(); c.features_len = 20
        m = base_Model(c).to(DEVICE)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)['model_state_dict']); m.eval()
        def fwd(xb):
            with torch.no_grad():
                _, f = m(xb)
            return f.reshape(f.shape[0], -1).cpu().numpy()
        return np.concatenate([encode_sliding(fwd, e_by_run[r], np.arange(0, lengths[i], STRIDE),
                                              WINDOW, ENCODE_BATCH)
                               for i, r in enumerate(run_ids)])

    if kind == 'simmtm':                                    # 평탄화 conv 특징 1,024
        isolate('SimMTM', 'SimMTM_Classification', 'code')
        from config_files.SleepEEG_Configs import Config
        from model import TFC
        from torch import nn
        c = Config(); c.CNNoutput_channel = 8; c.TSlength_aligned = WINDOW
        # 체크포인트가 'pre_train' 모드로 저장돼 awl/contrastive/aggregation/head/mse 를 포함한다.
        # 같은 모드로 만들어 load 한 뒤, forward 만 비사전학습 경로(h, z 반환)로 돌린다.
        # pre_train 모드는 ContrastiveWeight/AggregationRebuild 를 만들고 그들이 args 의
        # temperature·positive_nums 를 읽는다. 저장소 argparse 기본값과 같게 둔다.
        class A:
            training_mode = 'pre_train'
            temperature = 0.2
            positive_nums = 3
        m = TFC(c, A()); m.head = nn.Linear(1024, WINDOW)
        m.load_state_dict(torch.load(ck, map_location=DEVICE)['model_state_dict'])
        m = m.to(DEVICE).eval(); m.training_mode = 'eval'      # pretrain=False 경로 사용
        def fwd(xb):
            with torch.no_grad():
                h, _ = m(xb)
            return h.cpu().numpy()
        return np.concatenate([encode_sliding(fwd, e_by_run[r], np.arange(0, lengths[i], STRIDE),
                                              WINDOW, ENCODE_BATCH)
                               for i, r in enumerate(run_ids)])

    raise ValueError(kind)


# =============================================================================
# Step 2. 방법별로 표현 -> 학습 fold 로 표준화·PCA 적합 -> 2D 투영 + 진단 3종
# =============================================================================
store = {}
rows = []
for method, stage, kind in METHODS:
    t0 = time.time()
    try:
        z = build(kind, method, stage)
    except Exception as exc:
        print(f'[Step 2] {method}: FAILED  {type(exc).__name__}: {exc}', flush=True)
        continue
    d = z.shape[1]

    # 표준화·PCA 는 **학습 fold 로만** 적합한다 (GMM 과 같은 규약)
    sc = StandardScaler().fit(z[is_train])
    zs = sc.transform(z)
    if d >= 2:
        # 전 성분을 적합해 **설명력 곡선** 전체를 기록한다 — PC1+2 만 보면 "나머지가 얼마나
        # 남았는지" 를 알 수 없다. 2D 가 부족한지 판단하려면 50/80/90% 에 몇 개가 필요한지가 필요하다.
        pf = PCA(n_components=min(d, len(zs[is_train])), random_state=SEED).fit(zs[is_train])
        evr = pf.explained_variance_ratio_
        cum = np.cumsum(evr)
        z2 = pf.transform(zs)[:, :3]                        # PC3 까지 저장 (3D 그림 선택지용)
        ev2 = float(cum[1])
        ev1, ev3 = float(evr[0]), float(cum[2]) if len(cum) > 2 else float(cum[-1])
        n50, n80, n90 = (int(np.searchsorted(cum, t) + 1) for t in (0.5, 0.8, 0.9))
    else:                                                   # 1차원 — 투영하지 않는다
        z2 = np.column_stack([zs[:, 0], np.zeros(len(zs)), np.zeros(len(zs))])
        ev1 = ev2 = ev3 = 1.0
        n50 = n80 = n90 = 1

    # 그림의 대표성 진단 — **같은 서브샘플**에서 원 차원과 2D 를 모두 재야 비교가 성립한다.
    # A 실험의 auprc 는 전체 27만 시점 값이고 여기는 stride 10 의 2.7만 시점이라,
    # 그 값과 직접 비교하면 차이가 서브샘플링 인공물인지 구분되지 않는다.
    from sklearn.metrics import average_precision_score
    lab = meta.label.to_numpy()[is_val]
    keep = lab != 2                                         # ambiguous 는 채점에서 제외 (전 지표 규약)
    y = (lab[keep] == 0).astype(int)                         # non-steady = positive

    def auprc_of(zin):
        # float64 로 넘긴다 — 고차원에서 float32 는 diag GMM 적합을 깨뜨린다
        # (E[X^2]-E[X]^2 의 자리수 소실. docs/tstcc_config_choice.md §7.5 와 같은 원인)
        zin = np.asarray(zin, dtype=np.float64)
        g, si, _ = ev.fit_steady_gmm(zin[is_train], meta.e.to_numpy()[is_train], SEED)
        _, sc = ev.predict_steady(g, si, zin[is_val])
        return float(average_precision_score(y, sc[keep]))

    auprc_sub_full = auprc_of(zs)                            # 원 차원, 같은 서브샘플
    # **슬라이스를 반드시 자른다** — z2 에 PC3 까지 담겨 있어 그냥 넘기면 3D 를 재게 된다.
    auprc2 = auprc_of(z2[:, :1] if d == 1 else z2[:, :2])     # 2D (그림이 쓰는 평면)
    auprc3 = auprc_of(z2[:, :1] if d == 1 else z2[:, :3])     # 3D (2D 가 부족한지 판단용)

    store[f'{method}__z2'] = z2.astype(np.float32)
    rows.append({'method': method, 'dim': d,
                 'ev_pc1': ev1, 'ev2': ev2, 'ev_pc1to3': ev3,
                 'n_pc_50pct': n50, 'n_pc_80pct': n80, 'n_pc_90pct': n90,
                 'diag_auprc_fulldim': auprc_sub_full,
                 'diag_auprc_2d': auprc2, 'diag_auprc_3d': auprc3})
    print(f'[Step 2] {method:22s} d={d:5d}  PC1+2 분산 {ev2:6.1%}  '
          f'PC1~3 {ev3:5.1%}  50/80/90%% 에 필요한 PC {n50}/{n80}/{n90}개  '
          f'AUPRC(동일 서브샘플) 원차원 {auprc_sub_full:.3f} / 2D {auprc2:.3f} / 3D {auprc3:.3f}  ({time.time()-t0:.0f}s)', flush=True)

# 그 분할의 **정식 점수**를 참조용으로 붙인다 (이름에 official 을 박아 진단값과 구분한다)
#   A : 5 fold x 5 seed 전체 평균 (`_all_methods_full_table.csv`)
#   C : **그 fold(final_test) 의 seed 평균** (`_all_methods_by_fold.csv`) — 묶음을 가로질러
#       평균하면 안 되므로(§13.0 F6) fold 단위 값을 쓴다
if SPLIT == 'tension':
    full = pd.read_csv(f'{OUT_ROOT}/_all_methods_full_table.csv')[['method', 'pooled_auprc__mean']]
else:
    full = pd.read_csv(f'{OUT_ROOT}/_all_methods_by_fold.csv')
    full = (full[full.fold == FOLD][['method', 'pooled_auprc']]
            .rename(columns={'pooled_auprc': 'pooled_auprc__mean'}))
info = (pd.DataFrame(rows).merge(full, on='method', how='left')
        .rename(columns={'pooled_auprc__mean': 'official_auprc_fold_mean'}))
# 진단값은 **별도 CSV** 로도 저장한다 — 정식 결과 표(_paper_table.csv 등)와 섞지 않는다
info.assign(caveat=f'diagnostic only: stride {STRIDE} subsample, split {SPLIT} fold {FOLD} seed {SEED}, single run'
            ).to_csv(f'{OUT_ROOT}/_embedding_diagnostics_fold{FOLD}_seed{SEED}.csv', index=False)

# =============================================================================
# Step 3. 저장
# =============================================================================
os.makedirs(OUT_ROOT, exist_ok=True)
np.savez_compressed(OUT, info=info.to_records(index=False),
                    label=meta.label.to_numpy(), is_val=is_val, is_train=is_train, e=meta.e.to_numpy(),
                    speed=meta.speed.to_numpy(), tension=meta.tension.to_numpy(),
                    rewinder=meta.rewinder.to_numpy(), pos=meta.pos.to_numpy(),
                    run_id=meta.run_id.to_numpy().astype('U16'),
                    role=meta.role.to_numpy().astype('U8'), split=np.array(SPLIT),
                    **store)
print()
print(info.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
print(f'\n[save] {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)')
