# =============================================================================
# 03_baselines/11_embedding_plots.py : 방법별 임베딩 그림 2종을 각 방법 폴더에 저장한다
#
# 입력 : <분할 폴더>/_embeddings_fold{FOLD}_seed{SEED}.npz   (10_embeddings.py 산출)
# 출력 : <분할 폴더>/<단계폴더>/<방법>_embedding_*.png / .pdf
# 실험 : 인자로 고른다 — 'tension' (A) / 'c' (C). 분할마다 **그리는 그림이 다르다**:
#        A : 그림 A(seen/unseen) + 그림 B(시간 궤적)
#        C : 그림 A(seen/unseen) + 그림 C(테스트 속도별)   <- 2026-10-27 연구자 지시
#            C 에서 궤적(그림 B)은 그리지 않는다. 미학습 **고속**이 보이는 것이 목적이라
#            '속도를 뭉뚱그린 unseen' 대신 **속도마다 다른 색**으로 세분화한 그림 C 를 쓴다
# 실행 : python code/03_baselines/11_embedding_plots.py        (A 실험, 기본값)
#        python code/03_baselines/11_embedding_plots.py c      (C 실험)
#
# -----------------------------------------------------------------------------
# 그림 A — 학습 조건 vs 미학습 조건 (`_embedding_seen_unseen`)
# -----------------------------------------------------------------------------
# 묻는 것 : **조건이 달라도 상태별로 같은 자리에 임베딩되는가.**
#   대비되는 축이 분할마다 다르다:
#     A 실험 fold 1  : 학습 장력 4·5·6·7 kg vs 검증 **장력 3 kg (미학습 수준)**
#     C 실험 final_test : 학습 속도 50·75·100·125 vs 검증 **속도 150·175·200 (미학습 고속)**
#   조건 일반화가 되면 '미학습-steady' 가 '학습-steady' 와 겹쳐야 한다.
#
# 4군을 어떻게 구분하나 — **상태는 색, 학습/미학습은 표식(marker)** 으로 쪼갠다.
#   색만으로 4군을 구분하면 CVD 검증을 다시 해야 하고 "상태" 와 "조건" 이 한 채널에 섞여
#   읽기 어렵다. 상태(핵심 대비)에 색을 주고, 학습/미학습은 모양으로 둔다.
#     학습   : 작은 점 'o', 옅게        미학습 : 'x', 진하게 + 위에 그린다
#   패널 2개로 나란히 두어 겹침도 함께 본다 (좌: 겹쳐 그림 / 우: 미학습만).
#
# -----------------------------------------------------------------------------
# 그림 C — 테스트 속도별 (`_embedding_test_speed`)   **C 실험 전용**
# -----------------------------------------------------------------------------
# 묻는 것 : **미학습 고속이 속도마다 어디에 놓이는가.** 그림 A 는 검증 90 run 을 'unseen'
#   하나로 뭉치는데, 그러면 "속도 150 만 어긋나는지, 고속일수록 밀려나는지" 가 보이지 않는다.
#   `research_plan.md` 1.2 절이 기록한 **속도 150 문제**(steady 편차 증가)와 F5a 에서 실제로
#   관찰된 'final_test 보존률 하락이 속도 150 에 몰린다' 를 임베딩에서 확인하기 위한 그림이다.
#
# 색 = 속도. 속도는 **순서가 있는 값**이므로 categorical 이 아니라 sequential ramp 를 쓴다
#   (`evaluation.SEQ_CMAP` 의 3단계, 밝음 -> 어두움 = 느림 -> 빠름). 밝기로 구분되므로
#   색각이상에도 순서가 유지된다. 학습(저속) 구간은 중성 회색 배경으로 깔아 기준을 준다.
# 패널 3개 : 전체 / steady 만 / transient 만 — 상태를 표식에 겹쳐 싣지 않고 패널로 나눈다
#   (그림 A 에서 'o/x' 가 이미 seen/unseen 을 뜻하므로 같은 표식을 다른 의미로 쓰지 않는다).
#
# -----------------------------------------------------------------------------
# 그림 B — 조건별 상태 구분 + 시간 궤적 (`_embedding_trajectory`)   **A 실험 전용**
# -----------------------------------------------------------------------------
# 묻는 것 : **조건마다 steady / transient 가 갈리는가, 그리고 시간에 따라 어떻게 이동하는가.**
#   run 은 과도 -> 정상으로 **한 방향 수렴**한다(research_plan.md 3.1 (2)). 궤적이 그 구조를
#   보여주고, 진입 시점 오차(방법마다 -19 ~ +45 s)가 왜 생기는지 설명한다.
#   패널 = 대표 속도 조건. 각 패널에 그 조건의 run 궤적을 그린다.
#   시간은 **sequential ramp**(밝음 -> 어두움)로, 상태 경계는 참 라벨이 바뀌는 지점에 표식.
#
# -----------------------------------------------------------------------------
# 색 (dataviz 규약)
# -----------------------------------------------------------------------------
# 상태 = categorical 2색 + 중성 회색
#   steady `#0072b2` / transient(non-steady) `#d55e00` — Okabe-Ito 쌍. 검증기 통과:
#   CVD 분리 dE 21.9(protan) · 정상시각 31.2 · 명도대·채도 하한·대비 전부 PASS.
#   ambiguous 는 **채점에서 제외되는 범주**라 동급 식별자로 두지 않고 중성 회색 + 낮은 불투명도.
# 시간 = sequential 단일 hue (`evaluation.SEQ_CMAP`) — 시간은 순서 있는 값이므로 ramp 가 맞다.
#
# -----------------------------------------------------------------------------
# 제목에 진단값을 박는다 — 그림만 보고 오판하지 않게
# -----------------------------------------------------------------------------
#   `dim`   원 표현 차원 / `PC1+2` 2D 투영이 담은 분산 비율
#   PC1+2 가 낮으면 **그림에 안 보여도 고차원에서는 분리돼 있을 수 있다.**
#   AUPRC 는 **정식 점수(5 fold x 5 seed 전체 시점)** 만 표기한다. 2D 진단값은
#   `_embedding_diagnostics_*.csv` 에 따로 있고 점수로 쓰지 않는다.
#
# 1~2차원 방법은 투영하지 않는다 — `MovStd-Slope-fixed` 는 원 축(std vs |기울기|),
# `R-test-calibrated` 는 1차원이라 **그림 A·B 에서 제외**한다(산점도·궤적이 성립하지 않는다).
# =============================================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluation as ev                      # rcParams(논문 스타일) · SEQ_CMAP · PAGE_WIDTH
import splits

SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'                # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 모드의 npz 를 읽고 같은 루트에 그린다
FOLD = 1 if SPLIT == 'tension' else 'final_test'                       # 10_embeddings.py 와 같은 값
SEED = 0
OUT_ROOT = splits.out_root(SPLIT)
NPZ = f'{OUT_ROOT}/_embeddings_fold{FOLD}_seed{SEED}.npz'

# 그림 A 에서 '학습 조건 vs 미학습 조건' 을 가르는 설정값 — 분할마다 다르다
CONTRAST = ('tension', 'film tension', 'kg') if SPLIT == 'tension' else ('speed', 'web speed', 'mm/s')
SPEED_RAMP = [0.30, 0.60, 0.95]  # 그림 C 의 속도별 색 위치 (SEQ_CMAP 상). **[임의]** — 밝기 차이를
                                 # 충분히 벌려 3수준이 구분되게 잡았다. 밝음 -> 어두움 = 느림 -> 빠름
C_SEEN_BG = '#c9c8c0'            # 그림 C 의 학습(저속) 배경색 — 중성 회색 (상태·속도 색과 겹치지 않게)

C_STEADY, C_TRANS, C_AMB = '#0072b2', '#d55e00', '#b8b7ae'
NAME = {1: 'steady', 0: 'transient', 2: 'ambiguous'}
PLOT_PER_CLASS = 2500           # 군당 그릴 점 수 상한. **[임의]** — 과밀 방지. 클래스별로 균등히 뽑아야
                                # steady 85% 에 소수 클래스가 묻히지 않는다
TRAJ_SPEEDS = [50, 100, 150, 200]   # 그림 B 의 대표 속도. **[임의]** — 격자 양 끝 + 중간.
                                    # 150 은 'steady 편차 증가' 가 보고된 조건이라 포함한다
TRAJ_RUNS_PER_PANEL = 3         # 패널당 궤적 수. **[임의]** — 더 많으면 선이 엉킨다

STAGE_OF = {'MovStd-Slope-fixed': '01_stat_ssd', 'TS2Vec': '02_ts2vec', 'MOMENT': '03_moment',
            'CoST-W144': '04_cost', 'SoftCLT-fs-rule1b': '05_softclt',
            'TS-TCC': '07_tstcc', 'SimMTM': '08_simmtm'}

d = np.load(NPZ, allow_pickle=True)
info = pd.DataFrame(d['info'])
label, is_val = d['label'], d['is_val']
# 학습 = '검증이 아닌 전부' 가 아니다 (C 의 CV fold 에는 미사용 run 이 있다). 분할이 준 역할을 쓴다.
is_train = d['is_train'] if 'is_train' in d else ~is_val
speed, tension, run_id, pos = d['speed'], d['tension'], d['run_id'], d['pos']
# ambiguous 포함 모드의 npz 에는 라벨 2 가 없다 (전부 0/1 로 흡수됨) -> 범례에서 그 항목을 뺀다
HAS_AMB = bool((label == 2).any())
CONTRAST_VALS = {'tension': tension, 'speed': speed}[CONTRAST[0]]      # 그림 A 의 seen/unseen 목록용
OFFICIAL = ('official_auprc_fold_mean' if 'official_auprc_fold_mean' in info.columns else 'auprc_full')


def pick(mask, rng, per=PLOT_PER_CLASS):
    # 역할 : 클래스별로 같은 수를 뽑아 소수 클래스가 묻히지 않게 한다.
    out = []
    for lv in (2, 1, 0):
        c = np.flatnonzero(mask & (label == lv))
        out.append(rng.choice(c, min(per, len(c)), replace=False) if len(c) else c)
    return out                                      # [ambiguous, steady, transient] 순 (뒤가 위에 그려짐)


def axis_names(method):
    if method == 'MovStd-Slope-fixed':
        return 'rolling std (standardised)', '|slope| (standardised)'
    return 'PC1', 'PC2'


def short(name):
    # 역할 : 4 패널 그림에서 축 이름이 옆 패널과 겹치지 않도록 '(standardised)' 를 뗀다.
    return name.replace(' (standardised)', ' (z)')


def lim(z, m, pad=0.04):
    # 역할 : 이상치에 축이 끌려가지 않게 분위수로 범위를 잡는다.
    x0, x1 = np.percentile(z[m, 0], [0.5, 99.5]); y0, y1 = np.percentile(z[m, 1], [0.5, 99.5])
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


made = []
for _, r in info.iterrows():
    method, dim = str(r['method']), int(r['dim'])
    stage = STAGE_OF.get(method)
    if stage is None or f'{method}__z2' not in d:
        print(f'[skip] {method}: 1차원이거나 좌표가 없다 (산점도·궤적이 성립하지 않는다)')
        continue
    z = d[f'{method}__z2']
    xl, yl = axis_names(method)
    scope = '5 fold x 5 seed' if SPLIT == 'tension' else f'{FOLD}, 5 seeds'
    head = (f'{method}   dim={dim}   PC1+PC2 explains {r["ev2"]:.1%} of variance   '
            f'AUPRC (official, {scope}) {r[OFFICIAL]:.3f}')
    rng = np.random.default_rng(SEED)

    # =========================================================================
    # 그림 A — 학습 조건(장력 4~7) vs 미학습 조건(장력 3)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(ev.PAGE_WIDTH, 3.2), layout='constrained', sharex=True, sharey=True)
    seen = sorted({int(v) for v in CONTRAST_VALS[is_train]})
    unseen = sorted({int(v) for v in CONTRAST_VALS[is_val]})
    fig.suptitle(head + '\n' + f'seen conditions: {CONTRAST[1]} {seen} {CONTRAST[2]}    '
                 f'unseen: {unseen} {CONTRAST[2]}', fontsize=ev.FONT_SIZE)
    XL, YL = lim(z, np.ones(len(z), bool))
    for ax, (ttl, show_train) in zip(axes, [('seen + unseen (overlaid)', True),
                                            ('unseen condition only', False)]):
        if show_train:                                  # 학습 조건 — 옅은 점, 아래에
            for idx, lv in zip(pick(is_train, rng), (2, 1, 0)):
                if len(idx) == 0:
                    continue
                ax.scatter(z[idx, 0], z[idx, 1], s=3, marker='o', linewidths=0, zorder=1,
                           c=(C_AMB if lv == 2 else C_STEADY if lv == 1 else C_TRANS),
                           alpha=0.18 if lv == 2 else 0.30)
        for idx, lv in zip(pick(is_val, rng), (2, 1, 0)):   # 미학습 조건 — x 표식, 위에
            if len(idx) == 0:
                continue
            ax.scatter(z[idx, 0], z[idx, 1], s=9, marker='x', linewidths=0.6, zorder=3,
                       c=(C_AMB if lv == 2 else C_STEADY if lv == 1 else C_TRANS),
                       alpha=0.35 if lv == 2 else 0.85)
        ax.set_title(ttl, fontsize=ev.FONT_SIZE), ax.set_xlabel(xl)
        ax.set_xlim(*XL), ax.set_ylim(*YL)
    axes[0].set_ylabel(yl)
    # 범례는 축 밖(아래)에 둔다 — loc='best' 가 밀집 군집 위를 덮는다
    fig.legend(handles=[
        Line2D([], [], color=C_STEADY, marker='o', ls='', ms=4, alpha=0.5, label='seen . steady'),
        Line2D([], [], color=C_TRANS, marker='o', ls='', ms=4, alpha=0.5, label='seen . transient'),
        Line2D([], [], color=C_STEADY, marker='x', ls='', ms=5, mew=1.2, label='unseen . steady'),
        Line2D([], [], color=C_TRANS, marker='x', ls='', ms=5, mew=1.2, label='unseen . transient'),
    ] + ([Line2D([], [], color=C_AMB, marker='o', ls='', ms=4, alpha=0.5, label='ambiguous (excluded)')]
         if HAS_AMB else []),                       # ambiguous 포함 모드에서는 이 범주가 존재하지 않는다
        loc='outside lower center', ncol=5 if HAS_AMB else 4, frameon=False,
        fontsize=ev.FONT_SIZE - 1, handletextpad=0.4, borderpad=0.2, columnspacing=1.2)
    base = f'{OUT_ROOT}/{stage}/{method}_embedding_seen_unseen'
    fig.savefig(f'{base}.png', dpi=ev.DPI); fig.savefig(f'{base}.pdf'); plt.close(fig)
    print(f'[save] {base}.png / .pdf')

    # =========================================================================
    # 그림 C — 테스트 속도별 (C 실험 전용). 검증 속도마다 다른 색으로 그린다.
    # =========================================================================
    if SPLIT != 'tension':
        test_speeds = sorted({int(v) for v in speed[is_val]})
        cmap_pos = SPEED_RAMP if len(test_speeds) == len(SPEED_RAMP) else \
            list(np.linspace(0.30, 0.95, len(test_speeds)))
        col_of = {s: ev.SEQ_CMAP(p) for s, p in zip(test_speeds, cmap_pos)}
        fig, axes = plt.subplots(1, 3, figsize=(ev.PAGE_WIDTH, 2.9),
                                 layout='constrained', sharex=True, sharey=True)
        fig.suptitle(head + '\n' + f'grey = seen (training) speeds {seen} mm/s   '
                     f'colour = unseen test speed', fontsize=ev.FONT_SIZE)
        for ax, (ttl, keep_lab) in zip(axes, [('all test timepoints', None),
                                              ('test: steady only', 1),
                                              ('test: transient only', 0)]):
            bg = np.flatnonzero(is_train)                       # 배경: 학습(저속) 전체를 회색으로
            bg = rng.choice(bg, min(4000, len(bg)), replace=False)
            ax.scatter(z[bg, 0], z[bg, 1], s=3, marker='o', linewidths=0, zorder=1,
                       c=C_SEEN_BG, alpha=0.35)
            for s in test_speeds:                               # 속도마다 다른 색
                m = is_val & (speed == s)
                if keep_lab is not None:
                    m = m & (label == keep_lab)
                else:
                    m = m & (label != 2)                        # ambiguous 는 채점 제외 범주라 뺀다
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                idx = rng.choice(idx, min(PLOT_PER_CLASS, len(idx)), replace=False)
                ax.scatter(z[idx, 0], z[idx, 1], s=6, marker='o', linewidths=0, zorder=3,
                           c=[col_of[s]], alpha=0.65)
            ax.set_title(ttl, fontsize=ev.FONT_SIZE - 1)
            ax.set_xlabel(short(xl), fontsize=ev.FONT_SIZE - 1)
            ax.set_xlim(*XL), ax.set_ylim(*YL)
        axes[0].set_ylabel(short(yl), fontsize=ev.FONT_SIZE - 1)
        fig.legend(handles=[Line2D([], [], color=C_SEEN_BG, marker='o', ls='', ms=4,
                                   label=f'seen (train) {min(seen)}-{max(seen)} mm/s')] +
                   [Line2D([], [], color=col_of[s], marker='o', ls='', ms=4,
                           label=f'unseen {s} mm/s') for s in test_speeds],
                   loc='outside lower center', ncol=len(test_speeds) + 1, frameon=False,
                   fontsize=ev.FONT_SIZE - 1, handletextpad=0.4, borderpad=0.2, columnspacing=1.2)
        base = f'{OUT_ROOT}/{stage}/{method}_embedding_test_speed'
        fig.savefig(f'{base}.png', dpi=ev.DPI); fig.savefig(f'{base}.pdf'); plt.close(fig)
        print(f'[save] {base}.png / .pdf')
        made.append(method)
        continue                                                # C 는 궤적 그림을 그리지 않는다

    # =========================================================================
    # 그림 B — 조건별 상태 구분 + 시간 궤적 (A 실험 전용)
    # =========================================================================
    sp_use = [s for s in TRAJ_SPEEDS if (speed == s).any()]
    fig, axes = plt.subplots(1, len(sp_use), figsize=(ev.PAGE_WIDTH, 2.7),
                             layout='constrained', sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(head + '\n' + 'background = all timepoints (state colour)   lines = per-run time trajectory (light to dark = transient to steady)',
                 fontsize=ev.FONT_SIZE)
    for ax, s in zip(axes, sp_use):
        at = speed == s
        for idx, lv in zip(pick(at, rng, per=1200), (2, 1, 0)):      # 배경: 그 조건의 상태 분포
            if len(idx) == 0:
                continue
            ax.scatter(z[idx, 0], z[idx, 1], s=3, linewidths=0, zorder=1,
                       c=(C_AMB if lv == 2 else C_STEADY if lv == 1 else C_TRANS),
                       alpha=0.15 if lv == 2 else 0.28)
        runs = [u for u in pd.unique(run_id[at])][:TRAJ_RUNS_PER_PANEL]
        for u in runs:                                                # 궤적: run 내 시간 순서
            m = np.flatnonzero((run_id == u) & at)
            m = m[np.argsort(pos[m])]
            seg = np.stack([z[m, 0], z[m, 1]], axis=1)
            for i in range(len(seg) - 1):                             # 시간을 ramp 로
                ax.plot(seg[i:i + 2, 0], seg[i:i + 2, 1], lw=0.9, zorder=2,
                        color=ev.SEQ_CMAP(i / max(1, len(seg) - 2)), solid_capstyle='round')
            ax.scatter(*seg[0], s=22, marker='s', facecolor='none', edgecolor='#3b3a33',
                       linewidths=0.9, zorder=4)                      # 시작(과도)
            ax.scatter(*seg[-1], s=26, marker='*', color='#3b3a33', zorder=4)   # 끝(정상)
        # 4 패널이 6.5 in 를 나눠 쓰므로 한 줄로 쓰면 제목끼리 겹친다
        ax.set_title(f'speed {s} mm/s\n({len(runs)} runs)', fontsize=ev.FONT_SIZE - 1)
        # 4 패널이므로 짧은 축 이름을 쓴다 (긴 이름은 패널끼리 겹친다)
        ax.set_xlabel(short(xl), fontsize=ev.FONT_SIZE - 1)
        ax.set_xlim(*XL), ax.set_ylim(*YL)
    axes[0].set_ylabel(short(yl), fontsize=ev.FONT_SIZE - 1)
    fig.legend(handles=[
        Line2D([], [], color=C_STEADY, marker='o', ls='', ms=4, alpha=0.6, label='steady'),
        Line2D([], [], color=C_TRANS, marker='o', ls='', ms=4, alpha=0.6, label='transient'),
        Line2D([], [], color='#3b3a33', marker='s', ls='', ms=4, mfc='none', label='run start (transient)'),
        Line2D([], [], color='#3b3a33', marker='*', ls='', ms=6, label='run end (steady)'),
    ], loc='outside lower center', ncol=4, frameon=False,
        fontsize=ev.FONT_SIZE - 1, handletextpad=0.4, borderpad=0.2, columnspacing=1.2)
    base = f'{OUT_ROOT}/{stage}/{method}_embedding_trajectory'
    fig.savefig(f'{base}.png', dpi=ev.DPI); fig.savefig(f'{base}.pdf'); plt.close(fig)
    print(f'[save] {base}.png / .pdf')
    made.append(method)

print(f'\n[done] {len(made)} 방법 x 2 그림 = {len(made)*2} 장 (+ PDF). 방법: {made}')
print('주의: PC1+2 분산이 낮은 방법은 **그림에 안 보여도 고차원에서 분리돼 있을 수 있다.**')
print('      2D 진단값은 _embedding_diagnostics_*.csv 에 따로 있고 정식 점수가 아니다.')
