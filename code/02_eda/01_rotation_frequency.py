# =============================================================================
# 02_eda/01_rotation_frequency.py : steady 구간 주기 외란의 출처 식별과 창 길이 유도
#
# 목적 : 비교군·제안 방법의 이동창 길이를 라벨 없이 유도한다.
#   ① e 신호에 속도 비례 주기 성분이 있는지 확인                      -> 그림 1
#   ② 그 성분이 리와인더 롤에서 오는지 검증 (f ∝ 1/D 여야 한다)        -> 그림 2
#   ③ 역산 둘레가 전 조건에서 일정한지 확인 (고정 지름 롤러인가)        -> 그림 3
#   ④ 창이 1주기를 담지 못하면 성능이 나빠지는지 확인 (창 길이 하한)     -> 그림 4
#   ⑤ 창이 길면 steady 진입이 늦어지는지 확인 (창 길이 상한 방향)        -> 그림 5
#
# 라벨 사용 : 없다. 주파수는 **run 전체** 신호에서 측정한다 (steady 구간을 잘라내지 않는다).
#   docs/research_plan.md 1.2절의 기존 수치는 "run별 steady 앞 1024 샘플" 에서 측정한 것이라
#   라벨이 개입했다. 본 스크립트가 그것을 라벨 없이 대체한다.
#   그림 4·5 의 성능 수치는 이미 산출된 비교군 결과(outputs/03_baselines/)를 읽어 쓰는 것이며,
#   창 길이 결정에 쓰는 것이 아니라 결정 근거를 **사후 확인**하는 용도다.
#
# 입력 : data/data_tensionSetting_{3..7}.zip   (원본. Rewinder roll diameter 열이 필요해 zip 에서 직접 읽는다)
#        outputs/03_baselines/01_stat_ssd/*_run_metrics.csv, 04_cost/*_fold_metrics.csv  (그림 4·5)
# 출력 : outputs/02_eda/ 에 run별 스펙트럼 측정표 1개 + 그림 5종 (PNG 300 dpi + PDF)
# 실행 : python code/02_eda/01_rotation_frequency.py   (repo 루트에서)
# =============================================================================
import io
import os
import re
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# hyperparameter
# =============================================================================
DT = 0.109                      # 샘플 간격 (s). docs/data_dictionary.md
FS = 1.0 / DT                   # 샘플링 주파수 (Hz) ~= 9.174
BAND = (0.03, 0.30)             # 회전 1차 성분을 찾을 주파수 범위 (Hz).
                                # 하한 0.03 은 run 길이 140 s 에서 해상도 한계(0.007 Hz)의 4배,
                                # 상한 0.30 은 최고 속도(V=200)의 예상 주파수 0.26 Hz 보다 약간 위
T1 = 'Film Tension #1 (kg)'
TSET = 'Film-Tension Setting Value (kg)'
VS = 'Web Speed setting value (mm/sec)'
DIA = 'Rewinder roll diameter (mm)'

OUT_DIR = 'outputs/02_eda'
BASE_DIR = 'outputs/03_baselines'

FONT_SIZE, DPI, PAGE_WIDTH = 9, 300, 6.5          # 논문 그림 공통 (evaluation.py 와 동일)
mpl.rcParams.update({'font.size': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'axes.labelsize': FONT_SIZE,
                     'xtick.labelsize': FONT_SIZE, 'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
                     'axes.grid': True, 'grid.linewidth': 0.5, 'grid.color': '#e1e0d9',
                     'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8, 'savefig.bbox': 'tight'})
# 속도 7수준을 단일 색상(파랑) 밝음->어두움으로 (순서가 있는 양이므로 sequential)
SPEED_COLORS = ['#9ec5f4', '#86b6ef', '#5598e7', '#2a78d6', '#256abf', '#184f95', '#0d366b']
INK, ACCENT = '#0b0b0b', '#d55e00'


def save(fig, name):
    # 역할 : 그림을 PNG(300 dpi) + PDF 로 저장한다. 같은 이름이 있으면 덮어쓰지 않는다.
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = f'{OUT_DIR}/{name}.{ext}'
        assert not os.path.exists(p), f'이미 그림 파일이 있다 (덮어쓰기 금지): {p}'
        fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f'[save] {OUT_DIR}/{name}.png / .pdf')


def spectrum(e):
    # 역할 : run 1개의 e 신호에서 단측 파워 스펙트럼을 구한다.
    #        평균을 빼고 Hann 창을 적용한다 (유한 길이 신호의 스펙트럼 누설 억제).
    # 입력 : e (n,) run 전체의 오차 신호 (steady 구간을 자르지 않는다 = 라벨 미사용)
    # 출력 : (f (m,) Hz, P (m,) 파워)
    x = np.asarray(e, dtype=float)
    x = x - x.mean()
    n = len(x)
    P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    return np.fft.rfftfreq(n, DT), P


def peak_in_band(f, P):
    # 역할 : BAND 안에서 파워가 가장 큰 주파수를 찾는다 (회전 1차 성분 추정).
    # 출력 : (f_peak (Hz), 그 지점의 파워)
    m = (f > BAND[0]) & (f < BAND[1])
    i = np.argmax(P[m])
    return float(f[m][i]), float(P[m][i])


# =============================================================================
# Step 1. 원본 zip 에서 run 별 신호와 리와인더 지름을 읽어 스펙트럼 최대 주파수를 구한다
# =============================================================================
rows, spec_cache = [], {}
for t in (3, 4, 5, 6, 7):
    z = zipfile.ZipFile(f'data/data_tensionSetting_{t}.zip')
    for name in z.namelist():
        d = pd.read_excel(io.BytesIO(z.read(name)), usecols=[T1, TSET, VS, DIA])
        T, R, V = map(int, re.findall(r'=\s*(\d+)', name))
        e = (d[T1] - d[TSET]).to_numpy(dtype=float)
        f, P = spectrum(e)
        fp, _ = peak_in_band(f, P)
        rows.append({'run_id': f'T{T}_R{R}_V{V}', 'T': T, 'R': R, 'V': V, 'n': len(e),
                     'f_peak_Hz': fp, 'period_s': 1 / fp, 'period_samples': FS / fp,
                     'implied_circumference_mm': V / fp, 'implied_diameter_mm': V / fp / np.pi,
                     'rewinder_diameter_mean_mm': float(d[DIA].mean()),
                     'rewinder_diameter_min_mm': float(d[DIA].min()),
                     'rewinder_diameter_max_mm': float(d[DIA].max()),
                     'predicted_f_rewinder_Hz': V / (np.pi * float(d[DIA].mean()))})
        if (R == 4) and (f'V{V}' not in spec_cache):               # 그림 1 용 대표 run (리와인더 장력 4 kg 고정)
            spec_cache[f'V{V}'] = (f, P, f'T{T}_R{R}_V{V}')
    print(f'[Step 1] tension={t}: {len(z.namelist())} runs')
sp = pd.DataFrame(rows)

# 역산 둘레가 속도별 중앙값에서 크게 벗어난 run 은 회전 성분을 찾지 못한 것으로 본다.
# (진폭 이상 run 2개는 스펙트럼이 이상 진폭의 저주파 성분에 지배되고,
#  일부 run 은 인접 주파수 bin 을 잡는다. docs/periodic_disturbance.md §6.4)
# 둘레 산출에서만 제외하며, 비교군 평가에서는 210 run 전부를 쓴다.
OUTLIER_TOL = 0.10
_med = sp.groupby('V').implied_circumference_mm.transform('median')
sp['circumference_outlier'] = (sp.implied_circumference_mm - _med).abs() / _med > OUTLIER_TOL
os.makedirs(OUT_DIR, exist_ok=True)
sp.to_csv(f'{OUT_DIR}/01_rotation_frequency.csv', index=False)
clean = sp[~sp.circumference_outlier]
print(f'[Step 1] runs = {len(sp)}, 주파수 해상도 = {FS / sp.n.median():.4f} Hz, '
      f'둘레 이상치 {int(sp.circumference_outlier.sum())}개 제외')
print(f'[Step 1] 역산 둘레 = {clean.implied_circumference_mm.mean():.1f} mm '
      f'(표준편차 {clean.implied_circumference_mm.std():.1f}), 지름 = {clean.implied_diameter_mm.mean():.1f} mm')

# =============================================================================
# 그림 1. 속도별 스펙트럼 — 주기 성분이 존재하고 그 주파수가 속도에 비례하는가
# =============================================================================
fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.8), layout='constrained')
for i, V in enumerate(sorted(sp.V.unique())):
    f, P, rid = spec_cache[f'V{V}']
    m = (f > 0) & (f < 0.45)
    ax.semilogy(f[m], P[m] / P[m].max(), color=SPEED_COLORS[i], linewidth=1.0, label=f'{V} mm/s')
    fp, _ = peak_in_band(f, P)
    ax.plot([fp], [1.0], marker='v', color=SPEED_COLORS[i], markersize=5, clip_on=False)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Normalized power (log)')
ax.set_xlim(0, 0.45)
ax.legend(title='Web speed setting', ncol=4, loc='lower center', frameon=False, fontsize=FONT_SIZE - 2,
          title_fontsize=FONT_SIZE - 2)
save(fig, '01_spectrum_by_speed')

# =============================================================================
# 그림 2. 리와인더 롤 가설 검증 — V=50 에서 f_peak 가 1/D 에 비례하는가
# =============================================================================
g = sp[sp.V == 50].sort_values('rewinder_diameter_mean_mm')
fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 2.6), layout='constrained')
ax.plot(g.rewinder_diameter_mean_mm, g.predicted_f_rewinder_Hz, color=ACCENT, linewidth=1.5,
        linestyle='--', label='Predicted if rewinder roll:  $f=V/\\pi D$')
ax.plot(g.rewinder_diameter_mean_mm, g.f_peak_Hz, color='#0072b2', marker='o', markersize=4,
        linestyle='none', label='Measured $f_{peak}$')
ax.set_xlabel('Rewinder roll diameter, run mean (mm)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'Web speed = 50 mm/s, {len(g)} runs')
ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper right')
save(fig, '02_rewinder_hypothesis_test')
print(f'[그림 2] corr(f_peak, 1/D) = {np.corrcoef(g.f_peak_Hz, 1 / g.rewinder_diameter_mean_mm)[0, 1]:+.3f}, '
      f'f_peak 변동계수 {g.f_peak_Hz.std() / g.f_peak_Hz.mean():.3f} vs 예측 '
      f'{g.predicted_f_rewinder_Hz.std() / g.predicted_f_rewinder_Hz.mean():.3f}')

# =============================================================================
# 그림 3. 역산 둘레의 일관성 — 고정 지름 롤러인가 (전 210 run)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.6), layout='constrained')
for i, V in enumerate(sorted(sp.V.unique())):
    gg = sp[sp.V == V]
    axes[0].plot(gg.V + np.linspace(-3, 3, len(gg)), gg.implied_circumference_mm,
                 marker='o', markersize=3, linestyle='none', color=SPEED_COLORS[i])
C = clean.implied_circumference_mm.mean()                         # 이상치 제외 평균
axes[0].axhline(C, color=ACCENT, linewidth=1.2, linestyle='--', label=f'mean = {C:.0f} mm (outliers excluded)')
axes[0].set_xlabel('Web speed setting (mm/s)')
axes[0].set_ylabel('Implied circumference $V/f_{peak}$ (mm)')
axes[0].legend(frameon=False, fontsize=FONT_SIZE - 2)
axes[1].plot(clean.rewinder_diameter_mean_mm, clean.implied_diameter_mm, marker='o', markersize=3,
             linestyle='none', color='#0072b2', label=f'{len(clean)} runs (outliers excluded)')
lim = [120, 300]
axes[1].plot(lim, lim, color=INK, linewidth=0.8, linestyle=':', label='If rewinder roll (y = x)')
axes[1].set_xlim(lim)
axes[1].set_xlabel('Rewinder roll diameter, run mean (mm)')
axes[1].set_ylabel('Implied diameter $V/(\\pi f_{peak})$ (mm)')
axes[1].legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper left')
save(fig, '03_circumference_consistency')

# =============================================================================
# 그림 4. 창 길이 하한 근거 — 창이 1주기를 담지 못하는 속도에서 오염률이 올라가는가
# =============================================================================
W_FIXED = 64                                                      # 현재 MovStd-Slope 고정창 (임의로 정한 값)
W_RULE = int(np.ceil(FS / (50.0 / C)))                            # 규칙 1b 의 창 길이
per = clean.groupby('V').period_samples.median()
cont = {}
for lbl, stem in [('MovStd-Slope (fixed, W=64)', '01_stat_ssd/MovStd-Slope-fixed'),
                  ('MovStd-Slope (speed-scaled)', '01_stat_ssd/MovStd-Slope-speed')]:
    d = pd.read_csv(f'{BASE_DIR}/{stem}_run_metrics.csv')
    cont[lbl] = d.groupby(VS).contamination_rate.mean()
fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH * 0.72, 4.0), layout='constrained', sharex=True)
axes[0].axhline(1.0, color=ACCENT, linewidth=1.2, linestyle='--', label='One full period (criterion)')
axes[0].plot(per.index, W_FIXED / per.to_numpy(), color='#009e73', marker='s', markersize=5,
             label=f'Fixed window W={W_FIXED}')
axes[0].plot(per.index, W_RULE / per.to_numpy(), color='#0072b2', marker='^', markersize=5,
             label=f'Rule-1b window W={W_RULE}')
axes[0].set_ylabel('Periods captured\nby the window')
axes[0].legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper left')
for (lbl, s), col, mk in zip(cont.items(), ['#009e73', '#005a00'], ['s', 'D']):
    axes[1].plot(s.index, s.to_numpy(), color=col, marker=mk, markersize=5, label=lbl)
axes[1].set_xlabel('Web speed setting (mm/s)')
axes[1].set_ylabel('Contamination rate\n(run mean, measured)')
axes[1].legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper right')
save(fig, '04_window_lower_bound')

# =============================================================================
# 그림 5. 창 길이 상한 방향 근거 — 창이 길면 steady 진입이 늦어지는가 (CoST 실측)
# =============================================================================
ent = []
for p in sorted(__import__('glob').glob(f'{BASE_DIR}/04_cost/CoST-W*_fold_metrics.csv')):
    w = int(re.search(r'CoST-W(\d+)', os.path.basename(p)).group(1))
    d = pd.read_csv(p)
    ent.append({'W': w, 'bias': d.runmean_entry_err_s.mean(), 'mae': d.runmae_entry_err_s.mean()})
ent = pd.DataFrame(ent).sort_values('W')
fig, ax = plt.subplots(figsize=(PAGE_WIDTH * 0.62, 2.6), layout='constrained')
ax.axhline(0, color=INK, linewidth=0.8, linestyle=':')
ax.plot(ent.W, ent.bias, color='#d55e00', marker='<', markersize=5, label='Entry-time bias (signed)')
ax.plot(ent.W, ent.mae, color='#873f51', marker='*', markersize=7, label='Entry-time MAE')
ax.set_xlabel('CoST sliding-window length (samples)')
ax.set_ylabel('Steady-entry error (s)')
ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper left')
save(fig, '05_window_upper_bound')

# =============================================================================
# Step 2. 창 길이 유도 결과를 출력한다
# =============================================================================
f_min = 50.0 / C
print('\n' + '=' * 72)
print(f'역산 둘레 C         = {C:.1f} mm  (지름 {C / np.pi:.1f} mm, {len(clean)} run 표준편차 {clean.implied_circumference_mm.std():.1f} mm)')
print(f'최저 속도 V_min     = 50 mm/s')
print(f'최저 회전 주파수    = V_min / C = {f_min:.4f} Hz  (주기 {1 / f_min:.1f} s)')
print(f'창 길이 W           = ceil(fs / f_min) = {W_RULE} samples ({W_RULE / FS:.1f} s)')
print(f'주파수 해상도       = fs / n = {FS / clean.n.median():.4f} Hz  -> 둘레 +-{FS / clean.n.median() / f_min * 100 / 2:.0f}%, '
      f'W +-{int(round(W_RULE * FS / clean.n.median() / f_min / 2))} samples')
print('=' * 72)
print('[done] 주기 외란 분석 완료')
