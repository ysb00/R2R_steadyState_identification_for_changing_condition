# =============================================================================
# 01_merge.py : 데이터 준비 단계 (zip 5개 → 하나의 병합 데이터)
# 입력 : data/data_tensionSetting_{3..7}.zip  (각 zip = xlsx 42개, 1 xlsx = 1 운전조건 run)
# 출력 : outputs/merged_data.csv.gz      - 병합 데이터 (git 미추적, 이 스크립트로 재생성)
#        outputs/01_file_summary.csv     - run(파일)별 요약 표 (210행)
# 사용 컬럼 (연구자 지정 6종) → 짧은 이름으로 변경
#   Trigger 4                          → label   (0: 미안정, 1: steady state, 2: ambiguous; expert 수동 라벨)
#   Film Tension #1 (kg)               → T1      (target / monitoring 신호)
#   Film-Tension Setting Value (kg)    → T_set   (설정 조건)
#   ReWinder-Tension Setting Value (kg)→ R_set   (설정 조건)
#   Web Speed setting value (mm/sec)   → V_set   (설정 조건)
#   Web Current Speed (mm/sec)         → V_act   (실제 속도, 활용 여부 미정)
# 실행 : python code/01_merge.py   (repo 루트에서)
# =============================================================================
import zipfile, io, re, os
import pandas as pd
import numpy as np

COLS = {'Trigger 4': 'label', 'Film Tension #1 (kg)': 'T1',
        'Film-Tension Setting Value (kg)': 'T_set', 'ReWinder-Tension Setting Value (kg)': 'R_set',
        'Web Speed setting value (mm/sec)': 'V_set', 'Web Current Speed (mm/sec)': 'V_act'}
os.makedirs('outputs', exist_ok=True)

# --- Step 1. zip 안의 xlsx를 메모리에서 읽어 6개 컬럼만 추출 + run 식별자/샘플 순번(k) 부여 ---
runs = []
for t in [3, 4, 5, 6, 7]:
    z = zipfile.ZipFile(f'data/data_tensionSetting_{t}.zip')
    for name in z.namelist():
        d = pd.read_excel(io.BytesIO(z.read(name)), usecols=list(COLS)).rename(columns=COLS)
        T, R, V = map(int, re.findall(r'=\s*(\d+)', name))       # 파일명의 조건값 (설정값 컬럼과 일치하는지 Step 2에서 확인)
        d.insert(0, 'run_id', f'T{T}_R{R}_V{V}')
        d.insert(1, 'k', np.arange(len(d)))                       # run 내부 샘플 순번 (약 0.109 s 간격)
        d['fname_T'], d['fname_R'], d['fname_V'] = T, R, V
        runs.append(d)
    print(f'[Step 1] tension={t}: {len(z.namelist())} files read')
df = pd.concat(runs, ignore_index=True)
print(f'[Step 1] merged shape = {df.shape}, runs = {df.run_id.nunique()}')

# --- Step 2. 정합성 확인: 파일명 조건 == 설정값 컬럼, run 내 설정값 상수 여부, 결측치 ---
mismatch = ((df.fname_T != df.T_set) | (df.fname_R != df.R_set) | (df.fname_V != df.V_set)).sum()
nonconst = (df.groupby('run_id')[['T_set', 'R_set', 'V_set']].nunique() > 1).sum().sum()
print(f'[Step 2] filename-vs-setting mismatch rows = {mismatch}, non-constant setting runs = {nonconst}')
print(f'[Step 2] missing values per column:\n{df.isna().sum().to_string()}')
df = df.drop(columns=['fname_T', 'fname_R', 'fname_V'])
print(f'[Step 2] label share (%): {(df.label.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()}')

# --- Step 3. 병합 데이터 저장 ---
df = df.sort_values(['T_set', 'R_set', 'V_set', 'k']).reset_index(drop=True)
df.to_csv('outputs/merged_data.csv.gz', index=False)
print(f'[Step 3] saved outputs/merged_data.csv.gz ({os.path.getsize("outputs/merged_data.csv.gz") / 1e6:.1f} MB)')

# --- Step 4. run별 요약 표: 길이, 라벨 수, 첫 steady 시점, steady 구간의 T1 오차 통계 ---
df['e'] = df.T1 - df.T_set                                        # 설정값 대비 장력 오차
g = df.groupby('run_id')
summ = g[['T_set', 'R_set', 'V_set']].first()
summ['n'] = g.size()
for c in [0, 1, 2]:
    summ[f'n_label{c}'] = g.label.apply(lambda s, c=c: (s == c).sum())
summ['first_steady_k'] = df[df.label == 1].groupby('run_id').k.min()
summ['first_steady_sec'] = (summ.first_steady_k * 0.109).round(1)  # 평균 샘플 간격 0.109 s 가정 (근사값)
summ['unstable_after_steady'] = df.merge(summ.first_steady_k, on='run_id').query('label == 0 and k > first_steady_k').groupby('run_id').size()
summ['unstable_after_steady'] = summ.unstable_after_steady.fillna(0).astype(int)
summ['e_mean_steady'] = df[df.label == 1].groupby('run_id').e.mean().round(4)
summ['e_std_steady'] = df[df.label == 1].groupby('run_id').e.std().round(4)
summ['e_min_unstable'] = df[df.label == 0].groupby('run_id').e.min().round(4)
summ = summ.sort_values(['T_set', 'R_set', 'V_set'])
summ.to_csv('outputs/01_file_summary.csv')
print(f'[Step 4] saved outputs/01_file_summary.csv {summ.shape}')
print(summ.describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())
print('[Step 4] runs with unstable(0) after first steady:', summ.index[summ.unstable_after_steady > 0].tolist())
