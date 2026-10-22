# =============================================================================
# 03_baselines/00_folds.py : 교차검증 fold 배정 (제안 모델·전 비교군 공용, 1회 생성 후 고정)
# 입력 : outputs/merged_data.csv  (run_id + 설정값 3종만 사용)
# 출력 : outputs/03_baselines/folds.csv            - run 210개의 fold 배정 (이 파일 하나만 읽는다)
#        outputs/03_baselines/00_folds_config.csv  - 실행에 사용한 설정값 기록
# 배정 규칙 (docs/research_experiments_plan.md 4절)
#   fold_tension : 주 실험. Film Tension 설정값 수준별 leave-one-level-out 5-fold (검증 42 / 학습 168 runs)
#   fold_speed   : 보조 실험. Web Speed 설정값 수준별 leave-one-level-out 7-fold (검증 30 / 학습 180 runs)
#   각 run은 실험별로 정확히 1개 fold의 '검증' 쪽에 속한다. 즉 fold_* == f 인 run이 fold f의 검증 set,
#   나머지 전부가 학습 set 이다. 검증 조건의 설정값 수준이 학습에 나타나지 않도록 수준 단위로 끊는다.
#   난수 없음 (설정값 수준으로 결정되는 결정론적 배정) -> 랜덤 시드 불필요
# 실행 : python code/03_baselines/00_folds.py   (repo 루트에서)
# =============================================================================
import os, time
import pandas as pd

t0 = time.time()

# --- hyperparameter : fold 순서를 정하는 설정값 수준 목록 (데이터에 존재하는 수준 전체) ---
TENSION_LEVELS = [3, 4, 5, 6, 7]                       # 주 실험 fold 1~5 에 대응하는 Film Tension 설정값 (kg)
SPEED_LEVELS = [50, 75, 100, 125, 150, 175, 200]       # 보조 실험 fold 1~7 에 대응하는 Web Speed 설정값 (mm/sec)

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = 'outputs/03_baselines'

TS = 'Film-Tension Setting Value (kg)'                 # 원본 컬럼명 (변경하지 않음)
RS = 'ReWinder-Tension Setting Value (kg)'
VS = 'Web Speed setting value (mm/sec)'

os.makedirs(OUT_DIR, exist_ok=True)

# --- Step 1. run 목록 추출 (run_id 1개 = 운전조건 1개, 설정값은 run 내 상수) ---
df = pd.read_csv(IN_PATH, usecols=['run_id', TS, RS, VS])
runs = df.drop_duplicates('run_id').reset_index(drop=True)
print(f'[Step 1] runs = {len(runs)}, 조건 격자 = {runs[TS].nunique()} x {runs[RS].nunique()} x {runs[VS].nunique()}')

# --- Step 2. fold 배정 : 설정값 수준 -> fold 번호 (1부터) ---
runs['fold_tension'] = runs[TS].map({v: i + 1 for i, v in enumerate(TENSION_LEVELS)})
runs['fold_speed'] = runs[VS].map({v: i + 1 for i, v in enumerate(SPEED_LEVELS)})
print(f'[Step 2] 미배정 run = fold_tension {runs.fold_tension.isna().sum()}, fold_speed {runs.fold_speed.isna().sum()}')

# --- Step 3. 배정 결과 확인 : fold별 검증 run 수와 검증 수준 ---
for col, levels, setting in [('fold_tension', TENSION_LEVELS, TS), ('fold_speed', SPEED_LEVELS, VS)]:
    g = runs.groupby(col)
    print(f'[Step 3] {col}: ' + ', '.join(
        f'fold{int(f)}({setting.split()[0]}={levels[int(f) - 1]}) 검증 {len(d)} / 학습 {len(runs) - len(d)}'
        for f, d in g))

# --- Step 4. fold 배정 저장 ---
runs = runs.sort_values([TS, RS, VS]).reset_index(drop=True)
runs.to_csv(f'{OUT_DIR}/folds.csv', index=False)
print(f'[Step 4] saved {OUT_DIR}/folds.csv {runs.shape}, columns = {runs.columns.tolist()}')

# --- Step 5. 사용한 설정값 기록 (결과-설정 추적) ---
pd.DataFrame({'name': ['TENSION_LEVELS', 'SPEED_LEVELS', 'IN_PATH', 'n_runs'],
              'value': [str(TENSION_LEVELS), str(SPEED_LEVELS), IN_PATH, len(runs)]}
             ).to_csv(f'{OUT_DIR}/00_folds_config.csv', index=False)
print(f'[Step 5] saved {OUT_DIR}/00_folds_config.csv')
print(f'[done] elapsed = {time.time() - t0:.1f} s')
