# =============================================================================
# 03_baselines/00_folds.py : 교차검증 fold 배정 (제안 모델·전 비교군 공용, 1회 생성 후 고정)
# 입력 : outputs/merged_data.csv  (run_id + 설정값 3종만 사용)
# 출력 : outputs/03_baselines/folds.csv            - A 실험. run 210개의 fold 배정
#        outputs/03_baselines/00_folds_config.csv  - 실행에 사용한 설정값 기록
#        outputs/04_speed_extrap/folds_c.csv       - C 실험. fold 별 run 역할 (train/val/unused)
#        outputs/04_speed_extrap/00_folds_c_config.csv / _summary.csv
# 배정 규칙 (docs/research_experiments_plan.md 4절, docs/hyperparameters.md 13.0)
#
# [A 실험] 조건 축 = 장력. 210 run 전부 쓴다.
#   fold_tension : 주 실험. Film Tension 설정값 수준별 leave-one-level-out 5-fold (검증 42 / 학습 168 runs)
#   fold_speed   : 보조 실험. Web Speed 설정값 수준별 leave-one-level-out 7-fold (검증 30 / 학습 180 runs)
#                  **미실행** — 내삽이라 C 실험의 목표(고속 외삽)를 달성하지 못한다 (13.0)
#   각 run은 실험별로 정확히 1개 fold의 '검증' 쪽에 속한다. 즉 fold_* == f 인 run이 fold f의 검증 set,
#   나머지 전부가 학습 set 이다. 검증 조건의 설정값 수준이 학습에 나타나지 않도록 수준 단위로 끊는다.
#
# [C 실험] 조건 축 = 속도. **저속만 학습**하고 고속은 최종 test 에만 쓴다 (3분할).
#   저속 50/75/100/125 (120 run) = 학습 + CV / 고속 150/175/200 (90 run) = 최종 test
#   fold 8개 = 장력 LOO 5 + 속도 누적 외삽 2 + 최종 test 1. 방법당 8 fold x 5 seed = 40회.
#   여기서는 "학습 = 나머지" 가 성립하지 않는다 — CV fold 에서 고속 run 은 학습도 검증도 아니다.
#   그래서 fold 마다 run 의 역할을 train / val / unused 로 **명시**해 저장한다 (role__* 컬럼).
#
# 난수 없음 (설정값 수준으로 결정되는 결정론적 배정) -> 랜덤 시드 불필요
# 실행 : python code/03_baselines/00_folds.py   (repo 루트에서)
# =============================================================================
import os, time
import numpy as np
import pandas as pd

t0 = time.time()

# --- hyperparameter : fold 순서를 정하는 설정값 수준 목록 (데이터에 존재하는 수준 전체) ---
TENSION_LEVELS = [3, 4, 5, 6, 7]                       # 주 실험 fold 1~5 에 대응하는 Film Tension 설정값 (kg)
SPEED_LEVELS = [50, 75, 100, 125, 150, 175, 200]       # 보조 실험 fold 1~7 에 대응하는 Web Speed 설정값 (mm/sec)

# --- C 실험 (저속 학습 -> 고속 테스트) 의 속도 경계. 연구자 확정 2026-10-26 (worklog 9절) ---
LOW_SPEEDS = [50, 75, 100, 125]                        # 학습 + CV 에만 쓴다 (120 run)
HIGH_SPEEDS = [150, 175, 200]                          # 최종 test 에만 쓴다 (90 run)

# --- C 실험 CV 의 속도 외삽 fold. "누적 외삽" (연구자 결정 2026-10-26) ---
#     검증을 항상 '학습보다 빠른 속도' 로 두고 외삽 거리를 1수준(25 mm/s)으로 고정한다.
#     최종 test(저속 4수준 -> 고속 3수준)와 구조가 같아, CV 가 test 성능을 예측하는 근거가 된다.
SPEEDEXT_FOLDS = [('S1', [50, 75], [100]),             # (이름, 학습 속도, 검증 속도)
                  ('S2', [50, 75, 100], [125])]

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = 'outputs/03_baselines'
OUT_DIR_C = 'outputs/04_speed_extrap'                  # C 실험 결과를 A 와 섞지 않는다 (연구자 지시)
ROLE_PREFIX = 'role__'                                 # splits.py 가 읽는 역할 컬럼 접두사

TS = 'Film-Tension Setting Value (kg)'                 # 원본 컬럼명 (변경하지 않음)
RS = 'ReWinder-Tension Setting Value (kg)'
VS = 'Web Speed setting value (mm/sec)'

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(OUT_DIR_C, exist_ok=True)

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

# =============================================================================
# Step 6. C 실험 (저속 학습 -> 고속 테스트) 의 fold 별 역할표
#         A 와 달리 "학습 = 나머지" 가 성립하지 않으므로 run 마다 train/val/unused 를 명시한다.
# =============================================================================
c = runs[['run_id', TS, RS, VS]].copy()
c['speed_role'] = c[VS].map({**{v: 'low' for v in LOW_SPEEDS}, **{v: 'high' for v in HIGH_SPEEDS}})
assert c.speed_role.notna().all(), f'속도 수준 미분류: {sorted(set(c[VS]) - set(LOW_SPEEDS) - set(HIGH_SPEEDS))}'
is_low = (c.speed_role == 'low').to_numpy()

# (a) CV 1 — 저속 안에서 장력 leave-one-level-out 5-fold. 고속은 전부 unused.
#     A 실험과 같은 질문(새 장력 수준)을 저속 데이터만으로 묻는다. 학습 96 / 검증 24.
for i, lv in enumerate(TENSION_LEVELS):
    role = np.where(~is_low, 'unused', np.where(c[TS].to_numpy() == lv, 'val', 'train'))
    c[f'{ROLE_PREFIX}cv_T{i + 1}'] = role

# (b) CV 2 — 저속 안에서 속도 누적 외삽 2-fold. 검증은 항상 학습보다 빠르다.
for name, tr_v, va_v in SPEEDEXT_FOLDS:
    v = c[VS].to_numpy()
    role = np.where(np.isin(v, va_v), 'val', np.where(np.isin(v, tr_v), 'train', 'unused'))
    c[f'{ROLE_PREFIX}cv_{name}'] = role

# (c) 최종 test — 저속 전부로 학습하고 고속 전부로 평가한다. **한 번만 쓴다.**
c[f'{ROLE_PREFIX}final_test'] = np.where(is_low, 'train', 'val')

# --- Step 7. 역할표 검증 : 겹침·누락이 없는지 확인하고 fold 별 run 수를 센다 ---
role_cols = [k for k in c.columns if k.startswith(ROLE_PREFIX)]
assert c[role_cols].isin(['train', 'val', 'unused']).all().all(), '역할 값 오류'
for k in role_cols:                                     # 학습과 검증은 속도 수준이 겹치면 안 된다
    tr_v, va_v = set(c.loc[c[k] == 'train', VS]), set(c.loc[c[k] == 'val', VS])
    assert not (tr_v & va_v) or k.startswith(f'{ROLE_PREFIX}cv_T'), f'{k}: 속도 수준 누출 {tr_v & va_v}'
assert (c[f'{ROLE_PREFIX}final_test'] == 'val').sum() == len(HIGH_SPEEDS) * 30, '최종 test run 수 오류'

summary = pd.DataFrame([{'fold': k[len(ROLE_PREFIX):],
                         'n_train': int((c[k] == 'train').sum()),
                         'n_val': int((c[k] == 'val').sum()),
                         'n_unused': int((c[k] == 'unused').sum()),
                         'train_speeds': sorted(set(c.loc[c[k] == 'train', VS])),
                         'val_speeds': sorted(set(c.loc[c[k] == 'val', VS]))} for k in role_cols])
print('[Step 7] C 실험 fold:\n' + summary.to_string(index=False))

# --- Step 8. C 실험 분할 저장 ---
c.to_csv(f'{OUT_DIR_C}/folds_c.csv', index=False)
summary.to_csv(f'{OUT_DIR_C}/00_folds_c_summary.csv', index=False)
pd.DataFrame({'name': ['LOW_SPEEDS', 'HIGH_SPEEDS', 'TENSION_LEVELS', 'SPEEDEXT_FOLDS',
                       'n_folds', 'n_low_runs', 'n_high_runs', 'IN_PATH'],
              'value': [str(LOW_SPEEDS), str(HIGH_SPEEDS), str(TENSION_LEVELS), str(SPEEDEXT_FOLDS),
                        len(role_cols), int(is_low.sum()), int((~is_low).sum()), IN_PATH]}
             ).to_csv(f'{OUT_DIR_C}/00_folds_c_config.csv', index=False)
print(f'[Step 8] saved {OUT_DIR_C}/folds_c.csv {c.shape}, fold {len(role_cols)}개 '
      f'(방법당 {len(role_cols)} fold x seed)')
print(f'[done] elapsed = {time.time() - t0:.1f} s')
