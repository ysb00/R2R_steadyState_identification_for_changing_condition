# Data Dictionary

생성 스크립트: `code/01_merge.py`
원본: `data/data_tensionSetting_{3..7}.zip` (zip 5개 × xlsx 42개 = 210 run, 1 xlsx = 1 운전조건)

---

## 1. `outputs/merged_data.csv`

268,933 행 × 8 열. 원본 컬럼 6개는 **이름을 변경하지 않고** 그대로 사용하며, run 구분용 컬럼 2개를 추가했다.
정렬 순서: `Film-Tension Setting Value (kg)` → `ReWinder-Tension Setting Value (kg)` → `Web Speed setting value (mm/sec)` → `k`

### 1.1 원본 컬럼 (연구자 지정)

| 컬럼 | 역할 | 설명 |
|---|---|---|
| `Trigger 4` | label | expert 수동 라벨. 0 = 장비 신호가 아직 안정화되지 않은 영역, 1 = 장비가 steady state에 진입한 영역, 2 = ambiguous(라벨 부여가 모호). 제품 품질의 이상/정상을 뜻하지 않음 |
| `Film Tension #1 (kg)` | target | monitoring 대상 실측 신호 |
| `Film-Tension Setting Value (kg)` | 설정 조건 | 3, 4, 5, 6, 7 (run 내 상수) |
| `ReWinder-Tension Setting Value (kg)` | 설정 조건 | 2, 3, 4, 5, 6, 7 (run 내 상수) |
| `Web Speed setting value (mm/sec)` | 설정 조건 | 50, 75, 100, 125, 150, 175, 200 (run 내 상수) |
| `Web Current Speed (mm/sec)` | 실측 속도 | 활용 여부 미정 |

### 1.2 추가 컬럼

| 컬럼 | 설명 |
|---|---|
| `run_id` | run(원본 xlsx 1개) 식별자. 형식 `T{Film-Tension 설정}_R{ReWinder-Tension 설정}_V{Web Speed 설정}`. 예: `T3_R2_V50` |
| `k` | run 내부 샘플 순번 (0부터). 샘플 간격은 약 0.109 s (원본 `Date` 컬럼 기준 근사값, 0.099–0.119 s 범위) |

---

## 2. `outputs/01_file_summary.csv`

210 행(1 행 = 1 run). 설정값 컬럼 3개는 원본 이름을 사용한다.
이하에서 e = `Film Tension #1 (kg)` − `Film-Tension Setting Value (kg)` (설정값 대비 장력 오차).

| 컬럼 | 설명 |
|---|---|
| `run_id` | 위와 동일 |
| `n` | run의 샘플 수 (1275–1300) |
| `n_label0` | `Trigger 4` = 0 (미안정) 샘플 수. 0 라벨이 없는 run: 26개 |
| `n_label1` | `Trigger 4` = 1 (steady state) 샘플 수 |
| `n_label2` | `Trigger 4` = 2 (ambiguous) 샘플 수. 2 라벨이 없는 run: 6개 |
| `first_steady_k` | `Trigger 4`가 처음 1이 되는 `k`, 즉 expert가 판단한 steady state 진입 시점 |
| `first_steady_sec` | `first_steady_k` × 0.109 s. **근사값** (평균 샘플 간격 가정) |
| `unstable_after_steady` | 첫 steady 이후에 다시 0으로 표시된 샘플 수. > 0 이면 steady 진입 후 재불안정 run (`T3_R7_V200`, `T5_R6_V50`, `T6_R7_V50`) |
| `e_mean_steady` | steady(1) 구간 e의 평균: 설정값 대비 편향 |
| `e_std_steady` | steady(1) 구간 e의 표준편차: steady state의 변동 크기 |
| `e_min_unstable` | 미안정(0) 구간 e의 최솟값: 설정값 대비 최대 처짐. 0 라벨이 없는 run(26개)은 빈칸 |
