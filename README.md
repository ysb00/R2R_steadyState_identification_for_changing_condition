# R2R steady-state identification for changing condition

Roll-to-Roll 웹 공정에서 **운전 조건이 바뀔 때** 필름 장력의 정상상태(steady state)를 식별하는
연구 저장소. 최종 목표는 **저속 데이터만 학습하고 미학습 고속 신호에 적용**하는 것이다.

- **원시 데이터는 추적하지 않는다** (`data/rawdata/`, `.gitignore`). 병합 결과만 `outputs/` 에 둔다.
- **외부 비교군 저장소는 clone 만 하고 추적하지 않는다.** 커밋 hash 는
  `code/03_baselines/external_versions.csv` 에 기록한다. **원본은 한 줄도 수정하지 않고** 필요한
  변경은 런타임 shim 으로 처리한다.
- 모든 설정값은 출처(`[원논문]`/`[저장소]`/`[데이터]`/`[구조]`/`[임의]`)와 함께
  `docs/hyperparameters.md` 에 기록한다.

## 문서 — 여기서 시작

| 문서 | 내용 |
|---|---|
| **`docs/research_plan.md`** | 연구 전제·가정·한계, 진행 현황(9절), 결정이 필요한 사항 |
| **`docs/hyperparameters.md`** | **설정값 전체 + 열려 있는 문제(§11) + 앞으로 할 일(§13)**. §13.0 이 최종 목표 |
| `docs/research_experiments_plan.md` | 비교군 실험 설계, ablation 설계 |
| **`docs/baseline_cases.md`** | **비교군 23종이 각각 무엇인가** — 무엇/왜/설정 출처/주의사항 |
| `docs/outputs_guide.md` | 지표 정의, CSV 컬럼, 그림 읽는 법, 방법별 로직 |
| `docs/periodic_disturbance.md` | 주기 외란의 출처와 창 길이 유도(규칙 1b) |
| `docs/tstcc_config_choice.md` | TS-TCC config 를 Epilepsy 로 택한 근거 |
| `docs/data_dictionary.md` | 컬럼 설명 |
| `docs/coding_guidelines.md` | 코드 작성 규칙 (재현성·공정성·플롯) |
| `docs/worklog.md` | 날짜별 작업·결정 이력 |

## 데이터

210 run = **Film Tension 5수준 x ReWinder Tension 6수준 x Web Speed 7수준**, 조건당 run 정확히 1개.
run 길이 1,275~1,300 샘플 (약 140 s, 9.174 Hz). 입력은 단일 채널
`e = Film Tension #1 - Film-Tension Setting Value`.

## 평가 — 트랙 A (라벨 없음)

```
표현  ->  2-군집 GMM (학습 fold)  ->  |e 평균| 작은 군집 = steady  ->  판정
                                                     채점: Trigger 4 (ambiguous 제외)
```

전 방법이 `code/03_baselines/evaluation.py` 하나를 공유한다. **`Trigger 4` 라벨을 읽는 코드는
그 파일의 `_labels()` 한 곳뿐**이고, 비교군 어댑터는 `usecols` 에서 라벨을 제외한다.

## 실험

| 실험 | 질문 | 분할 | 상태 |
|---|---|---|---|
| **A** | 새 **장력** 설정값에 일반화되는가 | 210 run, 장력 leave-one-level-out 5-fold x seed 5 | ✅ **완료** (비교군 8개 / 변형 23종) |
| **C** | **저속만 학습**해도 **미학습 고속**에 되는가 | 저속 50~125 학습+CV / 고속 150~200 test | ⬜ 설계 확정, 미실행 (`hyperparameters.md` §13.0) |

## 코드

```
code/01_merge.py                  원시 zip -> 병합 CSV
code/02_eda/01_rotation_frequency.py   주기 외란 분석 -> 창 길이 규칙 1b
code/03_baselines/
  00_folds.py          fold 배정 (1회 생성 후 고정)
  evaluation.py        공통 평가 — 트랙 A 후단·채점·그림. 라벨을 읽는 유일한 지점
  01_stat_ssd.py       R-test 4종 + 이동창 std/기울기 4종
  02_ts2vec.py  03_moment.py  04_cost.py  05_softclt.py  07_tstcc.py  08_simmtm.py
  06_cost_ablation.py  CoST 성분 분해 2종 + 증강 제거 1종
  09_summary_tables.py 집계 표 4개 생성
```

## 결과

```
outputs/03_baselines/_paper_table.csv              논문 표에 바로 붙이는 형태 (평균±표준편차)
outputs/03_baselines/_all_methods_full_table.csv   전 지표 + 설정값
outputs/03_baselines/_all_methods_by_fold.csv      방법 x fold
outputs/03_baselines/_all_methods_by_fold_seed.csv 방법 x fold x seed
outputs/03_baselines/<단계>/<방법>_*.csv|png|pdf   방법별 run/fold/summary/config + heatmap + 손실곡선
```

읽는 법은 `docs/outputs_guide.md`, 각 행이 무엇인지는 `docs/baseline_cases.md`.

## 현재 결론 (A 실험)

**가장 강한 비교군은 CoST (창 144)** 이고 AUPRC(ns) 0.832±0.147 이다. 그 다음이
**2차원 통계 기준선(이동창 std+기울기)** 으로 0.718±0.049 — 표현학습 7종 중 CoST·SimMTM 을 뺀
나머지는 이 단순 기준선보다 못하다. Ablation 에서 **CoST 의 scaling/shift 증강을 제거하면
전 종합 지표가 개선**된다(0.853±0.157).

**알려진 한계**: 저장소 기본값을 그대로 쓰는 As-published 방침 때문에 **데이터 통과 횟수가
방법마다 3.17~38.1회로 다르고, 통과 횟수와 AUPRC 가 거의 역순**이다 — 방법 간 성능차에 학습량
효과가 섞여 있다 (`hyperparameters.md` §11 문제 2·9, §13.2).
