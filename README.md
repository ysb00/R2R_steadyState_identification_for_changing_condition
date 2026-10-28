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
| **C** | **저속만 학습**해도 **미학습 고속**에 되는가 | 저속 50~125 학습+CV(7 fold) / 고속 150~200 **final test** | ✅ **완료** (같은 23종, 8 fold x seed 5) |

**보조 평가 — ambiguous 포함** (2026-10-28). 위 두 실험은 `Trigger 4 == 2` (ambiguous, 6.6%)를
채점에서 뺀다. 같은 체크포인트로 **채점만 다시 해서** ambiguous 를 포함한 2클래스 결과도 둔다 —
ambiguous 를 **시간상 가장 가까운 라벨로 흡수**한다(앞뒤가 transient·steady 면 정확히 1/2 분할).
**재학습하지 않는다.** 규칙·주의사항은 `docs/hyperparameters.md` §13.9.

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
  09_summary_tables.py 집계 표 4개 생성 (A 전용)
  12_summary_tables_c.py  C 전용 집계 — 묶음을 가로질러 평균하지 않는다
  10_embeddings.py / 11_embedding_plots.py   임베딩 좌표 -> 그림
```

**실험은 인자로 고른다** — `python code/03_baselines/04_cost.py` (A, 기본값) /
`python code/03_baselines/04_cost.py c` (C). 분할 해석은 `splits.py` 한 곳에 모여 있다.

**분할 이름 뒤에 `amb` 를 붙이면** ambiguous 포함 재추론이다 — `... 04_cost.py c amb`.
학습을 생략하고 기존 체크포인트를 불러 쓰므로 **그 분할의 학습이 이미 끝나 있어야** 한다.
붙이지 않으면 기존 동작 그대로다 (기존 결과를 덮지 않는다).

```
# A 실험 전체 -> outputs/05_ambiguous_included_A/
for s in 01_stat_ssd 02_ts2vec 03_moment 04_cost 05_softclt 06_cost_ablation 07_tstcc 08_simmtm; do
  python code/03_baselines/$s.py tension amb
done
python code/03_baselines/09_summary_tables.py   amb      # 집계표 (방법이 전부 끝난 뒤)
python code/03_baselines/10_embeddings.py     tension amb
python code/03_baselines/11_embedding_plots.py tension amb

# C 실험 전체 -> outputs/06_ambiguous_included_C/   (12 는 A 집계표를 읽으므로 A 를 먼저 끝낼 것)
#   위와 같되 'tension' -> 'c', 09 대신 `python code/03_baselines/12_summary_tables_c.py amb`
```

## 결과

```
outputs/03_baselines/_paper_table.csv              논문 표에 바로 붙이는 형태 (평균±표준편차)
outputs/03_baselines/_all_methods_full_table.csv   전 지표 + 설정값
outputs/03_baselines/_all_methods_by_fold.csv      방법 x fold
outputs/03_baselines/_all_methods_by_fold_seed.csv 방법 x fold x seed
outputs/03_baselines/<단계>/<방법>_*.csv|png|pdf   방법별 run/fold/summary/config + heatmap + 손실곡선

outputs/04_speed_extrap/_paper_table_c.csv         C 논문 표 — 지표마다 cv_T / cv_S1 / cv_S2 / final_test 4열
outputs/04_speed_extrap/_all_methods_by_group.csv  방법 x 묶음 (묶음 안에서만 평균)
outputs/04_speed_extrap/_final_test_by_speed.csv   최종 test 를 속도 150/175/200 으로 쪼갠 표
outputs/04_speed_extrap/_final_test_by_speed.png   같은 것의 성능 곡선
outputs/04_speed_extrap/<단계>/<방법>_embedding_*.png  seen/unseen + 테스트 속도별 임베딩

outputs/05_ambiguous_included_A/   A 를 ambiguous 포함으로 재채점 (구조는 03_baselines 와 동일)
outputs/06_ambiguous_included_C/   C 를 ambiguous 포함으로 재채점 (구조는 04_speed_extrap 과 동일)
```

**ambiguous 포함 폴더의 AUPRC(ns) 는 기존 폴더와 직접 비교하면 안 된다** — 기준선이 유병률이라
non-steady 비율이 0.081 에서 0.106 으로 올라간다. 진입 오차도 참 진입 시점이 앞당겨져 달라진다.
학습을 하지 않으므로 그 폴더에는 손실 곡선과 학습량 컬럼이 없다.

**C 는 네 단위를 가로질러 평균하지 않는다** — `cv_T`(저속 안 새 장력) / `cv_S1`·`cv_S2`(1수준 속도
외삽, 학습량이 다름) / `final_test`(최종 목표)는 **묻는 질문이 서로 다르다.**

읽는 법은 `docs/outputs_guide.md`, 각 행이 무엇인지는 `docs/baseline_cases.md`.

## 현재 결론

### A 실험 (장력 축)

**가장 강한 비교군은 CoST (창 144)** 이고 AUPRC(ns) 0.832±0.147 이다. 그 다음이
**2차원 통계 기준선(이동창 std+기울기)** 으로 0.718±0.049 — 표현학습 7종 중 CoST·SimMTM 을 뺀
나머지는 이 단순 기준선보다 못하다. Ablation 에서 **CoST 의 scaling/shift 증강을 제거하면
전 종합 지표가 개선**된다(0.853±0.157).

### C 실험 (속도 축) — **최종 목표는 성립한다**

저속만 학습하고 미학습 고속에서 평가한 `final_test` 에서 **CoST-W144 가 AUPRC(ns) 0.907,
오염률 0.001, 진입 오차 4.2 s** 다. **순위 구조는 A 와 같다** — CoST 1위, 2차원 통계 기준선이
2위(0.847)로 표현학습 5종을 이긴다. 조건 축을 바꿔도 같은 관찰이 재현된다.

예상과 달랐던 것 2가지:

- **미학습 고속이 저속 CV 보다 오히려 쉽다** (23종 중 18종, 평균 +0.079). 미안정 구간 처짐이
  속도에 비례해 커져 non-steady 가 뚜렷해지는 것으로 보인다. 단 학습량도 96 -> 120 run 으로 달라
  교란이 섞여 있다.
- **어려운 조건은 속도 150 이었다.** `MovStd-Slope-fixed` 보존률이 150 에서 0.710, 175·200 에서
  0.971·0.984 인 반면 **CoST·SimMTM 은 세 속도 모두 0.98 이상**이다. 속도를 뭉뚱그려 보고했으면
  보이지 않았을 차이다.

### ambiguous 포함 재채점 — **결론이 바뀌지 않는다**

ambiguous 6.6% 를 빼고 채점한 것이 결과를 만든 것 아닌가 — 이 물음에 답하기 위해 같은
체크포인트로 ambiguous 를 포함해 다시 채점했다(2026-10-28, `hyperparameters.md` §13.9).

**23종 AUPRC 순위 상관이 A 0.989 / C final_test 0.981 로 순위 구조가 그대로다.**
CoST 계열이 상위, 2차원 통계 기준선이 그 뒤라는 관찰도 유지된다. 절대값은 채점 대상이
6.6% 늘면서 전반적으로 조금 낮아진다 — 예: `CoST-W144` final_test 오염률 0.001 → 0.007,
보존률 0.992 → 0.980, AUROC 0.989 → 0.964.

즉 **ambiguous 제외가 유리한 쪽으로 결과를 만든 것이 아니다.**

**알려진 한계**: 저장소 기본값을 그대로 쓰는 As-published 방침 때문에 **데이터 통과 횟수가
방법마다 3.17~38.1회로 다르고, 통과 횟수와 AUPRC 가 거의 역순**이다 — 방법 간 성능차에 학습량
효과가 섞여 있다 (`hyperparameters.md` §11 문제 2·9, §13.2).
