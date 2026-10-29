# `07_proposed_final_v1/code/` — Proposed (final v1) 코드 사본

final v1 과 표 2종을 재현하는 데 필요한 파이썬 코드다.

**정본은 저장소의 `code/` 아래이고 여기는 사본**이므로 수정은 정본에서 한다.

| 파일 | 역할 |
|---|---|
| `04_cost_dual_trunk.py` | ★ final v1 본체 — `nrot=1 freeze` 조합이 final v1 (정본 `code/04_proposed/04_cost_dual_trunk.py`) |
| `evaluation.py` | 공통 평가 후단 (GMM·채점·그림). 라벨을 읽는 유일한 지점 (정본 `code/03_baselines/evaluation.py`) |
| `splits.py` | 분할 해석 (A/C, ambiguous 모드) (정본 `code/03_baselines/splits.py`) |
| `06_final_v1_tables.py` | 표 2종과 이 사본들을 만든 스크립트 (정본 `code/04_proposed/06_final_v1_tables.py`) |

**외부 저장소(`code/03_baselines/external/CoST`)는 복사하지 않았다** — 추적 대상이 아니다.
쓴 커밋 hash 는 각 방법의 `_config.csv` 의 `repo` / `commit` 열에 있다.

### final v1 재현

```
python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze
python code/04_proposed/06_final_v1_tables.py
```
