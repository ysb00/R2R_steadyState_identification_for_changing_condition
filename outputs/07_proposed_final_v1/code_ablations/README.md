# `07_proposed_final_v1/code_ablations/` — ablation (표 2) 코드 사본

표 (2) 의 7행을 만든 어댑터다. 파일 이름에 ` (ablation)` 을 붙였다.
괄호·공백 때문에 python import 가 되지 않으니 **실행은 정본 `code/` 에서 한다.**

**정본은 저장소의 `code/` 아래이고 여기는 사본**이므로 수정은 정본에서 한다.

| 파일 | 역할 |
|---|---|
| `04_cost (ablation).py` | `CoST` 행 (정본 `code/03_baselines/04_cost.py`) |
| `06_cost_ablation (ablation).py` | `CoST (no scale/shift)` 행 (= ablation 기준) (정본 `code/03_baselines/06_cost_ablation.py`) |
| `01_cost_order (ablation).py` | `+ order tracking` 행 (정본 `code/04_proposed/01_cost_order.py`) |
| `04_cost_dual_trunk (ablation).py` | `+ trunk separation` / `+ trunk separation + freeze` / `+ trunk separation + order tracking` / `Proposed method` 행 (정본 `code/04_proposed/04_cost_dual_trunk.py`) |

### 행별 재현 명령

| 표 (2) 의 행 | 명령 |
|---|---|
| CoST | `python code/03_baselines/04_cost.py` |
| CoST (no scale/shift) | `python code/03_baselines/06_cost_ablation.py` |
| + trunk separation | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 noorder` |
| + order tracking | `python code/04_proposed/01_cost_order.py tension nrot=1` |
| + trunk separation + freeze | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 noorder freeze` |
| + trunk separation + order tracking | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1` |
| Proposed method | `python code/04_proposed/04_cost_dual_trunk.py tension nrot=1 freeze` |

같은 명령을 다시 돌리면 `{방법}_fold_metrics.csv` 가 있으면 **건너뛴다**
(재실행하려면 먼저 지운다).
