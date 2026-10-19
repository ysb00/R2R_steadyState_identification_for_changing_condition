# Worklog

날짜별 작업 기록. 새 작업은 맨 위에 날짜 섹션을 추가한다.

---

## 2026-10-19 — 리포 구성 및 작업 방식 합의

연구 코드 작성 전 단계. 데이터 분석이나 실험은 아직 시작하지 않았다.

### 1. 로컬 작업 환경을 GitHub 리포와 연결

로컬 폴더 `c:\Users\KIMM\research\informs-paper` 를 git 리포로 초기화하고 원격을 등록했다.

```
origin = https://github.com/ysb00/R2R_steadyState_identification_for_changing_condition.git
```

연결 전 상태:

| 위치 | 내용 |
|---|---|
| 원격 `main` | `README.md` + `data/data_tensionSetting_{3..7}.zip` (5개, 27.8 MB) |
| 원격 `claude/cool-johnson-ku8k6t` (PR #1, 미병합) | `code/01_merge.py`, `docs/research_plan.md`, `docs/data_dictionary.md`, `outputs/*.csv` |
| 로컬 | `data/rawdata/` 의 xlsx 210개 (37 MB) + `.vscode/` 만 존재. 코드·문서 없음 |

PR #1 브랜치는 2026-10-16 Claude Code 클라우드 세션이 만든 커밋 4개로, `main`을 조상으로 하는 선형 이력이었다. 따라서 fast-forward로 `main`에 병합했고 (`bdefa9e` → `b419e65`), push 시 PR #1은 merged 상태로 닫혔다. 현재 작업 브랜치는 `main` 하나다.

### 2. 원본 데이터 추적 정책

`data/rawdata/` 의 xlsx 210개는 Tension 3~7 × ReWinder × Speed 조건으로, 원격에 이미 있는 `data/*.zip` 5개를 압축 해제한 **동일 데이터**다. 둘 다 추적하면 같은 측정값이 37 MB 중복되므로 `.gitignore`에 추가했다 (커밋 `a1cea5f`).

`code/01_merge.py` 는 `data/rawdata/` 가 아니라 `data/data_tensionSetting_{3..7}.zip` 을 직접 읽으므로, rawdata 없이도 파이프라인은 재현된다. 다른 PC에서는 zip 만 clone 받으면 충분하다.

`.gitignore` 에서 제외하는 것: `__pycache__/`, `data/rawdata/`, `.vscode/`, `.ipynb_checkpoints/`, `*.egg-info/`, `.venv/`

### 3. 현재 리포에 있는 것

| 파일 | 내용 |
|---|---|
| `data/data_tensionSetting_{3..7}.zip` | 원본 데이터. zip 1개 = xlsx 42개, xlsx 1개 = 1 운전조건 run |
| `code/01_merge.py` | zip 5개 → 병합 CSV + run별 요약 생성 (repo 루트에서 `python code/01_merge.py`) |
| `outputs/merged_data.csv` | 병합 데이터. 268,933행 × 8열. 컬럼: `run_id`, `k`, `Film-Tension Setting Value (kg)`, `ReWinder-Tension Setting Value (kg)`, `Film Tension #1 (kg)`, `Web Speed setting value (mm/sec)`, `Web Current Speed (mm/sec)`, `Trigger 4` |
| `outputs/01_file_summary.csv` | run별 요약 210행 |
| `docs/data_dictionary.md` | 병합 데이터 컬럼 정의 |
| `docs/research_plan.md` | 연구 계획 **초안 v0.1** (아래 주의 참조) |

`research_plan.md` 는 2026-10-16 다른 세션이 작성한 초안이며, 문서 스스로 "설계 초안, 가정 검증 필요"로 표시하고 있다. 그 안의 알고리즘 선택과 실험 설계는 확정 사항이 아니라 참고용이다.

### 4. 작업 방식 (합의)

- 연구 아이디어는 연구자가 제공하고, Claude는 **구현만** 한다.
- 불필요한 예외처리, 요청하지 않은 임의의 설정값·옵션·기능은 추가하지 않는다.
- window 길이, 임계값, 분할 방식, 하이퍼파라미터처럼 **결과 해석에 영향을 주는 값은 Claude가 임의로 정하지 않는다.**
- 꼭 결정해야 하는 선택지, 해석이 둘 이상 가능한 모호한 지시, 기존 결과 덮어쓰기·장시간 연산 같은 위험은 **실행 전에** 먼저 알린다. 답이 필요 없는 부분은 멈추지 않고 진행한다.

### 5. 다음 작업

연구자가 제시할 아이디어를 받아 구현 착수. 분석 방향은 아직 미정이다.
