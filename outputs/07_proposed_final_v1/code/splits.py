# =============================================================================
# 03_baselines/splits.py : 실험별 분할을 읽어 어댑터에 넘기는 공용 모듈 (라이브러리, 실행 파일 아님)
#
# 왜 필요한가 — A 실험과 C 실험은 "학습 집합" 의 규칙이 다르다.
#   A (장력 외삽) : 검증 fold 를 뺀 **나머지 전부**가 학습이다. 210 run 이 항상 다 쓰인다.
#   C (속도 외삽) : 학습이 **저속으로 제한**된다. 고속 run 은 CV fold 에서 학습도 검증도 아닌
#                   '미사용(unused)' 이 되고, 최종 test 에서만 검증으로 쓰인다.
#   즉 C 에서는 "학습 = 나머지" 가 성립하지 않는다. 그래서 fold 마다 run 의 역할을
#   train / val / unused 세 값으로 **명시**하고, 어댑터는 그것만 읽는다.
#
# 분할 2종
#   'tension' : A 실험. outputs/03_baselines/folds.csv 의 fold_tension 을 그대로 쓴다.
#               역할은 (fold_tension == f) 면 val, 아니면 train 으로 유도한다.
#               **기존 동작과 완전히 같다** — A 결과를 다시 돌려도 값이 바뀌지 않는다.
#   'c'       : C 실험. outputs/04_speed_extrap/folds_c.csv 의 역할 컬럼 8개를 그대로 읽는다.
#               생성은 00_folds.py 가 한다 (정의의 단일 출처).
#
# 결과 저장 위치가 분할마다 다르다 — A 는 outputs/03_baselines/, C 는 outputs/04_speed_extrap/.
# 연구자 지시: C 결과가 A 결과와 섞이지 않게 별도 폴더에 둔다.
#
# ambiguous 포함 모드 ('amb') — 2026-10-28 연구자 결정. **기존 동작에 덧붙인 것이고 바꾸지 않는다**
#   기본 채점은 `Trigger 4 == 2` (ambiguous) 를 뺀다. 이 모드를 켜면 ambiguous 를 시간상
#   **가장 가까운 라벨**로 흡수해 2클래스로 채점한다 (규칙은 evaluation._nearest_label).
#   **재학습하지 않는다** — 저장된 인코더로 추론만 다시 한다. 그래서 플래그 하나가 세 가지를 한다:
#     ① 라벨 해석      evaluation._labels() 가 AMB_INCLUDED 를 보고 흡수본을 돌려준다
#     ② 결과 저장 위치  resolve() 가 AMB_OUT_DIR 로 간다 — 기존 결과와 섞이지 않게 루트를 분리
#     ③ 체크포인트 출처 ckpt_dir() 은 **항상 기존 루트**를 가리킨다 (새로 만들지 않는다)
#   이전 결정 C5("ambiguous 보조 평가 불필요", docs/hyperparameters.md §13.3)를 뒤집은 것이다.
#
# 사용법 (어댑터 공통)
#   import splits
#   sp = splits.load(SPLIT)                       # SPLIT = 'tension' | 'c'
#   run_ids = sp.run_ids                          # run 순서 (분할 파일의 순서)
#   for f in sp.folds:
#       tr_run, va_run = sp.run_role(f)           # (n_run,) bool 2개
#       tr_pt,  va_pt  = sp.point_role(f, lengths)  # 시점 단위로 펼친 것
#   INFER = splits.set_amb_mode(sys.argv[2:])     # 2번째 인자가 'amb' 면 ambiguous 포함 추론
# =============================================================================
import os
import numpy as np
import pandas as pd

# --- 분할 정의 (경로와 해석 방식) ---------------------------------------------
SPEC = {
    'tension': {'path': 'outputs/03_baselines/folds.csv',
                'out_dir': 'outputs/03_baselines',
                'kind': 'leave_one_out',          # fold 컬럼 1개, 학습 = 나머지
                'col': 'fold_tension',
                'desc': 'A 실험 — Film Tension 수준별 leave-one-level-out 5-fold (검증 42 / 학습 168)'},
    'c':       {'path': 'outputs/04_speed_extrap/folds_c.csv',
                'out_dir': 'outputs/04_speed_extrap',
                'kind': 'explicit_role',          # 역할 컬럼 n개, 값 = train/val/unused
                'col': None,
                'desc': 'C 실험 — 저속(50~125) 학습+CV / 고속(150~200) 최종 test'},
}

ROLE_PREFIX = 'role__'            # folds_c.csv 의 역할 컬럼 접두사 (fold 컬럼과 구분한다)


class Split:
    def __init__(self, name):
        if name not in SPEC:
            raise ValueError(f'알 수 없는 분할 {name!r}. 가능한 값: {list(SPEC)}')
        s = SPEC[name]
        if not os.path.exists(s['path']):
            raise FileNotFoundError(f'{s["path"]} 가 없다. 먼저 python code/03_baselines/00_folds.py 를 실행한다')
        self.name, self.out_dir, self.kind, self.desc = name, s['out_dir'], s['kind'], s['desc']
        self.table = pd.read_csv(s['path'])
        self.run_ids = self.table.run_id.tolist()
        self._idx = self.table.set_index('run_id')

        if self.kind == 'leave_one_out':
            self._col = s['col']
            self.folds = [int(v) for v in sorted(self.table[self._col].unique())]
        else:
            self.folds = [c[len(ROLE_PREFIX):] for c in self.table.columns if c.startswith(ROLE_PREFIX)]

    def run_role(self, fold):
        # 역할 : fold 하나에 대한 run 단위 역할을 bool 2개로 돌려준다.
        # 입력 : fold  self.folds 의 원소
        # 출력 : (is_train (n_run,) bool, is_val (n_run,) bool). 둘 다 False 면 그 run 은 미사용이다.
        if self.kind == 'leave_one_out':
            v = self._idx.loc[self.run_ids, self._col].to_numpy()
            is_val = v == fold
            return ~is_val, is_val                       # 학습 = 나머지 전부 (기존 동작)
        r = self._idx.loc[self.run_ids, ROLE_PREFIX + fold].to_numpy()
        return r == 'train', r == 'val'

    def point_role(self, fold, lengths):
        # 역할 : run 단위 역할을 시점 단위로 펼친다 (표현 배열과 같은 순서).
        # 입력 : fold / lengths (n_run,) run 별 시점 수 — run_ids 순서와 같아야 한다
        # 출력 : (is_train_pt, is_val_pt) 각각 (sum(lengths),) bool
        tr, va = self.run_role(fold)
        return np.repeat(tr, lengths), np.repeat(va, lengths)

    def summary(self):
        # 역할 : fold 별 학습/검증/미사용 run 수를 한 줄씩 적은 표를 만든다 (실행 로그·설정 기록용).
        rows = []
        for f in self.folds:
            tr, va = self.run_role(f)
            rows.append({'split': self.name, 'fold': f, 'n_train': int(tr.sum()),
                         'n_val': int(va.sum()), 'n_unused': int((~tr & ~va).sum())})
        return pd.DataFrame(rows)


def load(name):
    return Split(name)


def fold_group(fold):
    # 역할 : fold 이름을 '같은 질문을 묻는 묶음' 이름으로 바꾼다.
    #        C 는 cv_T* / cv_S* / final_test 가 서로 다른 질문이라 평균내거나 한 그림에 섞지 않는다
    #        (hyperparameters.md §13.0 F6). A 는 fold 가 전부 같은 질문이라 묶음이 하나다.
    # 입력 : fold  분할의 fold 이름 (A 는 정수, C 는 문자열)
    # 출력 : 묶음 이름 문자열. A 는 빈 문자열 (묶음 구분 없음 — 파일 이름에 접미사가 붙지 않는다)
    s = str(fold)
    if s.startswith('cv_T'):
        return 'cv_T'
    if s.startswith('cv_S'):
        return 'cv_S'
    if s == 'final_test':
        return 'final_test'
    return ''


def resolve(name, stage):
    # 역할 : 분할 이름과 단계 폴더 이름으로 출력 경로를 만든다. A 와 C 결과가 섞이지 않게 하는 지점이다.
    #        ambiguous 포함 모드(set_amb_mode)에서는 **별도 루트**로 간다 — 기존 결과를 덮지 않는다.
    # 입력 : name 분할 이름 / stage 어댑터별 폴더 이름 (예: '02_ts2vec')
    # 출력 : 출력 디렉터리 경로 (예: 'outputs/04_speed_extrap/02_ts2vec')
    return f'{out_root(name)}/{stage}'


# =============================================================================
# ambiguous 포함 모드 (2026-10-28 추가) — 위 헤더의 설명 참조
# =============================================================================
AMB_INCLUDED = False              # set_amb_mode() 만 이 값을 바꾼다. 기본값 False = 기존 동작 그대로
AMB_OUT_DIR = {'tension': 'outputs/05_ambiguous_included_A',    # A 실험 재추론 결과
               'c':       'outputs/06_ambiguous_included_C'}    # C 실험 재추론 결과
AMB_TOKEN = 'amb'                 # CLI 에서 모드를 켜는 토큰 (분할 이름 **뒤에** 온다)


def set_amb_mode(argv_rest):
    # 역할 : 어댑터·집계 스크립트 공통 — 분할 이름 뒤에 'amb' 가 붙었는지 보고 전역 모드를 켠다.
    #        분할 이름은 기존대로 `sys.argv[1]` 이 읽으므로 **기존 호출 방식이 그대로 동작한다**
    #        (`python ... 04_cost.py c` 는 argv_rest 가 비어 있어 False).
    # 입력 : argv_rest  sys.argv[2:] (분할 이름을 뺀 나머지)
    # 출력 : True 면 ambiguous 포함 + 재학습 없이 추론만. 전역 AMB_INCLUDED 도 함께 선다
    global AMB_INCLUDED
    unknown = [t for t in argv_rest if t not in (AMB_TOKEN, f'--{AMB_TOKEN}')]
    if unknown:
        raise ValueError(f'알 수 없는 인자 {unknown}. 쓸 수 있는 것: {AMB_TOKEN!r}')
    AMB_INCLUDED = len(argv_rest) > 0
    return AMB_INCLUDED


def out_root(name):
    # 역할 : 분할의 결과 루트. ambiguous 포함 모드면 별도 루트를 돌려준다.
    # 입력 : name 분할 이름 ('tension' | 'c')
    # 출력 : 루트 경로 문자열
    return AMB_OUT_DIR[name] if AMB_INCLUDED else SPEC[name]['out_dir']


def ckpt_dir(name, stage):
    # 역할 : 학습된 인코더를 **읽는** 곳. ambiguous 포함 모드에서도 **기존 실험의 체크포인트**를 쓴다
    #        — 재학습하지 않는 것이 이 모드의 전제이므로 out_root 를 따라가면 안 된다.
    # 입력 : name 분할 이름 / stage 단계 폴더 이름 (예: '04_cost')
    # 출력 : 체크포인트 폴더 경로 (예: 'outputs/04_speed_extrap/04_cost/checkpoints')
    return f'{SPEC[name]["out_dir"]}/{stage}/checkpoints'
