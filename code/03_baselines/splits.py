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
# 사용법 (어댑터 공통)
#   import splits
#   sp = splits.load(SPLIT)                       # SPLIT = 'tension' | 'c'
#   run_ids = sp.run_ids                          # run 순서 (분할 파일의 순서)
#   for f in sp.folds:
#       tr_run, va_run = sp.run_role(f)           # (n_run,) bool 2개
#       tr_pt,  va_pt  = sp.point_role(f, lengths)  # 시점 단위로 펼친 것
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


def resolve(name, stage):
    # 역할 : 분할 이름과 단계 폴더 이름으로 출력 경로를 만든다. A 와 C 결과가 섞이지 않게 하는 지점이다.
    # 입력 : name 분할 이름 / stage 어댑터별 폴더 이름 (예: '02_ts2vec')
    # 출력 : 출력 디렉터리 경로 (예: 'outputs/04_speed_extrap/02_ts2vec')
    return f'{SPEC[name]["out_dir"]}/{stage}'
