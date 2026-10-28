# =============================================================================
# 03_baselines/03_moment.py : MOMENT 비교군 어댑터 (파운데이션 모델, 추가 학습 없음)
#
# 방법 : MOMENT (Goswami et al., ICML 2024) — 대규모 시계열로 사전학습된 T5 기반 인코더.
#        본 데이터로 추가 학습하지 않고 사전학습 가중치로 표현만 뽑는다 (zero-shot).
#        따라서 fold 와 무관하게 표현은 한 번만 계산하고, 학습 fold 에서는 GMM 후단만 적합한다.
#
# 외부 저장소 : https://github.com/moment-timeseries-foundation-model/moment
#   clone 위치 : code/03_baselines/external/moment  (git 추적 제외)
#   커밋       : 38f7310ad594100747ca2a8357e9c7ca7d323e0e (2026-02-10)
#   가중치     : Hugging Face AutonLab/MOMENT-1-small
#   원본은 한 줄도 수정하지 않는다. 선언된 의존성 huggingface-hub, transformers 를 설치해 사용한다.
#
# 모델이 강제하는 제약 (임의 선택이 아니라 구조상 고정값)
#   - 입력 길이가 512 로 고정이다. run(1275~1300 샘플)을 512 길이 창으로 잘라 넣고,
#     마지막 창은 0 으로 채운 뒤 input_mask 로 유효 구간을 알려준다. 창은 겹치지 않게 자른다.
#   - 출력은 시점 단위가 아니라 patch 단위다 (patch_len=8, stride=8 -> 창당 64개 patch).
#     patch 표현을 그 patch 가 덮는 8개 시점에 그대로 복제해 시점 단위 표현으로 만든다.
#   - 창마다 내부에서 정규화(RevIN)가 걸리므로 창 단위 평균/스케일 정보는 제거된다.
#     본 과제의 구분 정보가 e 의 진폭에 있다는 점(research_plan 1.2)을 생각하면 불리하게 작용할 수 있다.
#
# hyperparameter : 사전학습 가중치를 그대로 쓴다. 학습이 없으므로 seed 는 GMM 후단에만 영향을 준다.
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        splits.py 가 분할에 맞는 fold 파일을 읽는다 (A: folds.csv / C: folds_c.csv)
# 출력 : A -> outputs/03_baselines/03_moment/ , C -> outputs/04_speed_extrap/03_moment/
#        run/fold/summary/config csv + 조건별 heatmap
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행
# 실험 : 인자로 고른다 — 'tension' (A: 장력 5-fold) / 'c' (C: 저속 학습 -> 고속 test, 8 fold)
# 실행 : python code/03_baselines/03_moment.py        (A 실험, 기본값)
#        python code/03_baselines/03_moment.py c      (C 실험)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'moment'))
import evaluation as ev
import splits
from momentfm import MOMENTPipeline                                    # 외부 저장소 원본


SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 채점 + 별도 폴더에 저장.
# MOMENT 는 zero-shot 이라 애초에 학습하지 않는다 (사전학습 가중치로 표현만 뽑는다). 그래서 추론
# 모드에서도 하는 일이 완전히 같고, 바뀌는 것은 **채점 라벨과 저장 위치**뿐이다 — 아래는 손대지 않는다.
# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값). 학습이 없어 GMM 후단에만 쓰인다
N_SEEDS = 5                     # seed 반복 횟수

MODEL_NAME = 'AutonLab/MOMENT-1-small'   # 사전학습 가중치 (계획 문서의 small/base 중 small)
DEVICE = 'cuda'
ENCODE_BATCH = 64               # 추론 배치 크기. 결과에 영향 없음(메모리/속도용). 임의의 값

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = splits.resolve(SPLIT, '03_moment')
METHOD = 'MOMENT'

T1 = 'Film Tension #1 (kg)'     # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS

# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 배정
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids                                                   # run 순서는 분할 파일을 따른다
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
lengths = np.array([len(by_run[r]) for r in run_ids])
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])     # 시점 순서 (표현과 같은 순서)
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 1] rows = {len(df)}, runs = {len(run_ids)}, run 길이 {lengths.min()}~{lengths.max()} '
      f'({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 사전학습 모델 로드 + 전체 run 인코딩 (추가 학습 없음, fold 와 무관하게 1회)
#         run -> 길이 512 창으로 분할 -> patch 표현 -> patch 를 덮는 8개 시점에 복제 -> run 길이로 자르기
# =============================================================================
t0 = time.time()
model = MOMENTPipeline.from_pretrained(MODEL_NAME, model_kwargs={'task_name': 'embedding'})
model.init()
model = model.to(DEVICE).eval()
L, P = model.config.seq_len, model.config.patch_len
print(f'[Step 2] {MODEL_NAME} 로드: 입력 길이 {L}, patch 길이 {P}, d_model {model.config.d_model}')

# run 마다 창으로 자르고 (마지막 창은 0 패딩 + mask 0), 전체 창을 한 배열에 모은다
win, msk, owner = [], [], []
for i, r in enumerate(run_ids):
    e = by_run[r].e.to_numpy()
    n_win = int(np.ceil(len(e) / L))
    buf = np.zeros(n_win * L)
    buf[:len(e)] = e
    m = np.zeros(n_win * L)
    m[:len(e)] = 1.0
    win.append(buf.reshape(n_win, L)), msk.append(m.reshape(n_win, L))
    owner += [i] * n_win
win, msk = np.concatenate(win), np.concatenate(msk)
print(f'[Step 2] 창 {win.shape[0]}개 (run당 {win.shape[0] / len(run_ids):.1f}개), 패딩 비율 {1 - msk.mean():.3f}')

reps = []
with torch.no_grad():
    for s in range(0, len(win), ENCODE_BATCH):
        xb = torch.tensor(win[s:s + ENCODE_BATCH], dtype=torch.float32, device=DEVICE).unsqueeze(1)   # (B, 1, L)
        mb = torch.tensor(msk[s:s + ENCODE_BATCH], dtype=torch.float32, device=DEVICE)
        out = model.embed(x_enc=xb, input_mask=mb, reduction='none').embeddings                       # (B, 1, L/P, d)
        reps.append(out.squeeze(1).repeat_interleave(P, dim=1).cpu().numpy())                         # patch -> 시점 복제
reps = np.concatenate(reps)                                                                           # (n_win, L, d)

owner = np.array(owner)
z_all = np.concatenate([reps[owner == i].reshape(-1, reps.shape[-1])[:lengths[i]] for i in range(len(run_ids))])
assert len(z_all) == len(e_flat), f'표현 {len(z_all)} != 시점 {len(e_flat)}'
print(f'[Step 2] 표현 {z_all.shape} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 3. fold × seed 별로 학습 fold 표현에 트랙 A 후단(GMM)을 적합하고 검증 fold 에 적용
#         (인코더는 고정이므로 fold 마다 다시 학습하지 않는다)
# =============================================================================
run_m, fold_m = [], []
for f in sp.folds:
    # C 실험에서는 '학습 = 검증 아닌 전부' 가 아니다 (고속 run 은 CV fold 에서 미사용)
    is_train_pt, is_val_pt = sp.point_role(f, lengths)
    for seed in range(SEED, SEED + N_SEEDS):
        t0 = time.time()
        gmm, sidx, emeans = ev.fit_steady_gmm(z_all[is_train_pt], e_flat[is_train_pt], seed)
        pred, score = ev.predict_steady(gmm, sidx, z_all[is_val_pt])
        r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
        fr['total_sec'] = time.time() - t0
        fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
        run_m.append(r), fold_m.append(fr)
        ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)       # 중간 저장
        ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
        print(f'[Step 3] fold {f} seed {seed}: {time.time() - t0:.0f} s, '
              f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
              f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

# =============================================================================
# Step 4. 결과 저장 + 조건별 heatmap
# =============================================================================
run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
ev.save_results(OUT_DIR, METHOD, run_m, fold_m,
                {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'MODEL_NAME': MODEL_NAME,
                 'DEVICE': DEVICE, 'ENCODE_BATCH': ENCODE_BATCH, 'seq_len': L, 'patch_len': P,
                 'repr_dims': int(z_all.shape[1]), 'finetuned': 'no (zero-shot)',
                 'repo': 'moment-timeseries-foundation-model/moment',
                 'commit': '38f7310ad594100747ca2a8357e9c7ca7d323e0e'})
for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
    ev.plot_condition_heatmap(run_m, metric, OUT_DIR, METHOD, label)
print('[done] MOMENT 평가 완료')
