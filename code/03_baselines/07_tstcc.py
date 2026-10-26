# =============================================================================
# 03_baselines/07_tstcc.py : TS-TCC 비교군 어댑터
#
# 방법 : TS-TCC (Eldele et al., IJCAI 2021) — 시계열에 weak/strong 두 증강을 걸고
#        ① Temporal Contrasting : 한 view 의 과거로 다른 view 의 미래를 맞추는 대조 과제(TC 모듈)
#        ② Contextual Contrasting : 두 view 의 문맥 벡터를 NT-Xent 로 끌어당기는 대조 과제
#        손실 = (TC loss 양방향) * 1 + NTXent * 0.7  (trainer.py:81-85, lambda1/lambda2 하드코딩)
#
# 외부 저장소 : https://github.com/emadeldeen24/TS-TCC
#   clone 위치 : code/03_baselines/external/TS-TCC  (git 추적 제외)
#   커밋       : 15bdfd6eced6aeb6c638839828effe1b4b2ea391  (external_versions.csv 참조)
#   원본은 한 줄도 수정하지 않는다. 본 파일이 어댑터(데이터 변환 + 호출 + 평가)다.
#
# ---------------------------------------------------------------------------
# config 기준을 Epilepsy 로 택한 이유  (docs/tstcc_config_choice.md 에 상세)
# ---------------------------------------------------------------------------
# TS-TCC 는 데이터셋별 config 4개를 주고 그 값들이 서로 크게 다르다. "저장소 기본값" 이
# 하나가 아니므로 어느 데이터셋의 것을 쓸지 **우리가 골라야 한다.** 성능을 보고 고르면
# 라벨 기반 선택이라 research_plan.md 0절 위반이므로, **데이터 특성만으로** 골랐다.
#
#            | 채널 | 원 길이 | kernel/stride | jitter_scale | jitter | max_seg
#   Epilepsy |   1  |   178   |     8 / 1     |    0.001     | 0.001  |    5     <- 채택
#   HAR      |   9  |   128   |     8 / 1     |    1.1       | 0.8    |    8
#   pFD      |   1  |  5120   |    32 / 4     |    2         | 0.1    |    5
#   sleepEDF |   1  |  3000   |    25 / 3     |    1.5       | 2      |   12
#
#   세 기준이 모두 Epilepsy 를 가리킨다 — ① 단일 채널(HAR 제외) ② 원 길이 178 이 본 창
#   144 에 가장 가깝다 ③ kernel 8 / stride 1 이 짧은 입력용이다.
#
# ---------------------------------------------------------------------------
# 구조가 강제하는 값 — 선택의 여지가 없다
# ---------------------------------------------------------------------------
# `features_len` : config 의 24 는 **입력 178 기준** 값이다. conv 3블록(각 MaxPool 2배 축소)을
#   통과한 특징맵 길이이므로 입력이 144 면 달라진다. 공식으로 추정하지 않고 **더미 입력으로
#   실측**해서 설정한다 (입력 178 -> 24 로 config 값이 재현되는 것까지 확인했다).
#   입력 144 -> 특징맵 (128, 20) -> features_len = 20, 평탄화 2,560 차원.
# `TC.timesteps` : 10. TC 모듈이 `randint(seq_len - timestep)` 을 쓰므로 특징맵 길이 20 > 10 이어야
#   한다. 충족한다.
# `num_classes` : 2 (Epilepsy 값 유지). 자기지도 모드에서는 `logits` 층이 쓰이지 않으므로
#   결과에 영향이 없다. 모델 생성에 필요해 형식상 둔다.
#
# ---------------------------------------------------------------------------
# 저장소 동작 중 기록해 둘 것 (원본 그대로 쓰되, 해석에 필요하다)
# ---------------------------------------------------------------------------
# 1. `scaling()` 의 곱셈 인자가 **N(loc=2.0, sigma)** 다 (augmentations.py:17). 즉 weak view 는
#    신호를 평균 2배로 키운다. Epilepsy 의 sigma=0.001 이면 거의 정확히 2배 고정이다.
#    -> weak view = 약 2*x,  strong view = permutation(x) + 거의 0 인 jitter.
#    즉 Epilepsy config 에서 TS-TCC 는 사실상 **"2배 스케일 vs permutation" 대조학습**이 된다.
#    (이 구조가 A6 ablation(permutation 제거)을 깨끗하게 만들어 준다 — 뺄 증강이 하나뿐이다)
# 2. 증강은 `Load_Dataset.__init__` 에서 **한 번만** 계산되어 전 epoch 동안 재사용된다
#    (dataloader.py:35). epoch 마다 다시 뽑지 않는다.
# 3. `NTXentLoss` 가 `batch_size` 고정 마스크를 만들므로 **drop_last=True 가 필수**다
#    (Epilepsy config 기본값이 True 이므로 그대로 쓴다).
#
# ---------------------------------------------------------------------------
# Trainer() 를 쓰지 않고 저장소의 model_train() 을 직접 epoch 루프로 돈다
# ---------------------------------------------------------------------------
# 이유: `Trainer()` 는 **아무것도 반환하지 않아** 손실을 기록할 수 없다(trainer.py:15-43).
# 자기지도 모드에서 Trainer 가 하는 일은 ① model_train 호출 ② model_evaluate(자기지도에서는
# `pass` 로 무동작, trainer.py:121) ③ 로깅 ④ 체크포인트 저장 뿐이고, **scheduler 는
# `training_mode != 'self_supervised'` 조건이라 step 되지 않는다**(trainer.py:26).
# 따라서 model_train 을 num_epoch 번 호출하는 것은 수치적으로 Trainer 와 동일하다.
# 손실은 epoch 단위로만 얻는다 (iteration 훅이 없다). `iters_per_epoch` 를 함께 기록해
# 다른 방법의 iteration 축과 환산할 수 있게 한다.
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        splits.py 가 분할에 맞는 fold 파일을 읽는다 (A: folds.csv / C: folds_c.csv)
# 출력 : A -> outputs/03_baselines/07_tstcc/ , C -> outputs/04_speed_extrap/07_tstcc/
#        run/fold/summary/config csv + heatmap + 손실곡선
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행
# 실험 : 인자로 고른다 — 'tension' (A: 장력 5-fold) / 'c' (C: 저속 학습 -> 고속 test, 8 fold)
# 실행 : python code/03_baselines/07_tstcc.py        (A 실험, 기본값)
#        python code/03_baselines/07_tstcc.py c      (C 실험)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'TS-TCC'))
import evaluation as ev
import splits
from config_files.Epilepsy_Configs import Config as EpilepsyConfig   # 외부 저장소 원본
from models.model import base_Model                                  # 외부 저장소 원본
from models.TC import TC                                             # 외부 저장소 원본
from dataloader.dataloader import Load_Dataset                       # 외부 저장소 원본 (증강 포함)
from trainer.trainer import model_train                              # 외부 저장소 원본 학습 1 epoch
import dataloader.augmentations as tstcc_aug                         # shim 대상 (아래)


# =============================================================================
# 런타임 shim — 원본 `permutation()` 이 numpy>=1.24 에서 깨진다
# =============================================================================
# 원인 : `np.random.permutation(splits)` 에서 splits 가 **길이가 다른 배열들의 리스트**다
#        (시간축을 임의 지점에서 쪼갠 조각들). numpy>=1.24 는 ragged 리스트를 배열로 만들 때
#        ValueError 를 낸다 (예전 numpy 는 object 배열로 받아 줬다).
#        -> `ValueError: setting an array element with a sequence. ... inhomogeneous shape`
#
# 처리 : 원본을 수정하지 않고 런타임에 함수를 바꿔 끼운다 (결정 3 "shim 우선").
#        `permutation(리스트)` 의 의미는 **리스트 원소(조각)의 순서를 섞는 것**이므로,
#        같은 의미의 인덱스 셔플로 바꾼다.
#
# 난수 동등성 : `np.random.permutation(길이 n 리스트)` 와 `np.random.permutation(n)` 은 모두
#        길이 n 의 1-D 에 Fisher-Yates 를 적용하므로 **난수 소비가 같다.** 따라서 예전 numpy 에서
#        원본이 냈을 결과와 비트 단위로 같다. 증강의 세기나 분포를 바꾸지 않는다.
# =============================================================================
def _permutation_numpy_compat(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            order = np.random.permutation(len(splits))               # <- 원본의 유일한 차이점
            warp = np.concatenate([splits[j] for j in order]).ravel()
            ret[i] = pat[0, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


tstcc_aug.permutation = _permutation_numpy_compat                    # DataTransform 이 모듈 전역에서 찾는다


SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값)
N_SEEDS = 5                     # seed 반복 횟수

# 창 길이 — 규칙 1b (docs/periodic_disturbance.md §7, docs/hyperparameters.md §12)
#   저장소 기본값은 데이터셋별 하드코딩(Epilepsy 178)이라 본 데이터에 의미가 없다.
#   가장 느린 조건(V=50)의 회전 외란 1주기를 담는 길이를 쓴다. 라벨을 쓰지 않는다.
ROLL_CIRCUMFERENCE_MM = 783.5   # 역산 둘레 (205 run 평균, 지름 249.4 mm). outputs/02_eda/ 참조
V_MIN = 50                      # 조건 격자의 최저 속도 (mm/s)
WINDOW = int(np.ceil((1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / V_MIN))     # = 144 samples (15.7 s)

ENCODE_BATCH = 512              # 표현 추출 배치 크기. **[임의]** — 결과에 영향 없음
DEVICE = 'cuda'                 # 저장소 기본값 (main.py --device)

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = splits.resolve(SPLIT, '07_tstcc')
METHOD = 'TS-TCC'

T1 = 'Film Tension #1 (kg)'     # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS


# =============================================================================
# 함수
# =============================================================================
def make_config(features_len):
    # 역할 : Epilepsy config 인스턴스를 만들고 **구조가 강제하는 값만** 덮어쓴다.
    #        원본 파일은 수정하지 않는다 (런타임 shim — 결정 3 "shim 우선").
    # 입력 : features_len  입력 창 WINDOW 에 대한 실측 특징맵 길이
    # 출력 : config 객체. input_channels 는 Epilepsy 가 이미 1 이라 바꿀 것이 없다.
    c = EpilepsyConfig()
    assert c.input_channels == 1, 'Epilepsy config 는 단일 채널이어야 한다'
    c.features_len = features_len                                   # 178 기준 24 -> WINDOW 기준 실측값
    return c


def measure_features_len(config, window):
    # 역할 : conv 3블록을 더미 입력으로 통과시켜 특징맵 길이를 **실측**한다.
    #        padding/MaxPool 조합을 손으로 계산하지 않기 위함이다.
    # 입력 : config  모델 생성용 config / window  입력 창 길이
    # 출력 : (features_len, 평탄화 차원)
    probe = base_Model(config).eval()
    with torch.no_grad():
        y = probe.conv_block3(probe.conv_block2(probe.conv_block1(torch.zeros(2, 1, window))))
    return int(y.shape[2]), int(y.shape[1] * y.shape[2])


def train_windows(e, window):
    # 역할 : run 1개를 겹치지 않는 창으로 잘라 학습 샘플을 만든다 (CoST 의 run 분할과 같은 취지).
    #        뒤쪽의 불완전한 꼬리는 버린다 — 앞쪽을 버리면 과도구간(소수 클래스)이 학습에서
    #        사라지므로, run 선두부터 채운다.
    # 입력 : e (n,) run 1개의 오차 신호 / window 창 길이
    # 출력 : (n_win, 1, window) 배열
    n_win = len(e) // window
    return e[:n_win * window].reshape(n_win, 1, window)


def encode_windows(model, e, window, batch):
    # 역할 : run 1개를 causal 이동창(stride 1)으로 시점별 인코딩한다.
    #        시점 i 의 창은 [i-window+1, i] 이므로 미래를 보지 않는다.
    #        창이 아직 다 차지 않은 앞 window-1 개 시점은 **첫 완전창의 표현**으로 채운다
    #        (MovStd-Slope 와 동일한 규약 — 창 길이가 달라도 채점 대상 시점 수가 같도록).
    #        표현은 특징맵을 평탄화한 것이다 — TS-TCC 자신의 분류 헤드가 쓰는 형태와 같다
    #        (model.py:34 `x_flat = x.reshape(x.shape[0], -1)`).
    #
    #        **float64 로 돌려준다.** float32 로는 트랙 A 후단의 GMM 적합이 실패한다 —
    #        2,560 차원 중 7개가 dead ReLU 로 항상 0 이고 그 밖에도 거의 퇴화한 차원이 있어,
    #        sklearn 이 분산을 E[X^2] - E[X]^2 로 계산할 때 float32 자리수 소실로 음수가 나오고
    #        `_compute_precision_cholesky` 가 ValueError 를 던진다
    #        ("some components have ill-defined empirical covariance ... passing float64 instead").
    #        실험으로 원인을 확정했다 (fold 2 seed 2 체크포인트로 재현):
    #          float32 / 2560차원        -> 실패
    #          float32 / std>0 만 2553차원 -> **실패** (0분산 차원 제거로는 해결되지 않는다)
    #          float64 / 2560차원        -> 성공
    #        즉 원인은 퇴화 차원이 아니라 **정밀도**다. 0분산 차원을 지우지 않는 이유도 이것이다.
    #
    #        주의 — 다른 비교군은 float32 로 후단에 들어간다 (인코더 출력이 torch 기본 dtype).
    #        TS-TCC 만 float64 인 것은 모델 선택이 아니라 **수치 정밀도 보정**이고, 이 표현이
    #        2,560 차원(다른 방법의 320~512 차원보다 5~8배)이라 필요해진 것이다. 전 방법을
    #        float64 로 통일하면 기존 결과가 미세하게 바뀌어 전면 재실행이 필요하다 -> 후일 작업.
    # 입력 : model 학습된 base_Model / e (n,) / window 창 길이 / batch 추론 배치 크기
    # 출력 : (n, features_len * final_out_channels) 시점별 표현, dtype float64
    n = len(e)
    starts = np.arange(0, n - window + 1)                           # 완전창의 시작 index
    idx = starts[:, None] + np.arange(window)                       # (n_full, window)
    out = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            x = torch.from_numpy(e[idx[b:b + batch]]).float().unsqueeze(1).to(DEVICE)
            _, feat = model(x)                                      # feat: (B, ch, features_len)
            out.append(feat.reshape(feat.shape[0], -1).cpu().numpy())
    z_full = np.concatenate(out).astype(np.float64)                 # (n-window+1, d) — float64 필수 (위 주석)
    head = np.repeat(z_full[:1], window - 1, axis=0)                # 앞 window-1 개를 첫 창으로 채움
    return np.concatenate([head, z_full])


# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 배정
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids                                                # run 순서는 분할 파일을 따른다
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
e_by_run = {r: by_run[r].e.to_numpy() for r in run_ids}
lengths = np.array([len(e_by_run[r]) for r in run_ids])
e_flat = np.concatenate([e_by_run[r] for r in run_ids])             # 표현과 같은 시점 순서
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 1] rows = {len(df)}, runs = {len(run_ids)}, run 길이 {lengths.min()}~{lengths.max()} '
      f'({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 구조값 실측 + 학습 샘플 구성
# =============================================================================
probe_cfg = make_config(1)
FEATURES_LEN, REPR_DIMS = measure_features_len(probe_cfg, WINDOW)
configs = make_config(FEATURES_LEN)
assert FEATURES_LEN > configs.TC.timesteps, \
    f'TC 모듈이 특징맵 길이 > timesteps({configs.TC.timesteps}) 를 요구한다 (현재 {FEATURES_LEN})'

win_by_run = {r: train_windows(e_by_run[r], WINDOW) for r in run_ids}
n_win_total = sum(len(v) for v in win_by_run.values())
print(f'[Step 2] 창 {WINDOW} samples ({WINDOW * ev.DT:.1f} s) | 특징맵 길이 {FEATURES_LEN} '
      f'-> 표현 {REPR_DIMS} 차원 | 학습 창 총 {n_win_total}개 (run 당 {lengths.min() // WINDOW}~{lengths.max() // WINDOW}개)')
print(f'[Step 2] 증강 (Epilepsy config): weak = scaling(loc=2.0, sigma={configs.augmentation.jitter_scale_ratio}), '
      f'strong = jitter(permutation(max_seg={configs.augmentation.max_seg}), sigma={configs.augmentation.jitter_ratio})')

# =============================================================================
# Step 3. fold × seed 별로 학습 fold 에서 자기지도 학습하고,
#         학습 fold 표현으로 트랙 A 후단(GMM)을 적합한 뒤 검증 fold 에 적용한다
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
if os.path.exists(f'{OUT_DIR}/{METHOD}_fold_metrics.csv'):
    print(f'[Step 3] {METHOD}: 이미 결과가 있어 건너뛴다 (다시 돌리려면 해당 파일들을 먼저 지울 것)')
    sys.exit(0)

run_m, fold_m, loss_m = [], [], []
for f in sp.folds:
    # C 실험에서는 '학습 = 검증 아닌 전부' 가 아니다 (고속 run 은 CV fold 에서 미사용)
    is_train_run, is_val_run = sp.run_role(f)
    is_train_pt, is_val_pt = sp.point_role(f, lengths)
    x_train = np.concatenate([win_by_run[r] for r, v in zip(run_ids, is_val_run) if not v])
    for seed in range(SEED, SEED + N_SEEDS):
        t0 = time.time()
        # 시드 초기화 — main.py:52-58 과 동일하게 둔다 (저장소 방식)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        # 저장소 원본 Dataset 에 넣는다. 증강은 여기서 한 번 계산되어 전 epoch 재사용된다.
        # 라벨은 자기지도 모드에서 쓰이지 않으므로 0 으로 채운다 (Dataset 이 요구하는 형식).
        ds = Load_Dataset({'samples': torch.from_numpy(x_train).float(),
                           'labels': torch.zeros(len(x_train), dtype=torch.long)},
                          configs, 'self_supervised')
        dl = torch.utils.data.DataLoader(ds, batch_size=configs.batch_size, shuffle=True,
                                         drop_last=configs.drop_last, num_workers=0)

        model = base_Model(configs).to(DEVICE)
        tc_model = TC(configs, DEVICE).to(DEVICE)
        # optimizer 2개 — main.py:139-140 과 동일 (lr, betas, weight_decay 전부 저장소 값)
        opt = torch.optim.Adam(model.parameters(), lr=configs.lr,
                               betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        tc_opt = torch.optim.Adam(tc_model.parameters(), lr=configs.lr,
                                  betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        criterion = nn.CrossEntropyLoss()                           # 자기지도에서는 쓰이지 않지만 인자로 필요

        epoch_loss = []
        for epoch in range(1, configs.num_epoch + 1):               # 저장소 기본값 40
            tr_loss, _ = model_train(model, tc_model, opt, tc_opt, criterion,
                                     dl, configs, DEVICE, 'self_supervised')
            epoch_loss.append(float(tr_loss))
        t_fit = time.time() - t0
        torch.save({'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': tc_model.state_dict()},
                   f'{OUT_DIR}/checkpoints/{METHOD}_fold{f}_seed{seed}.pt')

        z = np.concatenate([encode_windows(model, e_by_run[r], WINDOW, ENCODE_BATCH) for r in run_ids])
        gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
        pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
        r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
        del z

        iters_per_epoch = len(dl)                                   # drop_last 반영된 참값
        fr['window'], fr['features_len'], fr['repr_dims'] = WINDOW, FEATURES_LEN, REPR_DIMS
        fr['n_train_windows'] = len(x_train)
        fr['n_epochs'], fr['iters_per_epoch'] = configs.num_epoch, iters_per_epoch
        fr['n_iters'] = configs.num_epoch * iters_per_epoch
        # 데이터 통과 횟수 — drop_last 로 버려지는 나머지를 반영한 실효값
        fr['data_passes'] = configs.num_epoch * iters_per_epoch * configs.batch_size / len(x_train)
        fr['samples_seen'] = configs.num_epoch * iters_per_epoch * configs.batch_size * WINDOW
        fr['fit_sec'], fr['total_sec'] = t_fit, time.time() - t0
        fr['epoch_mean_loss_last'] = epoch_loss[-1]
        fr['final_loss'] = float(np.mean(epoch_loss[-3:]))          # 마지막 3 epoch 평균
        fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
        lo = pd.DataFrame({'fold': f, 'seed': seed,
                           'epoch': np.arange(1, len(epoch_loss) + 1), 'loss': epoch_loss})
        run_m.append(r), fold_m.append(fr), loss_m.append(lo)

        ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)    # 중간 저장 (끊겨도 여기까지 남는다)
        ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
        ev.append_progress(OUT_DIR, METHOD, 'loss', lo)
        print(f'[Step 3] {METHOD} fold {f} seed {seed}: 학습 {t_fit:.0f} s, 전체 {time.time() - t0:.0f} s, '
              f'epoch {configs.num_epoch} 손실 {epoch_loss[0]:.4f} -> {epoch_loss[-1]:.4f}, '
              f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
              f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

# =============================================================================
# Step 4. 산출물 저장
# =============================================================================
run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, METHOD,
                    'TC + contextual contrasting loss', x_col='epoch',
                    xlabel='Epoch (TS-TCC 는 iteration 손실 훅이 없다 — iters_per_epoch 는 fold_metrics 참조)')
ev.save_results(OUT_DIR, METHOD, run_m, fold_m,
                {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS,
                 'WINDOW': WINDOW, 'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
                 'config_base': 'Epilepsy (단일 채널·길이 178·kernel 8/stride 1 — docs/tstcc_config_choice.md)',
                 'FEATURES_LEN': FEATURES_LEN, 'REPR_DIMS': REPR_DIMS,
                 'repr_form': '특징맵 평탄화 (TS-TCC 자신의 분류 헤드와 같은 형태)',
                 'repr_dtype': 'float64 — float32 로는 GMM 적합이 실패한다(퇴화 차원 + 2,560 차원에서의 '
                               '자리수 소실). 다른 비교군은 float32. 모델 선택이 아니라 수치 정밀도 보정',
                 'NUM_EPOCH': configs.num_epoch, 'BATCH_SIZE': configs.batch_size, 'LR': configs.lr,
                 'BETAS': (configs.beta1, configs.beta2), 'WEIGHT_DECAY': 3e-4,
                 'DROP_LAST': configs.drop_last, 'DROPOUT': configs.dropout,
                 'KERNEL_SIZE': configs.kernel_size, 'STRIDE': configs.stride,
                 'FINAL_OUT_CHANNELS': configs.final_out_channels,
                 'TC_HIDDEN_DIM': configs.TC.hidden_dim, 'TC_TIMESTEPS': configs.TC.timesteps,
                 'NTXENT_TEMPERATURE': configs.Context_Cont.temperature,
                 'AUG_jitter_scale_ratio': configs.augmentation.jitter_scale_ratio,
                 'AUG_jitter_ratio': configs.augmentation.jitter_ratio,
                 'AUG_max_seg': configs.augmentation.max_seg,
                 'AUG_note': 'scaling() 의 곱셈 인자는 N(loc=2.0, sigma) 다 — weak view 는 신호를 약 2배로 키운다',
                 'LAMBDA_TC': 1, 'LAMBDA_NTXENT': 0.7,
                 'ENCODE_BATCH': ENCODE_BATCH, 'DEVICE': DEVICE,
                 'repo': 'emadeldeen24/TS-TCC',
                 'commit': '15bdfd6eced6aeb6c638839828effe1b4b2ea391'})
for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
    ev.plot_condition_heatmap(run_m, metric, OUT_DIR, METHOD, label)
print(f'[done] {METHOD} 평가 완료')
