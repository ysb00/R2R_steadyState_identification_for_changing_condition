# =============================================================================
# 03_baselines/08_simmtm.py : SimMTM 비교군 어댑터
#
# 방법 : SimMTM (Dong et al., NeurIPS 2023) — 마스킹된 시계열을 복원하는 사전학습.
#        한 시계열에서 마스킹본 여러 개(positive_nums=3)를 만들어
#        ① ContrastiveWeight : 원본-마스킹본 사이 유사도로 가중치를 학습
#        ② AggregationRebuild : 그 유사도로 표현을 가중 합산해 원 신호를 복원
#        손실 = AutomaticWeightedLoss(대조 손실, 복원 MSE)  (model.py:45, 두 손실의 가중치도 학습한다)
#        TS-TCC 처럼 '증강 불변성' 을 배우는 것이 아니라 **마스킹 복원**으로 표현을 만든다.
#
# 외부 저장소 : https://github.com/thuml/SimMTM
#   clone 위치 : code/03_baselines/external/SimMTM  (git 추적 제외)
#   커밋       : 169513bef74fb676e48d98a0e30f8823793f691c
#   원본은 한 줄도 수정하지 않는다. 본 파일이 어댑터(데이터 변환 + 호출 + 평가)다.
#   사용 브랜치 : SimMTM_Classification/code  (분류용 자기지도 사전학습).
#                 SimMTM_Forecasting 은 예측 과제용이라 표현학습 비교군에 맞지 않는다.
#
# ---------------------------------------------------------------------------
# config 는 SleepEEG 로 **강제된다** — TS-TCC 와 달리 고를 여지가 없다
# ---------------------------------------------------------------------------
# Classification 브랜치의 config 는 2개뿐이다 (Epilepsy / SleepEEG). `main.py` 는
# `--pretrain_dataset` 으로 config 를 고르고 기본값이 SleepEEG 다.
#
# **Epilepsy config 로는 사전학습 모델을 만들 수 없다** — `CNNoutput_channel` 속성이 없고
# `model.py:35` 의 `nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256)` 가
# 그것을 요구한다. Epilepsy 는 **target(미세조정) 데이터셋용** config 이고, 사전학습용은
# SleepEEG 하나다. 따라서 선택이 아니라 구조적 강제다.
#
# 그리고 **SleepEEG config 의 거친 증강은 사전학습에 쓰이지 않는다.** `model_pretrain` 은
# `data_transform_masked4cl(data, masking_ratio, lm, positive_nums)` 로 **마스킹만** 한다
# (`trainer.py:122`). jitter·scaling·permutation 은 사전학습 경로에 없다.
# 즉 TS-TCC 에서 문제였던 "jitter_ratio=2 가 진폭 정보를 덮는다" 가 여기서는 해당 없다.
# 마스킹 설정은 config 가 아니라 **argparse 기본값**에서 온다 (아래 MaskArgs).
#
# ---------------------------------------------------------------------------
# 구조가 강제하는 값 — 창 144 를 쓰려면 반드시 덮어써야 한다 (전부 실측으로 확정)
# ---------------------------------------------------------------------------
# conv 3블록(각 MaxPool 2배 축소)을 통과한 길이가 config 값과 맞아야 한다.
#   입력 178 (원 설정) -> conv 출력 (128, 10) -> 평탄화 1,280.  CNNoutput_channel=10 과 일치
#   입력 144 (규칙 1b) -> conv 출력 (128,  8) -> 평탄화 1,024.  CNNoutput_channel 을 8 로 바꿔야 한다
#
# `self.head = nn.Linear(1280, 178)` 이 **178 을 하드코딩**한다 (`model.py:46`). 그런데
# `loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1))` 이므로 head 출력이
# **입력 길이와 반드시 같아야** 한다. 창 144 에서는 `nn.Linear(1024, 144)` 로 바꿔야 하고,
# 안 바꾸면 shape 불일치로 크래시한다. 모델 생성 후 속성을 교체하는 런타임 shim 으로 처리한다.
#
# `TSlength_aligned` (=178) 는 저장소 dataloader 가 입력을 자르는 데만 쓴다
# (`dataloader.py:29`). 본 어댑터는 자체 dataloader 를 쓰므로 영향이 없다.
#
# ---------------------------------------------------------------------------
# 표현은 `h` (평탄화 conv 특징) 를 쓴다 — 저장소 자신의 분류 헤드와 같은 형태
# ---------------------------------------------------------------------------
# 비사전학습 모드의 `forward` 는 `(h, z)` 를 돌려준다 — h 는 평탄화 conv 특징(창 144 에서
# 1,024 차원), z 는 dense 출력(128 차원). 저장소의 `target_classifier` 가
# `nn.Linear(1280, 64)` 를 `emb.reshape(B,-1)` 에 적용하므로 **다운스트림이 쓰는 것은 h** 다
# (1,280 = 입력 178 에서의 h 차원). 그래서 h 를 표현으로 쓴다.
#
# **float64 로 후단에 넣는다.** TS-TCC(2,560 차원)에서 float32 가 GMM 적합을 깨뜨렸고
# (docs/tstcc_config_choice.md §7.5), 본 표현도 1,024 차원으로 다른 방법(320~512)보다 크다.
# 예방적으로 float64 를 쓴다. 다른 비교군은 float32 다 — 모델 선택이 아니라 수치 정밀도 보정이다.
#
# ---------------------------------------------------------------------------
# Trainer() 를 쓰지 않고 저장소의 model_pretrain() 을 직접 epoch 루프로 돈다
# ---------------------------------------------------------------------------
# `Trainer()` 는 사전학습 후 **분류 미세조정·테스트까지** 수행한다(`trainer.py:35`). 본 연구는
# 트랙 A(라벨 없는 GMM 후단)로 평가하므로 미세조정 단계가 필요 없고, 라벨을 쓰게 되므로
# 써서는 안 된다. `model_pretrain` 은 손실 3종을 그대로 반환하므로(`trainer.py:143`)
# TS-TCC 와 달리 손실 기록을 위한 우회가 아니라 **미세조정을 건너뛰기 위한** 선택이다.
# `model_scheduler.step()` 은 `model_pretrain` 안에서 호출된다(`trainer.py:142`).
#
# 주의 — `CosineAnnealingLR(T_max=pretrain_epoch)` 이므로 학습 끝에서 lr -> 0 이 된다.
# CoST 와 같은 구조이고, **손실 곡선이 끝에서 평탄해지는 것은 수렴의 증거가 아니다**
# (docs/worklog.md 2026-10-23 §2).
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        splits.py 가 분할에 맞는 fold 파일을 읽는다 (A: folds.csv / C: folds_c.csv)
# 출력 : A -> outputs/03_baselines/08_simmtm/ , C -> outputs/04_speed_extrap/08_simmtm/
#        run/fold/summary/config csv + heatmap + 손실곡선
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행
# 실험 : 인자로 고른다 — 'tension' (A: 장력 5-fold) / 'c' (C: 저속 학습 -> 고속 test, 8 fold)
# 실행 : python code/03_baselines/08_simmtm.py        (A 실험, 기본값)
#        python code/03_baselines/08_simmtm.py c      (C 실험)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'SimMTM', 'SimMTM_Classification', 'code'))
import evaluation as ev
import splits
from config_files.SleepEEG_Configs import Config as SleepEEGConfig       # 외부 저장소 원본
from model import TFC                                                   # 외부 저장소 원본
from trainer import model_pretrain                                       # 외부 저장소 원본 사전학습 1 epoch


SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만
# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값)
N_SEEDS = 5                     # seed 반복 횟수

# 창 길이 — 규칙 1b (docs/periodic_disturbance.md §7, docs/hyperparameters.md §12)
#   저장소 기본값 TSlength_aligned=178 은 타 데이터셋용 하드코딩이라 본 데이터에 의미가 없다.
ROLL_CIRCUMFERENCE_MM = 783.5   # 역산 둘레 (205 run 평균, 지름 249.4 mm). outputs/02_eda/ 참조
V_MIN = 50                      # 조건 격자의 최저 속도 (mm/s)
WINDOW = int(np.ceil((1.0 / ev.DT) * ROLL_CIRCUMFERENCE_MM / V_MIN))     # = 144 samples (15.7 s)

ENCODE_BATCH = 512              # 표현 추출 배치 크기. **[임의]** — 결과에 영향 없음
DEVICE = 'cuda'                 # 저장소 기본값 (main.py --device)

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = splits.resolve(SPLIT, '08_simmtm')
CKPT_DIR = splits.ckpt_dir(SPLIT, '08_simmtm')   # 인코더를 읽는 곳 (추론 모드에서도 기존 실험 것을 쓴다)
METHOD = 'SimMTM'

T1 = 'Film Tension #1 (kg)'     # 원본 컬럼명 (변경하지 않음)
TS, RS, VS = ev.TS, ev.RS, ev.VS


class MaskArgs:
    # 역할 : 저장소가 argparse 객체(`args`)로 받는 값들을 담는다. **전부 저장소 argparse 기본값**이고
    #        config 가 아니라 CLI 에서 오는 값이다 (main.py:38-41, 45).
    #        `model_pretrain` 과 `loss.py` 가 이 속성들을 읽는다.
    # 주의 : config 의 `lr`(SleepEEG 는 3e-8)은 **사전학습에 쓰이지 않는다.** main.py:114 가
    #        optimizer 를 `lr=args.pretrain_lr` 로 만든다 -> 실효값은 1e-4 다.
    training_mode = 'pre_train'                                         # [저장소] main.py --training_mode
    masking_ratio = 0.5                                                 # [저장소] main.py --masking_ratio
    lm = 3                                                              # [저장소] main.py --lm (평균 마스킹 길이)
    positive_nums = 3                                                   # [저장소] main.py --positive_nums
    temperature = 0.2                                                   # [저장소] main.py --temperature
    pretrain_lr = 0.0001                                                # [저장소] main.py --pretrain_lr
    pretrain_epoch = 10                                                 # [저장소] main.py --pretrain_epoch
    # model_pretrain 이 참조하지 않지만 Trainer 경로가 읽는 값들 (본 어댑터는 Trainer 를 쓰지 않는다)
    lr = 0.0001
    finetune_epoch = 300
    log_epoch = 5


# =============================================================================
# 함수
# =============================================================================
def conv_out_len(config, window):
    # 역할 : TFC 의 conv 3블록을 더미 입력으로 통과시켜 출력 길이를 **실측**한다.
    #        padding/MaxPool 조합을 손으로 계산하지 않기 위함이다. 모델을 만들기 전에 필요하므로
    #        conv 블록만 같은 정의로 재구성해 재는다 (model.py:10-31 과 동일한 구성).
    # 입력 : config 저장소 config / window 입력 창 길이
    # 출력 : (conv 출력 길이, 평탄화 차원)
    blocks = nn.Sequential(
        nn.Sequential(nn.Conv1d(config.input_channels, 32, kernel_size=config.kernel_size,
                                stride=config.stride, bias=False, padding=(config.kernel_size // 2)),
                      nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2, 1), nn.Dropout(config.dropout)),
        nn.Sequential(nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                      nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2, 1)),
        nn.Sequential(nn.Conv1d(64, config.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
                      nn.BatchNorm1d(config.final_out_channels), nn.ReLU(), nn.MaxPool1d(2, 2, 1))).eval()
    with torch.no_grad():
        y = blocks(torch.zeros(2, config.input_channels, window))
    return int(y.shape[2]), int(y.shape[1] * y.shape[2])


def make_config(window):
    # 역할 : SleepEEG config 인스턴스를 만들고 **구조가 강제하는 값만** 덮어쓴다.
    #        원본 파일은 수정하지 않는다 (런타임 shim — 결정 3 "shim 우선").
    # 입력 : window 입력 창 길이
    # 출력 : (config, conv 출력 길이, 평탄화 차원)
    c = SleepEEGConfig()
    assert c.input_channels == 1, 'SleepEEG config 는 단일 채널이어야 한다'
    out_len, repr_dims = conv_out_len(c, window)
    c.CNNoutput_channel = out_len                                       # 178 기준 10 -> window 기준 실측값
    c.TSlength_aligned = window                                         # 저장소 dataloader 용 (본 어댑터는 미사용)
    return c, out_len, repr_dims


def build_model(config, args, window, repr_dims):
    # 역할 : TFC 를 만들고 **복원 헤드의 출력 차원을 입력 길이에 맞춘다.**
    #        원본은 `nn.Linear(1280, 178)` 로 178 을 하드코딩하는데(model.py:46),
    #        복원 손실이 `mse(pred_x, x_in_t.reshape(B, -1))` 이라 출력이 입력 길이와 같아야 한다.
    #        모델 생성 후 속성만 교체하는 런타임 shim 이다 (원본 파일 미수정).
    # 입력 : config / args / window 입력 창 길이 / repr_dims 평탄화 conv 차원
    # 출력 : TFC 모델 (device 미이동)
    m = TFC(config, args)
    assert m.head.in_features == 1280 and m.head.out_features == 178, \
        f'원본 head 가 Linear(1280, 178) 이 아니다 — 저장소가 바뀌었는지 확인할 것 ({m.head})'
    m.head = nn.Linear(repr_dims, window)                               # 입력 길이에 맞춘 복원 헤드
    return m


def train_windows(e, window):
    # 역할 : run 1개를 겹치지 않는 창으로 잘라 학습 샘플을 만든다 (CoST·TS-TCC 와 같은 취지).
    #        뒤쪽의 불완전한 꼬리는 버린다 — 앞쪽을 버리면 과도구간(소수 클래스)이 학습에서 사라진다.
    # 입력 : e (n,) run 1개의 오차 신호 / window 창 길이
    # 출력 : (n_win, 1, window) 배열
    n_win = len(e) // window
    return e[:n_win * window].reshape(n_win, 1, window)


def encode_windows(model, e, window, batch):
    # 역할 : run 1개를 causal 이동창(stride 1)으로 시점별 인코딩한다.
    #        시점 i 의 창은 [i-window+1, i] 이므로 미래를 보지 않는다.
    #        창이 아직 다 차지 않은 앞 window-1 개 시점은 **첫 완전창의 표현**으로 채운다
    #        (MovStd-Slope·TS-TCC 와 동일한 규약 — 창 길이가 달라도 채점 시점 수가 같도록).
    #        표현은 `forward(pretrain=False)` 의 첫 반환값 h (평탄화 conv 특징)다.
    #        float64 로 돌려준다 (헤더의 정밀도 주석 참조).
    # 입력 : model 학습된 TFC / e (n,) / window 창 길이 / batch 추론 배치 크기
    # 출력 : (n, repr_dims) 시점별 표현, dtype float64
    n = len(e)
    starts = np.arange(0, n - window + 1)
    idx = starts[:, None] + np.arange(window)
    out = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            x = torch.from_numpy(e[idx[b:b + batch]]).float().unsqueeze(1).to(DEVICE)
            h, _ = model(x)                                             # pretrain=False -> (h, z)
            out.append(h.cpu().numpy())
    z_full = np.concatenate(out).astype(np.float64)
    head = np.repeat(z_full[:1], window - 1, axis=0)
    return np.concatenate([head, z_full])


# =============================================================================
# Step 1. 데이터 로드 (`Trigger 4` 는 읽지 않는다) + fold 배정
# =============================================================================
t0 = time.time()
df = pd.read_csv(IN_PATH, usecols=['run_id', 'k', T1, TS, RS, VS])
df['e'] = df[T1] - df[TS]
sp = splits.load(SPLIT)
df = df.sort_values(['run_id', 'k']).reset_index(drop=True)
run_ids = sp.run_ids                                                    # run 순서는 분할 파일을 따른다
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
e_by_run = {r: by_run[r].e.to_numpy() for r in run_ids}
lengths = np.array([len(e_by_run[r]) for r in run_ids])
e_flat = np.concatenate([e_by_run[r] for r in run_ids])                 # 표현과 같은 시점 순서
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 1] rows = {len(df)}, runs = {len(run_ids)}, run 길이 {lengths.min()}~{lengths.max()} '
      f'({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 구조값 실측 + 학습 샘플 구성
# =============================================================================
args = MaskArgs()
configs, CONV_OUT_LEN, REPR_DIMS = make_config(WINDOW)
win_by_run = {r: train_windows(e_by_run[r], WINDOW) for r in run_ids}
n_win_total = sum(len(v) for v in win_by_run.values())
print(f'[Step 2] 창 {WINDOW} samples ({WINDOW * ev.DT:.1f} s) | conv 출력 길이 {CONV_OUT_LEN} '
      f'(config 원래값 {SleepEEGConfig().CNNoutput_channel}) -> 표현 {REPR_DIMS} 차원 | '
      f'학습 창 총 {n_win_total}개 (run 당 {lengths.min() // WINDOW}~{lengths.max() // WINDOW}개)')
print(f'[Step 2] 사전학습: 마스킹 전용 (masking_ratio={args.masking_ratio}, lm={args.lm}, '
      f'positive_nums={args.positive_nums}) -> 실효 배치 {configs.batch_size * (args.positive_nums + 1)}'
      f' (원본 {configs.batch_size} + 마스킹 {configs.batch_size * args.positive_nums})')
print(f'[Step 2] 학습량: pretrain_epoch={args.pretrain_epoch}, pretrain_lr={args.pretrain_lr} '
      f'(config.lr={SleepEEGConfig().lr} 은 사전학습에 쓰이지 않는다), CosineAnnealingLR(T_max={args.pretrain_epoch})')

# =============================================================================
# Step 3. fold × seed 별로 학습 fold 에서 자기지도 사전학습하고,
#         학습 fold 표현으로 트랙 A 후단(GMM)을 적합한 뒤 검증 fold 에 적용한다
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
if not INFER:                                                           # 추론 모드는 체크포인트를 쓰지 않는다
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
if os.path.exists(f'{OUT_DIR}/{METHOD}_fold_metrics.csv'):
    print(f'[Step 3] {METHOD}: 이미 결과가 있어 건너뛴다 (다시 돌리려면 해당 파일들을 먼저 지울 것)')
    sys.exit(0)

run_m, fold_m, loss_m = [], [], []
for f in sp.folds:
    # C 실험에서는 '학습 = 검증 아닌 전부' 가 아니다 (고속 run 은 CV fold 에서 미사용)
    is_train_run, is_val_run = sp.run_role(f)
    is_train_pt, is_val_pt = sp.point_role(f, lengths)
    # 추론 모드에서는 학습 샘플이 필요 없다
    x_train = None if INFER else np.concatenate([win_by_run[r]
                                                 for r, v in zip(run_ids, is_val_run) if not v])
    for seed in range(SEED, SEED + N_SEEDS):
        t0 = time.time()
        # 시드 초기화 — main.py:48-54 (set_seed) 와 동일하게 둔다 (저장소 방식)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        if INFER:                                                       # 재학습 없이 저장된 인코더만 불러온다
            ck = f'{CKPT_DIR}/{METHOD}_fold{f}_seed{seed}.pt'
            assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습을 돌릴 것)'
            # 체크포인트는 'pre_train' 모드로 저장돼 awl/contrastive/aggregation/head 를 포함한다.
            # build_model 이 같은 구성(+ 입력 길이에 맞춘 head)을 만들므로 그대로 load 된다.
            model = build_model(configs, args, WINDOW, REPR_DIMS).to(DEVICE)
            model.load_state_dict(torch.load(ck, map_location=DEVICE)['model_state_dict'])
            model.training_mode = 'eval'                                # pretrain=False 경로 (h, z 반환)
            z = np.concatenate([encode_windows(model, e_by_run[r], WINDOW, ENCODE_BATCH) for r in run_ids])
            gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
            pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
            r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
            del z
            fr['window'], fr['conv_out_len'], fr['repr_dims'] = WINDOW, CONV_OUT_LEN, REPR_DIMS
            fr['total_sec'] = time.time() - t0
            fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
            run_m.append(r), fold_m.append(fr)
            ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)
            ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
            print(f'[Step 3] {METHOD} fold {f} seed {seed}: 체크포인트 재사용 (학습 생략), '
                  f'전체 {time.time() - t0:.0f} s, '
                  f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
                  f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                  f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)
            continue

        # 저장소 dataloader(data_generator)는 source+target 두 데이터셋을 요구하는 전이학습 전제라
        # 쓰지 않는다. model_pretrain 이 요구하는 것은 (data, labels) 를 주는 loader 뿐이다.
        # 라벨은 사전학습에서 쓰이지 않으므로 0 으로 채운다.
        ds = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                            torch.zeros(len(x_train), dtype=torch.long))
        dl = torch.utils.data.DataLoader(ds, batch_size=configs.batch_size, shuffle=True,
                                         drop_last=configs.drop_last, num_workers=0)

        model = build_model(configs, args, WINDOW, REPR_DIMS).to(DEVICE)
        # optimizer·scheduler — main.py:113-116 과 동일 (lr 은 args.pretrain_lr, weight_decay=0)
        opt = torch.optim.Adam([{'params': model.parameters()}], lr=args.pretrain_lr,
                               betas=(configs.beta1, configs.beta2), weight_decay=0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=args.pretrain_epoch)

        epoch_loss, epoch_cl, epoch_rb = [], [], []
        for epoch in range(1, args.pretrain_epoch + 1):                 # 저장소 기본값 10
            l, lcl, lrb = model_pretrain(model, opt, sched, dl, configs, args, DEVICE)
            epoch_loss.append(float(l)), epoch_cl.append(float(lcl)), epoch_rb.append(float(lrb))
        t_fit = time.time() - t0
        torch.save({'model_state_dict': model.state_dict()},
                   f'{OUT_DIR}/checkpoints/{METHOD}_fold{f}_seed{seed}.pt')

        # 표현 추출은 pretrain=False 경로여야 한다 (h, z 를 돌려준다)
        model.training_mode = 'eval'
        z = np.concatenate([encode_windows(model, e_by_run[r], WINDOW, ENCODE_BATCH) for r in run_ids])
        gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
        pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
        r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, METHOD, f, seed)
        del z

        iters_per_epoch = len(dl)                                       # drop_last 반영된 참값
        fr['window'], fr['conv_out_len'], fr['repr_dims'] = WINDOW, CONV_OUT_LEN, REPR_DIMS
        fr['n_train_windows'] = len(x_train)
        fr['n_epochs'], fr['iters_per_epoch'] = args.pretrain_epoch, iters_per_epoch
        fr['n_iters'] = args.pretrain_epoch * iters_per_epoch
        # 데이터 통과 횟수 — drop_last 로 버려지는 나머지를 반영한 실효값 (마스킹 사본은 제외한 원본 기준)
        fr['data_passes'] = args.pretrain_epoch * iters_per_epoch * configs.batch_size / len(x_train)
        fr['samples_seen'] = args.pretrain_epoch * iters_per_epoch * configs.batch_size * WINDOW
        fr['fit_sec'], fr['total_sec'] = t_fit, time.time() - t0
        fr['epoch_mean_loss_last'] = epoch_loss[-1]
        fr['final_loss'] = float(np.mean(epoch_loss[-3:]))              # 마지막 3 epoch 평균
        fr['final_loss_cl'], fr['final_loss_rb'] = epoch_cl[-1], epoch_rb[-1]
        fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
        lo = pd.DataFrame({'fold': f, 'seed': seed, 'epoch': np.arange(1, len(epoch_loss) + 1),
                           'loss': epoch_loss, 'loss_cl': epoch_cl, 'loss_rb': epoch_rb})
        run_m.append(r), fold_m.append(fr), loss_m.append(lo)

        ev.append_progress(OUT_DIR, METHOD, 'progress_fold', fr)        # 중간 저장 (끊겨도 여기까지 남는다)
        ev.append_progress(OUT_DIR, METHOD, 'progress_run', r)
        ev.append_progress(OUT_DIR, METHOD, 'loss', lo)
        print(f'[Step 3] {METHOD} fold {f} seed {seed}: 학습 {t_fit:.0f} s, 전체 {time.time() - t0:.0f} s, '
              f'epoch {args.pretrain_epoch} 손실 {epoch_loss[0]:.4f} -> {epoch_loss[-1]:.4f} '
              f'(대조 {epoch_cl[-1]:.4f} / 복원 {epoch_rb[-1]:.4f}), '
              f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
              f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
              f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

# =============================================================================
# Step 4. 산출물 저장
# =============================================================================
run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
if not INFER:                                                           # 추론 모드는 학습하지 않아 손실 곡선이 없다
    ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, METHOD,
                        'Masked reconstruction + contrastive loss', x_col='epoch',
                        xlabel='Epoch (SimMTM 은 iteration 손실 훅이 없다 — iters_per_epoch 는 fold_metrics 참조)')
ev.save_results(OUT_DIR, METHOD, run_m, fold_m,
                {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS,
                 'WINDOW': WINDOW, 'ROLL_CIRCUMFERENCE_MM': ROLL_CIRCUMFERENCE_MM,
                 'config_base': 'SleepEEG — **강제**. Epilepsy config 는 CNNoutput_channel 이 없어 '
                                '사전학습 모델을 만들 수 없다 (model.py:35)',
                 'branch': 'SimMTM_Classification/code (Forecasting 브랜치는 예측 과제용)',
                 'CONV_OUT_LEN': CONV_OUT_LEN, 'CNNoutput_channel_original': SleepEEGConfig().CNNoutput_channel,
                 'REPR_DIMS': REPR_DIMS,
                 'repr_form': 'h = 평탄화 conv 특징 (저장소 target_classifier 가 쓰는 형태)',
                 'repr_dtype': 'float64 — 고차원 표현에서 float32 가 GMM 적합을 깨뜨린 선례가 있다 '
                               '(docs/tstcc_config_choice.md §7.5). 다른 비교군은 float32',
                 'head_override': f'nn.Linear({REPR_DIMS}, {WINDOW}) — 원본은 nn.Linear(1280, 178) 로 '
                                  '178 을 하드코딩한다. 복원 손실이 입력 길이와 비교하므로 반드시 맞춰야 한다',
                 'PRETRAIN_EPOCH': args.pretrain_epoch, 'PRETRAIN_LR': args.pretrain_lr,
                 'config_lr_unused': SleepEEGConfig().lr,
                 'BATCH_SIZE': configs.batch_size,
                 'EFFECTIVE_BATCH': configs.batch_size * (args.positive_nums + 1),
                 'BETAS': (configs.beta1, configs.beta2), 'WEIGHT_DECAY': 0,
                 'SCHEDULER': f'CosineAnnealingLR(T_max={args.pretrain_epoch})',
                 'MASKING_RATIO': args.masking_ratio, 'LM': args.lm,
                 'POSITIVE_NUMS': args.positive_nums, 'TEMPERATURE': args.temperature,
                 'DROP_LAST': configs.drop_last, 'DROPOUT': configs.dropout,
                 'KERNEL_SIZE': configs.kernel_size, 'STRIDE': configs.stride,
                 'FINAL_OUT_CHANNELS': configs.final_out_channels,
                 'aug_note': 'config 의 jitter/scaling/permutation 설정은 사전학습에 쓰이지 않는다 — '
                             'model_pretrain 은 data_transform_masked4cl 로 마스킹만 한다 (trainer.py:122)',
                 'ENCODE_BATCH': ENCODE_BATCH, 'DEVICE': DEVICE,
                 'repo': 'thuml/SimMTM (SimMTM_Classification)',
                 'commit': '169513bef74fb676e48d98a0e30f8823793f691c'})
for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
    ev.plot_condition_heatmap(run_m, metric, OUT_DIR, METHOD, label)
print(f'[done] {METHOD} 평가 완료')
