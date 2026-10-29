# =============================================================================
# 03_baselines/04_cost.py : CoST 비교군 어댑터
#
# 방법 : CoST (Woo et al., ICLR 2022) — 추세(trend)·계절(season) 성분을 분리해 대조학습한다.
#        추세는 서로 다른 kernel 의 인과 conv 혼합 + MoCo 방식 queue 대조,
#        계절은 주파수 영역(진폭·위상) 대조로 학습한다.
#
# 외부 저장소 : https://github.com/salesforce/CoST
#   clone 위치 : code/03_baselines/external/CoST  (git 추적 제외)
#   커밋       : 7bb5271c2b66cd5a1d634c27f5df308f190b50ec
#   원본은 한 줄도 수정하지 않는다. 본 파일이 어댑터(데이터 변환 + 호출 + 평가)다.
#
# 구조가 강제하는 제약 (임의 선택이 아니다 — docs/worklog.md 2026-10-23 §5)
#   1) 출력이 창의 마지막 시점 1개뿐이다 (`cost.py:334` 가 `out[:, -1]` 로 하드코딩).
#      -> 시점별 표현을 얻으려면 이동창이 필요하다. 저장소 자체도 그렇게 쓴다
#         (`tasks/forecasting.py`: casual=True, sliding_length=1, sliding_padding=...).
#      (창 길이가 학습 때와 같아야 하므로 sliding_padding = W - 1 로 창 길이를 정확히 W 로 맞춘다)
#   2) 계절 성분이 `BandedFourierLayer(length=W)` 로 **창 길이에 맞춰 가중치 크기가 고정**된다.
#      즉 창 길이는 hyperparameter 가 아니라 모델 파라미터 shape 이다.
#      주의: 이 불일치는 코드가 막아주지 않는다 — 창이 W 보다 **길면** rfft 주파수 bin 을
#      앞에서 잘라 쓰면서 조용히 통과한다(주파수 대응이 어긋난 채 동작). 짧을 때만 einsum 에서
#      실패한다. 실측으로 확인했다 (win_len 257, 357 통과 / 129 실패). 따라서 어댑터가
#      sliding_padding = W - 1 을 정확히 지켜야 한다.
#   3) 추세 분기의 최장 kernel 이 128 이라 창이 128 보다 커야 실제 데이터를 본다.
#
# 창 길이 : 샘플링 주파수에서 유도한다. steady 구간 주기 성분의 최저 주파수가
#   V=50 에서 0.06 Hz 이므로(research_plan.md 1.2절), 창 하나가 1주기를 담으려면
#   W >= fs / f_min = 9.174 / 0.06 ~= 153 샘플. 제약 3)과 함께 만족하는 가장 작은
#   2의 거듭제곱이 256 이다. 민감도를 보기 위해 4종을 모두 돌린다 (본문 표는 256).
#
# hyperparameter : 창 길이를 뺀 나머지는 저장소 기본값 그대로 (train.py argparse).
#   반복수도 저장소 기본값을 쓴다 (docs/worklog.md 2026-10-23 §5 의 결정).
#   '손실 수렴까지 학습' 을 검토했으나 CoST 에는 적용하지 않기로 했다 —
#   `adjust_learning_rate` 가 총 n_iters 기준 cosine 감쇠라서 학습 끝에서 손실이 평탄해지는 것이
#   스케줄이 만든 인공물이고, N 을 2배로 늘리면 최종 손실이 계속 내려간다(실측: W=160 에서
#   N=10,000 -> 0.990, N=20,000 -> 0.742). 즉 '평탄함' 으로 수렴을 판정할 수 없고,
#   엄격히 적용하면 비용이 비현실적이 된다. 모든 비교군에 저장소 기본값을 쓰는
#   docs/research_experiments_plan.md 3절 원칙으로 돌아간다.
#
# 입력 : outputs/merged_data.csv        (`Trigger 4` 는 읽지 않는다 — 채점은 evaluation.py 안에서만)
#        splits.py 가 분할에 맞는 fold 파일을 읽는다 (A: folds.csv / C: folds_c.csv)
# 출력 : A -> outputs/03_baselines/04_cost/ , C -> outputs/04_speed_extrap/04_cost/
#        창 길이마다 run/fold/summary/config csv + heatmap + 손실곡선
# 실험 : 주 실험 (Film Tension 수준별 5-fold) 만 수행
# 실험 : 인자로 고른다 — 'tension' (A: 장력 5-fold) / 'c' (C: 저속 학습 -> 고속 test, 8 fold)
# 실행 : python code/03_baselines/04_cost.py        (A 실험, 기본값)
#        python code/03_baselines/04_cost.py c      (C 실험)
# =============================================================================
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'external', 'CoST'))
import evaluation as ev
import splits
from cost import CoST                                                   # 외부 저장소 원본
from utils import init_dl_program                                       # 외부 저장소 원본의 시드 초기화


SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'tension'   # 'tension' (A) | 'c' (C)
INFER = splits.set_amb_mode(sys.argv[2:])   # 2번째 인자 'amb' -> ambiguous 포함 채점 + 재학습 없이 추론만
# 예: python code/03_baselines/04_cost.py tension amb  /  ... 04_cost.py c amb
# =============================================================================
# hyperparameter
# =============================================================================
SEED = 0                        # 난수 시드 (seed 반복의 시작값)
N_SEEDS = 5                     # seed 반복 횟수

# 창 길이 5종. **본문 표는 144** — 회전 외란 주기에서 유도한 값이다
# (지름 약 249 mm 고정 롤러, 둘레 783.5 mm -> V=50 의 1주기 = 144 샘플.
#  docs/periodic_disturbance.md §7). 나머지 4종은 창 길이 민감도용이다.
# 5종 모두 5 fold × 5 seed 로 동등하게 돌린다.
WINDOWS = [144, 160, 256, 306, 512]
N_ITERS = None if SPLIT == 'tension' else 600
# A 실험 : 저장소 기본값 None. 내부 규칙(학습 데이터 원소 수 > 100,000 이면 600, 아니면 200)이 적용되고,
#          학습 168 run x 1300 = 218,400 이라 **항상 600** 이 선택됐다.
# C 실험 : **600 으로 고정한다** (연구자 결정 2026-10-26). fold 마다 학습 run 수가 60~120 으로 다른데
#          자동 규칙을 두면 cv_S1(60 run, 78,000) 만 200 이 되어 혼자 1/3 만 학습한다. 그러면 cv_S1 의
#          성능이 '외삽이 어려워서' 인지 '덜 학습해서' 인지 분리되지 않는다. 600 은 A 에서 저장소 규칙이
#          실제로 고른 값이므로 임의값을 새로 들인 것이 아니다.
#          **As-published 에서 벗어난 유일한 지점** (docs/hyperparameters.md 13.0)
                                # (학습 데이터 원소 수 > 100,000 이면 600, 아니면 200). 본 데이터는 600 이 된다

DEVICE = 'cuda'                 # 저장소 기본값 (train.py --gpu 0)
BATCH_SIZE = 8                  # 저장소 기본값 (train.py --batch-size)
LR = 0.001                      # 저장소 기본값 (train.py --lr)
REPR_DIMS = 320                 # 저장소 기본값 (train.py --repr-dims)
KERNELS = [1, 2, 4, 8, 16, 32, 64, 128]   # 저장소 기본값 (train.py --kernels)
ALPHA = 0.0005                  # 저장소 기본값 (train.py --alpha, 손실 가중)

IN_PATH = 'outputs/merged_data.csv'
OUT_DIR = splits.resolve(SPLIT, '04_cost (ablation)')
CKPT_DIR = splits.ckpt_dir(SPLIT, '04_cost (ablation)')   # 인코더를 읽는 곳 (추론 모드에서도 기존 실험 것을 쓴다)

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
run_ids = sp.run_ids                                                    # run 순서는 분할 파일을 따른다
print(f'[Step 1] rows = {len(df)}, runs = {len(run_ids)} ({time.time() - t0:.1f} s)')

# =============================================================================
# Step 2. 저장소 입력 형식으로 변환 : (n_instance, n_timestamps, n_features)
#         run 마다 길이가 1275~1300 으로 다르므로 최대 길이에 맞춰 NaN 으로 채운다
#         (CoST 는 TS2Vec 계열이라 결측을 NaN 으로 받도록 설계되어 있다)
# =============================================================================
t0 = time.time()
by_run = {r: g for r, g in df.groupby('run_id', sort=False)}
lengths = np.array([len(by_run[r]) for r in run_ids])
x = np.full((len(run_ids), lengths.max(), 1), np.nan)
for i, r in enumerate(run_ids):
    x[i, :lengths[i], 0] = by_run[r].e.to_numpy()
e_flat = np.concatenate([by_run[r].e.to_numpy() for r in run_ids])      # 패딩을 제거한 시점 순서 (표현과 같은 순서)
id_flat = pd.concat([by_run[r][['run_id', 'k', TS, RS, VS]] for r in run_ids], ignore_index=True)
print(f'[Step 2] 입력 배열 {x.shape}, run 길이 {lengths.min()}~{lengths.max()}, '
      f'패딩 비율 {np.isnan(x).mean():.3f} ({time.time() - t0:.1f} s)')


def unpad(z):
    # 역할 : (n_run, t_max, d) 표현에서 패딩 구간을 떼어내고 시점 순서로 이어 붙인다.
    # 입력 : z (n_run, t_max, d) encode 결과 / 전역 lengths 를 쓴다
    # 출력 : (sum(lengths), d) — e_flat, id_flat 과 같은 순서
    return np.concatenate([z[i, :lengths[i]] for i in range(len(lengths))])


# =============================================================================
# Step 3. 창 길이 × fold × seed 별로 학습 fold 에서 인코더를 자기지도 학습하고,
#         학습 fold 표현으로 트랙 A 후단(GMM)을 적합한 뒤 검증 fold 에 적용한다
# =============================================================================
#         중간 결과(fold×seed 지표, 학습 손실, 인코더 가중치)는 하나 끝날 때마다 바로 저장한다
if not INFER:                                                   # 추론 모드는 체크포인트를 쓰지 않는다
    os.makedirs(f'{OUT_DIR}/checkpoints', exist_ok=True)
for W in WINDOWS:
    method = f'CoST-W{W}'
    if os.path.exists(f'{OUT_DIR}/{method}_fold_metrics.csv'):      # 이미 돌린 창은 건너뛴다
        print(f'[Step 3] {method}: 이미 결과가 있어 건너뛴다', flush=True)
        continue
    run_m, fold_m, loss_m = [], [], []
    for f in sp.folds:
        # C 실험에서는 '학습 = 검증 아닌 전부' 가 아니다 (고속 run 은 CV fold 에서 미사용)
        is_train_run, is_val_run = sp.run_role(f)
        is_train_pt, is_val_pt = sp.point_role(f, lengths)
        for seed in range(SEED, SEED + N_SEEDS):
            t0 = time.time()
            init_dl_program(0 if DEVICE == 'cuda' else 'cpu', seed=seed)    # 저장소 원본의 시드 초기화
            # 손실은 **iteration 단위**로 받는다. 저장소의 fit 은 epoch 평균만 반환하는데,
            # CoST 는 max_train_length 가 창 길이와 같아야 해서 run 이 1300//W 조각으로 쪼개지고
            # 그 결과 1 epoch 의 iteration 수가 창마다 달라진다 (W=160 은 168, W=512 는 42).
            # 즉 epoch 축으로는 창 간 비교가 불가능하고, 600 iter 에서는 기록 점이 3~14개뿐이다.
            # after_iter_callback 은 저장소가 제공하는 확장점이므로 원본 수정이 아니다.
            iter_loss, epoch_iters = [], []
            model = CoST(input_dims=1, kernels=KERNELS, alpha=ALPHA, max_train_length=W,
                         output_dims=REPR_DIMS, device=DEVICE, lr=LR, batch_size=BATCH_SIZE,
                         after_iter_callback=lambda _m, l: iter_loss.append(l),
                         after_epoch_callback=lambda m, _l: epoch_iters.append(m.n_iters))
            if INFER:                                                   # 재학습 없이 저장된 인코더만 불러온다
                ck = f'{CKPT_DIR}/{method}_fold{f}_seed{seed}.pt'
                assert os.path.exists(ck), f'체크포인트가 없다: {ck} (먼저 같은 분할로 학습을 돌릴 것)'
                model.load(ck)
                loss_log = []
                t_fit = time.time() - t0
            else:
                loss_log = model.fit(x[is_train_run], n_iters=N_ITERS, verbose=False)   # 학습 fold 신호만, 라벨 없음
                t_fit = time.time() - t0
                model.save(f'{OUT_DIR}/checkpoints/{method}_fold{f}_seed{seed}.pt')    # 학습된 인코더 (재현·재분석용)

            # 이동창 encode : 창 길이 = sliding_padding + 1 이어야 BandedFourierLayer 의 길이와 맞는다.
            # casual=True 로 창을 [i-W+1, i] 로 두어 미래 시점을 보지 않게 한다 (인과).
            z = unpad(model.encode(x, mode='forecasting', casual=True,
                                   sliding_length=1, sliding_padding=W - 1, batch_size=len(run_ids)))
            gmm, sidx, emeans = ev.fit_steady_gmm(z[is_train_pt], e_flat[is_train_pt], seed)
            pred, score = ev.predict_steady(gmm, sidx, z[is_val_pt])
            r, fr = ev.score_fold(id_flat[is_val_pt], pred, score, method, f, seed)
            fr['window'] = W
            fr['total_sec'] = time.time() - t0
            if INFER:                                       # 학습량 기록은 학습했을 때만 의미가 있다
                fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
                run_m.append(r), fold_m.append(fr)
                ev.append_progress(OUT_DIR, method, 'progress_fold', fr)
                ev.append_progress(OUT_DIR, method, 'progress_run', r)
                print(f'[Step 3] W {W} fold {f} seed {seed}: 체크포인트 재사용 (학습 생략), '
                      f'전체 {time.time() - t0:.0f} s, '
                      f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
                      f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                      f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)
                continue
            fr['fit_sec'] = t_fit
            fr['n_epochs'], fr['n_iters'] = model.n_epochs, model.n_iters
            # 1 epoch 의 iteration 수. n_iters // n_epochs 로 구하면 마지막 epoch 이 중간에 끊겨
            # 참값보다 크게 나온다(600 iter, 3 epoch -> 200 이지만 참값은 168). 첫 epoch 끝의 누적 iter 가 참값이다.
            fr['iters_per_epoch'] = epoch_iters[0] if epoch_iters else np.nan
            # 같은 iteration 수에서도 창 길이에 따라 학습 데이터를 통과하는 횟수가 달라진다
            # (run 이 1300//W 조각으로 쪼개지므로). 창 길이 효과와 학습량 효과가 교란되는 지점이다.
            fr['data_passes'] = model.n_iters / epoch_iters[0] if epoch_iters else np.nan
            fr['samples_seen'] = model.n_iters * BATCH_SIZE * W
            fr['epoch_mean_loss_last'] = loss_log[-1] if len(loss_log) else np.nan   # 저장소가 주는 epoch 평균
            fr['final_loss'] = float(np.mean(iter_loss[-20:])) if iter_loss else np.nan   # 마지막 20 iter 평균
            fr['steady_cluster'], fr['e_mean_c0'], fr['e_mean_c1'] = sidx, emeans[0], emeans[1]
            lo = pd.DataFrame({'fold': f, 'seed': seed,
                               'iter': np.arange(1, len(iter_loss) + 1), 'loss': iter_loss})
            run_m.append(r), fold_m.append(fr), loss_m.append(lo)

            ev.append_progress(OUT_DIR, method, 'progress_fold', fr)     # 중간 저장 (끊겨도 여기까지 남는다)
            ev.append_progress(OUT_DIR, method, 'progress_run', r)
            ev.append_progress(OUT_DIR, method, 'loss', lo)
            loss_txt = f'{loss_log[0]:.4f} -> {loss_log[-1]:.4f}' if len(loss_log) else '기록 없음'
            print(f'[Step 3] W {W} fold {f} seed {seed}: 학습 {t_fit:.0f} s, 전체 {time.time() - t0:.0f} s, '
                  f'epoch {model.n_epochs} 손실 {loss_txt}, '
                  f'군집별 e 평균 {np.round(emeans, 3)} -> steady = 군집 {sidx}, '
                  f'steady 판정 비율 {pred.mean():.3f}, 오염률 {fr.pooled_contamination_rate.iloc[0]:.3f}, '
                  f'보존률 {fr.pooled_retention_rate.iloc[0]:.3f}', flush=True)

    # ---- 창 길이 하나가 끝날 때마다 최종 산출물을 저장한다 ----
    run_m, fold_m = pd.concat(run_m, ignore_index=True), pd.concat(fold_m, ignore_index=True)
    if not INFER:                                               # 추론 모드는 학습하지 않아 손실 곡선이 없다
        ev.plot_loss_curves(pd.concat(loss_m, ignore_index=True), OUT_DIR, method, 'CoST contrastive loss')
    ev.save_results(OUT_DIR, method, run_m, fold_m,
                    {**ev.EVAL_CONFIG, 'SPLIT': SPLIT, 'SEED': SEED, 'N_SEEDS': N_SEEDS, 'WINDOW': W, 'N_ITERS': N_ITERS,
                     'DEVICE': DEVICE, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR, 'REPR_DIMS': REPR_DIMS,
                     'KERNELS': KERNELS, 'ALPHA': ALPHA, 'SLIDING_PADDING': W - 1, 'CASUAL': True,
                     'repo': 'salesforce/CoST', 'commit': '7bb5271c2b66cd5a1d634c27f5df308f190b50ec'})
    for metric, label in [('contamination_rate', 'Contamination rate'), ('retention_rate', 'Retention rate')]:
        ev.plot_condition_heatmap(run_m, metric, OUT_DIR, method, label)
    print(f'[done] CoST W={W} 평가 완료', flush=True)

print('[done] CoST 전체 완료')
