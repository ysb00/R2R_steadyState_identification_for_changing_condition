# TS-TCC config 기준 선택 — 왜 Epilepsy 인가

- 작성일: 2026-10-26
- 결정: **`config_files/Epilepsy_Configs.py` 를 기준으로 쓴다** (사용자 결정)
- 관련 문서: `docs/hyperparameters.md`(설정값 전체), `docs/research_experiments_plan.md`(실험 설계), `docs/research_plan.md` 0절(라벨 미사용 전제), `docs/periodic_disturbance.md`(창 길이 유도)
- 적용 코드: `code/03_baselines/07_tstcc.py`

---

## 1. 왜 선택이 필요한가 — "저장소 기본값" 이 하나가 아니다

다른 비교군은 저장소 기본값이 단일하다. TS2Vec·CoST·SoftCLT 는 `train.py` 의 argparse 기본값
하나이고, MOMENT 는 사전학습 가중치 그대로다.

**TS-TCC 는 다르다.** 데이터셋별 config 파일 4개(`Epilepsy` / `HAR` / `pFD` / `sleepEDF`)를 주고,
각각이 **모델 구조·학습·증강 값을 따로 들고 있다.** 공용 기본값이 없다.

그래서 "As-published 로 돌린다" 고 정해도 **어느 데이터셋의 기본값인지를 우리가 골라야 한다.**
이 선택은 피할 수 없다.

### 1.1 config 4개가 실제로 얼마나 다른가

| 항목 | Epilepsy | HAR | pFD | sleepEDF |
|---|---|---|---|---|
| `input_channels` | **1** | 9 | 1 | 1 |
| 원 데이터 길이 | **178** | 128 | 5,120 | 3,000 |
| `kernel_size` / `stride` | **8 / 1** | 8 / 1 | 32 / 4 | 25 / 3 |
| `features_len` | 24 | 18 | 162 | 127 |
| `num_classes` | 2 | 6 | 3 | 5 |
| `num_epoch` | 40 | 40 | 40 | 40 |
| `batch_size` | 128 | 128 | **64** | 128 |
| `lr` | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| `dropout` | 0.35 | 0.35 | 0.35 | 0.35 |
| `TC.hidden_dim` | 100 | 100 | 64 | 64 |
| `TC.timesteps` | **10** | 6 | 50 | 50 |
| **`jitter_scale_ratio`** (weak 증강) | **0.001** | 1.1 | **2** | 1.5 |
| **`jitter_ratio`** (strong 증강) | **0.001** | 0.8 | 0.1 | **2** |
| **`max_seg`** (permutation 분할 수 상한) | 5 | 8 | 5 | **12** |

`num_epoch`·`lr`·`dropout` 은 네 config 가 같지만, **구조(커널·stride·채널)와 증강 세기는 크게 다르다.**
증강 파라미터는 **최대 2,000배** 차이가 난다 (`jitter_scale_ratio` 0.001 vs 2).

## 2. 왜 성능을 보고 고를 수 없는가

`research_plan.md` **0절**이 라벨 사용을 금지한다. 네 config 를 모두 돌려 보고 가장 성능이 좋은
것을 고르면 **라벨로 하이퍼파라미터를 선택한 것**이 되고, 그것은 전제 위반이다.
다른 모든 비교군에 적용한 "라벨로 조정하지 않는다" 원칙과도 어긋난다.

그래서 **데이터 특성만으로** 골랐다. 성능은 선택 근거에 들어가지 않았다.

## 3. 선택 근거 — 세 기준이 모두 Epilepsy 를 가리킨다

| # | 기준 | 본 데이터 | 판정 |
|---|---|---|---|
| 1 | **채널 수** | 단일 채널 (e = `Film Tension #1` − 설정값) | Epilepsy 1 ✅ / pFD 1 ✅ / sleepEDF 1 ✅ / **HAR 9 ❌** |
| 2 | **입력 길이 규모** | 창 **144** (규칙 1b 유도값) | **Epilepsy 178 ✅ (가장 가깝다)** / HAR 128 / sleepEDF 3,000 / pFD 5,120 |
| 3 | **커널·stride 규모** | 짧은 입력에 맞는 조합이 필요 | **Epilepsy 8 / 1 ✅** / HAR 8 / 1 ✅ / sleepEDF 25 / 3 / pFD 32 / 4 |

기준 1 이 HAR 을 배제하고, 기준 2·3 이 남은 셋 중 Epilepsy 를 가리킨다.

**기준 3 이 왜 중요한가**: `kernel_size=32, stride=4`(pFD)나 `25, 3`(sleepEDF)은 길이 수천 샘플의
입력을 전제한 설정이다. 창 144 에 stride 4 를 쓰면 첫 conv 에서 길이가 1/4 로 줄고, 이후
MaxPool 3회(각 1/2)를 더 거쳐 특징맵이 과도하게 짧아진다. 그러면 `TC` 모듈이 요구하는
**특징맵 길이 > `TC.timesteps`** 조건이 위태로워진다 (pFD·sleepEDF 는 `timesteps=50` 이라 더 심하다).
Epilepsy 는 `timesteps=10` 이고 창 144 에서 특징맵 길이가 20 이라 조건을 만족한다.

### 3.1 선택에 쓰지 않은 것

- **성능** — 0절 위반이므로 보지 않았다.
- **저장소에서 가장 많이 쓰인 config** — 논문 대표 데이터셋은 HAR 이지만 9채널이라 구조가 맞지 않는다.
- **증강 세기** — 이것이 결과에 큰 영향을 주지만(아래 §4), **영향이 크다는 이유로 고르면 결국
  성능 기준이 된다.** 그래서 선택 근거에서 제외하고, 선택의 **결과**로만 기록한다.

## 4. 이 선택의 결과 — Epilepsy config 에서 TS-TCC 는 사실상 "2배 스케일 vs permutation" 이 된다

증강 세기가 결과를 크게 바꿀 수 있으므로, 선택의 귀결을 명시해 둔다.

### 4.1 증강 구조

```python
# dataloader/augmentations.py:5-9
def DataTransform(sample, config):
    weak_aug   = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg),
                        config.augmentation.jitter_ratio)
    return weak_aug, strong_aug
```

### 4.2 `scaling()` 의 곱셈 인자는 평균이 **2.0** 이다

```python
# dataloader/augmentations.py:16-23
def scaling(x, sigma=1.1):
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ...
```

`loc=2.` 이므로 **weak view 는 신호를 평균 2배로 키운다.** 이것은 Epilepsy 만의 특성이 아니고
네 config 공통이다 — 바뀌는 것은 그 2배 주변의 산포(`sigma`)뿐이다.

Epilepsy 의 `jitter_scale_ratio = 0.001` 이면 인자가 N(2.0, 0.001) 이므로 **거의 정확히 2배 고정**이다.
실측으로 확인했다 — 학습 창에서 `weak / 원본` 의 중앙값이 **2.000** 이었다.

### 4.3 `jitter_ratio = 0.001` 은 본 데이터에서 사실상 0 이다

strong view 의 가산 잡음 표준편차가 0.001 kg 인데, steady 구간 e 의 표준편차가 **0.041 kg** 이다.
즉 잡음이 신호 변동의 **2.4%** 수준이라 무시할 수 있다. 실측에서도 strong view 의 값 범위가
원본과 거의 같았다 (원본 −2.065~0.605 vs strong −2.066~0.606).

### 4.4 그래서 두 view 의 차이는 거의 전부 permutation 에서 온다

| view | Epilepsy config 에서의 실제 내용 |
|---|---|
| **weak** | 원본 × 약 2.0 (거의 결정론적) |
| **strong** | `permutation(원본)` + 무시 가능한 잡음 |

**결과적으로 TS-TCC 는 "2배 스케일된 원본" 과 "구간 순서가 섞인 원본" 을 같은 상태로 보도록
학습한다.** 대조 과제가 자명해지지는 않는다 — permutation 이 `max_seg=5` 로 충분히 강한 변형이기
때문이다 (시간축을 1~4 조각으로 임의 분할해 순서를 섞는다).

### 4.5 다른 config 를 골랐다면

| config | 본 데이터에서의 귀결 |
|---|---|
| **sleepEDF** | `jitter_ratio = 2` 는 steady e 표준편차 0.041 kg 의 **49배** 다. 구분 정보(진폭)를 완전히 덮어쓴다 |
| **pFD** | `jitter_scale_ratio = 2` 는 인자가 N(2, 2) 이라 **약 16% 확률로 부호가 뒤집힌다**. 진폭 정보가 심하게 교란된다 |
| **HAR** | `jitter_ratio = 0.8` 는 0.041 kg 의 **20배**. 그리고 9채널 구조라 애초에 맞지 않는다 |

**Epilepsy 는 네 config 중 증강이 가장 약하고, 따라서 e 의 진폭 정보를 가장 많이 보존한다.**
본 과제의 구분 정보가 진폭에 있으므로(미안정 0.209 kg vs steady 0.041 kg) 이는 TS-TCC 에 유리한
방향일 수 있다. **다만 이것은 선택의 근거가 아니라 선택의 결과다** — 근거는 §3 의 세 기준이다.

## 5. 부수 효과 — A6 ablation 이 깨끗해진다

`research_experiments_plan.md` §8.3 의 **A6 (TS-TCC permutation 제거 ablation)** 은
"입력 구간의 수렴 순서를 보존하는 것이 중요한지" 를 묻는다.

Epilepsy config 에서는 **두 view 의 차이가 거의 전부 permutation 에서 오므로**, permutation 을
제거하면 그 기여가 거의 단독으로 측정된다. 다른 config 라면 큰 jitter·scaling 이 섞여 있어
permutation 의 효과가 묻힌다.

**단, 주의**: Epilepsy config 에서 permutation 을 완전히 제거하면 두 view 가
"원본 × 2.0" vs "원본" 이 되어 **거의 같은 쌍**이 된다. 이는 `research_experiments_plan.md` §8.4 가
경고한 "증강을 전부 제거하면 대조 과제가 자명해진다" 에 해당한다. A6 설계 시 이 점을 반영해야
한다 (예: weak 의 scaling 은 남기고 strong 의 permutation 만 제거 — 그러면 "2배 vs 원본" 이 되는데,
`scaling` 의 산포가 0.001 이라 사실상 상수배여서 인코더가 쉽게 분리할 수 있다. **A6 을 돌리기 전에
이 자명화 여부를 손실 곡선으로 먼저 확인할 것.**)

## 6. config 외에 우리가 정한 값

Epilepsy config 를 쓰더라도 본 데이터에 맞춰 **구조가 강제하는 값**은 바꿔야 한다. 전부 런타임
shim 이고 원본 파일은 수정하지 않는다.

| 값 | config | 본 실험 | 성격 |
|---|---|---|---|
| 입력 창 길이 | 178 (Epilepsy 데이터 길이) | **144** | **[데이터]** 규칙 1b (회전 외란 1주기). 저장소 값은 타 데이터셋용 하드코딩이라 본 데이터에 의미 없음 |
| `features_len` | 24 | **20** | **[구조]** conv 3블록 통과 후 특징맵 길이. 입력 길이가 바뀌면 따라 바뀐다. **공식 추정이 아니라 더미 입력으로 실측**했고, 입력 178 에서 config 값 24 가 재현되는 것까지 확인했다 |
| `input_channels` | 1 | 1 | 변경 없음 (Epilepsy 가 이미 단일 채널) |
| `num_classes` | 2 | 2 | 자기지도 모드에서 `logits` 층이 쓰이지 않아 **결과에 영향 없음.** 모델 생성에 필요해 형식상 유지 |
| 표현 형태 | — | 특징맵 **평탄화 2,560 차원** (20 × 128) | TS-TCC **자신의 분류 헤드와 같은 형태**다 (`model.py:34` `x_flat = x.reshape(x.shape[0], -1)`) |
| 학습 샘플 | — | run 을 겹치지 않는 144 창으로 자름 (총 1,345개) | CoST 의 run 분할과 같은 취지. **뒤쪽 불완전 꼬리를 버린다** — 앞쪽을 버리면 과도구간(소수 클래스)이 학습에서 사라진다 |
| 표현 추출 | — | causal 이동창 stride 1 | `research_experiments_plan.md` 3절. 창이 안 찬 앞 143 시점은 첫 완전창 표현으로 채운다 (MovStd-Slope 와 동일 규약 — 채점 시점 수를 방법 간 동일하게) |

## 7. 저장소 코드에서 기록해 둘 것

### 7.1 numpy 비호환 — `permutation()` 이 깨진다

```
ValueError: setting an array element with a sequence.
            The requested array has an inhomogeneous shape after 1 dimensions.
```

`augmentations.py:42` 의 `np.random.permutation(splits)` 에서 `splits` 가 **길이가 다른 배열들의
리스트**다(시간축을 임의 지점에서 쪼갠 조각들). numpy>=1.24 는 ragged 리스트의 배열화를 거부한다.

**처리**: 원본 미수정 원칙에 따라 런타임 shim 으로 함수를 바꿔 끼웠다
(`07_tstcc.py: _permutation_numpy_compat`). `permutation(리스트)` 의 의미는 **리스트 원소의 순서를
섞는 것**이므로 동일 의미의 인덱스 셔플(`np.random.permutation(len(splits))`)로 대체했다.

**난수 동등성**: `np.random.permutation(길이 n 리스트)` 와 `np.random.permutation(n)` 은 모두 길이 n
1-D 에 Fisher-Yates 를 적용하므로 **난수 소비가 같다.** 따라서 예전 numpy 에서 원본이 냈을 결과와
비트 단위로 같고, 증강의 세기나 분포를 바꾸지 않는다.

### 7.2 증강은 epoch 마다 다시 뽑지 않는다

`Load_Dataset.__init__` 에서 `DataTransform` 을 **한 번만** 호출한다(`dataloader.py:35`).
즉 `aug1`/`aug2` 가 전 epoch 동안 고정된다. 보통의 대조학습이 epoch 마다 증강을 재추출하는 것과
다르므로, 학습 다양성이 그만큼 낮다. **원본 동작이므로 그대로 둔다.**

### 7.3 `drop_last=True` 가 필수다

`NTXentLoss._get_correlated_mask()` 가 `batch_size` 고정 마스크를 만들므로(`loss.py:23-29`)
마지막 배치가 작으면 깨진다. Epilepsy config 가 `drop_last = True` 이므로 그대로 쓴다.
그 결과 학습창 1,345개 / batch 128 → **iter/epoch = 10** (나머지 65개는 매 epoch 버려진다).

### 7.4 `Trainer()` 를 쓰지 않는다

`Trainer()` 는 **아무것도 반환하지 않아** 손실을 기록할 수 없다(`trainer.py:15-43`).
자기지도 모드에서 Trainer 가 하는 일은 ① `model_train` 호출 ② `model_evaluate`(자기지도에서는
`pass` 로 무동작, `trainer.py:121`) ③ 로깅 ④ 체크포인트 저장뿐이고,
**scheduler 는 `training_mode != 'self_supervised'` 조건이라 step 되지 않는다**(`trainer.py:26`).

따라서 저장소의 `model_train` 을 `num_epoch` 번 호출하는 것은 **수치적으로 Trainer 와 동일하다.**
손실은 epoch 단위로만 얻는다(iteration 훅이 없다). `iters_per_epoch` 를 함께 기록해 다른 방법의
iteration 축과 환산할 수 있게 했다.

### 7.5 무해한 경고 1건

`TC.py:14` 의 `nn.LogSoftmax()` 가 `dim` 을 지정하지 않아 deprecation 경고가 난다.
2-D 입력에서는 기존 동작과 같으므로 결과에 영향이 없다. 원본 미수정.

## 8. 학습량 — As-published 그대로

| 항목 | 값 | 출처 |
|---|---|---|
| `num_epoch` | **40** | [저장소] 네 config 모두 40 |
| `batch_size` | 128 | [저장소] Epilepsy |
| `lr` / `betas` / `weight_decay` | 3e-4 / (0.9, 0.99) / 3e-4 | [저장소] config + `main.py:139` |
| iter/epoch | 10 | 학습창 1,345 ÷ 128, `drop_last` 반영 |
| 총 iteration | 400 | 40 × 10 |
| **데이터 통과 횟수** | **약 38.1** | 40 × 10 × 128 ÷ 1,345. `drop_last` 로 버려지는 나머지 반영 |

데이터 통과 횟수 38.1 은 CoST(3.17)·TS2Vec·SoftCLT(28.6)·SimMTM(10 예정)과 다르다.
**이 비동등성은 As-published 방침의 알려진 한계이고, 동일 epoch 예산 정렬은 후일 작업으로 미뤘다**
(`docs/hyperparameters.md` §13.2). 두 설정을 병기해 순위 안정성을 보고하는 것이 계획이다.

---

## 9. 요약

1. TS-TCC 는 데이터셋별 config 4개를 주므로 **"저장소 기본값" 이 하나가 아니고**, 선택이 불가피하다.
2. 성능으로 고르면 **라벨 기반 선택**이 되어 0절 위반이므로, **데이터 특성만으로** 골랐다.
3. **채널 수(단일) · 입력 길이(178 이 창 144 에 가장 가깝다) · 커널 규모(8/1) 세 기준이 모두
   Epilepsy 를 가리킨다.**
4. 그 결과 Epilepsy config 에서 TS-TCC 는 사실상 **"2배 스케일 vs permutation"** 대조학습이 되고,
   네 config 중 **증강이 가장 약해 진폭 정보를 가장 많이 보존한다.** 이는 선택의 **결과**이지 근거가 아니다.
5. 창 144·`features_len` 20·표현 평탄화·학습창 분할은 **구조가 강제하거나 전 방법 공통 규약**이며,
   원본 저장소는 수정하지 않고 런타임 shim 으로만 처리했다.
6. numpy>=1.24 에서 `permutation()` 이 깨지는 것을 **난수 소비가 동일한 shim** 으로 처리했다.
