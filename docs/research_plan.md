# Research Plan (v0.1)

**주제**: 운전조건 변화에 강건한, 표현학습 기반 정상상태(steady state) 구간 자동 라벨링 및 non-steady state 탐지
(Representation learning–based auto-labeling of steady-state segments under unseen roll-to-roll setting conditions)

- 작성일: 2026-10-16
- 상태: 설계 초안 (탐색적 분석 전, 가정 검증 필요)
- 용어: SSI = Steady-State Identification. 본 연구에서 "이상"은 제품 이상이 아니라 **non-steady state(장비 신호 미안정)** 를 뜻한다.

---

## 1. 연구 동기

1. **정상 데이터 선별 비용**: 이상(non-steady) 탐지·품질 모델 학습에는 steady state 데이터만 골라야 하며, 현재는 전문가가 수작업으로 선별한다 (본 데이터의 `Trigger 4`).
2. **운전조건 변화**: 장력·속도 레시피가 바뀌면 신호 수준, 과도 형태, 안정화 시간(0–52 s)이 달라진다. 모든 조건을 실험·라벨링할 수 없다.
3. **기존 SSI의 한계**: 통계 기반 SSI는 조건별 임계값 재튜닝이 필요하다.
   - Cao & Rhinehart (1995), *J. Process Control* 5(6):363–374. https://doi.org/10.1016/0959-1524(95)00009-F
   - Kelly & Hedengren (2013), *J. Process Control* 23(3):326–331. https://scholarsarchive.byu.edu/facpub/1713 (DOI 미확인)

## 2. 기대 효과 / 기여

- **라벨링 비용 절감**: 새 레시피·장비에서 전문가 없이 steady 데이터 확보
- **조건 일반화**: 일부 조건 학습 → 미학습 조건 적용 가능성을 정량 검증 (내삽/외삽 구분)
- **보수적 자동 라벨**: "보류(abstain)" 판정으로 steady 집합의 오염 최소화
- **활용처**: 이상 탐지 학습 데이터 생성, 실시간 SSI, 측정·품질분석 시작 시점 판단

## 3. 문제 정의

- **입력**: `Film Tension #1 (kg)` window + 설정값 3종 (`Film-Tension Setting Value (kg)`, `ReWinder-Tension Setting Value (kg)`, `Web Speed setting value (mm/sec)`). `Web Current Speed (mm/sec)`는 ablation으로만 사용.
- **출력**: 시점별 steady(1) / non-steady(0) / 보류
- **연구 질문**
  - **RQ1**: 라벨 없이(또는 소수 라벨로) 자동 라벨링 시 오염률·보존률이 전문가 라벨에 얼마나 근접하는가?
  - **RQ2**: 미학습 조건(내삽/외삽)에서도 성능이 유지되는가?
  - **RQ3**: 자동 라벨 steady 데이터로 학습한 non-steady 탐지기가 전문가 라벨 학습(상한) 대비 어느 수준인가?

## 4. 후보 알고리즘 (단계별)

| 단계 | 역할 | 후보 | 비고 |
|---|---|---|---|
| S0 전처리 | 정규화·window | e = T1 − 설정값 또는 T1/설정값; window 32/64/128 샘플 (≈3.5/7/14 s) | 설정값 정규화가 조건 일반화의 첫 단계 |
| S1 자기지도 사전학습 | 장력 window 표현 학습 | TNC [arXiv 2106.00750](https://arxiv.org/abs/2106.00750), TS2Vec [arXiv 2106.10466](https://arxiv.org/abs/2106.10466), TS-TCC [arXiv 2106.14112](https://arxiv.org/abs/2106.14112) | 소형 1D-CNN/TCN; 설정값 FiLM 주입 유무 비교 |
| S2 자동 라벨링 | steady/non-steady/보류 판정 | (a) PU 학습: run 후반 = positive, 나머지 = unlabeled (Elkan & Noto, KDD 2008; nnPU, Kiryo et al., NeurIPS 2017, https://arxiv.org/abs/1703.00593) (b) steady prototype 거리 + 보수적 임계값 (c) 임베딩 군집화(GMM) (d) 시간 일관성 제약(run당 진입 1회 가정) | (a)가 데이터 구조를 가장 잘 활용. "모든 run이 steady로 끝난다"는 가정은 코드로 재확인 필요 |
| S3 one-class 탐지 | 자동 라벨 steady로 학습 → non-steady 탐지 | Deep SVDD (Ruff et al., ICML 2018, https://proceedings.mlr.press/v80/ruff18a.html), 임베딩 kNN 거리 | non-steady = `Trigger 4` = 0 |
| 기준선 | 비교 | R-test, Kelly–Hedengren, 이동창 표준편차 임계값; 상한: 전문가 라벨 지도학습; 하한: 선별 없이 전체 사용 | 동일 분할 |

참고(내용 미검토): [arXiv 2608.24194](https://arxiv.org/pdf/2608.24194), [arXiv 2506.04204](https://arxiv.org/pdf/2506.04204) — steady interval 검출 관련 최신 연구로 보임.

## 5. 실험 설정

- **데이터**: 210 runs (Film Tension 5 × ReWinder 6 × Speed 7). 분할은 **항상 run/조건 단위**.
- **분할 시나리오**
  - E1 (주 실험): Tension leave-one-level-out, 5-fold
  - E2 (내삽): 격자 내부 조건 제외 (예: T5 전체, V125 전체)
  - E3 (외삽): 격자 끝 조건 제외 (예: T7 전체, V175–200 전체)
  - validation도 조건 단위로 분리
- **라벨 사용 수준**: 0개(run 후반 가정만) / 소수(전문가 라벨 run 5·10·20개) / 전체(상한)
- **Ambiguous(2)**: 주 평가 제외, 보조 평가에서 양쪽 모두 정답 처리
- **지표**
  - 자동 라벨 품질: 오염률(steady로 고른 구간 중 실제 non-steady 비율, 최우선), 보존률, steady 진입 시점 오차(s)
  - 탐지: AUROC, AUPRC(non-steady = positive), F1, 탐지 지연
  - 조건별 성능 heatmap
- **통계**: seed 5회 평균 ± 표준편차, 평가 run 단위 paired 검정(Wilcoxon)
- **Ablation**: window 길이, 설정값 정규화, 조건 주입 방식, 사전학습 기법, `Web Current Speed` 포함 여부
- **계산 자원**: 소형 인코더 기준 CPU 학습 가능 예상 (미검증)

## 6. 위험과 대응

| 위험 | 대응 |
|---|---|
| 단순 기법(표준편차 임계값 등)이 이미 충분할 수 있음 | 기준선 선행; 차이가 작으면 외삽·라벨 0개 설정에서 기여 탐색 |
| 전문가 라벨의 비일관성 | 진입 시점 오차 분포 분석, ambiguous 라벨 활용 |
| 경계(window가 두 상태에 걸침) 처리 | window 라벨 규칙(마지막 시점 / 순수 window만) ablation |

## 7. 다음 단계

1. `02_eda.py`: S2 핵심 가정(모든 run이 steady로 종료) 확인, 단순 기준선 성능 확인
2. 환경: matplotlib, scikit-learn, PyTorch(CPU) 설치 — 클라우드 환경 네트워크 설정 변경 필요
3. S1 사전학습 착수
