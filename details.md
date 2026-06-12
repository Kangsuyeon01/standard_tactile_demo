# Roughness Generation Model

CNN + GRU + FiLM 기반 표면 거칠기 진동 생성 모델.  
입력(가속도 히스토리, 힘, 속도, 거칠기)에서 미래 가속도 신호를 예측한다.

---

> **⚡ 실시간 추론 최적화 (권장)**
>
> `realtime.py`는 **ONNX Runtime** 으로 구동하면 PyTorch 대비 빠른 추론이 가능합니다.  
>
> ```bash
> pip install onnx onnxruntime
> ```
>
> ONNX 모델은 첫 실행 시 자동 export되며, 이후 자동으로 사용됩니다.  
> 수동 export:
> ```bash
> python -m scripts.realtime --pt-path pt_files/runs/<run_id>/best_model.pt --cache-path pt_files/inference_cache_allinone.npz --onnx-path pt_files/runs/<run_id>/best_model.onnx
> ```

---

## 프로젝트 구조

```
roughness_model/
├── scripts/
│   ├── preprocess.py               # 1단계: CSV → NPZ 캐시 생성
│   ├── resplit_npz.py              # 기존 NPZ를 참가자 기반으로 재분할
│   ├── train.py                    # 2단계: 모델 학습 (학습 후 자동 평가 4종 실행)
│   ├── eval_test_samples.py        # [auto] 실제 vs 예측 신호 비교
│   ├── eval_comprehensive.py       # [auto] roughness × 조건/force/velocity 그리드 평가 + spectrogram
│   ├── eval_roughness_signal.py    # [auto] 고정 F/V에서 roughness별 신호 비교
│   ├── gen_ramp_plots.py           # [auto] force/velocity ramp 시나리오 시각화 (spectrogram 포함)
│   ├── gen_sweep_plot.py           # roughness × force × velocity 스윕 플롯
│   ├── analyze_fv_distribution.py  # 학습 데이터 Force/Velocity 분포 박스플롯
│   ├── analyze_velocity_spectrum.py # 속도 구간별 주파수 스펙트럼 분석
│   ├── benchmark_inference.py      # PyTorch vs ONNX Runtime 추론 속도 비교
│   ├── demo.py                     # 대화형 진동 출력 루프 (NI-DAQ)
│   └── realtime.py                 # Unity 소켓 서버 + 라이브 플롯 모드 (ONNX 지원)
├── pt_files/
│   ├── inference_cache_allinone.npz    # 전처리 캐시 (roughness label 기반 split)
│   ├── inference_cache_participant.npz # 참가자 기반 재분할 캐시
│   └── runs/
│       └── YYYYMMDD-NNN/              # run 별 결과
│           ├── best_model.pt
│           ├── report.json
│           ├── training_history.png
│           ├── test_samples/          # eval_test_samples.py 출력 (R005~R100.png)
│           └── roughness_eval/        # eval_roughness_signal.py 출력
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
conda create -n roughness python=3.10 -y
conda activate roughness
```

CPU:
```bash
pip install -r requirements.txt
```

ONNX Runtime (권장 — 추론 속도 ~24x 향상):
```bash
pip install onnx onnxruntime
```

GPU (CUDA 12.1):
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy pandas matplotlib
```

> `nidaqmx`는 NI-DAQ 하드웨어가 연결된 Windows 환경에서만 필요합니다.

## 데이터 형식

`pilot-data/` 아래에 CSV 파일이 있어야 합니다.

```
pilot-data/
  data_participant_1_roughness_5_trial_1.csv
  data_participant_1_roughness_45_trial_1.csv
  ...
```

각 CSV는 `Acceleration`, `Force`, `Velocity` 컬럼을 포함해야 합니다.  
전처리 완료된 NPZ 캐시가 있으면 `pt_files/inference_cache_allinone.npz`에 배치하고 전처리 단계를 건너뜁니다.

거칠기 분할:

| Split | Roughness |
|-------|-----------|
| Train | 5, 45, 100 |
| Val   | 12, 58 |
| Test  | 23, 66 |

## 전체 실행 순서

**Option A — 참가자 기반 split (권장: 모든 거칠기를 학습에 활용)**

```bash
cd roughness_model

# 1. 전처리 (캐시가 이미 있으면 생략)
python -m scripts.preprocess

# 2. 기존 NPZ를 참가자 기반으로 재분할 (전처리 재실행 불필요)
python -m scripts.resplit_npz

# 3. 학습 (재분할된 NPZ 사용)
python -m scripts.train --npz pt_files/inference_cache_participant.npz --note "participant split"

# 4. 거칠기별 신호 비교 평가
python -m scripts.eval_roughness_signal

# 5. 표준 평가 (RMSE / MAE / Corr, 특성 중요도)
python -m scripts.evaluate

# 6. 분석 플롯
python -m scripts.plot_analysis
```

**Option B — 거칠기 기반 split (원래 방식)**

```bash
cd roughness_model

python -m scripts.preprocess
python -m scripts.train
python -m scripts.eval_roughness_signal
```

---

## scripts/train.py — 모델 학습

```bash
python -m scripts.train
```

실행하면 `pt_files/runs/YYYYMMDD-NNN/` 폴더가 자동 생성되고 아래 파일이 저장됩니다.

| 파일 | 설명 |
|------|------|
| `best_model.pt` | 최적 체크포인트 |
| `report.json` | run ID, 인자, config, 학습 곡선, 테스트 지표, roughness별 RMS |
| `training_history.png` | 학습/검증 loss 곡선 |

학습이 끝나면 `eval_test_samples.py`와 `eval_comprehensive.py`가 자동으로 실행되어 `runs/{run_id}/` 아래에 평가 결과가 저장됩니다.

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | `pt_files` | 결과 저장 루트 |
| `--npz` | 자동 탐색 | NPZ 캐시 경로 직접 지정 |
| `--run-id` | 자동 생성 | 수동 지정 시 해당 ID 사용 |
| `--note` | `""` | report.json에 기록할 메모 |
| `--lambda-rms` | `10.0` | roughness RMS 보정 loss 가중치 |
| `--lambda-contrast` | `0.6` | roughness contrastive loss 가중치 |
| `--lambda-hf` | `0.0` | HF 에너지 비율 contrastive loss 가중치 (0=비활성) |
| `--lambda-centroid` | `0.0` | 스펙트럼 무게중심 loss 가중치 (0=비활성) |
| `--lambda-force` | `1.0` | force contrastive loss 가중치 |
| `--lambda-profile` | `2.0` | spectral profile loss 가중치 |
| `--lambda-rough-cls` | `1.0` | roughness classification auxiliary loss 가중치 |
| `--zero-augment-ratio` | `0.0` | zero-contact 샘플 augmentation 비율 |

```bash
# 참가자 split NPZ로 학습 (권장)
python -m scripts.train --npz pt_files/inference_cache_participant.npz --note "participant split"

# lambda 조정 예시 (run 008 기준)
python -m scripts.train \
  --npz pt_files/inference_cache_participant.npz \
  --lambda-rms 5.0 --lambda-contrast 3.0 --lambda-hf 1.0 \
  --zero-augment-ratio 0.05 --note "stronger_rms_contrast_hf"
```

> ⚠️ roughness label 기반 split(`inference_cache_allinone.npz`)에서 `--lambda-rms` / `--lambda-contrast`를 크게 올리면 val 손실이 폭발할 수 있습니다. val roughness(12, 58)가 train(5, 45, 100)에 없어서 rms/contrast loss가 오히려 잘못된 방향으로 학습됩니다. 이 경우 lambda를 낮추거나 participant split을 사용하세요.

---

## scripts/eval_roughness_signal.py — 거칠기별 신호 비교

고정된 force/velocity 조건에서 roughness만 바꿔 신호를 생성하고, 모델이 거칠기를 실제로 반영하는지 시각적으로 확인합니다.

```bash
# 최신 run 자동 사용
python -m scripts.eval_roughness_signal

# 특정 run 지정
python -m scripts.eval_roughness_signal --run-id 20260610-001

# NPZ 데이터의 평균 신호와 비교 추가
python -m scripts.eval_roughness_signal --run-id 20260610-001 --compare-data

# force/velocity 직접 지정
python -m scripts.eval_roughness_signal --force 1.96 --vel 0.067
```

### 출력 (`pt_files/runs/{run_id}/roughness_eval/`)

| 파일 | 설명 |
|------|------|
| `roughness_eval_combined.png` | 파형 + FFT + 지표 종합 1장 |
| `waveforms.png` | roughness별 파형 겹쳐서 비교 |
| `fft_spectrum.png` | roughness별 FFT 스펙트럼 |
| `metrics_bar.png` | RMS, HF ratio, ZCR, Crest Factor, Centroid 막대 그래프 |
| `metrics.csv` | 수치 정리 CSV |

### 확인 포인트

RMS와 HF ratio(>200Hz)가 roughness 증가에 따라 단조증가하면 FiLM conditioning이 올바르게 동작하는 것입니다.

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--run-id` | 최신 run | 평가할 run ID |
| `--force` | 데이터 median | 고정 force 값 |
| `--vel` | 데이터 median | 고정 velocity 값 |
| `--compare-data` | off | NPZ 데이터 평균 신호와 비교 추가 |
| `--out-dir` | `pt_files` | 결과 루트 디렉토리 |

---

## scripts/eval_test_samples.py — 실제 vs 예측 신호 비교

전체 데이터(train+val+test)에서 roughness label별로 Small/Mid/Large force 조건 대표 세그먼트를 선택해 원본(파란색) vs 모델 예측(주황색)을 비교합니다. 학습 종료 후 자동 실행됩니다.

```bash
# 최신 run 자동 사용
python -m scripts.eval_test_samples

# 특정 run 지정
python -m scripts.eval_test_samples --run-id 20260611-007

# NPZ 직접 지정
python -m scripts.eval_test_samples --run-id 20260611-007 --npz pt_files/inference_cache_allinone.npz
```

### 출력 (`pt_files/runs/{run_id}/test_samples/`)

| 파일 | 설명 |
|------|------|
| `R005.png` ~ `R100.png` | roughness별 3행(Small/Mid/Large) × 4열(Acc/Force/Vel/FFT) 비교 |

---

## scripts/eval_test_samples_blended.py — Reference-blended 신호 비교

`eval_test_samples.py`와 동일한 레이아웃이지만, `RealtimeReferenceGuidedGenerator`를 사용해 **realtime.py와 동일한 조건** (model + reference guide 블렌딩)으로 신호를 생성합니다. 실제 haptic 출력이 어떻게 보이는지 평가할 때 사용합니다.

```bash
# 기본 실행 (ref_blend=0.20, realtime.py 기본값과 동일)
python -m scripts.eval_test_samples_blended --run-id 20260611-007

# blend 비율 직접 지정
python -m scripts.eval_test_samples_blended --ref-blend 0.10

# pure model 출력 (reference 혼합 없음, eval_test_samples.py와 동일한 결과)
python -m scripts.eval_test_samples_blended --ref-blend 0.0
```

### 출력 (`pt_files/runs/{run_id}/test_samples_blended_blend020/`)

| 파일 | 설명 |
|------|------|
| `R005.png` ~ `R100.png` | roughness별 3행 × 4열, orange=blended 출력 |

### eval_test_samples vs eval_test_samples_blended 비교

| | eval_test_samples | eval_test_samples_blended |
|---|---|---|
| 생성 방식 | 배치 추론 (윈도우 단위) | 스트리밍 추론 (40샘플씩) |
| Reference 혼합 | 없음 | ref_blend 비율로 혼합 |
| 실제 haptic 출력과 일치 | ✗ | ✓ (ref_blend=0.20 시) |
| 모델 순수 성능 평가 | ✓ | ✗ (reference 포함) |

---

## scripts/eval_comprehensive.py — 종합 조건 그리드 평가

roughness × 조건/force/velocity 격자에서 모델 출력을 시각화합니다. 학습 종료 후 자동 실행됩니다.

```bash
python -m scripts.eval_comprehensive --run-id 20260611-007
```

### 출력

| 파일 | 설명 |
|------|------|
| `conditions_grid.png` | roughness × [Zero/Low/Mid/High] 조건 그리드 |
| `force_sweep_grid.png` | roughness × force 단계별 그리드 |
| `vel_sweep_grid.png` | roughness × velocity 단계별 그리드 |
| `force_ramp_grid.png` | force 0→max ramp, roughness별 |
| `vel_ramp_grid.png` | velocity 0→max ramp, roughness별 |
| `summary_rms.png` | RMS 히트맵 (이상적: Zero≈0, 행 기준 단조증가) |

---

## scripts/evaluate.py — 표준 평가

```bash
python -m scripts.evaluate
```

RMSE / MAE / Corr, Permutation Feature Importance, Saliency 히트맵을 계산하고 `pt_files/`에 저장합니다.

> ⚠️ 구버전 모델 경로(`best_model_light_wo71.pt`, `in_ch=4`)를 참조합니다. 새 run ID 기반 모델을 쓰려면 파일 상단 경로를 `runs/{run_id}/best_model.pt`로 수정하세요.

---

## scripts/analyze_fv_distribution.py — 학습 데이터 Force/Velocity 분포 분석

학습 데이터에서 거칠기별 Force/Velocity 분포를 박스플롯으로 시각화합니다.

```bash
python -m scripts.analyze_fv_distribution
python -m scripts.analyze_fv_distribution --npz pt_files/inference_cache_allinone.npz --out pt_files/fv_distribution.png
```

### 분석 결과 요약 (inference_cache_allinone.npz 기준)

| 항목 | 최솟값 | 평균 | 최댓값 (p99) |
|------|--------|------|--------------|
| Velocity (m/s) | 0.034 | 0.057 | 0.164 |
| Force (N) | 0.56 | 1.29 | 8.3 |

> **주의**: 모델은 학습 데이터 범위 안에서만 신뢰할 수 있는 출력을 냅니다. Velocity 0.164 m/s 이상에서는 출력이 수렴(saturate)합니다. 속도 효과를 제대로 학습시키려면 속도를 의도적으로 제어한 데이터가 필요합니다.

---

## scripts/realtime.py — Unity 실시간 소켓 서버

학습된 모델을 Unity와 연결해 실시간으로 촉각 진동 파형을 생성합니다.

### 통신 프로토콜

```
Unity → Python : struct.pack('<7f', roughness, force, vx, vy, vz, user_id, fingerIdx)  (28 bytes)
Python → Unity : uint16 파형 샘플 (little endian), IIR 필터 적용
  user_id 홀수 → 80 samples (5Glove,    Glove5_CH*)
  user_id 짝수 → 40 samples (KrissGlove, Glove3_CH*)
```

### 글로브 채널

| 기기 | user_id | fingerIdx |
|------|---------|-----------|
| KrissGlove (Glove3) | 짝수 | 0, 1, 2 → CH1, CH2, CH3 |
| 5Glove (Glove5)     | 홀수 | 0, 1 → CH1, CH2 |

### 실행

```bash
# 기본 실행
python -m scripts.realtime

# ONNX Runtime으로 실행 (권장 — 23x 빠름, 0.25ms/추론)
python -m scripts.realtime --pt-path pt_files/runs/20260612-003/best_model.pt --onnx-path pt_files/runs/20260612-003/best_model.onnx

# GPU 사용
python -m scripts.realtime --device cuda

# 특정 run 모델 사용
python -m scripts.realtime --pt-path pt_files/runs/20260612-003/best_model.pt

# 오프라인 신호 생성 테스트 (Unity 불필요)
python -m scripts.realtime --save-test-signal

# 라이브 플롯 모드 (슬라이더로 R/F/V 실시간 조절)
python -m scripts.realtime --live-plot --pt-path pt_files/runs/20260612-003/best_model.pt --cache-path pt_files/inference_cache_allinone.npz
```

### 라이브 플롯 모드 (`--live-plot`)

소켓 서버 없이 단독으로 실행하며, matplotlib 슬라이더로 Roughness / Force / Velocity를 실시간으로 조절하면서 모델 가속도 출력을 확인할 수 있습니다.

- **Roughness 슬라이더** (0~100): 2 이상 변화 시 reference guide 자동 재로드
- **Force 슬라이더** (0~10 N): 즉시 반영
- **Velocity 슬라이더** (0~0.25 m/s): 즉시 반영, target RMS 기준선도 같이 이동
- Force = 0 또는 Velocity = 0에 가까우면 출력이 0으로 감쇄 (`force_velocity_gate`)
- 상단: 롤링 파형 + 실시간 RMS / target RMS 표시
- 하단: RMS 이력 그래프

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--roughness-val` | `58.0` | 초기 roughness |
| `--force-val` | `1.96` | 초기 force (N) |
| `--vel-val` | `0.067` | 초기 velocity (m/s) |
| `--plot-window` | `8000` | 표시 샘플 수 (1초 = 8000) |
| `--plot-interval` | `20` | 갱신 주기 (ms) |

### 추론 시간 출력

소켓 서버 실행 중 패킷마다 아래 형식으로 추론 시간이 출력됩니다.

```
[TIME] samples=40 | predict=1.23ms | total=2.87ms | budget=5.00ms
```

- `predict`: 모델 추론 시간
- `total`: 패킷 수신 → 전송 전체 시간
- `budget`: 해당 샘플 수의 실시간 마감 시간 (num_samples / 8000 × 1000 ms)

> `--onnx-path`에 지정한 파일이 없으면 자동으로 export합니다. PyTorch CPU(~6ms) 대비 ONNX Runtime(~0.25ms)으로 **약 24x 빠릅니다.**

### 추론 속도 벤치마크

```bash
pip install onnx onnxruntime

# PyTorch vs ONNX Runtime 비교 (200회 측정)
python -m scripts.benchmark_inference \
  --pt-path pt_files/runs/20260610-004/best_model.pt \
  --use-onnx
```

오프라인 테스트(`--save-test-signal`)는 roughness=66, force=1.96, speed=0.067 조건으로 100 iteration을 생성하고 두 파일로 저장합니다:

```
test_raw_model_output_66.0_1.96_0.067.txt   ← 모델 원본 출력
test_filtered_signal.txt                    ← IIR 필터 적용 후 uint16
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--host` | `127.0.0.1` | 바인드 주소 |
| `--port` | `65432` | 수신 포트 |
| `--device` | `cpu` | `cpu` 또는 `cuda` |
| `--pt-path` | `pt_files/best_model_light.pt` | 모델 파일 경로 |
| `--mode` | `safe` | `safe`: 모델 + reference 블렌딩 / `pure`: 모델 출력만 |
| `--ref-blend` | `0.20` | reference 혼합 비율 (0~1) |
| `--roughness-change-threshold` | `0.25` | 이 이상 변화 시 guide 재구성 |
| `--no-enhance-roughness` | off | roughness별 고주파 강화 비활성화 |
| `--no-output-limit` | off | 출력 RMS 스케일링 비활성화 |
| `--save-test-signal` | off | 오프라인 신호 생성 후 종료 |
| `--live-plot` | off | 라이브 플롯 모드 (소켓 서버 대신 단독 실행) |

---

## 모델 구조 (`LiteSeq2SeqCNNGRU_AttnPool`)

- **입력**: `[B, 4, 400]` — ch0: Acceleration, ch1: Force, ch2: Velocity, ch3: Roughness(0~1)
- **FiLM conditioning**: roughness scalar → `film_net` → gamma/beta로 conv feature map을 직접 scale & shift
  - ch0~2(동적 신호)만 Conv1D에 입력, ch3(roughness)는 FiLM으로 별도 처리
- **구조**: Conv1D(3→24) → Conv1D(24→32) → FiLM → GRU(32) → Attention Pooling → MLP Head
- **출력**: `[B, 40]` — 미래 40 타임스텝 가속도 예측
- **정규화**: ch0~2만 z-score 정규화; ch3(roughness)는 0~1 그대로 FiLM에 입력

## Loss 함수

| Loss | 설명 |
|------|------|
| `weighted_point_loss` | 진폭 가중 MSE |
| `diff_loss` | 1차 차분 MAE (연속성) |
| `spectral_loss` | FFT 크기 차이 |
| `envelope_loss` | RMS 차이 |
| `roughness_rms_calibration_loss` | roughness별 목표 RMS에 유도. 타겟은 Y_STD로 정규화하여 모델 출력 공간과 일치시킴. zero-contact 샘플(force < 0.1N) 제외 |
| `roughness_contrastive_loss` | roughness 차이가 클수록 출력 RMS 차이도 크게 |
| `hf_energy_ratio_loss` | roughness가 높을수록 HF(>cutoff) 에너지 비율이 높도록 강제 (방향성 제약) |
| `force_contrastive_loss` | force가 높은 쪽의 출력 RMS가 더 크도록 |
| `spectral_profile_loss` | roughness별 평균 FFT 프로파일에 맞게 유도 |
| `roughness_cls_loss` | auxiliary roughness 분류 head (train-only, 7-class) |

### rms_calib 주요 수정 이력

- **Y_STD 정규화**: pred_rms(z-score 정규화 공간 ≈0.6)와 raw target(≈0.5) 비교 오류 수정 → target을 Y_STD로 나눠 동일 공간에서 비교
- **zero-contact 제외**: F=0, Y=0 zero-augment 샘플에 rms_calib 적용 시 zero-output 학습 목표와 충돌 → force < 0.1N 샘플 rms_calib에서 제외

---

## scripts/resplit_npz.py — 참가자 기반 재분할

전처리를 다시 실행하지 않고, 기존 NPZ 캐시를 참가자(PID) 기반으로 재분할합니다.  
**거칠기 기반 split의 문제점**: train이 roughness {5, 45, 100}만 보므로 중간값 일반화가 어렵습니다.  
**참가자 기반 split**: 각 참가자가 모든 거칠기 조건을 수행했으므로, 모든 roughness가 train에 포함됩니다.

```bash
# 자동 비율 분리 (PID 정렬 기준 train 70% / val 15% / test 15%)
python -m scripts.resplit_npz

# 특정 PID를 val/test로 지정
python -m scripts.resplit_npz --val-pids 8 9 --test-pids 10 11

# 입출력 경로 직접 지정
python -m scripts.resplit_npz \
    --in-npz  pt_files/inference_cache_allinone.npz \
    --out-npz pt_files/inference_cache_participant.npz
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--in-npz` | `pt_files/inference_cache_allinone.npz` | 원본 NPZ |
| `--out-npz` | `pt_files/inference_cache_participant.npz` | 저장 경로 |
| `--val-pids` | 자동 | val로 지정할 PID 목록 |
| `--test-pids` | 자동 | test로 지정할 PID 목록 |

> ⚠️ 구버전 NPZ(`window_meta__pid`, `window_meta__split` 컬럼 없음)는 지원되지 않습니다. 이 경우 `preprocess.py`를 다시 실행하세요.
