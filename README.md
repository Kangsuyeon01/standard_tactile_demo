# Roughness Generation Model

CNN + GRU + FiLM 기반 표면 거칠기 진동 생성 모델.  
입력(가속도 히스토리, 힘, 속도, 거칠기)에서 미래 가속도 신호를 예측한다.

## 프로젝트 구조

```
roughness_model/
├── src/
│   ├── config.py           # 전역 하이퍼파라미터 / 경로 설정
│   ├── data.py             # CSV 파싱, 세그먼트 검출, 윈도우 생성
│   ├── model.py            # LiteSeq2SeqCNNGRU_AttnPool, Loss 함수, Dataset
│   ├── cache.py            # NPZ 캐시 저장/로드
│   ├── inference.py        # 신호 생성, DAQ 출력
│   └── visualise.py        # 시각화 유틸리티
├── scripts/
│   ├── preprocess.py       # 1단계: CSV → NPZ 캐시 생성
│   ├── resplit_npz.py      # 기존 NPZ를 참가자 기반으로 재분할 (전처리 재실행 불필요)
│   ├── train.py            # 2단계: 모델 학습 (run ID 리포트 포함)
│   ├── evaluate.py         # 3단계: RMSE/MAE/Corr, 특성 중요도, Saliency
│   ├── eval_roughness_signal.py  # 거칠기별 생성 신호 비교 시각화
│   ├── demo.py             # 대화형 진동 출력 루프 (NI-DAQ)
│   ├── plot_analysis.py    # LOW/MID/HIGH + 생성 신호 비교 플롯
│   └── realtime.py         # Unity 소켓 서버 (실시간 촉각 생성)
├── pt_files/
│   ├── inference_cache_allinone.npz   # 전처리 캐시
│   └── runs/
│       └── YYYYMMDD-NNN/              # run 별 결과
│           ├── best_model.pt
│           ├── report.json
│           ├── training_history.png
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

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--out-dir` | `pt_files` | 결과 저장 루트 |
| `--npz` | 자동 탐색 | NPZ 캐시 경로 직접 지정 |
| `--run-id` | 자동 생성 | 수동 지정 시 해당 ID 사용 |
| `--note` | `""` | report.json에 기록할 메모 |
| `--lambda-rms` | `2.0` | roughness RMS 보정 loss 가중치 |
| `--lambda-contrast` | `0.6` | roughness contrastive loss 가중치 |
| `--lambda-hf` | `0.5` | HF 에너지 비율 contrastive loss 가중치 |
| `--lambda-centroid` | `0.0` | 스펙트럼 무게중심 loss 가중치 (0=비활성) |

```bash
# 참가자 split NPZ로 학습
python -m scripts.train --npz pt_files/inference_cache_participant.npz --note "participant split"

# 결과 폴더 직접 지정
python -m scripts.train --out-dir /data/results
```

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

## scripts/evaluate.py — 표준 평가

```bash
python -m scripts.evaluate
```

RMSE / MAE / Corr, Permutation Feature Importance, Saliency 히트맵을 계산하고 `pt_files/`에 저장합니다.

> ⚠️ 현재 구버전 모델 경로(`best_model_light_wo71.pt`, `in_ch=4`)를 참조합니다. 새 run ID 기반 모델을 쓰려면 파일 상단 경로를 `runs/{run_id}/best_model.pt`로 수정하세요.

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

# GPU 사용
python -m scripts.realtime --device cuda

# 특정 run 모델 사용
python -m scripts.realtime --pt-path pt_files/runs/20260610-001/best_model.pt

# 오프라인 신호 생성 테스트 (Unity 불필요)
python -m scripts.realtime --save-test-signal
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
| `--apply-output-limit` | off | 출력 RMS 제한 적용 |
| `--save-test-signal` | off | 오프라인 신호 생성 후 종료 |

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
| `roughness_rms_calibration_loss` | roughness별 목표 RMS에 유도 |
| `roughness_contrastive_loss` | roughness 차이가 클수록 출력 RMS 차이도 크게 |
| `hf_energy_ratio_loss` | roughness가 높을수록 HF(>cutoff) 에너지 비율이 높도록 강제 (방향성 제약) |

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
