"""
Global configuration / hyperparameters
"""
from pathlib import Path
import random
import numpy as np
import torch

# Paths
ROOT_DIR = Path("./pilot-data")
OUT_DIR = Path("./pt_files")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segment / window parameters
SEG_TARGET_LEN  = 4000
INPUT_STEPS     = 400
OUTPUT_STEPS    = 40
STRIDE          = 20

# Scraping-segment detection
FORCE_THRESHOLD    = 0.5
VEL_THRESHOLD      = 0.03
MIN_SEG_LEN        = INPUT_STEPS + OUTPUT_STEPS
MERGE_GAP          = 500
MARGIN             = 200
SMOOTH_W           = 101
POST_MERGE_MIN_LEN = 4000

# Peak filter
MAX_ABS_PEAK = 4.0

# Excluded roughness values
EXCLUDE_ROUGHNESS = [71]

# Split mode
# "roughness"   : roughness 값으로 train/val/test 분리 (원래 방식)
#                 단점: 학습 roughness 3개만 보고 중간값 일반화 어려움
# "participant" : 참가자 ID로 분리 -> 모든 roughness 학습 가능
#                 val/test 참가자를 PARTICIPANT_SPLIT에 지정;
#                 비워두면 자동 비율 분리 (55/70/100% cutoff)
SPLIT_MODE = "roughness"

# Roughness-based split  (SPLIT_MODE="roughness" 일 때)
ROUGHNESS_SPLIT = {
    "train": [5, 45, 100],
    "val":   [12, 58],
    "test":  [23, 66],
}

# Participant-based split  (SPLIT_MODE="participant" 일 때)
# 직접 지정 예: {"val": [8, 9], "test": [10, 11]}
# 비워두면 sorted PID 기준 상위 30% -> test+val 자동 분리
PARTICIPANT_SPLIT = {
    "val":  [],
    "test": [],
}

# Training hyperparameters
BATCH_SIZE    = 64
EPOCHS        = 60
LR            = 1e-3
WEIGHT_DECAY  = 1e-4

# Loss weights
AMP_WEIGHT_ALPHA = 1.0
LAMBDA_POINT     = 1.0
LAMBDA_DIFF      = 0.3
LAMBDA_SPEC      = 0.3
LAMBDA_ENV       = 0.2

# DAQ / inference
SAMPLE_RATE    = 8000
PLAY_SECONDS   = 3.0
VOLTAGE_SCALE  = 1.2
AO_CHANNEL     = "Dev1/ao0"

# Global y-axis limits (pre-computed from full dataset)
ACC_LIM   = (-1, 1)
FORCE_LIM = (-2, 6)
VEL_LIM   = (-0.1, 1.5)
FFT_LIM   = (0, 500)
