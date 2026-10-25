"""
Global Configuration for ECG Usability Assessment Project
"Cross-dataset ECG usability assessment with a single-pass multimodal decision model"
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "cache" / "datasets"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Local copies of the PhysioNet datasets and the d1-3B weights (override with environment variables)
LOCAL_DATASETS_DIR = Path(os.environ.get("ECG_DATASETS_ROOT", "/Volumes/KIOXIA/Datasets"))
D1_MODEL_PATH = os.environ.get("D1_MODEL_PATH", "/Volumes/KIOXIA/AI Models/liquid-ai/d1-3B")

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ECG Signal Parameters
DEFAULT_WINDOW_SEC = 10.0   # Standard analysis window duration (seconds)
TARGET_FS = 500             # Standardized sampling rate for model inputs (Hz)
QRS_TOLERANCE_MS = 50.0     # Temporal matching tolerance for QRS detector agreement

# Class Mapping
# Binary Usability:
# Class 0: Usable (suitable for QRS-based rhythm analysis)
# Class 1: Unusable (severe noise, baseline wander, saturation, requires human review)
CLASS_USABLE = 0
CLASS_UNUSABLE = 1
CLASS_NAMES = ["Usable", "Unusable"]

# BUT QDB Details
BUTQDB_URL = "https://physionet.org/files/butqdb/1.0.0/"
BUTQDB_RECORDS = [
    "100001", "100002", "103001", "103002", "103003",
    "104001", "105001", "111001", "113001", "114001",
    "115001", "118001", "121001", "122001", "123001",
    "124001", "125001", "126001"
]
# Mapping of record to subject ID (first 3 digits represent patient)
def get_subject_id(record_name: str) -> str:
    clean_name = record_name.split("_")[0]
    return clean_name[:3]

# CinC 2011 Details
CINC2011_URL = "https://physionet.org/files/challenge-2011/1.0.0/set-a/"
CINC2011_DEFAULT_LEAD = 1  # Lead II (index 1 in standard 12-lead layout)

# NSTDB Details
NSTDB_URL = "https://physionet.org/files/nstdb/1.0.0/"
NSTDB_SNR_LEVELS = [24, 18, 12, 6, 0, -6]
NSTDB_RECORDS = ["118", "119"]

# Image Rendering Settings
DEFAULT_IMAGE_DPI = 100
DEFAULT_IMAGE_SIZE = (512, 256) # Width, Height in pixels
DEFAULT_LINE_WIDTH = 1.0
DEFAULT_GRID_STYLE = "clinical" # 'clinical' (red/pink grid), 'gray', 'none'

# Reproducibility
RANDOM_SEED = 42
