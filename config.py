"""
Configuration for the Knee Osteoarthritis (KOA) Wearable Sensor Inference API
==============================================================================
All constants, unit conversions, sensor channel mappings, stride-detection
parameters, imputation strategy and result thresholds live here.

WHAT THE MODEL ACTUALLY PREDICTS (from OA_Sensor_v8.ipynb)
  * Training data: KOA patients only (no healthy controls).
  * Target: WOMAC symptom-severity tier -> Low tertile (0) vs High tertile (1).
  * So the output is a *symptom-severity tier*, NOT "KOA vs healthy".
"""

import os
from typing import Dict, List, Literal, Optional

# ==============================================================================
# BASE PATHS  (model artifacts exported by Part 12 of OA_Sensor_v8.ipynb)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("KOA_MODEL_DIR", os.path.join(BASE_DIR, "model"))


def _resolve_artifact(stem: str) -> str:
    """The notebook saves *.pkl; older instructions said *.joblib. Accept both."""
    for ext in (".pkl", ".joblib"):
        candidate = os.path.join(MODEL_DIR, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(MODEL_DIR, stem + ".pkl")


MODEL_PATH = _resolve_artifact("oa_sensor_model")
SCALER_PATH = _resolve_artifact("oa_sensor_scaler")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "oa_sensor_features.json")

# ==============================================================================
# SAMPLING & DATA VALIDATION
# ==============================================================================
# Training data (Xsens) was recorded at 100 Hz. Incoming data is ALWAYS
# resampled to this rate using its t_ms timestamps, whatever the client claims.
DEFAULT_SAMPLING_RATE_HZ: float = 100.0

MIN_DURATION_SECONDS: float = 5.0
MIN_READINGS_COUNT: int = 100

# Length of the trials the model was trained on (seconds).
# Stride counts and spectral-energy features grow with recording length, so the
# live recording should be about as long as a training trial.
#   Find it in the notebook:   print(data["total_stride_count"].describe())
#   or:                        len(load_trial_processed_data(trial_folders[0])) / 100
# If set: longer recordings are cropped to this window, and shorter ones
# (< 80%) trigger a warning. If None: no cropping and no length warning.
TRAIN_TRIAL_SECONDS: Optional[float] = None

# ==============================================================================
# UNIT CONVERSION
# ==============================================================================
# ESP32 IMUs stream acceleration in 'g' and gyroscope in 'deg/s'.
# The training dataset (Xsens) uses SI units: m/s^2 and rad/s.
ACCEL_UNIT_IN: str = "g"
ACCEL_CONVERSION_FACTOR: float = 9.80665           # g -> m/s^2

GYRO_UNIT_IN: str = "deg/s"
GYRO_CONVERSION_FACTOR: float = 0.017453292519943295   # deg/s -> rad/s

# ==============================================================================
# CHANNEL MAPPING
# ==============================================================================
# Training setup (4 Xsens IMUs): HE=Head, LB=Lower Back, LF=Left Foot, RF=Right Foot.
# Live setup (2 ESP32 IMUs):     shin, thigh.
#   shin  -> RF  (closest lower-limb sensor; a foot IMU is NOT identical to a shin IMU)
#   thigh -> LB  (no thigh sensor existed in training; LB is only a rough stand-in)
# These are approximations. The real fix is retraining on shin+thigh data.
SENSOR_MAPPING: Dict[str, str] = {
    "shin": "RF",
    "thigh": "LB",
}

ALL_TRAINING_SENSORS: List[str] = ["HE", "LB", "LF", "RF"]
ALL_SIGNAL_TYPES = ["Acc", "FreeAcc", "Gyr"]
AXES = ["X", "Y", "Z"]

# ==============================================================================
# MISSING-FEATURE IMPUTATION
# ==============================================================================
# Any feature the live hardware cannot provide (HE, LF, any Mag column, and the
# left-leg gait features) is replaced so it contributes a z-score of 0 after the
# StandardScaler, i.e. raw value = scaler.mean_.
#   "zero_after_scaling" / "train_mean": raw value = scaler.mean_   (z = 0)
#   "raw_zero": raw value = 0.0 (becomes -mean/std after scaling; not neutral)
# (The notebook filled its own NaNs with the median; the difference is small.)
FillModeType = Literal["zero_after_scaling", "train_mean", "raw_zero"]
FILL_MODE: FillModeType = "zero_after_scaling"

# ==============================================================================
# STRIDE DETECTION (peak detection on shin gyroscope magnitude)
# ==============================================================================
STRIDE_PEAK_MIN_DISTANCE_MS: int = 450      # min time between two strides
STRIDE_PEAK_PROMINENCE: float = 0.40        # absolute floor for peak prominence (rad/s)
STRIDE_PEAK_REL_PROMINENCE: float = 0.35    # also require >= 35% of the 95th-percentile gyro magnitude
MIN_STRIDES_REQUIRED: int = 2               # minimum peaks to continue
# Tune the three values above on REAL ESP32 recordings; the defaults were only
# checked on synthetic walking data.

# Only ONE leg is instrumented. Inventing the other leg (50% phase shift) would
# make stride asymmetry always ~0, which is fake information. Keep this False:
# the left-leg / asymmetry features are then treated as missing and imputed.
ESTIMATE_CONTRALATERAL_STRIDES: bool = False

# ==============================================================================
# RESULT TIERS  (probability of the "High severity" class)
# ==============================================================================
# Mirrors the notebook's Part 13: scores within 0.15 of 0.5 are "Monitor".
LABEL_HIGH: str = "Higher severity"
LABEL_LOW: str = "Lower severity"
LABEL_BORDERLINE: str = "Borderline"

RISK_THRESHOLD_HIGH: float = 0.65
RISK_THRESHOLD_LOW: float = 0.35

INTERPRETATION: str = (
    "Estimated WOMAC-based symptom-severity tier for people with knee osteoarthritis. "
    "This is not a diagnosis. Prototype model trained on KOA patients only, "
    "with part of its inputs unavailable from the live sensors."
)

# True while the live shin+thigh model is not yet trained
IS_PROTOTYPE: bool = True
