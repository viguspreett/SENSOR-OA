"""
Preprocessing Pipeline for KOA Sensor Data
===========================================
  1. Parse + validate the JSON payload (multi-sensor or legacy single-array).
  2. Convert units (g -> m/s^2, deg/s -> rad/s) and derive an approximate FreeAcc.
  3. Resample every sensor to a uniform 100 Hz grid (using t_ms, never the client's claimed rate).
  4. Detect strides from the shin gyroscope.
  5. Map shin/thigh onto the notebook's RF/LB columns and extract the notebook features.
  6. Impute every feature the live hardware cannot measure (z = 0 after scaling).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

import config as cfg
from features import extract_all_trial_features, get_canonical_feature_names


class PreprocessingError(Exception):
    """Raised when validation, duration or stride detection fails."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ------------------------------------------------------------------------------
# 1. PARSING & VALIDATION
# ------------------------------------------------------------------------------
def parse_and_validate_payload(
    body: Dict[str, Any],
) -> Tuple[Dict[str, List[Dict[str, float]]], float]:
    """
    Standard schema: {"sampling_rate_hz": 100, "sensors": {"shin": [...], "thigh": [...]}}
    Legacy schema:   {"readings": [...], "sensor_position": "shin", "sampling_rate_hz": 100}
    The shin sensor is mandatory (stride detection needs it).
    """
    claimed_rate = float(body.get("sampling_rate_hz") or cfg.DEFAULT_SAMPLING_RATE_HZ)
    if claimed_rate <= 0:
        raise PreprocessingError("sampling_rate_hz must be positive.", status_code=400)

    if isinstance(body.get("sensors"), dict) and body["sensors"]:
        raw = body["sensors"]
    elif isinstance(body.get("readings"), list):
        pos = str(body.get("sensor_position") or "shin")
        raw = {pos: body["readings"]}
    else:
        raise PreprocessingError(
            "Invalid request body. Expected a non-empty 'sensors' object or a 'readings' array.",
            status_code=400,
        )

    sensors_dict = {str(k).lower(): v for k, v in raw.items()}

    unknown = [p for p in sensors_dict if p not in cfg.SENSOR_MAPPING]
    if unknown:
        raise PreprocessingError(
            f"Unknown sensor position(s) {unknown}. Use only {sorted(cfg.SENSOR_MAPPING)}.",
            status_code=400,
        )
    if "shin" not in sensors_dict:
        raise PreprocessingError(
            "The 'shin' sensor is required (strides are detected from it).", status_code=400
        )

    required_keys = ("t_ms", "ax", "ay", "az", "gx", "gy", "gz")
    for pos, readings in sensors_dict.items():
        if not isinstance(readings, list) or len(readings) == 0:
            raise PreprocessingError(f"Sensor '{pos}' contains no readings.", status_code=400)
        if len(readings) < cfg.MIN_READINGS_COUNT:
            raise PreprocessingError(
                f"Sensor '{pos}' has {len(readings)} readings. "
                f"Minimum required is {cfg.MIN_READINGS_COUNT} samples.",
                status_code=400,
            )

        t_prev: Optional[float] = None
        for i, item in enumerate(readings):
            if not isinstance(item, dict):
                raise PreprocessingError(
                    f"Sensor '{pos}' reading at index {i} must be a JSON object.", status_code=400
                )
            missing = [k for k in required_keys if k not in item]
            if missing:
                raise PreprocessingError(
                    f"Sensor '{pos}' reading at index {i} is missing fields: {missing}",
                    status_code=400,
                )
            for k in required_keys:
                val = item[k]
                if (
                    isinstance(val, bool)
                    or not isinstance(val, (int, float))
                    or np.isnan(val)
                    or np.isinf(val)
                ):
                    raise PreprocessingError(
                        f"Non-numeric or invalid value at '{pos}[{i}].{k}': {val}", status_code=400
                    )
            t_curr = float(item["t_ms"])
            if t_prev is not None and t_curr < t_prev:
                raise PreprocessingError(
                    f"Sensor '{pos}' timestamps must not decrease "
                    f"(index {i}: {t_curr} < {t_prev}).",
                    status_code=400,
                )
            t_prev = t_curr

        duration_sec = (float(readings[-1]["t_ms"]) - float(readings[0]["t_ms"])) / 1000.0
        if duration_sec < cfg.MIN_DURATION_SECONDS:
            raise PreprocessingError(
                f"Sensor '{pos}' duration is {duration_sec:.2f}s. "
                f"Minimum required duration is {cfg.MIN_DURATION_SECONDS:.1f} seconds for gait analysis.",
                status_code=400,
            )

    return sensors_dict, claimed_rate


# ------------------------------------------------------------------------------
# 2-3. UNIT CONVERSION + RESAMPLING
# ------------------------------------------------------------------------------
def convert_and_resample_sensor_df(
    readings: List[Dict[str, float]],
    target_sampling_rate: float = cfg.DEFAULT_SAMPLING_RATE_HZ,
    window_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """
    g -> m/s^2, deg/s -> rad/s, approximate FreeAcc (mean removed), then linear
    resampling onto an exact uniform grid. If window_seconds is given, the
    recording is cropped to its first window_seconds.
    """
    df = pd.DataFrame(readings)

    for ax in ("x", "y", "z"):
        df[f"a{ax}_si"] = df[f"a{ax}"].astype(float) * cfg.ACCEL_CONVERSION_FACTOR
        df[f"g{ax}_si"] = df[f"g{ax}"].astype(float) * cfg.GYRO_CONVERSION_FACTOR
        # Approximation of Xsens FreeAcc (true FreeAcc needs orientation estimation)
        df[f"free_a{ax}_si"] = df[f"a{ax}_si"] - df[f"a{ax}_si"].mean()

    df_dedup = df.groupby("t_ms", as_index=False).mean()
    t_start = float(df_dedup["t_ms"].iloc[0])
    t_end = float(df_dedup["t_ms"].iloc[-1])
    if window_seconds is not None:
        t_end = min(t_end, t_start + window_seconds * 1000.0)

    step_ms = 1000.0 / target_sampling_rate
    t_uniform = np.arange(t_start, t_end + 1e-6, step_ms)

    signal_cols = [
        "ax_si", "ay_si", "az_si",
        "free_ax_si", "free_ay_si", "free_az_si",
        "gx_si", "gy_si", "gz_si",
    ]
    out = {"t_ms": t_uniform}
    for col in signal_cols:
        fn = interp1d(df_dedup["t_ms"], df_dedup[col], kind="linear", fill_value="extrapolate")
        out[col] = fn(t_uniform)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------
# 4. STRIDE DETECTION
# ------------------------------------------------------------------------------
def detect_strides_from_shin(
    shin_df: pd.DataFrame,
    sampling_rate: float = cfg.DEFAULT_SAMPLING_RATE_HZ,
) -> Tuple[List[List[int]], Dict[str, float]]:
    """Peak detection on shin gyroscope magnitude; consecutive peaks bound one stride."""
    gyro_mag = np.sqrt(
        shin_df["gx_si"] ** 2 + shin_df["gy_si"] ** 2 + shin_df["gz_si"] ** 2
    ).values

    min_distance = max(1, int((cfg.STRIDE_PEAK_MIN_DISTANCE_MS / 1000.0) * sampling_rate))
    prominence = max(
        cfg.STRIDE_PEAK_PROMINENCE,
        cfg.STRIDE_PEAK_REL_PROMINENCE * float(np.percentile(gyro_mag, 95)),
    )

    peaks, _ = find_peaks(gyro_mag, distance=min_distance, prominence=prominence)
    if len(peaks) < cfg.MIN_STRIDES_REQUIRED:
        peaks, _ = find_peaks(gyro_mag, distance=min_distance, prominence=max(0.15, prominence * 0.5))

    if len(peaks) < cfg.MIN_STRIDES_REQUIRED:
        raise PreprocessingError(
            f"Could not detect sufficient gait cycles from shin sensor. "
            f"Found {len(peaks)} stride peak(s); minimum required is {cfg.MIN_STRIDES_REQUIRED}. "
            "Please ensure the subject is walking continuously and the sensor is securely attached.",
            status_code=422,
        )

    events = [[int(peaks[i]), int(peaks[i + 1])] for i in range(len(peaks) - 1)]
    durations_s = [(e[1] - e[0]) / sampling_rate for e in events]
    mean_dur = float(np.mean(durations_s))
    std_dur = float(np.std(durations_s))

    metrics = {
        "stride_count": len(events),
        "mean_stride_duration_s": round(mean_dur, 3),
        "stride_duration_std_s": round(std_dur, 3),
        "stride_duration_cv": round(std_dur / mean_dur, 3) if mean_dur > 0 else 0.0,
        "cadence_steps_per_min": round((60.0 / mean_dur) * 2.0, 1) if mean_dur > 0 else 0.0,
        "shin_peak_angular_velocity_rad_s": round(float(np.max(gyro_mag)), 2),
    }
    return events, metrics


# ------------------------------------------------------------------------------
# 5. CHANNEL MAPPING
# ------------------------------------------------------------------------------
def build_wide_dataframe(
    resampled_sensors: Dict[str, pd.DataFrame],
    sensor_mapping: Dict[str, str] = cfg.SENSOR_MAPPING,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Wide table with the notebook's {SENSOR}_{SIGNAL}_{AXIS} columns (e.g. RF_Gyr_Z)."""
    min_len = min(len(df) for df in resampled_sensors.values())
    wide_df = pd.DataFrame({"PacketCounter": np.arange(min_len)})

    mapped_prefixes: List[str] = []
    for pos, df in resampled_sensors.items():
        prefix = sensor_mapping.get(pos.lower())
        if not prefix:
            continue
        mapped_prefixes.append(prefix)
        sub = df.iloc[:min_len]
        for ax in ("X", "Y", "Z"):
            lo = ax.lower()
            wide_df[f"{prefix}_Acc_{ax}"] = sub[f"a{lo}_si"].values
            wide_df[f"{prefix}_FreeAcc_{ax}"] = sub[f"free_a{lo}_si"].values
            wide_df[f"{prefix}_Gyr_{ax}"] = sub[f"g{lo}_si"].values

    missing = [s for s in cfg.ALL_TRAINING_SENSORS if s not in mapped_prefixes]
    return wide_df, mapped_prefixes, missing


# Left-leg / two-leg features that cannot be measured with one instrumented leg
_UNMEASURED_WITH_ONE_LEG = re.compile(
    r"^(left_stride_.*|stride_asymmetry|total_stride_count|.*_left_stridecv_(mean|peak))$"
)


# ------------------------------------------------------------------------------
# FULL PIPELINE
# ------------------------------------------------------------------------------
def preprocess_and_extract_features(
    payload: Dict[str, Any],
    scaler: Any = None,
    expected_feature_cols: Optional[List[str]] = None,
    fill_mode: str = cfg.FILL_MODE,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """Returns (feature_vector [1, N] in model column order, summary_metrics, warnings)."""
    warnings: List[str] = []
    target_rate = cfg.DEFAULT_SAMPLING_RATE_HZ  # always the training rate

    sensors_dict, _claimed_rate = parse_and_validate_payload(payload)

    # Recording-length handling (stride counts / spectral energy depend on it)
    window = cfg.TRAIN_TRIAL_SECONDS
    shin_readings = sensors_dict["shin"]
    duration_s = (float(shin_readings[-1]["t_ms"]) - float(shin_readings[0]["t_ms"])) / 1000.0
    if window is not None and duration_s < 0.8 * window:
        warnings.append(
            f"Recording is {duration_s:.1f}s but the model was trained on ~{window:.0f}s trials; "
            "length-dependent features (stride counts, spectral energy) are out of range."
        )

    resampled: Dict[str, pd.DataFrame] = {
        pos: convert_and_resample_sensor_df(r, target_rate, window_seconds=window)
        for pos, r in sensors_dict.items()
    }

    shin_df = resampled["shin"]
    detected_events, stride_metrics = detect_strides_from_shin(shin_df, sampling_rate=target_rate)

    # Gait metadata: shin -> RF (right). Left leg is not instrumented.
    gait_meta: Dict[str, Any] = {"rightGaitEvents": detected_events, "leftGaitEvents": []}
    if cfg.ESTIMATE_CONTRALATERAL_STRIDES:
        left_events = []
        for ev in detected_events:
            half = int((ev[1] - ev[0]) / 2)
            ls = min(ev[0] + half, len(shin_df) - 2)
            le = min(ev[1] + half, len(shin_df) - 1)
            if le > ls:
                left_events.append([ls, le])
        gait_meta["leftGaitEvents"] = left_events
        warnings.append("Left-leg strides were ESTIMATED (50% phase shift); asymmetry is not a real measurement.")

    wide_df, mapped_prefixes, missing_prefixes = build_wide_dataframe(resampled)

    if "LB" in mapped_prefixes:
        warnings.append(
            "Thigh IMU is mapped to the training lower-back (LB) slot; no thigh sensor existed in training data."
        )
    if missing_prefixes:
        warnings.append(
            f"Missing training sensor locations: {missing_prefixes}. "
            f"Imputing them using FILL_MODE='{fill_mode}'."
        )

    raw_features = extract_all_trial_features(wide_df, gait_meta, sampling_rate_hz=target_rate)

    force_missing = not cfg.ESTIMATE_CONTRALATERAL_STRIDES
    if force_missing:
        for name in list(raw_features):
            if _UNMEASURED_WITH_ONE_LEG.match(name):
                raw_features[name] = np.nan
        warnings.append(
            "Only one leg is instrumented: left-leg stride features and stride asymmetry were treated as missing."
        )

    if expected_feature_cols is None:
        expected_feature_cols = get_canonical_feature_names()

    feature_vector = np.zeros(len(expected_feature_cols), dtype=float)
    imputed = 0
    scaler_means = getattr(scaler, "mean_", None) if scaler is not None else None

    for idx, col in enumerate(expected_feature_cols):
        val = raw_features.get(col)
        if val is None or not np.isfinite(val):
            imputed += 1
            if fill_mode in ("zero_after_scaling", "train_mean") and scaler_means is not None:
                feature_vector[idx] = float(scaler_means[idx])
            else:
                feature_vector[idx] = 0.0
        else:
            feature_vector[idx] = float(val)

    imputed_fraction = imputed / max(1, len(expected_feature_cols))
    if imputed:
        warnings.append(
            f"{imputed} of {len(expected_feature_cols)} model features ({imputed_fraction:.0%}) "
            "were imputed because the live hardware cannot measure them."
        )

    summary: Dict[str, Any] = dict(stride_metrics)
    summary["mapped_sensors"] = mapped_prefixes
    summary["missing_sensors"] = missing_prefixes
    summary["imputed_feature_fraction"] = round(imputed_fraction, 3)

    asym = raw_features.get("stride_asymmetry")
    summary["stride_asymmetry"] = (
        round(float(asym), 3) if asym is not None and np.isfinite(asym) else None
    )

    summary["shin_mean_accel_norm_m_s2"] = round(
        float(np.sqrt(shin_df["ax_si"] ** 2 + shin_df["ay_si"] ** 2 + shin_df["az_si"] ** 2).mean()), 2
    )
    if "thigh" in resampled:
        t = resampled["thigh"]
        summary["thigh_peak_angular_velocity_rad_s"] = round(
            float(np.sqrt(t["gx_si"] ** 2 + t["gy_si"] ** 2 + t["gz_si"] ** 2).max()), 2
        )

    return feature_vector.reshape(1, -1), summary, warnings
