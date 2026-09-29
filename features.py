"""
Feature Extraction for Knee Osteoarthritis (KOA) Sensor API
============================================================
Extracted directly from OA_Sensor_v8.ipynb (Part 3).
Computes:
  1. Signal statistics (mean, std, min, max, range, rms, skew, kurtosis,
     zcr, dom_freq, spec_energy) for each 3D sensor channel.
  2. Stride rhythm features (stride count, mean/std/cv duration, stride asymmetry).
  3. Stride signal variability (stride-to-stride CV of acceleration and gyro magnitude).
"""

import re
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

SIGNAL_COL_PATTERN = re.compile(r"^(HE|LB|LF|RF)_(Acc|FreeAcc|Gyr|Mag)_[XYZ]$", re.IGNORECASE)


def extract_axis_features(signal: np.ndarray, sr: float, prefix: str) -> Dict[str, float]:
    """
    Extracts time-domain and frequency-domain statistics from a single 1D signal axis.
    Exact implementation from OA_Sensor_v8.ipynb.
    """
    signal = np.asarray(signal, dtype=float)
    signal = signal[~np.isnan(signal)]
    keys = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis", "zcr", "dom_freq", "spec_energy"]
    if len(signal) < 4:
        return {f"{prefix}_{k}": np.nan for k in keys}

    feats = {}
    feats[f"{prefix}_mean"] = float(np.mean(signal))
    feats[f"{prefix}_std"] = float(np.std(signal))
    feats[f"{prefix}_min"] = float(np.min(signal))
    feats[f"{prefix}_max"] = float(np.max(signal))
    feats[f"{prefix}_range"] = float(np.ptp(signal))
    feats[f"{prefix}_rms"] = float(np.sqrt(np.mean(signal**2)))
    feats[f"{prefix}_skew"] = float(stats.skew(signal))
    feats[f"{prefix}_kurtosis"] = float(stats.kurtosis(signal))

    # Zero-crossing rate
    mean_val = signal.mean()
    zero_crossings = np.where(np.diff(np.sign(signal - mean_val)))[0]
    feats[f"{prefix}_zcr"] = float(len(zero_crossings) / len(signal))

    # Spectral features (FFT)
    freqs = rfftfreq(len(signal), d=1.0 / sr)
    mags = np.abs(rfft(signal - mean_val))
    if len(mags) > 1:
        feats[f"{prefix}_dom_freq"] = float(freqs[1:][np.argmax(mags[1:])])
        feats[f"{prefix}_spec_energy"] = float(np.sum(mags**2))
    else:
        feats[f"{prefix}_dom_freq"] = 0.0
        feats[f"{prefix}_spec_energy"] = 0.0

    return feats


def extract_signal_features(df: pd.DataFrame, sr: float) -> Dict[str, float]:
    """
    Extracts axis features for all matching {SENSOR}_{SIGNAL}_{AXIS} columns present in df.
    """
    row = {}
    signal_cols = [c for c in df.columns if SIGNAL_COL_PATTERN.match(c)]
    for col in signal_cols:
        row.update(extract_axis_features(df[col].values, sr, col))
    return row


def extract_gait_event_features(meta: Dict[str, Any], sampling_rate_hz: float) -> Dict[str, float]:
    """
    Computes stride duration, variability, and left/right asymmetry from gait events.
    Exact implementation from OA_Sensor_v8.ipynb.
    """
    feats = {}
    means = {}
    for side, key in [("left", "leftGaitEvents"), ("right", "rightGaitEvents")]:
        events = meta.get(key, []) or []
        durations = [(e[1] - e[0]) / sampling_rate_hz for e in events if len(e) == 2]
        feats[f"{side}_stride_count"] = float(len(durations))
        if durations:
            feats[f"{side}_stride_dur_mean"] = float(np.mean(durations))
            feats[f"{side}_stride_dur_std"] = float(np.std(durations))
            feats[f"{side}_stride_dur_cv"] = float(
                np.std(durations) / np.mean(durations) if np.mean(durations) > 0 else np.nan
            )
            means[side] = np.mean(durations)
        else:
            feats[f"{side}_stride_dur_mean"] = np.nan
            feats[f"{side}_stride_dur_std"] = np.nan
            feats[f"{side}_stride_dur_cv"] = np.nan

    if "left" in means and "right" in means and (means["left"] + means["right"]) > 0:
        feats["stride_asymmetry"] = float(
            abs(means["left"] - means["right"]) / ((means["left"] + means["right"]) / 2.0)
        )
    else:
        feats["stride_asymmetry"] = np.nan

    feats["total_stride_count"] = feats["left_stride_count"] + feats["right_stride_count"]
    return feats


def extract_stride_signal_variability(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    sensors: List[str] = ["HE", "LB", "LF", "RF"],
) -> Dict[str, float]:
    """
    Computes stride-to-stride CV of acceleration and gyroscope magnitude per sensor.
    Exact implementation from OA_Sensor_v8.ipynb.
    """
    if df is None or "PacketCounter" not in df.columns:
        return {}

    row = {}
    for sensor in sensors:
        for mag_name in ["FreeAcc", "Gyr"]:
            cols = [f"{sensor}_{mag_name}_{ax}" for ax in ["X", "Y", "Z"]]
            if not all(c in df.columns for c in cols):
                continue

            # Orientation-invariant vector magnitude
            magnitude = np.sqrt((df[cols].astype(float) ** 2).sum(axis=1))

            for side, key in [("left", "leftGaitEvents"), ("right", "rightGaitEvents")]:
                events = meta.get(key, []) or []
                stride_means, stride_peaks = [], []
                for ev in events:
                    if len(ev) != 2:
                        continue
                    start, end = int(ev[0]), int(ev[1])
                    seg = magnitude[(df["PacketCounter"] >= start) & (df["PacketCounter"] <= end)]
                    if len(seg) < 2:
                        continue
                    stride_means.append(seg.mean())
                    stride_peaks.append(seg.max())

                prefix = f"{sensor}_{mag_name}_{side}"
                if len(stride_means) >= 2 and np.mean(stride_means) != 0:
                    row[f"{prefix}_stridecv_mean"] = float(np.std(stride_means) / abs(np.mean(stride_means)))
                else:
                    row[f"{prefix}_stridecv_mean"] = np.nan

                if len(stride_peaks) >= 2 and np.mean(stride_peaks) != 0:
                    row[f"{prefix}_stridecv_peak"] = float(np.std(stride_peaks) / abs(np.mean(stride_peaks)))
                else:
                    row[f"{prefix}_stridecv_peak"] = np.nan

    return row


def extract_all_trial_features(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    sampling_rate_hz: float,
) -> Dict[str, float]:
    """
    Extracts the complete feature dictionary (signal statistics, gait events,
    and stride signal variability) as performed in extract_trial_features() in the notebook.
    """
    feats = extract_signal_features(df, sampling_rate_hz)
    feats.update(extract_gait_event_features(meta, sampling_rate_hz))
    feats.update(extract_stride_signal_variability(df, meta))
    return feats


def get_canonical_feature_names() -> List[str]:
    """
    Generates the complete, deterministic list of 438 feature names matching the
    notebook's training matrix.
    """
    names = []

    # 1. 36 Signal Columns x 11 Axis Features = 396 features
    sensors = ["HE", "LB", "LF", "RF"]
    signals = ["Acc", "FreeAcc", "Gyr"]
    axes = ["X", "Y", "Z"]
    stat_keys = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis", "zcr", "dom_freq", "spec_energy"]

    for s in sensors:
        for sig in signals:
            for ax in axes:
                col = f"{s}_{sig}_{ax}"
                for k in stat_keys:
                    names.append(f"{col}_{k}")

    # 2. Gait event features = 10 features
    names.extend([
        "left_stride_count",
        "left_stride_dur_mean",
        "left_stride_dur_std",
        "left_stride_dur_cv",
        "right_stride_count",
        "right_stride_dur_mean",
        "right_stride_dur_std",
        "right_stride_dur_cv",
        "stride_asymmetry",
        "total_stride_count",
    ])

    # 3. Stride signal variability features = 32 features
    for s in sensors:
        for mag_name in ["FreeAcc", "Gyr"]:
            for side in ["left", "right"]:
                names.append(f"{s}_{mag_name}_{side}_stridecv_mean")
                names.append(f"{s}_{mag_name}_{side}_stridecv_peak")

    return names
