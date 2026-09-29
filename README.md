# KOA Wearable Sensor Inference API (prototype)

FastAPI service that takes raw IMU readings from two ESP32 nodes (**shin** and **thigh**),
detects strides, extracts the same features as `OA_Sensor_v8.ipynb`, and returns a
**symptom-severity tier**.

## What the model predicts (read this first)

* The model was trained on **knee-OA patients only** (no healthy controls).
* Its target is the **WOMAC severity tier**: low third (0) vs high third (1).
* So the output is *Lower severity / Borderline / Higher severity*. It is **not** a diagnosis and it does **not** say "has OA vs healthy".

## Known limitations (prototype)

| Limitation | What the API does about it |
|---|---|
| Model trained on 4 lab Xsens IMUs (head, lower back, left foot, right foot); live hardware has 2 (shin, thigh) | shin -> `RF`, thigh -> `LB` (rough stand-in). Head/left-foot features are imputed. |
| About half the model inputs (~52%) cannot be measured | Imputed so their scaled value is 0. Reported in `features.imputed_feature_fraction` and `warnings`. |
| Only one leg is instrumented | Left-leg stride features and stride asymmetry are treated as missing (not invented). `stride_asymmetry` is `null`. |
| FreeAcc is approximated (mean removed), not Xsens orientation-corrected | Documented approximation. |
| Stride counts / spectral energy grow with recording length | Set `TRAIN_TRIAL_SECONDS` in `config.py` (see below). |
| Stride-detection thresholds were only checked on synthetic data | Tune `STRIDE_PEAK_*` in `config.py` on real ESP32 recordings. |

The real fix is retraining on data recorded with shank and thigh IMUs.

## Files

```
sensor/
  sensor_api.py     FastAPI app, request/response models, model loading, tiers
  preprocess.py     validation, unit conversion, 100 Hz resampling, stride detection, channel mapping, imputation
  features.py       feature extraction (verified identical to the notebook)
  config.py         all constants and thresholds
  requirements.txt
  test_request.py
  model/            put the 3 files from the notebook here (see below)
```

## 1. Put the model files in `sensor/model/`

Part 12 of the notebook saves these to Drive (`/content/drive/MyDrive/SENSOR DATASET`). Copy:

```
oa_sensor_model.pkl
oa_sensor_scaler.pkl
oa_sensor_features.json
```

(`.joblib` names also work.) On startup the API refuses to serve if the feature list, scaler and
model disagree on the number of features (`/health` then shows `model_loaded: false`).

**Pin scikit-learn** in `requirements.txt` to the version used in the notebook:
`import sklearn; print(sklearn.__version__)`.

**Check `oa_sensor_features.json`:** if it contains names with `_Mag_`, those features cannot come from the ESP32
and will be imputed (the imputed share will be higher than 52%).

## 2. Set the training recording length

Run in the notebook: `print(data["total_stride_count"].describe())`
(or `len(load_trial_processed_data(trial_folders[0])) / 100` for seconds).
Then set `TRAIN_TRIAL_SECONDS` in `config.py`. Longer recordings are cropped to that window, and shorter
ones add a warning. Leave it `None` to disable.

## API

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_type": "LogisticRegression", "features_count": 438, "fill_mode": "zero_after_scaling"}
```

### `POST /analyze-sensor`
No API key. JSON body. **`shin` is required**, `thigh` is optional.
Units: acceleration in **g**, gyroscope in **deg/s**, `t_ms` in milliseconds.
Data is always resampled to 100 Hz using `t_ms` (`sampling_rate_hz` is informational only).
Minimum 5 s of data.

Standard request:
```json
{
  "sensors": {
    "shin":  [{"t_ms": 0, "ax": 0.02, "ay": -0.98, "az": 0.11, "gx": 1.2, "gy": -0.4, "gz": 0.0}],
    "thigh": [{"t_ms": 0, "ax": 0.01, "ay": -0.95, "az": 0.08, "gx": 0.8, "gy": 0.2, "gz": -0.1}]
  }
}
```
(each array needs at least 100 samples spanning 5 s or more). Legacy request:
`{"readings": [...], "sensor_position": "shin"}`.

Response `200`:
```json
{
  "status": "complete",
  "prediction": "Higher severity",
  "confidence": 0.74,
  "risk_score": 74.0,
  "risk_tier": "high",
  "interpretation": "Estimated WOMAC-based symptom-severity tier ... This is not a diagnosis. ...",
  "features": {
    "stride_count": 6,
    "mean_stride_duration_s": 0.907,
    "stride_duration_cv": 0.015,
    "cadence_steps_per_min": 132.4,
    "stride_asymmetry": null,
    "shin_peak_angular_velocity_rad_s": 2.02,
    "shin_mean_accel_norm_m_s2": 9.93,
    "thigh_peak_angular_velocity_rad_s": 1.01,
    "mapped_sensors": ["RF", "LB"],
    "missing_sensors": ["HE", "LF"],
    "imputed_feature_fraction": 0.521
  },
  "warnings": ["..."],
  "prototype": true
}
```

| `risk_score` (P of higher severity) | `risk_tier` | `prediction` |
|---|---|---|
| >= 65% | high | Higher severity |
| 35% - 65% | monitor | Borderline |
| <= 35% | low | Lower severity |

(Same idea as the notebook's Part 13: scores within 0.15 of 0.5 are "monitor".)

Errors (JSON, no stack traces): `400` invalid payload / missing shin / unknown sensor / under 5 s,
`422` no strides detected or schema error, `503` model not loaded.

## Run locally

```bash
cd sensor
pip install -r requirements.txt
uvicorn sensor_api:app --host 0.0.0.0 --port 8001
python test_request.py      # in a second terminal, or on its own
```

## Run in Google Colab with ngrok

```python
# Cell 1: get the code and model files
!git clone <YOUR_REPO_URL> repo
%cd repo/sensor
!pip install -q -r requirements.txt
!mkdir -p model && cp "/content/drive/MyDrive/SENSOR DATASET/oa_sensor_"* model/   # after mounting Drive

# Cell 2: start the server + tunnel
import threading, uvicorn
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
threading.Thread(target=lambda: uvicorn.run("sensor_api:app", host="127.0.0.1", port=8001), daemon=True).start()
tunnel = ngrok.connect(8001)
print(tunnel.public_url + "/analyze-sensor")
```

* Free ngrok URLs **change every time the runtime restarts**; keep the URL in the backend's `SENSOR_API_URL` env var.
* Every backend request through ngrok must send the header `ngrok-skip-browser-warning: true`.

### Node.js example
```javascript
const axios = require('axios');
const res = await axios.post(`${process.env.SENSOR_API_URL}/analyze-sensor`,
  { sensors: { shin: readings.shin, thigh: readings.thigh } },
  { headers: { 'ngrok-skip-browser-warning': 'true' }, timeout: 15000 });
console.log(res.data.prediction, res.data.risk_score, res.data.warnings);
```

## Swapping in the retrained shin + thigh model

1. Train on shank + thigh recordings; drop or per-second-normalise length-dependent features (stride counts, spectral energy).
2. Save the model, scaler and feature list to `sensor/model/` with the same names.
3. In `config.py` set `ALL_TRAINING_SENSORS = ["shin", "thigh"]` and `SENSOR_MAPPING = {"shin": "shin", "thigh": "thigh"}`,
   and update the column-name handling in `preprocess.build_wide_dataframe` / `features.py` to match the new column prefixes.
4. Set `IS_PROTOTYPE = False` only after validating on real ESP32 data.
