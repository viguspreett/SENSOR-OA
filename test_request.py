"""
Test Suite for the KOA Sensor Inference API
============================================
Runs against a live server (http://127.0.0.1:8001) if one is up, otherwise
in-process with FastAPI's TestClient (needs `httpx`, see requirements.txt).
Requires the model files to be present in sensor/model/.

Tests:
  1. GET  /health
  2. POST /analyze-sensor  standard schema (shin + thigh, 6 s)
  3. POST /analyze-sensor  legacy flat schema (shin only)
  4. POST /analyze-sensor  too short (< 5 s)            -> 400
  5. POST /analyze-sensor  no shin sensor               -> 400
  6. POST /analyze-sensor  unknown sensor name          -> 400
"""

import json
import numpy as np
import requests

SERVER_URL = "http://127.0.0.1:8001"
ALLOWED_PREDICTIONS = {"Higher severity", "Lower severity", "Borderline"}


def generate_walking_sensor_data(
    duration_s: float = 6.0,
    sampling_rate_hz: float = 100.0,
    cadence_strides_per_sec: float = 1.1,
) -> dict:
    """Synthetic shin + thigh walking telemetry (g and deg/s), for pipeline testing only."""
    n = int(duration_s * sampling_rate_hz)
    t_s = np.linspace(0, duration_s, n)
    t_ms = (t_s * 1000.0).tolist()
    phase = 2.0 * np.pi * cadence_strides_per_sec * t_s
    rng = np.random.default_rng(0)

    # Shin
    shin_gy = np.maximum(100.0 * np.sin(phase) + 30.0 * np.sin(2 * phase), -10.0) + rng.normal(0, 2.0, n)
    shin_gx = 10.0 * np.cos(phase) + rng.normal(0, 1.0, n)
    shin_gz = 5.0 * np.sin(phase) + rng.normal(0, 1.0, n)
    shin_ay = -0.98 + 0.35 * np.cos(phase) + rng.normal(0, 0.05, n)
    shin_ax = 0.05 + 0.20 * np.sin(phase) + rng.normal(0, 0.04, n)
    shin_az = 0.10 + 0.15 * np.sin(2 * phase) + rng.normal(0, 0.03, n)

    # Thigh
    thigh_gy = 55.0 * np.sin(phase + 0.3) + rng.normal(0, 1.5, n)
    thigh_gx = 8.0 * np.cos(phase) + rng.normal(0, 1.0, n)
    thigh_gz = 4.0 * np.sin(phase) + rng.normal(0, 0.8, n)
    thigh_ay = -0.95 + 0.20 * np.cos(phase + 0.3) + rng.normal(0, 0.04, n)
    thigh_ax = 0.03 + 0.12 * np.sin(phase) + rng.normal(0, 0.03, n)
    thigh_az = 0.08 + 0.10 * np.sin(phase) + rng.normal(0, 0.03, n)

    def pack(ax, ay, az, gx, gy, gz):
        return [
            {
                "t_ms": round(t_ms[i], 1),
                "ax": round(float(ax[i]), 4), "ay": round(float(ay[i]), 4), "az": round(float(az[i]), 4),
                "gx": round(float(gx[i]), 2), "gy": round(float(gy[i]), 2), "gz": round(float(gz[i]), 2),
            }
            for i in range(n)
        ]

    return {
        "shin": pack(shin_ax, shin_ay, shin_az, shin_gx, shin_gy, shin_gz),
        "thigh": pack(thigh_ax, thigh_ay, thigh_az, thigh_gx, thigh_gy, thigh_gz),
    }


def run_tests():
    def banner(title):
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

    def execute_suite(http_get, http_post):
        banner("TEST 1: GET /health")
        res = http_get("/health")
        print(res.status_code, json.dumps(res.json(), indent=2))
        assert res.status_code == 200
        assert res.json().get("model_loaded") is True, "Model not loaded (check sensor/model/)"

        data = generate_walking_sensor_data(6.0)

        banner("TEST 2: standard schema (shin + thigh)")
        res = http_post("/analyze-sensor", {"sensors": data})
        print(res.status_code, json.dumps(res.json(), indent=2))
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["status"] == "complete"
        assert body["prediction"] in ALLOWED_PREDICTIONS
        assert body["risk_tier"] in {"high", "monitor", "low"}
        assert body["features"]["stride_count"] >= 2
        assert body["features"]["stride_asymmetry"] is None  # one leg only -> not measurable
        assert body["prototype"] is True

        banner("TEST 3: legacy flat schema (shin only)")
        res = http_post("/analyze-sensor", {"readings": data["shin"], "sensor_position": "shin"})
        print(res.status_code)
        assert res.status_code == 200, res.text

        banner("TEST 4: too short (2 s) -> 400")
        res = http_post("/analyze-sensor", {"sensors": generate_walking_sensor_data(2.0)})
        print(res.status_code, res.json().get("message"))
        assert res.status_code == 400
        assert "Minimum required duration" in res.json().get("detail", "")

        banner("TEST 5: no shin sensor -> 400")
        res = http_post("/analyze-sensor", {"sensors": {"thigh": data["thigh"]}})
        print(res.status_code, res.json().get("message"))
        assert res.status_code == 400

        banner("TEST 6: unknown sensor name -> 400")
        res = http_post("/analyze-sensor", {"sensors": {"shin": data["shin"], "wrist": data["thigh"]}})
        print(res.status_code, res.json().get("message"))
        assert res.status_code == 400

        banner("ALL TESTS PASSED")

    try:
        requests.get(f"{SERVER_URL}/health", timeout=1.0)
        print(f"[*] Using live server at {SERVER_URL}")
        execute_suite(
            lambda p: requests.get(f"{SERVER_URL}{p}"),
            lambda p, d: requests.post(f"{SERVER_URL}{p}", json=d),
        )
    except requests.exceptions.ConnectionError:
        print("[*] No live server found. Using in-process FastAPI TestClient...")
        from fastapi.testclient import TestClient
        from sensor_api import app

        with TestClient(app) as client:
            execute_suite(lambda p: client.get(p), lambda p, d: client.post(p, json=d))


if __name__ == "__main__":
    run_tests()
