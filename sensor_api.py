"""
Knee Osteoarthritis (KOA) Wearable IMU Sensor Inference API
=============================================================
FastAPI service for wearable IMU data (shin + thigh ESP32 nodes).

Endpoints:
  GET  /health           : service health and model load status
  POST /analyze-sensor   : JSON sensor streams -> gait features -> ML model -> severity tier

IMPORTANT: the model predicts a WOMAC-based symptom-severity tier for people who
already have knee OA (trained on KOA patients only). It is not a diagnosis.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

import config
from preprocess import preprocess_and_extract_features, PreprocessingError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sensor_api")


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================
class SensorReadingPoint(BaseModel):
    """One IMU sample."""
    t_ms: float = Field(..., description="Timestamp in milliseconds (non-decreasing)")
    ax: float = Field(..., description="Acceleration X (g)")
    ay: float = Field(..., description="Acceleration Y (g)")
    az: float = Field(..., description="Acceleration Z (g)")
    gx: float = Field(..., description="Gyroscope X (deg/s)")
    gy: float = Field(..., description="Gyroscope Y (deg/s)")
    gz: float = Field(..., description="Gyroscope Z (deg/s)")


class AnalyzeSensorRequest(BaseModel):
    """Either 'sensors' (recommended) or the legacy flat 'readings' array."""
    sampling_rate_hz: Optional[float] = Field(
        default=config.DEFAULT_SAMPLING_RATE_HZ,
        description="Informational only. Data is always resampled to 100 Hz using t_ms.",
    )
    sensors: Optional[Dict[str, List[SensorReadingPoint]]] = Field(
        default=None,
        description="Map of sensor location ('shin' required, 'thigh' optional) to readings.",
    )
    readings: Optional[List[SensorReadingPoint]] = Field(
        default=None, description="Legacy flat array of readings (single sensor)."
    )
    sensor_position: Optional[str] = Field(
        default="shin", description="Location of the legacy flat 'readings' ('shin' required)."
    )


class ClinicalFeaturesSummary(BaseModel):
    """Interpretable gait metrics for display."""
    stride_count: int
    mean_stride_duration_s: float
    stride_duration_cv: float
    cadence_steps_per_min: float
    stride_asymmetry: Optional[float] = None  # None = not measurable with one instrumented leg
    shin_peak_angular_velocity_rad_s: float
    shin_mean_accel_norm_m_s2: float
    thigh_peak_angular_velocity_rad_s: Optional[float] = None
    mapped_sensors: List[str]
    missing_sensors: List[str]
    imputed_feature_fraction: float = Field(
        ..., description="Share of model inputs that were placeholders, not measured (0-1)"
    )


class AnalyzeSensorResponse(BaseModel):
    status: str = "complete"
    prediction: str = Field(..., description="'Higher severity', 'Lower severity' or 'Borderline'")
    confidence: float = Field(..., description="max(p, 1-p) of the model probability [0.5 - 1.0]")
    risk_score: float = Field(..., description="Probability of the higher-severity class, in percent")
    risk_tier: str = Field(..., description="'high', 'monitor' or 'low'")
    interpretation: str
    features: ClinicalFeaturesSummary
    warnings: List[str] = Field(default_factory=list)
    prototype: bool = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    features_count: Optional[int] = None
    fill_mode: str


# ==============================================================================
# RESULT INTERPRETATION (pure function)
# ==============================================================================
def interpret_probability(prob_high: float) -> Tuple[str, str, float]:
    """Maps P(high severity) to (risk_tier, prediction_label, confidence)."""
    prob_high = max(0.0, min(1.0, float(prob_high)))
    confidence = max(prob_high, 1.0 - prob_high)
    if prob_high >= config.RISK_THRESHOLD_HIGH:
        return "high", config.LABEL_HIGH, confidence
    if prob_high <= config.RISK_THRESHOLD_LOW:
        return "low", config.LABEL_LOW, confidence
    return "monitor", config.LABEL_BORDERLINE, confidence


# ==============================================================================
# LIFESPAN: load model artifacts once
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading sensor model artifacts from %s ...", config.MODEL_DIR)
    app.state.model = None
    app.state.scaler = None
    app.state.feature_cols = None
    app.state.model_loaded = False

    try:
        if os.path.exists(config.FEATURE_COLUMNS_PATH):
            with open(config.FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                app.state.feature_cols = json.load(f)
            logger.info("Loaded %d feature names.", len(app.state.feature_cols))
        else:
            logger.warning("Feature list not found at %s", config.FEATURE_COLUMNS_PATH)

        if os.path.exists(config.SCALER_PATH):
            app.state.scaler = joblib.load(config.SCALER_PATH)
            logger.info("Loaded scaler from %s", config.SCALER_PATH)
        else:
            logger.warning("Scaler not found at %s", config.SCALER_PATH)

        # TODO: replace with the retrained shin+thigh model when available
        if os.path.exists(config.MODEL_PATH):
            app.state.model = joblib.load(config.MODEL_PATH)
            logger.info("Loaded model %s from %s", type(app.state.model).__name__, config.MODEL_PATH)
        else:
            logger.warning("Model not found at %s", config.MODEL_PATH)

        n_cols = len(app.state.feature_cols) if app.state.feature_cols else None
        n_scaler = getattr(app.state.scaler, "n_features_in_", None)
        n_model = getattr(app.state.model, "n_features_in_", None)
        consistent = app.state.model is not None and all(
            n is None or n_cols is None or n == n_cols for n in (n_scaler, n_model)
        )
        if app.state.model is not None and not consistent:
            logger.error(
                "Feature count mismatch: feature list=%s, scaler=%s, model=%s. Refusing to serve.",
                n_cols, n_scaler, n_model,
            )
        app.state.model_loaded = bool(consistent)
    except Exception as e:  # noqa: BLE001
        logger.error("Error loading artifacts: %s", e, exc_info=True)

    yield
    logger.info("Shutting down sensor API.")


# ==============================================================================
# APP
# ==============================================================================
app = FastAPI(
    title="KOA Wearable Sensor Inference API",
    description="Dual-IMU (shin + thigh) gait analysis -> symptom-severity tier (prototype).",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origins cannot be combined with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# ERROR HANDLERS (clean JSON, no stack traces)
# ==============================================================================
@app.exception_handler(PreprocessingError)
async def preprocessing_error_handler(request: Request, exc: PreprocessingError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_type": "ValidationError" if exc.status_code == 400 else "GaitDetectionError",
            "message": exc.message,
            "detail": exc.message,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    parts = []
    for err in exc.errors()[:10]:
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        parts.append(f"{field}: {err.get('msg', 'invalid value')}")
    msg = "; ".join(parts)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "error_type": "RequestValidationError", "message": msg, "detail": msg},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred during sensor signal processing.",
            "detail": "See server logs.",
        },
    )


# ==============================================================================
# ENDPOINTS
# ==============================================================================
@app.get("/health", response_model=HealthResponse)
def health_check():
    model = getattr(app.state, "model", None)
    features = getattr(app.state, "feature_cols", None)
    return HealthResponse(
        status="ok",
        model_loaded=bool(model is not None and app.state.model_loaded),
        model_type=type(model).__name__ if model is not None else None,
        features_count=len(features) if features else None,
        fill_mode=config.FILL_MODE,
    )


@app.post("/analyze-sensor", response_model=AnalyzeSensorResponse)
def analyze_sensor(request_data: AnalyzeSensorRequest):
    """
    1. Validate payload (shin required, >= 5 s, non-decreasing timestamps).
    2. Convert units and resample to 100 Hz.
    3. Detect strides from the shin gyroscope.
    4. Extract the notebook features; impute what the live hardware cannot measure.
    5. Scale, predict, and map the probability to a severity tier.
    """
    if not app.state.model_loaded or app.state.model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_type": "ModelNotLoaded",
                "message": "Model not loaded. Ensure the model, scaler and feature list exist in sensor/model/.",
                "detail": "Model artifacts missing or inconsistent; see server logs.",
            },
        )

    payload = request_data.model_dump(exclude_none=True)

    feature_matrix, summary_metrics, warnings = preprocess_and_extract_features(
        payload=payload,
        scaler=app.state.scaler,
        expected_feature_cols=app.state.feature_cols,
        fill_mode=config.FILL_MODE,
    )

    X = app.state.scaler.transform(feature_matrix) if app.state.scaler is not None else feature_matrix
    model = app.state.model

    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", [0, 1]))
        prob_high = float(model.predict_proba(X)[0][classes.index(1)])
    else:
        prob_high = float(1.0 / (1.0 + np.exp(-float(model.decision_function(X)[0]))))

    risk_tier, label, confidence = interpret_probability(prob_high)

    return AnalyzeSensorResponse(
        status="complete",
        prediction=label,
        confidence=round(confidence, 4),
        risk_score=round(max(0.0, min(1.0, prob_high)) * 100.0, 2),
        risk_tier=risk_tier,
        interpretation=config.INTERPRETATION,
        features=ClinicalFeaturesSummary(**summary_metrics),
        warnings=warnings,
        prototype=config.IS_PROTOTYPE,
    )
