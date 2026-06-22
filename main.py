from __future__ import annotations

import logging

import math

import os

import joblib

import numpy as np

import holidays

from datetime import datetime, timezone

from functools import lru_cache

from pathlib import Path

from typing import Any

from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import FileResponse, JSONResponse

from pydantic import BaseModel, Field, field_validator, model_validator

import uvicorn



# 1. Ayarlar ve Loglama

LOGGER = logging.getLogger("traffic_api")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "traffic_model.pkl"

SCALER_PATH = BASE_DIR / "scaler.pkl"

ISTANBUL_TIMEZONE = ZoneInfo("Europe/Istanbul")



# 2. FastAPI Kurulumu

app = FastAPI(title="AI Traffic Prediction API", version="1.0.0")



app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)



def setup_logging() -> None:

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")



# 3. Pydantic Modelleri

class TrafficPredictionRequest(BaseModel):

    origin_latitude: float = Field(..., ge=36.0, le=42.0)

    origin_longitude: float = Field(..., ge=26.0, le=45.0)

    destination_latitude: float = Field(..., ge=36.0, le=42.0)

    destination_longitude: float = Field(..., ge=26.0, le=45.0)

    request_datetime: datetime



class PredictionData(BaseModel):

    estimated_minutes: float

    distance_km: float

    confidence_score: float



class PredictionResponse(BaseModel):

    status: str

    data: PredictionData

    timestamp: str



# 4. Model Yönetimi

class ModelBundle:

    def __init__(self, payload: dict[str, Any], scaler: Any) -> None:

        self.estimator = payload["estimator"]

        self.residual_std = float(payload.get("residual_std", 4.0))

        self.scaler = scaler



MODEL_BUNDLE: ModelBundle | None = None



def load_model_bundle() -> ModelBundle:

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():

        raise RuntimeError("Model dosyaları bulunamadı!")

    return ModelBundle(joblib.load(MODEL_PATH), joblib.load(SCALER_PATH))



# 5. Yardımcı Fonksiyonlar

def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0088

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2

    return 2 * R * math.asin(math.sqrt(a))



@lru_cache(maxsize=16)

def get_holiday_checker(year: int):

    return holidays.Turkey(years=[year])



# 6. Endpointler

@app.on_event("startup")

def startup_event():

    setup_logging()

    global MODEL_BUNDLE

    MODEL_BUNDLE = load_model_bundle()

    LOGGER.info("Model başarıyla yüklendi.")



@app.get("/")

async def serve_frontend():

    return FileResponse(BASE_DIR / "index.html")



@app.post("/api/v1/predict-duration", response_model=PredictionResponse)

async def predict_duration(payload: TrafficPredictionRequest):

    if MODEL_BUNDLE is None:

        raise HTTPException(status_code=503, detail="Model yüklenemedi")



    # Özellik Mühendisliği

    dist = haversine_km(payload.origin_latitude, payload.origin_longitude,

                        payload.destination_latitude, payload.destination_longitude)

    dt = payload.request_datetime.astimezone(ISTANBUL_TIMEZONE)

   

    features = [[

        payload.origin_latitude, payload.origin_longitude, payload.destination_latitude,

        payload.destination_longitude, dist, float(dt.hour), float(dt.minute),

        float(dt.weekday()), 1.0 if dt.weekday() >= 5 else 0.0,

        1.0 if dt.date() in get_holiday_checker(dt.year) else 0.0,

        1.0 # Peak multiplier (basitleştirilmiş)

    ]]

   

    scaled = MODEL_BUNDLE.scaler.transform(features)

    pred = float(MODEL_BUNDLE.estimator.predict(scaled)[0])

   

    return PredictionResponse(

        status="success",

        data=PredictionData(estimated_minutes=round(pred, 2), distance_km=round(dist, 2), confidence_score=0.95),

        timestamp=datetime.now(timezone.utc).isoformat()

    )



if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)