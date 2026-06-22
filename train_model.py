from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor


LOGGER = logging.getLogger("traffic_training")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
RANDOM_STATE = 42
N_SAMPLES = 30000


ISTANBUL_BOUNDS = {
    "lat_min": 40.802,
    "lat_max": 41.468,
    "lon_min": 28.315,
    "lon_max": 29.445,
}

ANKARA_BOUNDS = {
    "lat_min": 39.731,
    "lat_max": 40.305,
    "lon_min": 32.469,
    "lon_max": 33.140,
}

CITY_BOUNDS = {"istanbul": ISTANBUL_BOUNDS, "ankara": ANKARA_BOUNDS}


@dataclass(frozen=True)
class RouteProfile:
    name: str
    base_speed_kmh: float
    congestion_multiplier: float
    variability: float


ROUTE_PROFILES = {
    "same_city": RouteProfile("same_city", base_speed_kmh=41.0, congestion_multiplier=1.00, variability=0.22),
    "cross_city": RouteProfile("cross_city", base_speed_kmh=76.0, congestion_multiplier=0.92, variability=0.15),
    "dense_central": RouteProfile("dense_central", base_speed_kmh=33.0, congestion_multiplier=1.18, variability=0.28),
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    earth_radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def sample_coordinates(bounds: dict[str, float], size: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.random.uniform(bounds["lat_min"], bounds["lat_max"], size=size)
    lon = np.random.uniform(bounds["lon_min"], bounds["lon_max"], size=size)
    return lat, lon


def is_turkish_holiday(day: pd.Timestamp) -> bool:
    fixed_holidays = {
        (1, 1),
        (4, 23),
        (5, 1),
        (5, 19),
        (7, 15),
        (8, 30),
        (10, 29),
    }
    return (day.month, day.day) in fixed_holidays


def build_peak_multiplier(hour: np.ndarray, minute: np.ndarray, is_weekend: np.ndarray, is_holiday: np.ndarray) -> np.ndarray:
    minutes_of_day = hour * 60 + minute
    morning_peak = (minutes_of_day >= 7 * 60 + 30) & (minutes_of_day <= 9 * 60 + 30)
    evening_peak = (minutes_of_day >= 17 * 60) & (minutes_of_day <= 19 * 60 + 30)
    lunch_peak = (minutes_of_day >= 12 * 60) & (minutes_of_day <= 13 * 60 + 30)

    multiplier = np.ones_like(hour, dtype=float)
    multiplier += np.where(morning_peak, 0.32, 0.0)
    multiplier += np.where(evening_peak, 0.38, 0.0)
    multiplier += np.where(lunch_peak, 0.08, 0.0)
    multiplier -= np.where(is_weekend, 0.10, 0.0)
    multiplier -= np.where(is_holiday, 0.14, 0.0)
    return np.clip(multiplier, 0.62, 1.95)


def build_city_pressure(origin_lat: np.ndarray, origin_lon: np.ndarray, destination_lat: np.ndarray, destination_lon: np.ndarray) -> np.ndarray:
    istanbul_center = (41.0082, 28.9784)
    ankara_center = (39.9334, 32.8597)

    origin_ist = haversine_km(origin_lat, origin_lon, np.full_like(origin_lat, istanbul_center[0]), np.full_like(origin_lon, istanbul_center[1]))
    destination_ist = haversine_km(destination_lat, destination_lon, np.full_like(destination_lat, istanbul_center[0]), np.full_like(destination_lon, istanbul_center[1]))
    origin_ank = haversine_km(origin_lat, origin_lon, np.full_like(origin_lat, ankara_center[0]), np.full_like(origin_lon, ankara_center[1]))
    destination_ank = haversine_km(destination_lat, destination_lon, np.full_like(destination_lat, ankara_center[0]), np.full_like(destination_lon, ankara_center[1]))

    origin_city = np.where(origin_ist <= origin_ank, 1.0, 0.0)
    destination_city = np.where(destination_ist <= destination_ank, 1.0, 0.0)
    same_city = origin_city == destination_city
    dense_central = (origin_ist < 8.0) & (destination_ist < 8.0)

    pressure = np.where(same_city, 1.0, 0.88)
    pressure += np.where(dense_central, 0.18, 0.0)
    pressure += np.where(~same_city, 0.06, 0.0)
    return np.clip(pressure, 0.82, 1.28)


def generate_synthetic_data(n_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    route_types = np.array(["same_city", "cross_city", "dense_central"])
    route_weights = np.array([0.58, 0.18, 0.24])
    selected_routes = rng.choice(route_types, size=n_samples, p=route_weights)
    origin_city_choices = rng.choice(["istanbul", "ankara"], size=n_samples)
    destination_city_choices = np.where(rng.random(n_samples) < 0.26, np.where(origin_city_choices == "istanbul", "ankara", "istanbul"), origin_city_choices)

    origin_lat = np.empty(n_samples)
    origin_lon = np.empty(n_samples)
    destination_lat = np.empty(n_samples)
    destination_lon = np.empty(n_samples)

    for city in CITY_BOUNDS:
        city_mask = origin_city_choices == city
        lat, lon = sample_coordinates(CITY_BOUNDS[city], int(city_mask.sum()))
        origin_lat[city_mask] = lat
        origin_lon[city_mask] = lon

    for city in CITY_BOUNDS:
        city_mask = destination_city_choices == city
        lat, lon = sample_coordinates(CITY_BOUNDS[city], int(city_mask.sum()))
        destination_lat[city_mask] = lat
        destination_lon[city_mask] = lon

    raw_day = rng.integers(1, 367, size=n_samples)
    raw_hour = rng.integers(0, 24, size=n_samples)
    raw_minute = rng.integers(0, 60, size=n_samples)
    base_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(raw_day - 1, unit="D")
    is_weekend = np.asarray(base_dates.dayofweek >= 5, dtype=int)
    is_holiday = np.array([1 if is_turkish_holiday(ts) else 0 for ts in base_dates], dtype=int)

    distance_km = haversine_km(origin_lat, origin_lon, destination_lat, destination_lon)
    route_pressure = build_city_pressure(origin_lat, origin_lon, destination_lat, destination_lon)
    peak_multiplier = build_peak_multiplier(raw_hour, raw_minute, is_weekend, is_holiday)

    profile_map = np.vectorize(lambda name: ROUTE_PROFILES[name].base_speed_kmh)(selected_routes).astype(float)
    variability = np.vectorize(lambda name: ROUTE_PROFILES[name].variability)(selected_routes).astype(float)
    congestion = np.vectorize(lambda name: ROUTE_PROFILES[name].congestion_multiplier)(selected_routes).astype(float)

    effective_speed = profile_map * (1.0 - 0.20 * (peak_multiplier - 1.0)) * route_pressure * congestion
    effective_speed = np.clip(effective_speed, 14.0, 95.0)

    base_minutes = (distance_km / np.maximum(effective_speed, 1e-6)) * 60.0
    density_delay = np.where(selected_routes == "dense_central", 5.5, 0.0)
    cross_city_delay = np.where(selected_routes == "cross_city", 7.0, 0.0)
    time_penalty = np.where(is_weekend == 1, -2.4, 0.0) + np.where(is_holiday == 1, -3.6, 0.0)

    peak_boost = np.where((raw_hour >= 7) & (raw_hour <= 9), 1.18, 1.0) * np.where((raw_hour >= 17) & (raw_hour <= 19), 1.12, 1.0)
    noise = rng.normal(0.0, 1.7 + variability * 5.0, size=n_samples)

    duration_minutes = (
        base_minutes * peak_multiplier * peak_boost
        + density_delay
        + cross_city_delay
        + time_penalty
        + noise
    )
    duration_minutes = np.clip(duration_minutes, 4.0, None)

    df = pd.DataFrame(
        {
            "origin_latitude": origin_lat,
            "origin_longitude": origin_lon,
            "destination_latitude": destination_lat,
            "destination_longitude": destination_lon,
            "hour": raw_hour,
            "minute": raw_minute,
            "day_of_week": np.asarray(base_dates.dayofweek, dtype=int),
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "distance_km": distance_km,
            "peak_multiplier": peak_multiplier,
            "traffic_duration_minutes": duration_minutes,
        }
    )
    return df


def train_and_select_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict[str, Any], int, dict[str, Any]]:
    param_grid = list(
        ParameterGrid(
            {
                "n_estimators": [500, 800],
                "learning_rate": [0.03, 0.05],
                "max_depth": [4, 5, 6],
                "subsample": [0.8, 0.9],
                "colsample_bytree": [0.8, 0.9],
                "min_child_weight": [1, 3],
                "reg_alpha": [0.0, 0.1],
                "reg_lambda": [1.0, 2.0],
            }
        )
    )
    random.Random(RANDOM_STATE).shuffle(param_grid)
    param_grid = param_grid[:12]

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score = -math.inf
    best_params: dict[str, Any] | None = None
    best_iterations: list[int] = []

    for params in param_grid:
        fold_scores: list[float] = []
        fold_iterations: list[int] = []
        for fold_index, (fit_index, valid_index) in enumerate(kfold.split(X_train), start=1):
            fold_X_fit = X_train.iloc[fit_index]
            fold_X_valid = X_train.iloc[valid_index]
            fold_y_fit = y_train.iloc[fit_index]
            fold_y_valid = y_train.iloc[valid_index]

            scaler = StandardScaler()
            fold_X_fit_scaled = scaler.fit_transform(fold_X_fit)
            fold_X_valid_scaled = scaler.transform(fold_X_valid)

            training_params = {
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "seed": RANDOM_STATE,
                "learning_rate": params["learning_rate"],
                "max_depth": params["max_depth"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "min_child_weight": params["min_child_weight"],
                "alpha": params["reg_alpha"],
                "lambda": params["reg_lambda"],
                "nthread": -1,
            }
            dtrain = xgb.DMatrix(fold_X_fit_scaled, label=fold_y_fit.to_numpy())
            dvalid = xgb.DMatrix(fold_X_valid_scaled, label=fold_y_valid.to_numpy())
            booster = xgb.train(
                training_params,
                dtrain,
                num_boost_round=int(params["n_estimators"]),
                evals=[(dvalid, "validation")],
                early_stopping_rounds=35,
                verbose_eval=False,
            )

            best_iteration = int(getattr(booster, "best_iteration", params["n_estimators"] - 1))
            predictions = booster.predict(dvalid, iteration_range=(0, best_iteration + 1))
            fold_scores.append(r2_score(fold_y_valid, predictions))
            fold_iterations.append(best_iteration + 1)
            LOGGER.info("Fold %s completed for params %s with R2=%.4f", fold_index, params, fold_scores[-1])

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_iterations = fold_iterations

    if best_params is None:
        raise RuntimeError("No parameter combination produced a valid model")

    tuned_params = dict(best_params)
    tuned_params["n_estimators"] = int(max(best_iterations)) if best_iterations else tuned_params["n_estimators"]
    tuned_params["n_estimators"] = max(tuned_params["n_estimators"], 300)

    best_n_estimators = int(max(best_iterations)) if best_iterations else int(tuned_params["n_estimators"])
    final_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=best_n_estimators,
        learning_rate=tuned_params["learning_rate"],
        max_depth=tuned_params["max_depth"],
        subsample=tuned_params["subsample"],
        colsample_bytree=tuned_params["colsample_bytree"],
        min_child_weight=tuned_params["min_child_weight"],
        reg_alpha=tuned_params["reg_alpha"],
        reg_lambda=tuned_params["reg_lambda"],
    )
    return tuned_params, best_n_estimators, {"best_params": tuned_params, "cv_r2": best_score}


def main() -> None:
    setup_logging()
    LOGGER.info("Synthetic traffic dataset generation started")
    data = generate_synthetic_data(N_SAMPLES)

    feature_columns = [
        "origin_latitude",
        "origin_longitude",
        "destination_latitude",
        "destination_longitude",
        "distance_km",
        "hour",
        "minute",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "peak_multiplier",
    ]
    X = data[feature_columns].copy()
    y = data["traffic_duration_minutes"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tuned_params, best_n_estimators, search_meta = train_and_select_model(
        pd.DataFrame(X_train_scaled, columns=feature_columns),
        y_train.reset_index(drop=True),
    )
    validation_split = train_test_split(
        pd.DataFrame(X_train_scaled, columns=feature_columns),
        y_train.reset_index(drop=True),
        test_size=0.15,
        random_state=RANDOM_STATE,
    )
    X_fit, X_valid, y_fit, y_valid = validation_split

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=best_n_estimators,
        learning_rate=tuned_params["learning_rate"],
        max_depth=tuned_params["max_depth"],
        subsample=tuned_params["subsample"],
        colsample_bytree=tuned_params["colsample_bytree"],
        min_child_weight=tuned_params["min_child_weight"],
        reg_alpha=tuned_params["reg_alpha"],
        reg_lambda=tuned_params["reg_lambda"],
    )
    model.fit(
        X_fit,
        y_fit,
    )

    test_predictions = model.predict(X_test_scaled)
    r2 = r2_score(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    residual_std = float(np.std(y_valid.to_numpy() - model.predict(X_valid), ddof=1))

    LOGGER.info("Model training completed")
    LOGGER.info("Best CV R2: %.4f", search_meta["cv_r2"])
    LOGGER.info("Test R2: %.4f", r2)
    LOGGER.info("Test MSE: %.4f", mse)
    LOGGER.info("Test MAE: %.4f", mae)

    model_payload = {
        "estimator": model,
        "feature_columns": feature_columns,
        "residual_std": residual_std,
        "metrics": {
            "cv_r2": float(search_meta["cv_r2"]),
            "test_r2": float(r2),
            "test_mse": float(mse),
            "test_mae": float(mae),
        },
        "best_params": search_meta["best_params"],
        "best_n_estimators": best_n_estimators,
    }

    joblib.dump(model_payload, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    LOGGER.info("Saved model to %s", MODEL_PATH)
    LOGGER.info("Saved scaler to %s", SCALER_PATH)


if __name__ == "__main__":
    main()