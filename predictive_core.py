
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

try:
    from prophet import Prophet  # type: ignore
except Exception:  # pragma: no cover
    Prophet = None  # type: ignore

try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
except Exception:  # pragma: no cover
    ARIMA = None  # type: ignore

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore
    Sequential = LSTM = Dense = None  # type: ignore

try:
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    StandardScaler = None  # type: ignore


@dataclass
class ForecastMetrics:
    rmse: float
    mae: float
    directional_accuracy: float


def _set_seeds(seed: int = 42) -> None:
    if tf is not None:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
    np.random.seed(seed)


def _prepare_series(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    series = df[column].dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series


# ---------------- Prophet core ----------------

def fit_prophet(series: pd.Series) -> Tuple[Any, pd.Series]:
    if Prophet is None:
        raise RuntimeError("Prophet not available")
    df = series.reset_index()
    df.columns = ["ds", "y"]
    # Ensure ds column is timezone-naive
    if isinstance(df['ds'].dtype, pd.DatetimeTZDtype):
        df['ds'] = df['ds'].dt.tz_localize(None)
    elif pd.api.types.is_datetime64_any_dtype(df['ds']):
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None) if hasattr(pd.to_datetime(df['ds']), 'dt') else df['ds']
    model = Prophet()
    model.fit(df)
    forecast = model.predict(df)["yhat"]
    forecast.index = series.index
    return model, forecast


def forecast_prophet(model: Any, periods: int, last_date: pd.Timestamp, freq: str = "B") -> pd.Series:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    fc = model.predict(future)
    tail = fc.tail(periods)
    series = pd.Series(tail["yhat"].values, index=pd.to_datetime(tail["ds"]))
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series


# ---------------- LSTM residual model ----------------

def _build_lstm(input_len: int) -> Any:
    if Sequential is None or LSTM is None or Dense is None:
        raise RuntimeError("TensorFlow/Keras not available")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(input_len, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_on_residuals(residuals: pd.Series, lookback: int = 20, epochs: int = 10) -> Tuple[Any, Any]:
    if StandardScaler is None:
        raise RuntimeError("scikit-learn not available for scaling")
    
    lookback = int(lookback)
    if len(residuals) < lookback + 10:
        raise ValueError(f"Insufficient data: need at least {lookback + 10} points, got {len(residuals)}")
    
    _set_seeds(42)
    r = np.asarray(residuals.to_numpy(), dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(r)
    X, y = [], []
    
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i, 0])
        y.append(scaled[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        raise ValueError("No training samples generated - data too short")
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = _build_lstm(lookback)
    model.fit(X, y, epochs=int(epochs), batch_size=16, verbose=0)
    return model, scaler


def forecast_lstm_residuals(model: Any, scaler: Any, residuals: pd.Series, steps: int, lookback: int = 20) -> np.ndarray:
    lookback = int(lookback)
    if len(residuals) < lookback:
        raise ValueError(f"Insufficient data for forecasting: need at least {lookback} points, got {len(residuals)}")
    
    r = np.asarray(residuals.to_numpy(), dtype=float).reshape(-1, 1)
    scaled = scaler.transform(r)
    last_seq = scaled[-lookback:]
    
    if len(last_seq) < lookback:
        raise ValueError(f"Last sequence too short: expected {lookback}, got {len(last_seq)}")
    
    preds = []
    for _ in range(steps):
        inp = last_seq.reshape((1, lookback, 1))
        pred = model.predict(inp, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)
    inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return inv


# ---------------- Hybrid model ----------------

def hybrid_lstm_prophet_forecast(df: pd.DataFrame, steps: int, lookback: int = 20, epochs: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Hybrid residual model (paper requirement):
      1) Prophet on price => y_prophet(t)
      2) Residuals ε(t) = y_actual(t) - y_prophet(t)
      3) LSTM on residuals
      4) Predict residuals and add to Prophet forecast
    Returns (prophet_forecast, residual_forecast, hybrid_forecast)
    """
    series = _prepare_series(df)
    lookback = int(lookback)
    
    if len(series) < lookback + 20:
        raise ValueError(f"Insufficient data for hybrid model: need at least {lookback + 20} points, got {len(series)}")
    
    model, yhat = fit_prophet(series)
    residuals = series - yhat
    lstm_model, scaler = train_lstm_on_residuals(residuals, lookback=lookback, epochs=epochs)
    res_fc = forecast_lstm_residuals(lstm_model, scaler, residuals, steps=steps, lookback=lookback)
    prophet_fc = forecast_prophet(model, steps, series.index[-1])
    res_idx = prophet_fc.index
    res_series = pd.Series(res_fc, index=res_idx)
    hybrid = prophet_fc + res_series
    return prophet_fc, res_series, hybrid


def lstm_residual_only_forecast(df: pd.DataFrame, steps: int, lookback: int = 20, epochs: int = 10) -> pd.Series:
    """
    LSTM-only forecast on Prophet residuals (no direct price training).
    Final series uses last actual level plus residual forecast to remain in price scale.
    """
    series = _prepare_series(df)
    lookback = int(lookback)
    
    if len(series) < lookback + 20:
        raise ValueError(f"Insufficient data for LSTM forecast: need at least {lookback + 20} points, got {len(series)}")
    
    model, yhat = fit_prophet(series)
    residuals = series - yhat
    lstm_model, scaler = train_lstm_on_residuals(residuals, lookback=lookback, epochs=epochs)
    res_fc = forecast_lstm_residuals(lstm_model, scaler, residuals, steps=steps, lookback=lookback)
    idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
    base = float(series.iloc[-1])
    return pd.Series(base + res_fc, index=idx)


# ---------------- ARIMA ----------------

def arima_forecast(df: pd.DataFrame, steps: int, order: Tuple[int, int, int]) -> pd.Series:
    if ARIMA is None:
        raise RuntimeError("ARIMA not available")
    series = _prepare_series(df)
    model = ARIMA(series, order=tuple(order))
    fit = model.fit()
    fc = fit.forecast(steps=steps)
    fc.index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="B")
    return fc


# ---------------- Evaluation ----------------

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series, last_train: Optional[float] = None) -> ForecastMetrics:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    y_true_arr = np.asarray(y_true.to_numpy(), dtype=float)
    y_pred_arr = np.asarray(y_pred.to_numpy(), dtype=float)
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    # Directional accuracy
    if last_train is None:
        last_train = float(y_true.iloc[0])
    true_dirs = np.sign(np.diff(np.concatenate([np.array([last_train]), y_true_arr])))
    pred_dirs = np.sign(np.diff(np.concatenate([np.array([last_train]), y_pred_arr])))
    directional_accuracy = float(np.mean(true_dirs == pred_dirs))
    return ForecastMetrics(rmse=rmse, mae=mae, directional_accuracy=directional_accuracy)


def compare_models(df: pd.DataFrame, steps: int, order: Tuple[int, int, int], lookback: int, epochs: int) -> Dict[str, ForecastMetrics]:
    """
    Compare ARIMA vs Prophet vs LSTM vs Hybrid on a holdout of length `steps`.
    LSTM is trained on Prophet residuals per paper requirement.
    """
    series = _prepare_series(df)
    if len(series) <= steps + 10:
        raise ValueError("Not enough history for evaluation")
    train = series.iloc[:-steps]
    test = series.iloc[-steps:]
    last_train = float(train.iloc[-1])

    # ARIMA
    arima_pred = None
    if ARIMA is not None:
        arima_model = ARIMA(train, order=tuple(order))
        arima_fit = arima_model.fit()
        arima_pred = arima_fit.forecast(steps=steps)
        arima_pred.index = test.index

    # Prophet
    prop_model, prop_fit = fit_prophet(train)
    prop_pred = forecast_prophet(prop_model, steps, train.index[-1])
    prop_pred.index = test.index

    # LSTM residual-only
    lstm_pred = lstm_residual_only_forecast(train.to_frame(name="Close"), steps=steps, lookback=lookback, epochs=epochs)
    lstm_pred.index = test.index

    # Hybrid
    _, res_fc, hybrid = hybrid_lstm_prophet_forecast(train.to_frame(name="Close"), steps=steps, lookback=lookback, epochs=epochs)
    hybrid.index = test.index

    metrics: Dict[str, ForecastMetrics] = {}
    if arima_pred is not None:
        metrics["ARIMA"] = evaluate_forecast(test, arima_pred, last_train=last_train)
    metrics["Prophet"] = evaluate_forecast(test, prop_pred, last_train=last_train)
    metrics["LSTM"] = evaluate_forecast(test, lstm_pred, last_train=last_train)
    metrics["Hybrid"] = evaluate_forecast(test, hybrid, last_train=last_train)
    return metrics
