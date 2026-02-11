"""
Isolation Forest–based anomaly detection with Z-score standardized log returns and volume.
anomaly detection and risk criterion C3a.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest  # type: ignore
except Exception:  # pragma: no cover
    IsolationForest = None  # type: ignore


@dataclass
class AnomalyOutput:
    frame: pd.DataFrame
    threshold: float


def _zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std()
    if sd == 0 or np.isnan(sd):
        return (x - mu).fillna(0.0)
    return (x - mu) / sd


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 42) -> AnomalyOutput:
    if IsolationForest is None:
        raise RuntimeError("scikit-learn not available for IsolationForest")

    close = df["Close"].astype(float).copy()
    vol = df["Volume"].astype(float).copy()
    log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index)
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    z_ret = _zscore(log_ret)
    z_vol = _zscore(vol)
    feats = pd.DataFrame({"z_log_return": z_ret, "z_volume": z_vol}).fillna(0.0)

    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=random_state)
    model.fit(feats)
    raw_score = model.decision_function(feats)
    pred = model.predict(feats)

    out = df.copy()
    out["anomaly_score"] = -raw_score  # higher = more anomalous
    out["anomaly_flag"] = pred == -1

    # Threshold as high quantile of anomaly_score for non-compensatory rule
    threshold = float(np.quantile(out["anomaly_score"], 0.97))
    return AnomalyOutput(frame=out, threshold=threshold)


def compute_volatility(series: pd.Series, window: int = 20) -> float:
    rets = series.pct_change().dropna()
    if rets.empty:
        return 0.0
    vol = float(rets.rolling(window).std().iloc[-1])
    if np.isnan(vol):
        return 0.0
    return vol
