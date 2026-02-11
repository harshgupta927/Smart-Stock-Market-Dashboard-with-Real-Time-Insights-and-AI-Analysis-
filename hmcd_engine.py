"""

Mathematical Model:
    HMCD_score = Σ(w_i × s_i) - w_risk × r
    where:
        s_i ∈ [0, 1]: normalized criterion scores
        w_i: criterion weights (Σw_i = 1.0)
        r ∈ [0, 1]: risk penalty score
        
Decision Thresholds:
    BUY:  HMCD_score > +0.25
    SELL: HMCD_score < -0.25
    HOLD: -0.25 ≤ HMCD_score ≤ +0.25
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np


@dataclass
class HMCDWeights:
    """
    Criterion weights for HMCD decision framework.
    Total weight = 100%, with risk as negative contribution.
    """
    predictive_strength: float = 0.40      # Model forecast accuracy
    technical_confirmation: float = 0.30   # RSI, MACD signals
    investor_sentiment: float = 0.20       # News + social sentiment
    risk_penalty: float = 0.10             # Anomaly & volatility (negative)
    
    def validate(self) -> bool:
        """Ensure weights sum to 1.0 (excluding risk penalty)."""
        total = self.predictive_strength + self.technical_confirmation + self.investor_sentiment
        return abs(total - 0.90) < 0.01  # Allow 1% tolerance


@dataclass
class HMCDOutput:
    """Explainable HMCD decision output."""
    score: float                           # Final HMCD score in [-1, +1]
    decision: str                          # BUY / HOLD / SELL
    confidence: float                      # Decision confidence [0, 1]
    rationale: str                         # Human-readable explanation
    contributions: Dict[str, float]        # Individual criterion contributions
    signals_breakdown: Dict[str, str]      # Per-criterion interpretations


def normalize_0_1(x: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    """Normalize value to [0, 1] range with bounds checking."""
    if max_v == min_v:
        return 0.5
    return float(np.clip((x - min_v) / (max_v - min_v), 0.0, 1.0))


def compute_predictive_strength(metrics: Dict[str, float]) -> float:
    """
    Compute model forecast quality score ∈ [0, 1].
    
    Combines:
        - Inverse RMSE (lower error = higher score)
        - Inverse MAE (lower error = higher score)
        - Directional accuracy (hit rate)
    
    Args:
        metrics: {"rmse": float, "mae": float, "directional_accuracy": float}
    
    Returns:
        Normalized score where 1.0 = perfect predictions, 0.0 = worst
    """
    rmse = float(metrics.get("rmse", 1.0))
    mae = float(metrics.get("mae", 1.0))
    da = float(metrics.get("directional_accuracy", 0.5))
    
    # Inverse error with soft normalization
    rmse_score = 1.0 / (1.0 + rmse)
    mae_score = 1.0 / (1.0 + mae)
    
    # Directional accuracy is already in [0, 1]
    da_score = float(np.clip(da, 0.0, 1.0))
    
    # Equal weighting of all three metrics
    return float(np.clip(np.mean([rmse_score, mae_score, da_score]), 0.0, 1.0))


def compute_technical_confirmation(rsi: float, macd: float, macd_signal: float, ema_short: Optional[float] = None, ema_long: Optional[float] = None) -> float:
    """
    Compute technical indicator consensus score ∈ [0, 1].
    
    Key improvement: RSI 45-55 is treated as neutral (0.5), not bearish.
    
    Components:
        - RSI: 70+ → 0.0 (overbought), 30- → 1.0 (oversold), 50 → 0.5 (neutral)
        - MACD: positive divergence → 1.0, negative → 0.0
        - EMA crossover: bullish → 1.0, bearish → 0.0 (optional)
    
    Returns:
        Score where 1.0 = strong bullish, 0.0 = strong bearish, 0.5 = neutral
    """
    # RSI: Map to [0, 1] with 50 = neutral
    # Below 50 → bullish zone (0.5-1.0), Above 50 → bearish zone (0.0-0.5)
    if rsi <= 30:
        rsi_score = 1.0  # Oversold → bullish signal
    elif rsi >= 70:
        rsi_score = 0.0  # Overbought → bearish signal
    elif rsi < 50:
        # 30 to 50: map [30, 50] → [1.0, 0.5]
        rsi_score = 1.0 - (rsi - 30) / 40
    else:
        # 50 to 70: map [50, 70] → [0.5, 0.0]
        rsi_score = 0.5 - (rsi - 50) / 40
    
    rsi_score = float(np.clip(rsi_score, 0.0, 1.0))
    
    # MACD: Bullish divergence → 1.0, bearish → 0.0
    macd_score = 1.0 if macd > macd_signal else 0.0
    
    # EMA crossover (if provided)
    scores = [rsi_score, macd_score]
    if ema_short is not None and ema_long is not None:
        ema_score = 1.0 if ema_short > ema_long else 0.0
        scores.append(ema_score)
    
    return float(np.mean(scores))


def compute_investor_sentiment(news_sentiment: float, youtube_fame: float) -> float:
    """
    Aggregate investor sentiment from multiple sources ∈ [0, 1].
    
    Args:
        news_sentiment: VADER compound score, normalized to [0, 1]
        youtube_fame: Social media engagement score [0, 1]
    
    Returns:
        Combined sentiment where 1.0 = very positive, 0.0 = very negative
    """
    # Ensure inputs are in [0, 1]
    news_norm = float(np.clip(news_sentiment, 0.0, 1.0))
    fame_norm = float(np.clip(youtube_fame, 0.0, 1.0))
    
    # Weighted average: news is more important than social fame
    sentiment_score = 0.7 * news_norm + 0.3 * fame_norm
    
    return float(np.clip(sentiment_score, 0.0, 1.0))


def compute_risk_score(anomaly_score: float, volatility: float) -> float:
    """
    Compute market risk penalty score ∈ [0, 1].
    
    Higher risk → higher penalty → reduces HMCD score.
    
    Args:
        anomaly_score: Isolation Forest score (unbounded)
        volatility: Historical volatility (unbounded)
    
    Returns:
        Risk penalty where 1.0 = maximum risk, 0.0 = minimal risk
    """
    # Log-scale normalization for unbounded inputs
    a = float(np.log1p(max(anomaly_score, 0.0)))
    v = float(np.log1p(max(volatility, 0.0)))
    
    if a == 0 and v == 0:
        return 0.0
    
    # Soft normalization: asymptotic approach to 1.0
    anomaly_component = a / (a + 1.0)
    volatility_component = v / (v + 1.0)
    
    # Weight anomaly detection more than volatility
    risk_score = 0.6 * anomaly_component + 0.4 * volatility_component
    
    return float(np.clip(risk_score, 0.0, 1.0))


def hmcd_score_calculation(
    predictive_strength: float,
    technical_confirmation: float,
    investor_sentiment: float,
    risk_penalty: float,
    weights: HMCDWeights
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate HMCD score using weighted compensatory model.
    
    Mathematical Formula:
        HMCD = w₁·s₁ + w₂·s₂ + w₃·s₃ - w_r·r
    
    Where:
        s₁ = predictive_strength ∈ [0, 1]
        s₂ = technical_confirmation ∈ [0, 1]
        s₃ = investor_sentiment ∈ [0, 1]
        r = risk_penalty ∈ [0, 1]
        w₁ = 0.40, w₂ = 0.30, w₃ = 0.20, w_r = 0.10
    
    Score is then mapped from [0, 1] → [-1, +1] for symmetric thresholds.
    
    Returns:
        (hmcd_score, contributions_dict)
        hmcd_score ∈ [-1, +1]: negative = bearish, positive = bullish
        contributions_dict: individual weighted contributions
    """
    # Ensure all inputs are in [0, 1]
    ps = float(np.clip(predictive_strength, 0.0, 1.0))
    tc = float(np.clip(technical_confirmation, 0.0, 1.0))
    sent = float(np.clip(investor_sentiment, 0.0, 1.0))
    risk = float(np.clip(risk_penalty, 0.0, 1.0))
    
    # Calculate individual contributions
    contrib_ps = weights.predictive_strength * ps
    contrib_tc = weights.technical_confirmation * tc
    contrib_sent = weights.investor_sentiment * sent
    contrib_risk = weights.risk_penalty * risk  # This will be subtracted
    
    # Raw score in [0, 1] range (before risk penalty)
    raw_score = contrib_ps + contrib_tc + contrib_sent
    
    # Apply risk penalty
    final_score_0_1 = raw_score - contrib_risk
    
    # Map from [0, 1] → [-1, +1] for symmetric decision boundaries
    # 0.5 → 0.0 (neutral), 0.0 → -1.0 (max bearish), 1.0 → +1.0 (max bullish)
    hmcd_final = 2.0 * final_score_0_1 - 1.0
    hmcd_final = float(np.clip(hmcd_final, -1.0, 1.0))
    
    contributions = {
        "predictive_strength": contrib_ps,
        "technical_confirmation": contrib_tc,
        "investor_sentiment": contrib_sent,
        "risk_penalty": -contrib_risk,  # Negative to show it reduces score
        "raw_score": raw_score,
        "final_score": hmcd_final
    }
    
    return hmcd_final, contributions


def interpret_signals(
    predictive_strength: float,
    technical_confirmation: float,
    investor_sentiment: float,
    risk_penalty: float
) -> Dict[str, str]:
    """Generate human-readable interpretations for each signal."""
    interpretations = {}
    
    # Predictive strength
    if predictive_strength >= 0.7:
        interpretations["predictive"] = "Strong forecast accuracy"
    elif predictive_strength >= 0.5:
        interpretations["predictive"] = "Moderate forecast reliability"
    else:
        interpretations["predictive"] = "Weak forecast confidence"
    
    # Technical confirmation
    if technical_confirmation >= 0.7:
        interpretations["technical"] = "Strong bullish indicators"
    elif technical_confirmation >= 0.55:
        interpretations["technical"] = "Mild bullish trend"
    elif technical_confirmation >= 0.45:
        interpretations["technical"] = "Neutral technical signals"
    elif technical_confirmation >= 0.3:
        interpretations["technical"] = "Mild bearish trend"
    else:
        interpretations["technical"] = "Strong bearish indicators"
    
    # Investor sentiment
    if investor_sentiment >= 0.7:
        interpretations["sentiment"] = "Very positive market sentiment"
    elif investor_sentiment >= 0.55:
        interpretations["sentiment"] = "Moderately positive sentiment"
    elif investor_sentiment >= 0.45:
        interpretations["sentiment"] = "Neutral investor sentiment"
    elif investor_sentiment >= 0.3:
        interpretations["sentiment"] = "Moderately negative sentiment"
    else:
        interpretations["sentiment"] = "Very negative market sentiment"
    
    # Risk penalty
    if risk_penalty >= 0.7:
        interpretations["risk"] = "High market risk detected"
    elif risk_penalty >= 0.4:
        interpretations["risk"] = "Moderate risk levels"
    else:
        interpretations["risk"] = "Low risk environment"
    
    return interpretations


def hmcd_decision(
    predictive_strength: float,
    technical_confirmation: float,
    investor_sentiment: float,
    risk_penalty: float,
    weights: Optional[HMCDWeights] = None
) -> HMCDOutput:
    """
    Human-Machine Collaborative Decision (HMCD) Framework.
    
    Balanced compensatory model with explainable outputs.
    Designed for IEEE publication and practical deployment.
    
    Args:
        predictive_strength: Model forecast quality [0, 1]
        technical_confirmation: Technical indicator consensus [0, 1]
        investor_sentiment: Combined sentiment score [0, 1]
        risk_penalty: Market risk score [0, 1]
        weights: Custom weight allocation (optional)
    
    Returns:
        HMCDOutput with decision, score, confidence, and rationale
    
    Decision Rules:
        BUY:  score > +0.25
        SELL: score < -0.25
        HOLD: -0.25 ≤ score ≤ +0.25
    """
    if weights is None:
        weights = HMCDWeights()
    
    # Calculate HMCD score
    score, contributions = hmcd_score_calculation(
        predictive_strength,
        technical_confirmation,
        investor_sentiment,
        risk_penalty,
        weights
    )
    
    # Make decision based on thresholds
    if score > 0.25:
        decision = "BUY"
        confidence = min((score - 0.25) / 0.75, 1.0)  # Normalize to [0, 1]
    elif score < -0.25:
        decision = "SELL"
        confidence = min((abs(score) - 0.25) / 0.75, 1.0)
    else:
        decision = "HOLD"
        confidence = 1.0 - (abs(score) / 0.25)  # Higher confidence near center
    
    # Generate interpretations
    signals = interpret_signals(
        predictive_strength,
        technical_confirmation,
        investor_sentiment,
        risk_penalty
    )
    
    # Build rationale
    rationale_parts = []
    
    # Identify dominant factors
    sorted_contribs = sorted(
        [(k, v) for k, v in contributions.items() if k not in ["raw_score", "final_score"]],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    if decision == "BUY":
        rationale_parts.append(f"BUY recommendation (score: {score:.3f})")
        rationale_parts.append(f"Strongest positive factor: {sorted_contribs[0][0].replace('_', ' ').title()}")
        if risk_penalty > 0.6:
            rationale_parts.append("⚠️ Note: Elevated risk, but outweighed by positive signals")
    elif decision == "SELL":
        rationale_parts.append(f"SELL recommendation (score: {score:.3f})")
        if risk_penalty > 0.5:
            rationale_parts.append("Primary driver: High risk penalty")
        else:
            rationale_parts.append(f"Primary driver: {sorted_contribs[-1][0].replace('_', ' ').title()}")
    else:
        rationale_parts.append(f"HOLD recommendation (score: {score:.3f})")
        rationale_parts.append("Signals are mixed or neutral")
    
    rationale = " | ".join(rationale_parts)
    
    return HMCDOutput(
        score=score,
        decision=decision,
        confidence=confidence,
        rationale=rationale,
        contributions=contributions,
        signals_breakdown=signals
    )
