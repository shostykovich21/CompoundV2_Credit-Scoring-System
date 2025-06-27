import os, joblib, logging
from typing import Dict, Optional
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)

MODEL_FEATURES = {
    "health": ["health_factor","repayment_ratio","log_net_collateral_flow"],
    "trust":  ["log_account_age_days","log_unique_active_days","log_median_tx_gap_hours"],
    "risk":   ["liquidation_count","log_tx_freq_per_day","log_usd_amount_std","asset_diversity"],
}
NEG_FEATURES = {"liquidation_count","log_tx_freq_per_day","log_usd_amount_std"}

def calculate_scores(
    F: pd.DataFrame,
    weights: Dict[str,float],
    scaler_path: Optional[str]=None
) -> pd.DataFrame:
    """
    Returns DataFrame with added columns:
    health_score, trust_score, risk_score, raw_score, score (0–100).
    """
    if set(weights) != set(MODEL_FEATURES) or abs(sum(weights.values())-1)>1e-6:
        raise ValueError("Weights must be exactly health, trust, risk and sum to 1")

    F = F.copy()
    feats = [f for comp in MODEL_FEATURES.values() for f in comp if f in F]
    if not feats:
        raise ValueError("No features found for scoring")

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(F[feats])
        logger.info("Loaded scaler from %s", scaler_path)
    else:
        n_q = max(2, min(100, len(F)//10))
        scaler = QuantileTransformer(output_distribution="uniform",
                                     n_quantiles=n_q, random_state=42)
        X = scaler.fit_transform(F[feats])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            logger.info("Saved scaler (%d quantiles) to %s", n_q, scaler_path)

    S = pd.DataFrame(X, index=F.index, columns=[f"{c}_scaled" for c in feats])
    for neg in NEG_FEATURES:
        col = f"{neg}_scaled"
        if col in S:
            S[col] = 1 - S[col]

    for comp, flist in MODEL_FEATURES.items():
        cols = [f"{f}_scaled" for f in flist if f in feats]
        F[f"{comp}_score"] = S[cols].mean(axis=1) if cols else 0.0

    # Raw & overrides
    F["raw_score"] = (
        F["health_score"]*weights["health"]
      + F["trust_score"] *weights["trust"]
      + F["risk_score"]  *weights["risk"]
    )
    n_liq = (F["ever_liquidated"]==1).sum()
    n_low = (F["repayment_ratio"]<0.8).sum()
    logger.info("Applying overrides: %d liquidations, %d low-repay", n_liq, n_low)
    F.loc[F["ever_liquidated"]==1, "raw_score"] *= 0.2
    F.loc[F["repayment_ratio"]<0.8, "raw_score"] *= 0.5

    F["score"] = (F["raw_score"].clip(0,1) * 100).round().astype(int)
    return F.sort_values("score", ascending=False)
