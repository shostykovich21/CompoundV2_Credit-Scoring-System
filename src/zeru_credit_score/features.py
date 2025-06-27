# src/zeru_credit_score/features.py

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-9

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer wallet-level features from cleaned transaction DataFrame.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning empty features DataFrame.")
        return pd.DataFrame()

    logger.info("Starting feature engineering for %d records", len(df))

    g = df.groupby("wallet")
    F = pd.DataFrame(index=g.size().index)

    raw = df.groupby(["wallet","action_type"])["amountUSD"] \
            .sum().unstack(fill_value=0)
    F["raw_deposits_usd"] = raw.get("deposit", 0.0)
    F["raw_borrows_usd"]  = raw.get("borrow",  0.0)

    rep = raw.get("repay", 0.0)
    F["repayment_ratio"] = rep / (F["raw_borrows_usd"] + EPS)
    zero_mask = F["raw_borrows_usd"] < EPS
    F.loc[zero_mask, "repayment_ratio"] = 1.0
    F["repayment_ratio"] = F["repayment_ratio"].clip(upper=1.0)

    F["net_collateral_flow"] = F["raw_deposits_usd"] - raw.get("withdraw", 0.0)
    F["health_factor"]       = F["net_collateral_flow"] / (F["raw_borrows_usd"] + EPS)

    cnt = df.groupby(["wallet","action_type"])["tx_hash"] \
            .nunique().unstack(fill_value=0)
    F["liquidation_count"] = cnt.get("liquidated_event", 0) 
    F["ever_liquidated"]   = (F["liquidation_count"] > 0).astype(int)

    first_ts = g["timestamp"].min()
    last_ts  = g["timestamp"].max()
    F["account_age_days"] = (last_ts - first_ts).dt.days

    dates = (
      df.assign(day=df["timestamp"].dt.floor("d"))
        .drop_duplicates(["wallet","day"])
        .groupby("wallet")
        .size()
    )
    F["unique_active_days"] = dates.reindex(F.index, fill_value=0)
    F["tx_count"]          = g.size()
    F["asset_diversity"]   = g["asset"].nunique()
    F["usd_amount_std"]    = g["amountUSD"].std().fillna(0.0)

    diffs = df.sort_values(["wallet","timestamp"]) \
             .groupby("wallet")["timestamp"] \
             .diff().dt.total_seconds()
    F["median_tx_gap_hours"] = (
        diffs.div(3600)
             .groupby(df["wallet"])
             .median()
             .fillna(0.0)
    )
    F["tx_freq_per_day"] = F["tx_count"] / F["unique_active_days"].replace(0, 1)

    # Log-features
    log_map = {
      "log_net_collateral_flow": ("net_collateral_flow", -F["net_collateral_flow"].min()+EPS),
      "log_usd_amount_std":      ("usd_amount_std", EPS),
      "log_account_age_days":    ("account_age_days", EPS),
      "log_unique_active_days":  ("unique_active_days", EPS),
      "log_median_tx_gap_hours": ("median_tx_gap_hours", EPS),
      "log_tx_freq_per_day":     ("tx_freq_per_day", EPS),
    }
    for out, (col, shift) in log_map.items():
        F[out] = np.log1p(F[col] + shift)
        p99 = F[out].quantile(0.99)
        F[out] = F[out].clip(upper=p99)
        logger.debug("Capped %s at %.4f", out, p99)

    logger.info("Completed features: %d wallets × %d features", *F.shape)
    return F.fillna(0)