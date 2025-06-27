# src/zeru_credit_score/loader.py

import os
import json
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_transactions(
    data_dir: str,
    ts_unit: Optional[str] = None
) -> pd.DataFrame:
    """
    Load and clean Compound V2 transaction data from JSON files.
    """
    dfs = []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(data_dir, fn)
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except Exception as e:
            logger.error(f"Skipping {path}: JSON parse error ({e})")
            continue

        for action, recs in raw.items():
            if not isinstance(recs, list) or not recs:
                logger.debug(f"No records for action '{action}' in {fn}")
                continue

            df = pd.DataFrame(recs)
            df["action_type"] = action.rstrip("s")

            if 'asset' in df.columns:
                df['asset'] = df['asset'].apply(lambda x: x.get('symbol') if isinstance(x, dict) else x)

            df.rename(columns={
                "hash": "tx_hash",
                "transactionHash": "tx_hash",
                "blockTimestamp": "timestamp"
            }, inplace=True, errors="ignore")

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors='coerce')

            if action == "liquidates":
                liq_df = df.copy()
                if 'liquidator' in liq_df.columns:
                    liq_df['wallet'] = liq_df['liquidator'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
                    liq_df["action_type"], liq_df["relation"] = "liquidator_action", "none"
                else:
                    liq_df = pd.DataFrame()

                ld_df = df.copy()
                if 'user' in ld_df.columns:
                    ld_df['wallet'] = ld_df['user'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
                    ld_df["action_type"], ld_df["relation"] = "liquidated_event", "none"
                else:
                    ld_df = pd.DataFrame()
                df = pd.concat([liq_df, ld_df], ignore_index=True)
            else:
                if 'account' in df.columns:
                    df['wallet'] = df['account'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
                if action == "repays":
                     df["relation"] = np.where(df.get("payer") == df["wallet"], "self_pay", "third_party_pay")
                else:
                     df["relation"] = "none"

            if 'asset' not in df.columns:
                logger.error(f"{fn}/{action}: missing 'asset' field")
                continue

            ts = df["timestamp"]
            if pd.api.types.is_numeric_dtype(ts):
                unit = "ms" if ts.max() > 1e12 else "s"
                df["timestamp"] = pd.to_datetime(ts, unit=unit, errors="coerce")

            dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid transaction records could be processed from {data_dir}")

    full = pd.concat(dfs, ignore_index=True)

    cols = ["wallet","tx_hash","timestamp","action_type","amountUSD","asset","relation"]
    full = full[[c for c in cols if c in full.columns]]

    full["amountUSD"] = pd.to_numeric(full["amountUSD"], errors="coerce")
    full.dropna(subset=["wallet","tx_hash","timestamp","amountUSD","asset"], inplace=True)

    full.sort_values("timestamp", inplace=True)
    before = len(full)
    full.drop_duplicates(subset=["tx_hash"], keep="first", inplace=True)
    logger.info("Dropped %d duplicate transactions", before - len(full))
    if not full.empty:
      full["wallet"] = full["wallet"].str.lower()
    return full.sort_values("timestamp").reset_index(drop=True)