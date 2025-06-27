# src/zeru_credit_score/main.py

import os
import argparse
import logging
from typing import Dict
from zeru_credit_score.logging_config import setup_logging
from zeru_credit_score.loader import load_transactions
from zeru_credit_score.features import engineer_features
from zeru_credit_score.scoring import calculate_scores

logger = logging.getLogger(__name__)

def parse_weights(spec: str) -> Dict[str, float]:
    d = {}
    if not isinstance(spec, str):
        raise argparse.ArgumentTypeError("Weight spec must be a string.")
    for kv in spec.split(","):
        if "=" not in kv:
            raise argparse.ArgumentTypeError(f"Bad weight '{kv}' (expected key=val)")
        k, v = kv.split("=", 1)
        try:
            d[k.strip()] = float(v.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f"Bad weight value '{v}' for '{k}'")
    return d

def main():
    setup_logging()

    p = argparse.ArgumentParser(description="Zeru Finance Credit Scoring")
    p.add_argument("--data-dir",   default="data",  help="JSON input directory")
    p.add_argument("--output-dir", default="output",help="CSV & scaler output directory")
    p.add_argument(
        "--weights",
        type=parse_weights,
        default="health=0.45,trust=0.35,risk=0.20",
        help="Comma-separated health=…,trust=…,risk=… (sum=1)"
    )
    p.add_argument("--topk",  type=int, default=1000, help="Number of top wallets to export")
    p.add_argument(
        "--ts-unit",
        choices=["s","ms","auto"],
        default="auto",
        help="Timestamp unit: s, ms, or auto-detect"
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print feature summary (mean,std,skew) and exit"
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scaler_path = os.path.join(args.output_dir, "scaler.joblib")

    logger.info("Loading transactions…")
    tsu = None if args.ts_unit=="auto" else args.ts_unit
    df = load_transactions(args.data_dir, ts_unit=tsu)

    logger.info("Engineering features…")
    F = engineer_features(df)

    if F.empty:
        logger.warning("Feature engineering resulted in an empty DataFrame. No scores to calculate. Exiting.")
        return

    if args.profile:
        desc = F.describe().T
        desc["skew"] = F.skew()
        print(desc[["mean","std","skew"]].round(4))
        return

    logger.info("Calculating scores…")
    weights = args.weights
    C = calculate_scores(F, weights, scaler_path=scaler_path)

    out_csv = os.path.join(args.output_dir, f"top_{args.topk}_wallets.csv")
    if os.path.exists(out_csv):
        logger.warning("%s exists; overwriting", out_csv)
    C.head(args.topk)[["score"]].to_csv(out_csv, index_label="wallet")
    logger.info("Wrote top %d to %s", args.topk, out_csv)

if __name__ == "__main__":
    main()