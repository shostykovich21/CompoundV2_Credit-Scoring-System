#!/usr/bin/env python

import os
import argparse
import logging

from zeru_credit_score.logging_config import setup_logging
from zeru_credit_score.loader         import load_transactions
from zeru_credit_score.features       import engineer_features
from zeru_credit_score.scoring        import calculate_scores

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run weight‐sensitivity analysis (Jaccard) on the Zeru scoring model"
    )
    parser.add_argument("--data-dir",   default="data",
                        help="JSON input directory")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for scaler & outputs")
    parser.add_argument("--ts-unit", choices=["s","ms","auto"], default="auto",
                        help="Timestamp unit: s, ms, or auto")
    parser.add_argument("--topk", type=int, default=1000,
                        help="Top-K wallets to include in Jaccard sets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tsu = None if args.ts_unit=="auto" else args.ts_unit

    logger.info("Loading transactions from %s (ts_unit=%s)…", args.data_dir, tsu or "auto")
    df = load_transactions(args.data_dir, ts_unit=tsu)

    logger.info("Engineering features…")
    F = engineer_features(df)

    scaler_path = os.path.join(args.output_dir, "sensitivity_scaler.joblib")
    weight_sets = [
        ({"health":0.45,"trust":0.35,"risk":0.20}, "base"),
        ({"health":0.60,"trust":0.20,"risk":0.20}, "alt1"),
        ({"health":0.40,"trust":0.40,"risk":0.20}, "alt2"),
    ]

    C_base = calculate_scores(F, weight_sets[0][0], scaler_path=scaler_path)
    tops = {"base": set(C_base.head(args.topk).index)}

    for weights, name in weight_sets[1:]:
        logger.info("Scoring with weights %s (%s)…", weights, name)
        C_alt = calculate_scores(F, weights, scaler_path=scaler_path)
        tops[name] = set(C_alt.head(args.topk).index)

    j1 = jaccard(tops["base"], tops["alt1"])
    j2 = jaccard(tops["base"], tops["alt2"])
    print(f"Jaccard(base, alt1) = {j1:.3f}")
    print(f"Jaccard(base, alt2) = {j2:.3f}")
    if j1 > 0.8 and j2 > 0.8:
        print("Stable leaderboard under weight variations.")
    else:
        print("Significant sensitivity detected.")
    
if __name__ == "__main__":
    main()
