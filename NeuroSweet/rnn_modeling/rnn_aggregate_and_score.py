#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_aggregate_and_score.py
Description: Step 3 of the SLURM pipeline (after rnn_array.sh's per-session, per-architecture
    array tasks complete for all 3 architectures): concatenates each architecture's
    per-task results CSVs (rnn_batch_results_<architecture>_task*.csv) into one CSV per
    architecture, scores each architecture against the real-data reference signature via
    rnn_architecture_scoring.py, and reports the winning architecture (plan.md step 3).
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import glob
import json
import argparse

import pandas as pd

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from NeuroSweet.rnn_modeling.rnn_architecture_scoring import score_architecture, select_winning_architecture


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop_directory", type=str, required=True)
    parser.add_argument("--architectures", type=str, nargs="+",
                         default=["unmixed", "semi-mixed", "fully-mixed"])
    return parser.parse_args()


def main():
    args = cli_parser()
    architecture_scores = {}
    combined_frames = {}

    for architecture in args.architectures:
        pattern = os.path.join(args.drop_directory, f"rnn_batch_results_{architecture}_task*.csv")
        task_csvs = sorted(glob.glob(pattern))
        if not task_csvs:
            print(f"WARNING: no per-task CSVs found for architecture={architecture} "
                  f"(pattern={pattern}) -- skipping.")
            continue
        frames = [pd.read_csv(p) for p in task_csvs]
        combined = pd.concat(frames, ignore_index=True)
        combined_path = os.path.join(args.drop_directory, f"rnn_batch_results_{architecture}_ALL.csv")
        combined.to_csv(combined_path, index=False)
        combined_frames[architecture] = combined
        print(f"architecture={architecture}: {len(task_csvs)} task CSVs -> {len(combined)} rows "
              f"-> {combined_path}")

        session_summaries = combined.to_dict("records")
        architecture_scores[architecture] = score_architecture(session_summaries)

    if not architecture_scores:
        print("ERROR: no architectures had any results to score.")
        sys.exit(1)

    winner, ranked = select_winning_architecture(architecture_scores)
    print("\n=== Architecture comparison (lower match_score = better real-data match) ===")
    for name, score in ranked:
        print(f"  {name}: match_score={score['match_score']:.5f} components={score['components']}")
    print(f"\nWinning architecture: {winner}")

    out = {
        "winner": winner,
        "ranked": [{"architecture": name, **score} for name, score in ranked],
    }
    out_path = os.path.join(args.drop_directory, "rnn_architecture_comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
