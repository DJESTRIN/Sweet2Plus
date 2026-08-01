#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_stats.py
Description: Final step of plan.md (step 5) -- feeds session-level RNN decoder/encoder
    summary metrics (one row per session x seed, as produced by rnn_batch_run.py /
    rnn_aggregate_and_score.py) into a group x day mixed-model analysis, following the SAME
    Day-0-normalization convention used everywhere else in this repo
    (NeuroSweet.statistics.state_dynamics.StateStatistics.normalize_to_day0) -- each session
    RNN metric is z-scored relative to that SAME SUBJECT's own day-0 sessions before testing
    group x day effects, since this dataset's cohort/batch structure otherwise confounds Group.

    Each (subject, day, seed) RNN fit is collapsed to one (subject, day) observation by
    averaging across seeds first (the seed-averaging step required by plan.md's "3-5 random
    seed refits per session, report mean +/- spread" design, so RNN-training stochasticity is
    not counted as if it were extra independent data).

    Uses a linear mixed model (dependent_var ~ group * day, random intercept per subject) via
    statsmodels, mirroring the family of models used in state_dynamics.run_group_day_stats,
    but simplified since RNN summary metrics are one scalar per (subject, day) rather than one
    per (subject, day, odor_pair) -- no odor_pair random/fixed effect term is needed here.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def average_across_seeds(df, metric_cols):
    """Collapse (subject, day, seed) rows to one (subject, day) row per metric by averaging
    across seeds -- required before any group x day test, since seeds are repeats of the same
    underlying (subject, day) RNN fit, not independent biological observations."""
    group_cols = ["group", "day", "mouse", "session_id"]
    return df.groupby(group_cols, as_index=False)[metric_cols].mean()


def normalize_to_day0(df, metric, mouse_col="mouse"):
    """Per-subject Day-0 z-score normalization -- same logic/rationale as
    StateStatistics.normalize_to_day0: removes cohort/batch confounds and per-subject scale
    differences by anchoring each subject's own day-0 value(s) at z~0."""
    day0_all = df.loc[df["day"].astype(str) == "0", metric]
    global_day0_sd = day0_all.std(ddof=1) if len(day0_all) > 1 else np.nan

    out_col = f"{metric}_day0z"
    z_values = []
    for mouse_id, sub in df.groupby(mouse_col):
        day0_vals = sub.loc[sub["day"].astype(str) == "0", metric]
        if len(day0_vals) >= 1 and np.isfinite(global_day0_sd) and global_day0_sd > 0:
            mu = day0_vals.mean()
            z = (sub[metric] - mu) / global_day0_sd
        else:
            z = pd.Series(np.nan, index=sub.index)
        z_values.append(z)
    df[out_col] = pd.concat(z_values).reindex(df.index)
    return df, out_col


def run_group_day_stats(df, metric, drop_directory="."):
    """Fits dependent_var ~ group * day (random intercept per subject via MixedLM), after
    Day-0 normalizing `metric` and excluding day 0 (the normalization anchor, not an
    independent observation) -- mirrors state_dynamics.run_group_day_stats's convention.
    Returns the fitted MixedLMResults object, or None if too few groups/days to fit."""
    os.makedirs(drop_directory, exist_ok=True)
    df = df.copy()
    df, normalized_col = normalize_to_day0(df, metric)
    df = df.dropna(subset=[normalized_col])
    df = df[df["day"].astype(str) != "0"]

    if df["group"].nunique() < 2 or df["day"].nunique() < 2 or len(df) < 6:
        print(f"WARNING: insufficient data to fit group x day model for {metric} "
              f"(n={len(df)}, groups={df['group'].unique()}, days={df['day'].unique()}) -- skipping.")
        return None

    df["group"] = df["group"].astype("category")
    df["day"] = df["day"].astype(str).astype("category")

    model = smf.mixedlm(f"{normalized_col} ~ group * day", data=df, groups=df["mouse"])
    result = model.fit()

    summary_path = os.path.join(drop_directory, f"rnn_{metric}_mixedmodel_summary.csv")
    result.summary().tables[1].to_csv(summary_path)

    emmeans = df.groupby(["group", "day"], observed=True)[normalized_col].agg(["mean", "sem", "count"]).reset_index()
    emmeans_path = os.path.join(drop_directory, f"rnn_{metric}_group_day_emmeans.csv")
    emmeans.to_csv(emmeans_path, index=False)

    print(f"{metric}: fit ok -> {summary_path}, {emmeans_path}")
    return result


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_csv", type=str, required=True,
                         help="A rnn_batch_results_<architecture>_ALL.csv from rnn_aggregate_and_score.py")
    parser.add_argument("--drop_directory", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs="+",
                         default=["p_k_decoder_ge2", "tuned_noise_corr", "agreement_rho",
                                  "mean_k_decoder", "mean_k_encoder"])
    return parser.parse_args()


def main():
    args = cli_parser()
    df = pd.read_csv(args.results_csv)
    df_avg = average_across_seeds(df, args.metrics)
    for metric in args.metrics:
        run_group_day_stats(df_avg, metric, drop_directory=args.drop_directory)


if __name__ == "__main__":
    main()
