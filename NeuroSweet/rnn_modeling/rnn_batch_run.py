#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_batch_run.py
Description: Orchestrates the full in-silico RNN pipeline across many sessions: for each
    session (subject x day) and a given input architecture, trains N seed-refits, extracts
    hidden-unit decoder/encoder summary stats (rnn_decoder_encoder.py), and writes one row per
    (session, seed) to a results CSV. This is the "many independent RNN fits" step referenced
    in plan.md step 1/5 -- each session x seed is one independent observation, ready to feed
    into rnn_architecture_scoring.py (per-architecture comparison) and eventually
    NeuroSweet.statistics.state_dynamics.run_group_day_stats (final group x day stats).

    Designed to be run either:
      (a) directly (a simple for-loop over a session manifest) for local/small-scale runs, or
      (b) as one SLURM array task per (session, architecture) combination -- see
          NeuroSweet/cluster_scripts/rnn_array.sh, which calls this script with
          --manifest_row_index selecting a single row to process per array task, following the
          same one-task-per-chunk pattern as glm_encoder_array.sh.

Caps hidden_size at --max_hidden_size (default 128) for sessions with very large real neuron
counts (some sessions in this dataset have 500-1800+ neurons) purely for tractable runtime;
this is a compute-budget cap, not a modeling assumption, and should be revisited if compute
resources allow larger runs on SLURM.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from NeuroSweet.rnn_modeling.rnn_data import list_sessions, build_session_dataset, load_one_session, \
    _onset_indicator, TRIAL_LIST
from NeuroSweet.rnn_modeling.rnn_train import train_one_session, get_hidden_states
from NeuroSweet.rnn_modeling.rnn_decoder_encoder import build_decoder_encoder_summary


def normalize_group_label(raw_group):
    """Session json 'group' field values vary in casing/spelling across the dataset (e.g.
    'cort' vs 'Cort'); normalize to lowercase 'control'/'cort' for consistent grouping."""
    g = str(raw_group).strip().lower()
    if "cort" in g and "control" not in g:
        return "cort"
    if "control" in g or "ctrl" in g or "veh" in g:
        return "control"
    return g


def run_one(session_row, architecture, seed, epochs, max_hidden_size, window_len, stride,
            drop_directory):
    path = session_row["path"]
    channels, target, meta = build_session_dataset(path, architecture=architecture, seed=seed)
    hidden_size = min(meta["n_neurons"], max_hidden_size)
    label = f"{normalize_group_label(meta['group'])}_{meta['mouse']}_day{meta['day']}_{architecture}_seed{seed}"

    t0 = time.time()
    model, history = train_one_session(
        channels, target, hidden_size=hidden_size, epochs=epochs, seed=seed,
        window_len=window_len, stride=stride, progress_label=label)
    hidden_states = get_hidden_states(model, channels)

    _, all_evts_imagetime, _ = load_one_session(path)
    n_timepoints = hidden_states.shape[1]
    per_odor_onsets = np.stack([
        _onset_indicator(all_evts_imagetime[i], n_timepoints) for i in range(len(TRIAL_LIST))
    ], axis=0)

    summary, decoder_importance, encoder_tuning = build_decoder_encoder_summary(
        hidden_states, per_odor_onsets, seed=seed)

    k_decoder = np.array(list(summary["k_decoder_counts"].values()))
    k_decoder_keys = np.array([int(k) for k in summary["k_decoder_counts"].keys()])
    n_units = summary["n_units"]
    p_k_decoder_ge2 = float(np.sum(k_decoder[k_decoder_keys >= 2]) / n_units) if n_units else 0.0

    tuned_mask = (encoder_tuning > 0).any(axis=1)
    if tuned_mask.sum() >= 2:
        tuned_hidden = hidden_states[tuned_mask]
        corr = np.corrcoef(tuned_hidden)
        iu = np.triu_indices_from(corr, k=1)
        tuned_noise_corr = float(np.nanmean(corr[iu]))
    else:
        tuned_noise_corr = np.nan

    row = {
        "session_id": f"{meta['group']}_{meta['cage']}_{meta['mouse']}_day{meta['day']}",
        "group": normalize_group_label(meta["group"]),
        "day": meta["day"],
        "mouse": meta["mouse"],
        "architecture": architecture,
        "seed": seed,
        "hidden_size": hidden_size,
        "n_real_neurons": meta["n_neurons"],
        "train_time_s": time.time() - t0,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "p_k_decoder_ge2": p_k_decoder_ge2,
        "tuned_noise_corr": tuned_noise_corr,
        "agreement_rho": summary["decoder_encoder_agreement_spearman_r"],
        "mean_k_decoder": summary["mean_k_decoder"],
        "mean_k_encoder": summary["mean_k_encoder"],
    }

    if drop_directory:
        os.makedirs(drop_directory, exist_ok=True)
        with open(os.path.join(drop_directory, label + "_summary.json"), "w") as f:
            json.dump(row, f, indent=2)

    return row


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--drop_directory", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="unmixed",
                         choices=["unmixed", "semi-mixed", "fully-mixed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--max_hidden_size", type=int, default=128)
    # window_len/stride tuned during the training-accuracy audit: shorter, denser windows
    # (100/20 vs the original 200/100) give ~5x more training windows per session, which
    # matters a lot given mini-batch training now takes many steps per epoch. epochs raised
    # to 400 with early stopping (see rnn_train.train_one_session) since actual epoch count
    # used will typically be far lower -- early stopping halts once val loss stops improving.
    parser.add_argument("--window_len", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--manifest_row_index", type=int, default=None,
                         help="If set, only process this one row of the session manifest "
                              "(used for SLURM array tasks -- one array task per session).")
    parser.add_argument("--max_sessions", type=int, default=None,
                         help="If set, only process the first N sessions (for quick local "
                              "smoke tests, not full runs).")
    parser.add_argument("--results_csv", type=str, default=None,
                         help="Path to append/write results row(s). Default: "
                              "<drop_directory>/rnn_batch_results_<architecture>.csv")
    return parser.parse_args()


def main():
    args = cli_parser()
    os.makedirs(args.drop_directory, exist_ok=True)
    results_csv = args.results_csv or os.path.join(
        args.drop_directory, f"rnn_batch_results_{args.architecture}.csv")

    manifest = list_sessions(args.data_directory)
    if args.max_sessions is not None:
        manifest = manifest.iloc[:args.max_sessions]

    if args.manifest_row_index is not None:
        rows_to_run = [manifest.iloc[args.manifest_row_index]]
    else:
        rows_to_run = [manifest.iloc[i] for i in range(len(manifest))]

    all_rows = []
    for session_row in rows_to_run:
        for seed in args.seeds:
            print(f"=== session={session_row['session_id']} architecture={args.architecture} seed={seed} ===")
            try:
                row = run_one(session_row, args.architecture, seed, args.epochs,
                               args.max_hidden_size, args.window_len, args.stride,
                               args.drop_directory)
                all_rows.append(row)
            except Exception as e:
                print(f"FAILED session={session_row['session_id']} seed={seed}: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        if os.path.isfile(results_csv):
            existing = pd.read_csv(results_csv)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(results_csv, index=False)
        print(f"Wrote {len(all_rows)} new row(s) -> {results_csv} ({len(df)} total rows)")
    else:
        print("No rows produced (all runs failed or empty manifest selection).")


if __name__ == "__main__":
    main()
