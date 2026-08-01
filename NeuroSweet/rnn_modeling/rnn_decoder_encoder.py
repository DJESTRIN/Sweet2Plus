#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_decoder_encoder.py
Description: Applies the SAME two statistical analyses used in the original report's
    decoder/encoder pipeline (report.tex Findings 1-5) to an in-silico RNN's hidden-unit
    activity, so RNN-derived population-coding statistics can be directly compared to the
    real in-vivo Findings 1-2/5.

    - DECODER: population-level statistic. Bootstrapped elastic-net logistic regression
      (same family/hyperparameters as circuit_regression.process_recording in
      NeuroSweet.statistics.circuit_coefficient_clustering) predicting each odor's onset from
      the FULL population of hidden units simultaneously, giving each hidden unit a
      population-level "relative contribution" (|coef|, bootstrapped for stability).
    - ENCODER: single-unit statistic. Per-hidden-unit logistic regression of that unit's own
      activity trace against each odor's onset (independent per-unit fits), analogous to
      Engelhard et al.-style single-neuron tuning (a lighter-weight reimplementation of
      engelhardglm.py's spline-based GLM, avoiding that module's heavier statsmodels/spline/
      permutation machinery + venn/tqdm_joblib deps not installed in the RNN-training conda
      env, but preserving the same conceptual question: "is this unit's activity, taken
      alone, significantly related to a given odor?").

    From these two fits we recompute the exact summary statistics the report's Findings 1-2
    are built from: per-unit "number of odors it's tuned/decodes for" (k in {0,1,2,3,4}), and
    decoder-vs-encoder agreement (Spearman correlation among units' decoder-importance vs
    encoder-tuning-strength rankings).

Usage: import build_decoder_encoder_summary(hidden_states, external_inputs, channel_names)
    from rnn_train.py's saved hidden_states.npy + the same session's onset channels.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _windowed_binary_labels(onset_indicator, n_windows_per_bin=1):
    """Collapse a per-timepoint binary onset indicator into per-bin (window) labels by
    taking the max within each bin -- a bin is labeled "1" for a given odor if that odor's
    onset window overlaps the bin at all."""
    return onset_indicator


def build_binned_dataset(hidden_states, per_odor_onsets, bin_size=20):
    """
    hidden_states : (n_units, T)
    per_odor_onsets : (n_odors, T) binary indicator (1 during odor-onset windows)
    Returns X (n_bins, n_units), Y (n_bins, n_odors) by averaging hidden-unit activity and
    max-pooling onset labels within non-overlapping bins of `bin_size` timepoints -- keeps the
    decoder/encoder fits computationally light while preserving the odor-onset structure.
    """
    n_units, T = hidden_states.shape
    n_odors = per_odor_onsets.shape[0]
    n_bins = T // bin_size
    X = np.zeros((n_bins, n_units), dtype=np.float32)
    Y = np.zeros((n_bins, n_odors), dtype=np.int32)
    for b in range(n_bins):
        s, e = b * bin_size, (b + 1) * bin_size
        X[b] = hidden_states[:, s:e].mean(axis=1)
        Y[b] = (per_odor_onsets[:, s:e].max(axis=1) > 0).astype(np.int32)
    return X, Y


def fit_decoder(X, Y, n_bootstraps=20, seed=0):
    """Population-level decoder: one elastic-net logistic regression per odor, using ALL
    units simultaneously, bootstrapped for coefficient stability. Returns
    decoder_importance: (n_units, n_odors) matrix of mean |bootstrapped coefficient|."""
    rng = np.random.RandomState(seed)
    n_units, n_odors = X.shape[1], Y.shape[1]
    decoder_importance = np.zeros((n_units, n_odors))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for odor_idx in range(n_odors):
        y = Y[:, odor_idx]
        if len(np.unique(y)) < 2:
            continue  # this odor never/always occurs in this session -- can't fit
        boot_coefs = []
        for _ in range(n_bootstraps):
            idx = rng.choice(np.arange(X_scaled.shape[0]), size=X_scaled.shape[0], replace=True)
            X_boot, y_boot = X_scaled[idx], y[idx]
            if len(np.unique(y_boot)) < 2:
                continue
            model = LogisticRegression(penalty="elasticnet", l1_ratio=0.5, class_weight="balanced",
                                        solver="saga", max_iter=5000)
            model.fit(X_boot, y_boot)
            boot_coefs.append(np.abs(model.coef_.flatten()))
        if boot_coefs:
            decoder_importance[:, odor_idx] = np.mean(np.stack(boot_coefs), axis=0)
    return decoder_importance


def fit_encoder(X, Y, alpha=0.05):
    """Single-unit encoder: independent logistic regression per (unit, odor) pair, using
    ONLY that one unit's own activity as the predictor (Engelhard-style single-neuron tuning,
    lightweight reimplementation). Returns encoder_tuning: (n_units, n_odors) matrix of
    |coefficient| (0 if the unit's fit did not reach significance via a Wald-style p-value
    approximation, or if the odor is degenerate for this session)."""
    n_units, n_odors = X.shape[1], Y.shape[1]
    encoder_tuning = np.zeros((n_units, n_odors))
    for unit_idx in range(n_units):
        x = X[:, unit_idx].reshape(-1, 1)
        x = (x - x.mean()) / (x.std() + 1e-8)
        for odor_idx in range(n_odors):
            y = Y[:, odor_idx]
            if len(np.unique(y)) < 2:
                continue
            model = LogisticRegression(class_weight="balanced", max_iter=2000)
            model.fit(x, y)
            # crude significance proxy: is |coef| distinguishable from a null obtained by
            # refitting on 20 label permutations? (much cheaper than engelhardglm's 500-perm
            # circular-lag test, but same idea: compare real fit strength to a shuffled null)
            real_coef = np.abs(model.coef_.flatten()[0])
            null_coefs = []
            rng = np.random.RandomState(odor_idx * 1000 + unit_idx)
            for _ in range(20):
                y_perm = rng.permutation(y)
                if len(np.unique(y_perm)) < 2:
                    continue
                m = LogisticRegression(class_weight="balanced", max_iter=2000)
                m.fit(x, y_perm)
                null_coefs.append(np.abs(m.coef_.flatten()[0]))
            if null_coefs:
                p_value = (np.sum(np.array(null_coefs) >= real_coef) + 1) / (len(null_coefs) + 1)
                if p_value < alpha:
                    encoder_tuning[unit_idx, odor_idx] = real_coef
    return encoder_tuning


def summarize(decoder_importance, encoder_tuning, decoder_thresh_pct=50):
    """Recomputes the report's Finding-1/2-style summary statistics:
      - k_decoder[unit] = number of odors for which this unit is in the top
        `decoder_thresh_pct` percent of decoder importance for that odor.
      - k_encoder[unit] = number of odors for which this unit has nonzero (significant)
        encoder tuning.
      - agreement = Spearman correlation between each unit's total decoder importance
        (summed across odors) and total encoder tuning strength (summed across odors) --
        mirrors the report's weak-agreement finding (|r| <= 0.065 in real data).
    """
    n_units, n_odors = decoder_importance.shape
    decoder_binary = np.zeros_like(decoder_importance, dtype=int)
    for odor_idx in range(n_odors):
        col = decoder_importance[:, odor_idx]
        if col.sum() == 0:
            continue
        thresh = np.percentile(col[col > 0], 100 - decoder_thresh_pct) if np.any(col > 0) else np.inf
        decoder_binary[:, odor_idx] = (col >= thresh) & (col > 0)

    k_decoder = decoder_binary.sum(axis=1)
    k_encoder = (encoder_tuning > 0).sum(axis=1)

    decoder_total = decoder_importance.sum(axis=1)
    encoder_total = encoder_tuning.sum(axis=1)
    if np.std(decoder_total) > 0 and np.std(encoder_total) > 0:
        rho, pval = spearmanr(decoder_total, encoder_total)
    else:
        rho, pval = np.nan, np.nan

    return {
        "n_units": int(n_units),
        "k_decoder_counts": pd.Series(k_decoder).value_counts().sort_index().to_dict(),
        "k_encoder_counts": pd.Series(k_encoder).value_counts().sort_index().to_dict(),
        "mean_k_decoder": float(np.mean(k_decoder)),
        "mean_k_encoder": float(np.mean(k_encoder)),
        "decoder_encoder_agreement_spearman_r": float(rho) if not np.isnan(rho) else None,
        "decoder_encoder_agreement_pvalue": float(pval) if not np.isnan(pval) else None,
    }


def build_decoder_encoder_summary(hidden_states, per_odor_onsets, bin_size=20, n_bootstraps=20, seed=0):
    """Top-level convenience function: hidden_states (n_units, T), per_odor_onsets (n_odors, T)
    -> summary dict (see summarize())."""
    X, Y = build_binned_dataset(hidden_states, per_odor_onsets, bin_size=bin_size)
    decoder_importance = fit_decoder(X, Y, n_bootstraps=n_bootstraps, seed=seed)
    encoder_tuning = fit_encoder(X, Y)
    summary = summarize(decoder_importance, encoder_tuning)
    return summary, decoder_importance, encoder_tuning


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden_states_npy", type=str, required=True)
    parser.add_argument("--history_json", type=str, required=True,
                         help="The _history.json saved alongside hidden_states.npy by rnn_train.py "
                              "(used to recover channel_names/meta, NOT the per-odor onsets -- those "
                              "must be rebuilt from the original session_json).")
    parser.add_argument("--session_json", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="unmixed",
                         choices=["unmixed", "semi-mixed", "fully-mixed"])
    parser.add_argument("--out_json", type=str, required=True)
    return parser.parse_args()


def main():
    args = cli_parser()
    from NeuroSweet.rnn_modeling.rnn_data import build_session_dataset, TRIAL_LIST

    hidden_states = np.load(args.hidden_states_npy)
    # Rebuild the per-ODOR (not per-channel) onset indicators directly from the raw session,
    # since the decoder/encoder comparison should always be phrased in terms of the 4 real
    # odors regardless of which input architecture (unmixed/semi-mixed/fully-mixed) was used
    # to drive the RNN.
    _, _, meta = build_session_dataset(args.session_json, architecture="unmixed")
    from NeuroSweet.rnn_modeling.rnn_data import load_one_session, _onset_indicator
    ztraces, all_evts_imagetime, sess_meta = load_one_session(args.session_json)
    n_timepoints = ztraces.shape[1]
    per_odor_onsets = np.stack([
        _onset_indicator(all_evts_imagetime[i], n_timepoints) for i in range(len(TRIAL_LIST))
    ], axis=0)

    summary, decoder_importance, encoder_tuning = build_decoder_encoder_summary(
        hidden_states, per_odor_onsets)

    with open(args.history_json, "r") as f:
        history = json.load(f)

    out = {
        "session_meta": history.get("meta", sess_meta),
        "architecture": args.architecture,
        "summary": summary,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved -> {args.out_json}")


if __name__ == "__main__":
    main()
