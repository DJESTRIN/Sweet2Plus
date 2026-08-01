#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_odor_sweep.py
Description: Step 4 of plan.md -- using the WINNING external-input architecture (identified by
    rnn_aggregate_and_score.py), systematically up/down-weight each individual odor input
    channel and pairwise combination on an already-trained session RNN, re-running the
    decoder/encoder pipeline after each re-weighting to map which specific input pathway(s)
    most strongly drive the control-vs-cort divergence in RNN hidden-unit statistics.

    Uses MPFCModelRNN.set_input_channel_weight() (see rnn_model.py) to multiplicatively
    rescale one already-trained model's input weights post-hoc (no retraining of the
    recurrent core -- only the external input strength for a given channel is perturbed),
    then re-extracts hidden states and re-runs the decoder/encoder summary. This directly
    answers "does up/down-weighting the TMT (or any other odor / combination) input pathway
    change how similar/dissimilar the model's control-fit vs cort-fit signature becomes to the
    real divergence" -- covering ALL 4 odors and their pairwise combinations, not only TMT.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from NeuroSweet.rnn_modeling.rnn_data import build_session_dataset, load_one_session, \
    _onset_indicator, TRIAL_LIST
from NeuroSweet.rnn_modeling.rnn_model import MPFCModelRNN
from NeuroSweet.rnn_modeling.rnn_train import get_hidden_states
from NeuroSweet.rnn_modeling.rnn_decoder_encoder import build_decoder_encoder_summary


def load_trained_model(model_path, n_input_channels, hidden_size, n_neurons):
    model = MPFCModelRNN(n_input_channels=n_input_channels, hidden_size=hidden_size, n_neurons=n_neurons)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def sweep_channels(model_path, session_json, architecture, channel_names, hidden_size, n_neurons,
                    scales=(0.25, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """For each individual channel AND each pairwise combination of channels in
    `channel_names`, reload a fresh copy of the trained model, rescale that channel's (or
    channel-pair's, applied jointly) input weight by each factor in `scales`, and recompute the
    decoder/encoder summary. scale=1.0 is the unperturbed baseline for reference.
    Returns a long-format DataFrame: one row per (channel_or_pair, scale, summary metric)."""
    n_input_channels = len(channel_names)

    from NeuroSweet.rnn_modeling.rnn_data import build_input_channels
    ztraces, all_evts_imagetime, meta = load_one_session(session_json)
    channels, _names = build_input_channels(all_evts_imagetime, ztraces.shape[1], architecture=architecture)
    n_timepoints = channels.shape[1]
    per_odor_onsets = np.stack([
        _onset_indicator(all_evts_imagetime[i], n_timepoints) for i in range(len(TRIAL_LIST))
    ], axis=0)

    # Targets to sweep: every individual channel, plus every pairwise combination of channels
    # (jointly rescaled together) -- covers "all 4 odors and their combinations," not just TMT.
    singles = [(i,) for i in range(n_input_channels)]
    pairs = list(combinations(range(n_input_channels), 2))
    targets = singles + pairs

    rows = []
    for target in targets:
        target_label = "+".join(channel_names[i] for i in target)
        for scale in scales:
            model = load_trained_model(model_path, n_input_channels, hidden_size, n_neurons)
            for ch_idx in target:
                model.set_input_channel_weight(ch_idx, scale)

            hidden_states = get_hidden_states(model, channels)
            summary, _, encoder_tuning = build_decoder_encoder_summary(hidden_states, per_odor_onsets)

            k_decoder = np.array(list(summary["k_decoder_counts"].values()))
            k_decoder_keys = np.array([int(k) for k in summary["k_decoder_counts"].keys()])
            n_units = summary["n_units"]
            p_k_decoder_ge2 = float(np.sum(k_decoder[k_decoder_keys >= 2]) / n_units) if n_units else 0.0

            rows.append({
                "swept_channel(s)": target_label,
                "scale": scale,
                "p_k_decoder_ge2": p_k_decoder_ge2,
                "mean_k_decoder": summary["mean_k_decoder"],
                "mean_k_encoder": summary["mean_k_encoder"],
                "agreement_rho": summary["decoder_encoder_agreement_spearman_r"],
            })
    return pd.DataFrame(rows)


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True,
                         help="Path to a _model.pt saved by rnn_train.py/rnn_batch_run.py, "
                              "trained with the winning architecture.")
    parser.add_argument("--session_json", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True,
                         choices=["unmixed", "semi-mixed", "fully-mixed"],
                         help="Must match the architecture the model at --model_path was trained with.")
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--n_neurons", type=int, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


def main():
    args = cli_parser()
    from NeuroSweet.rnn_modeling.rnn_data import build_input_channels
    ztraces, all_evts_imagetime, meta = load_one_session(args.session_json)
    _, channel_names = build_input_channels(all_evts_imagetime, ztraces.shape[1], architecture=args.architecture)

    df = sweep_channels(args.model_path, args.session_json, args.architecture, channel_names,
                         args.hidden_size, args.n_neurons)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"Saved -> {args.out_csv}")


if __name__ == "__main__":
    main()
