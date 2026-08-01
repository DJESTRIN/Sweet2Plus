#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_data.py
Description: Per-session data preparation for the in-silico RNN model of mPFC (see
    session plan.md, "In-Silico RNN Model of mPFC"). Loads ONE session (subject x day) at a
    time -- deliberately avoids any cross-day neuron matching, since the RNN unit of modeling
    is "one RNN per session," not "one RNN per subject." Builds:
      - target tensor: the session's own real neuron calcium traces (neurons x time), the
        supervised training target (next-timestep prediction).
      - external input tensor: odor onset/identity regressors, built according to one of three
        architectures (unmixed / semi-mixed / fully-mixed), per user-specified design.

Reuses NeuroSweet.statistics.glms.run_full_comparison.lightweight_gather_data's obj*.json
schema reader so this module has zero dependency on suite2p/torch/optuna at data-loading time
(only needed once we actually build/train the torch RNN, in rnn_train.py).
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Canonical odor/trial order used throughout the repo's decoder/encoder pipeline.
TRIAL_LIST = ["Vanilla", "PeanutButter", "Water", "FoxUrine"]

# Onset->offset window (in frames) during which an odor's input channel is considered "on".
# Matches the tsoffset_frames convention used in engelhardglm.currate_data (10 frames).
ONSET_WINDOW_FRAMES = 10


def list_sessions(parent_data_directory, file_indicator="obj"):
    """Enumerate every obj*.json file under parent_data_directory (one per session) and
    return a manifest DataFrame with subject/day metadata parsed straight from the json,
    WITHOUT loading each file's (large, ~100MB) neuronal activity array yet."""
    objfiles = sorted(glob.glob(os.path.join(parent_data_directory, f"**/{file_indicator}*.json"), recursive=True))
    rows = []
    for path in objfiles:
        with open(path, "r") as f:
            big_list = json.load(f)
        day, cage, mouse, group = big_list[19], big_list[20], big_list[21], big_list[22]
        n_neurons = int(np.asarray(big_list[9]).shape[0])
        rows.append({
            "path": path,
            "day": day,
            "cage": cage,
            "mouse": mouse,
            "group": group,
            "n_neurons": n_neurons,
            "session_id": f"{group}_{cage}_{mouse}_day{day}",
        })
    manifest = pd.DataFrame(rows)
    return manifest


def load_one_session(path):
    """Load a single obj*.json file's ztraces (neurons x time) and behavioral timestamps
    (list of per-trial-type onset-frame lists), matching lightweight_gather_data's schema
    (index 9 = ztraces, 11 = all_evts_imagetime, 19-22 = day/cage/mouse/group)."""
    with open(path, "r") as f:
        big_list = json.load(f)
    ztraces = np.asarray(big_list[9], dtype=np.float32)
    all_evts_imagetime = big_list[11]
    day, cage, mouse, group = big_list[19], big_list[20], big_list[21], big_list[22]
    meta = {"day": day, "cage": cage, "mouse": mouse, "group": group, "path": path}
    return ztraces, all_evts_imagetime, meta


def _onset_indicator(timestamps, n_timepoints, window=ONSET_WINDOW_FRAMES):
    """Binary 1D indicator (n_timepoints,) that is 1 for `window` frames after each onset."""
    indicator = np.zeros(n_timepoints, dtype=np.float32)
    for ts in timestamps:
        ts = int(ts)
        if 0 <= ts < n_timepoints:
            end = min(ts + window, n_timepoints)
            indicator[ts:end] = 1.0
    return indicator


def build_input_channels(all_evts_imagetime, n_timepoints, architecture="unmixed",
                          trial_list=TRIAL_LIST, window=ONSET_WINDOW_FRAMES, seed=0):
    """Build the external-input tensor (n_channels, n_timepoints) for one session, according
    to one of three architectures (see plan.md section 3):

      - "unmixed":    one channel per odor (n_channels == len(trial_list)); each channel
                      carries exactly that odor's onset indicator.
      - "semi-mixed": pairwise-combination channels -- for each unordered pair of odors, one
                      channel that is "on" whenever EITHER odor in the pair is presented
                      (so information about 2 odors is relayed jointly through one channel).
                      n_channels == C(len(trial_list), 2).
      - "fully-mixed": a single small set of channels, each a random positive linear mixture
                      (fixed, non-trainable mixing matrix) of all per-odor onset indicators --
                      i.e. every channel carries blended/overlapping information about all
                      odors simultaneously (conjunctive coding at the input stage). Uses
                      n_channels == len(trial_list) so architectures are directly comparable
                      in channel count; only how information is organized differs.

    Returns
    -------
    channels : np.ndarray, shape (n_channels, n_timepoints), float32
    channel_names : list[str] describing what each channel encodes (for bookkeeping /
        odor-sweep step 4, so we know which channel to up/down-weight for "TMT", etc.)
    """
    per_odor = np.stack([
        _onset_indicator(all_evts_imagetime[i], n_timepoints, window=window)
        for i in range(len(trial_list))
    ], axis=0)  # (n_odors, T)

    if architecture == "unmixed":
        channels = per_odor
        channel_names = list(trial_list)

    elif architecture == "semi-mixed":
        from itertools import combinations
        pairs = list(combinations(range(len(trial_list)), 2))
        channels = np.stack([
            np.clip(per_odor[i] + per_odor[j], 0, 1) for i, j in pairs
        ], axis=0)
        channel_names = [f"{trial_list[i]}+{trial_list[j]}" for i, j in pairs]

    elif architecture == "fully-mixed":
        rng = np.random.RandomState(seed)
        n_channels = len(trial_list)
        # Fixed (non-trainable) positive mixing matrix -- represents input-stage blending
        # that happens BEFORE the trainable input weights (in rnn_train.py) act on it, i.e.
        # this encodes "how are inputs organized", while the trainable weights encode
        # "how strongly does each organized channel drive the RNN".
        mix = rng.uniform(0.2, 1.0, size=(n_channels, len(trial_list))).astype(np.float32)
        mix = mix / mix.sum(axis=1, keepdims=True)  # rows sum to 1 -> still a "presence" signal in [0,1]
        channels = mix @ per_odor
        channel_names = [f"mix{k}({','.join(trial_list)})" for k in range(n_channels)]

    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Must be one of "
                          f"'unmixed', 'semi-mixed', 'fully-mixed'.")

    return channels.astype(np.float32), channel_names


def build_session_dataset(path, architecture="unmixed", trial_list=TRIAL_LIST,
                           window=ONSET_WINDOW_FRAMES, seed=0):
    """Build the full (input_channels, target_activity, meta) tuple for one session, ready to
    hand to rnn_train.py. target_activity is (n_neurons, T); input_channels is
    (n_channels, T); both are frame-aligned (same T)."""
    ztraces, all_evts_imagetime, meta = load_one_session(path)
    n_timepoints = ztraces.shape[1]
    channels, channel_names = build_input_channels(
        all_evts_imagetime, n_timepoints, architecture=architecture,
        trial_list=trial_list, window=window, seed=seed)
    meta = dict(meta)
    meta["architecture"] = architecture
    meta["channel_names"] = channel_names
    meta["n_neurons"] = ztraces.shape[0]
    meta["n_timepoints"] = n_timepoints
    return channels, ztraces, meta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--architecture", type=str, default="unmixed",
                         choices=["unmixed", "semi-mixed", "fully-mixed"])
    args = parser.parse_args()

    manifest = list_sessions(args.data_directory)
    print(f"Found {len(manifest)} sessions:")
    print(manifest[["session_id", "day", "group", "n_neurons"]].to_string(index=False))

    if len(manifest) > 0:
        row = manifest.iloc[0]
        channels, target, meta = build_session_dataset(row["path"], architecture=args.architecture)
        print(f"\nExample session {row['session_id']}: input {channels.shape}, target {target.shape}")
        print(f"Channel names ({args.architecture}): {meta['channel_names']}")
