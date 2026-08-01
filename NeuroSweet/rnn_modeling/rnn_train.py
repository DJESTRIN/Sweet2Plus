#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_train.py
Description: Supervised training loop for one session's MPFCModelRNN (see rnn_model.py) and
    rnn_data.py). Fits the RNN's next-timestep calcium-prediction loss against the session's
    own real recorded activity.

    Provides a live `rich`-based CLI table (epoch, train loss, val loss, ETA/time remaining)
    when run in an interactive terminal (local prototyping, Phase A of plan.md), and a plain
    periodic-print fallback (no rich rendering) when stdout is not a live terminal -- e.g. when
    redirected to a SLURM .out log file (Phase B), since rich's live-updating table does not
    render meaningfully in a static log file.

Usage (single session, single seed -- local prototype):
    python -m NeuroSweet.rnn_modeling.rnn_train \
        --session_json <path/to/objfile.json> --architecture unmixed --seed 0 \
        --epochs 200 --hidden_size 64 --drop_directory <outdir>

Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from NeuroSweet.rnn_modeling.rnn_data import build_session_dataset, TRIAL_LIST
from NeuroSweet.rnn_modeling.rnn_model import MPFCModelRNN


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_windows(channels, target, window_len=200, stride=100):
    """Chop a single long (n_channels, T) / (n_neurons, T) session recording into overlapping
    windows so the RNN trains on many shorter sequences (mini-batches) instead of one huge
    sequence -- standard practice for RNN training stability/speed, and lets us hold out
    windows for a validation split.
    Returns X (n_windows, window_len, n_channels), Y (n_windows, window_len, n_neurons).
    """
    n_channels, T = channels.shape
    n_neurons = target.shape[0]
    starts = list(range(0, max(T - window_len, 1), stride))
    if not starts:
        starts = [0]
        window_len = T
    X, Y = [], []
    for s in starts:
        e = min(s + window_len, T)
        if e - s < window_len:
            continue
        X.append(channels[:, s:e].T)   # (window_len, n_channels)
        Y.append(target[:, s:e].T)     # (window_len, n_neurons)
    if not X:  # fallback: single window covering whatever we have
        X = [channels.T]
        Y = [target.T]
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)


def _is_live_terminal():
    """True if stdout is an interactive terminal (local run) vs redirected to a file (SLURM
    .out log), used to pick the rich live-table UI vs a plain print fallback."""
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def train_one_session(channels, target, hidden_size=64, epochs=200, lr=1e-3, seed=0,
                       window_len=200, stride=100, val_frac=0.2, device="cpu",
                       progress_label="session", cell_type="gru", batch_size=8,
                       weight_decay=1e-3, early_stop_patience=30):
    """Trains one MPFCModelRNN on one session's data. Returns (model, history) where history
    is a list of dicts with per-epoch train/val loss.

    NOTE on batch_size: training was originally full-batch (one optimizer step per epoch,
    across all ~36-45 windows of a session at once). Diagnostic testing showed this, combined
    with a vanilla tanh RNN, could not beat a trivial "predict-the-mean" baseline even after
    2000 epochs. Training now takes multiple mini-batch gradient steps per epoch (shuffled
    each epoch), giving many more optimizer updates for the same epoch budget -- standard
    practice, and cheap here since windows are small.

    NOTE on weight_decay/early_stop_patience: mini-batching alone let the model drive train
    loss down but caused val loss to steadily WORSEN (classic overfitting -- a single session
    only has ~30-40 independent training windows). L2 weight decay plus early stopping
    (restore the best-val-loss model seen so far if val loss hasn't improved for
    `early_stop_patience` epochs) keeps the model from memorizing training windows."""
    set_seed(seed)
    n_channels = channels.shape[0]
    n_neurons = target.shape[0]

    X, Y = make_windows(channels, target, window_len=window_len, stride=stride)
    n_windows = X.shape[0]
    n_val = max(1, int(n_windows * val_frac)) if n_windows > 1 else 0
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_windows)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:  # too few windows to split -- train on everything, val==train
        train_idx = idx
        val_idx = idx

    X_train = torch.from_numpy(X[train_idx]).to(device)
    Y_train = torch.from_numpy(Y[train_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    Y_val = torch.from_numpy(Y[val_idx]).to(device)

    model = MPFCModelRNN(n_input_channels=n_channels, hidden_size=hidden_size,
                          n_neurons=n_neurons, cell_type=cell_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    eff_batch_size = min(batch_size, n_train)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_best = 0

    history = []
    use_rich = _is_live_terminal()

    if use_rich:
        from rich.live import Live
        from rich.table import Table
        from rich.console import Console

        console = Console()

        def render_table(epoch, train_loss, val_loss, t_elapsed, t_per_epoch):
            table = Table(title=f"RNN training: {progress_label}")
            table.add_column("Epoch")
            table.add_column("Train Loss")
            table.add_column("Val Loss")
            table.add_column("Elapsed (s)")
            table.add_column("ETA (s)")
            eta = t_per_epoch * (epochs - epoch - 1)
            table.add_row(f"{epoch + 1}/{epochs}", f"{train_loss:.5f}", f"{val_loss:.5f}",
                          f"{t_elapsed:.1f}", f"{eta:.1f}")
            return table

        t0 = time.time()
        with Live(console=console, refresh_per_second=4) as live:
            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(n_train)
                epoch_losses = []
                for b0 in range(0, n_train, eff_batch_size):
                    b_idx = perm[b0:b0 + eff_batch_size]
                    optimizer.zero_grad()
                    pred, _ = model(X_train[b_idx])
                    loss = loss_fn(pred, Y_train[b_idx])
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                train_loss = float(np.mean(epoch_losses))

                model.eval()
                with torch.no_grad():
                    val_pred, _ = model(X_val)
                    val_loss = loss_fn(val_pred, Y_val).item()

                t_elapsed = time.time() - t0
                t_per_epoch = t_elapsed / (epoch + 1)
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                                 "elapsed_s": t_elapsed})
                live.update(render_table(epoch, train_loss, val_loss, t_elapsed, t_per_epoch))

                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                    if early_stop_patience is not None and epochs_since_best >= early_stop_patience:
                        break
    else:
        # Plain periodic-print fallback for non-interactive (e.g. SLURM .out log) contexts.
        t0 = time.time()
        print_every = max(1, epochs // 20)
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train)
            epoch_losses = []
            for b0 in range(0, n_train, eff_batch_size):
                b_idx = perm[b0:b0 + eff_batch_size]
                optimizer.zero_grad()
                pred, _ = model(X_train[b_idx])
                loss = loss_fn(pred, Y_train[b_idx])
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            train_loss = float(np.mean(epoch_losses))

            model.eval()
            with torch.no_grad():
                val_pred, _ = model(X_val)
                val_loss = loss_fn(val_pred, Y_val).item()

            t_elapsed = time.time() - t0
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                             "elapsed_s": t_elapsed})
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                t_per_epoch = t_elapsed / (epoch + 1)
                eta = t_per_epoch * (epochs - epoch - 1)
                print(f"[{progress_label}] epoch {epoch + 1}/{epochs} "
                      f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                      f"elapsed={t_elapsed:.1f}s eta={eta:.1f}s", flush=True)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if early_stop_patience is not None and epochs_since_best >= early_stop_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        history.append({"epoch": "restored_best", "train_loss": None,
                         "val_loss": best_val_loss, "elapsed_s": history[-1]["elapsed_s"]})

    return model, history


def get_hidden_states(model, channels, device="cpu"):
    """Run the full (untruncated) session through the trained model once (no gradient) and
    return hidden-unit activity (n_hidden, T) for downstream decoder/encoder analysis."""
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(channels.T[None, :, :]).to(device)  # (1, T, n_channels)
        _, hidden_states = model(x)
    return hidden_states[0].cpu().numpy().T  # (n_hidden, T)


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session_json", type=str, required=True,
                         help="Path to one session's objfile*.json")
    parser.add_argument("--architecture", type=str, default="unmixed",
                         choices=["unmixed", "semi-mixed", "fully-mixed"])
    parser.add_argument("--drop_directory", type=str, required=True)
    parser.add_argument("--hidden_size", type=int, default=None,
                         help="Default: match session's real neuron count")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window_len", type=int, default=200)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--cell_type", type=str, default="gru", choices=["gru", "rnn"])
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = cli_parser()
    os.makedirs(args.drop_directory, exist_ok=True)

    channels, target, meta = build_session_dataset(args.session_json, architecture=args.architecture,
                                                     seed=args.seed)
    hidden_size = args.hidden_size or meta["n_neurons"]
    label = f"{meta['group']}_{meta['mouse']}_day{meta['day']}_{args.architecture}_seed{args.seed}"

    print(f"Session: {label} | neurons={meta['n_neurons']} T={meta['n_timepoints']} "
          f"channels={channels.shape[0]} hidden_size={hidden_size}")

    model, history = train_one_session(
        channels, target, hidden_size=hidden_size, epochs=args.epochs, lr=args.lr,
        seed=args.seed, window_len=args.window_len, stride=args.stride,
        cell_type=args.cell_type, batch_size=args.batch_size,
        progress_label=label)

    hidden_states = get_hidden_states(model, channels)

    out_prefix = os.path.join(args.drop_directory, label)
    torch.save(model.state_dict(), out_prefix + "_model.pt")
    np.save(out_prefix + "_hidden_states.npy", hidden_states)
    with open(out_prefix + "_history.json", "w") as f:
        json.dump({"history": history, "meta": {k: v for k, v in meta.items() if k != "channel_names"},
                   "channel_names": meta["channel_names"]}, f, indent=2)

    print(f"Done. Final train_loss={history[-1]['train_loss']:.5f} "
          f"val_loss={history[-1]['val_loss']:.5f}")
    print(f"Saved -> {out_prefix}_model.pt / _hidden_states.npy / _history.json")


if __name__ == "__main__":
    main()
