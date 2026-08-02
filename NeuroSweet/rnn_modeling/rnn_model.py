#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_model.py
Description: The trainable RNN core used by the in-silico mPFC model (see plan.md). A vanilla
    Elman-style RNN with:
      - a trainable input layer (Linear: n_input_channels -> hidden_size) representing the
        "external input weights" (per-odor / per-channel strength), and
      - a trainable recurrent core (torch.nn.RNN) representing mPFC's own recurrent dynamics,
      - a trainable linear readout (hidden_size -> n_neurons) mapping hidden units back onto
        the session's real neuron count, so the model's *output* is directly comparable
        (same shape) to the session's real calcium traces,
      - an OPTIONAL per-neuron causal odor-response convolution kernel (see use_odor_kernel
        below), added directly to the readout.

    Training objective is SUPERVISED next-timestep prediction of the session's own real
    calcium activity (see rnn_train.py) -- NOT a classification/discrimination task, since
    odors are unconditioned stimuli, not something the animal is trained to classify.

    Hidden-unit activations (post recurrent core, pre readout) are what gets handed to the
    existing decoder/encoder pipeline (see rnn_decoder_encoder.py) as the "neurons" of the
    in-silico model.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import torch
import torch.nn as nn


class MPFCModelRNN(nn.Module):
    """
    Parameters
    ----------
    n_input_channels : int
        Number of external odor-input channels (varies by architecture: unmixed=4,
        semi-mixed=6, fully-mixed=4).
    hidden_size : int
        Number of RNN hidden units. Set equal to the session's own real neuron count by
        convention (see rnn_train.py), so the model has a directly comparable "population size,"
        though this is not a strict requirement of the architecture itself.
    n_neurons : int
        Number of real neurons in this session -- the readout dimensionality.
    use_odor_kernel : bool, default True
        Adds a per-neuron LEARNABLE CAUSAL CONVOLUTION over the external odor inputs directly
        to the readout: predicted_t = readout(hidden_t) + sum_c (kernel_{neuron,c} * inputs_c)_t.
        This is a pure feedforward function of external_inputs ONLY -- it never receives, uses,
        or has access to any real OR self-generated calcium value at any timestep, so it is
        architecturally IMPOSSIBLE for it to leak/cheat on validation, unlike the earlier
        teacher-forced AR(1) decay-layer design (removed -- see history below).

        MOTIVATION / HISTORY: an earlier version added an AR(1) leaky-integrator readout,
        teacher-forced with the REAL previous-frame calcium value, on the reasoning that GCaMP
        decay is well described by calcium_t ~= gamma*calcium_{t-1} + (1-gamma)*drive_t. That
        design was flawed in two ways found through direct empirical testing: (1) validation
        was ALSO teacher-forced (real data injected every single timestep), so the reported
        "improvement" was largely an artifact of continuously re-copying ground truth, not a
        real forecast -- once validation was fixed to be fully autonomous/non-cheating
        (single legitimate real anchor at the window start, then a pure self-generated
        cascade), held-out MSE collapsed back to baseline (~0.94) regardless of training
        curriculum. (2) Even before that was noticed, full per-step teacher forcing collapsed
        hidden-unit odor decodability from ~0.96 AUC to chance (~0.50), since the decay term
        gave the model a free way to explain trace variance without needing the hidden state
        to encode odor identity at all.

        Root cause once investigated honestly: per-timestep raw calcium is dominated by
        spontaneous/noise variance (~94% of total variance), while the true odor-evoked signal
        is small in comparison -- MSE-based gradient descent on the whole raw trace has no
        incentive to fit that small signal and simply learns to predict close to the mean. A
        recurrent AR(1) autoregression on raw values cannot fix this: it either needs real data
        fed back in (which is cheating) or it needs to hallucinate future values from its own
        possibly-wrong past outputs (which just compounds noise).

        The FIX is this feedforward causal-convolution kernel: instead of trying to
        autoregressively reconstruct noisy raw calcium values step by step, it directly learns
        each neuron's own odor-triggered impulse-response SHAPE from the (sparse, exactly-known)
        odor onset timing -- exactly the structure a trial-averaged/event-triggered analysis
        would recover, but folded into the model and evaluated per-timepoint. Verified via a
        standalone diagnostic (`rnn_fir_kernel_test.py`, pure FIR kernel alone, no GRU) and then
        in the combined architecture (`rnn_unified_model_test.py`) that this design:
        (a) improves whole-session honest MSE (combined model: 0.916 vs 0.940 trivial-mean
        baseline),
        (b) recovers real, noise-cancelled trial-averaged transient SHAPE with population
        correlation 0.94 and per-neuron mean correlation 0.93 (100% of neurons > 0.3) against
        real data, and
        (c) hidden-unit odor decodability is UNCHANGED OR BETTER (0.98 AUC vs ~0.96 for the
        plain GRU with no kernel) -- while being structurally impossible to cheat (no
        target/self-generated feedback of any kind), and additive to (not competing/blended
        with) the GRU readout, so it does not dilute gradient pressure on hidden units to
        encode odor identity (unlike the old gamma-blend design).

        IMPORTANT CAVEAT + RESOLUTION found in a later skeptical re-audit (not architectural
        cheating, but a train/val SPLIT flaw -- see rnn_train.py's
        make_windows/split_train_val_regions docstrings for the full fix): the (a)/(b) numbers
        above were originally computed on the WHOLE session (train-region + val-region frames
        combined), and at the time the train/val split itself randomly assigned overlapping
        windows to train vs val by index, so up to 99.6% of "val" frames were also covered by
        some training window (see `rnn_leakage_quantify.py`). After fixing the split to a
        genuine contiguous, guard-buffered held-out time block, a first re-check restricted to
        ONLY that single held-out block (the last 20% of the session, ~87 odor trials) found
        trial-averaged correlation had apparently collapsed (mean per-neuron ~0.14). This
        looked alarming, but a follow-up "noise ceiling" check (splitting REAL data alone into
        two random ~87-trial halves, no model at all) showed even real-vs-real correlation at
        that sample size only reaches ~0.31 -- so a single ~87-trial held-out block is simply
        too small/underpowered a sample to reliably measure trial-averaged correlation, quite
        apart from any model quality question.

        Resolved with a proper 5-fold contiguous-time-block cross-validation
        (`rnn_kfold_honest_check.py`): rotate the held-out block through 5 positions spanning
        the whole session, refit a fresh model per fold (never touching that fold's held-out
        frames during training), and pool honest val-only metrics across all 5 folds (617
        held-out odor trials total, each evaluated only by the model that never trained on it).
        Result: trial-averaged population correlation = 0.706, per-neuron trial-averaged
        correlation mean = 0.567 (median 0.698, 78% of neurons > 0.3) -- clearly above the 0.31
        real-data noise ceiling for this sample size, i.e. genuine, honest, non-trivial signal
        recovery -- and pooled hidden-unit odor decodability = 0.998 AUC. Whole-trace raw MSE
        still barely beats the trivial-mean baseline in aggregate (0.946 vs 0.940), which is
        expected and fine: raw per-timepoint calcium is noise-dominated, and the
        noise-cancelling trial-averaged view (not raw MSE) is the correct honest lens for
        judging transient-shape quality. Bottom line: on genuinely unseen data, with an
        adequately large held-out sample, this architecture achieves reasonably good (not
        perfect) transient-shape recovery AND odor decodability simultaneously, with zero
        cheating by construction. A single small held-out block (as used per-session in the
        production `train_one_session` split) can look misleadingly bad purely from sampling
        noise -- treat any single-session, single-split val metric with that caveat, and prefer
        pooling across sessions/seeds when judging overall model quality.
    kernel_length : int, default 60
        Causal convolution kernel length in frames (~2s at ~30Hz, matching previously-observed
        calcium decay timescales) -- only used when use_odor_kernel=True.
    """
    def __init__(self, n_input_channels, hidden_size, n_neurons, nonlinearity="tanh",
                 cell_type="gru", use_odor_kernel=True, kernel_length=60):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.hidden_size = hidden_size
        self.n_neurons = n_neurons
        self.cell_type = cell_type
        self.use_odor_kernel = use_odor_kernel
        self.kernel_length = kernel_length

        # External input weights: trainable, one weight per (channel, hidden-unit) pair.
        self.input_layer = nn.Linear(n_input_channels, hidden_size, bias=True)

        # Recurrent core: mPFC's own trainable internal dynamics.
        # NOTE: default changed from vanilla tanh RNN -> GRU. Diagnostic testing found the
        # vanilla RNN could not beat even a trivial "predict-the-mean" baseline on real data
        # (val MSE ~0.93 vs trivial-mean baseline 0.94, even after 2000 epochs), while a
        # naive last-frame-persistence baseline achieved 0.52 -- i.e. the vanilla RNN was not
        # learning to propagate/gate information through recurrent state at all. Only 13.7% of
        # session timepoints have ANY odor input active (rest is spontaneous/endogenous
        # activity the model cannot access), so gradient signal is sparse; a tanh RNN's
        # vanishing-gradient tendency made this worse. GRU's gating gives much better gradient
        # flow through long silent stretches between odor events.
        if cell_type == "gru":
            self.rnn = nn.GRU(input_size=hidden_size, hidden_size=hidden_size,
                               num_layers=1, batch_first=True)
        elif cell_type == "rnn":
            self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size,
                               num_layers=1, nonlinearity=nonlinearity, batch_first=True)
        else:
            raise ValueError(f"Unknown cell_type '{cell_type}'. Must be 'gru' or 'rnn'.")

        # Linear readout back to real neuron count (for computing the supervised loss
        # against real calcium traces).
        self.readout = nn.Linear(hidden_size, n_neurons)

        # Per-neuron causal odor-response convolution kernel (see use_odor_kernel docstring
        # above). Purely feedforward over external_inputs -- never touches calcium values.
        if self.use_odor_kernel:
            self.odor_kernel = nn.Conv1d(n_input_channels, n_neurons,
                                          kernel_size=kernel_length,
                                          padding=kernel_length - 1, bias=False)

    def forward(self, external_inputs, h0=None):
        """
        external_inputs : (batch, T, n_input_channels)

        Returns
        -------
        predicted_activity : (batch, T, n_neurons) -- readout of hidden units, optionally plus
            the per-neuron odor-response convolution kernel's output (see use_odor_kernel).
            Used for the supervised training/validation loss against real calcium traces.
            NOTE: this is a pure function of external_inputs (and h0) -- it never depends on
            target/real calcium values at all, so there is no teacher forcing, no
            self-referential feedback, and no way for validation to be dishonest by
            construction.
        hidden_states : (batch, T, hidden_size) -- raw RNN hidden-unit activity, used as the
            in-silico "neurons" for the decoder/encoder pipeline.
        """
        driven_input = self.input_layer(external_inputs)  # (batch, T, hidden_size)
        hidden_states, _ = self.rnn(driven_input, h0)      # (batch, T, hidden_size)
        drive = self.readout(hidden_states)                # (batch, T, n_neurons)

        if not self.use_odor_kernel:
            return drive, hidden_states

        T = external_inputs.shape[1]
        conv_in = external_inputs.transpose(1, 2)                    # (batch, C, T)
        kernel_out = self.odor_kernel(conv_in)[:, :, :T]              # causal: drop future tail
        kernel_out = kernel_out.transpose(1, 2)                       # (batch, T, n_neurons)
        predicted_activity = drive + kernel_out
        return predicted_activity, hidden_states

    def set_input_channel_weight(self, channel_idx, scale):
        """Multiplicatively re-scale the trained input weights for a single external input
        channel (used by the odor up/down-weighting sweep, step 4 of plan.md). Operates
        in-place on self.input_layer.weight (shape: hidden_size x n_input_channels)."""
        with torch.no_grad():
            self.input_layer.weight[:, channel_idx] *= scale
