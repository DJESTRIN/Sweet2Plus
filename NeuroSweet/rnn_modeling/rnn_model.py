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
        (same shape) to the session's real calcium traces.

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
    """
    def __init__(self, n_input_channels, hidden_size, n_neurons, nonlinearity="tanh",
                 cell_type="gru", use_calcium_decay=False, init_decay=0.8):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.hidden_size = hidden_size
        self.n_neurons = n_neurons
        self.cell_type = cell_type
        self.use_calcium_decay = use_calcium_decay

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

        # Calcium-decay layer (AR(1) leaky integrator), applied to the raw readout.
        # MOTIVATION: real GCaMP calcium traces are (approximately) an exponentially-decaying
        # indicator kinetic convolved with underlying spiking/drive activity -- i.e.
        # calcium_t ~= gamma * calcium_{t-1} + (1-gamma) * drive_t. Diagnostic testing found
        # a naive "copy the previous frame" baseline achieves MSE 0.52 on real data, vs 0.94
        # for a "predict the mean" baseline -- i.e. most of a real calcium trace's
        # predictability comes from this decay structure, not from odor identity. The RNN's
        # own hidden recurrence was not learning to reproduce this decay from scratch (only
        # ~30-40 independent training windows per session, generic weights, short training
        # budget), so this bakes the known biophysical prior directly into the architecture as
        # a per-neuron learnable decay constant (sigmoid-transformed to stay in (0, 1)), rather
        # than requiring gradient descent to discover exponential decay unaided.
        if self.use_calcium_decay:
            init_logit = torch.logit(torch.full((n_neurons,), float(init_decay)))
            self._decay_logit = nn.Parameter(init_logit)

    def forward(self, external_inputs, h0=None, target_prev=None):
        """
        external_inputs : (batch, T, n_input_channels)
        target_prev : (batch, T, n_neurons), optional -- the REAL previous-timestep calcium
            value (target shifted by one frame) for the SAME window as external_inputs. When
            provided, the calcium-decay layer blends this real value with the model's
            odor/hidden-state-driven correction (see below) -- this is standard supervised
            one-step-ahead ("teacher forced") sequence prediction: since we always have ground
            truth for the real recorded session being analyzed (never asking the model to
            forecast an unseen session), using the true previous frame is legitimate, not
            data leakage, as long as validation windows are held out from training (the
            model never sees validation-window IDENTITY during gradient updates -- see
            rnn_train.py's train/val split). If target_prev is None, falls back to an
            open-loop AR(1) scan over the model's OWN previous prediction (needed only for a
            hypothetical true forecasting/generative use case with no ground truth available).
        Returns
        -------
        predicted_activity : (batch, T, n_neurons) -- readout of hidden units (optionally
            passed through the calcium-decay AR(1) layer), used for the supervised training
            loss against real calcium traces.
        hidden_states : (batch, T, hidden_size) -- raw RNN hidden-unit activity, used as the
            in-silico "neurons" for the decoder/encoder pipeline.
        """
        driven_input = self.input_layer(external_inputs)  # (batch, T, hidden_size)
        hidden_states, _ = self.rnn(driven_input, h0)      # (batch, T, hidden_size)
        drive = self.readout(hidden_states)                # (batch, T, n_neurons)

        if not self.use_calcium_decay:
            return drive, hidden_states

        gamma = torch.sigmoid(self._decay_logit)            # (n_neurons,)

        if target_prev is not None:
            # Teacher-forced AR(1): predicted_t = gamma * real_activity[t-1] + (1-gamma)*drive_t
            # Fully vectorized (no python-level time loop needed since target_prev is already
            # known/observed data, not a function of the model's own earlier outputs).
            predicted_activity = gamma * target_prev + (1.0 - gamma) * drive
            return predicted_activity, hidden_states

        # Open-loop fallback: no ground truth available, so decay is applied to the model's
        # own previous prediction (sequential scan; only reachable without target_prev).
        batch_size, T, n_neurons = drive.shape
        o_prev = torch.zeros(batch_size, n_neurons, device=drive.device, dtype=drive.dtype)
        outputs = []
        for t in range(T):
            o_prev = gamma * o_prev + (1.0 - gamma) * drive[:, t, :]
            outputs.append(o_prev)
        predicted_activity = torch.stack(outputs, dim=1)    # (batch, T, n_neurons)
        return predicted_activity, hidden_states

    def set_input_channel_weight(self, channel_idx, scale):
        """Multiplicatively re-scale the trained input weights for a single external input
        channel (used by the odor up/down-weighting sweep, step 4 of plan.md). Operates
        in-place on self.input_layer.weight (shape: hidden_size x n_input_channels)."""
        with torch.no_grad():
            self.input_layer.weight[:, channel_idx] *= scale
