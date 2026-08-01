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
    def __init__(self, n_input_channels, hidden_size, n_neurons, nonlinearity="tanh"):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.hidden_size = hidden_size
        self.n_neurons = n_neurons

        # External input weights: trainable, one weight per (channel, hidden-unit) pair.
        self.input_layer = nn.Linear(n_input_channels, hidden_size, bias=True)

        # Recurrent core: mPFC's own trainable internal dynamics.
        self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size,
                           num_layers=1, nonlinearity=nonlinearity, batch_first=True)

        # Linear readout back to real neuron count (for computing the supervised loss
        # against real calcium traces).
        self.readout = nn.Linear(hidden_size, n_neurons)

    def forward(self, external_inputs, h0=None):
        """
        external_inputs : (batch, T, n_input_channels)
        Returns
        -------
        predicted_activity : (batch, T, n_neurons) -- readout of hidden units, used for the
            supervised training loss against real calcium traces.
        hidden_states : (batch, T, hidden_size) -- raw RNN hidden-unit activity, used as the
            in-silico "neurons" for the decoder/encoder pipeline.
        """
        driven_input = self.input_layer(external_inputs)  # (batch, T, hidden_size)
        hidden_states, _ = self.rnn(driven_input, h0)      # (batch, T, hidden_size)
        predicted_activity = self.readout(hidden_states)   # (batch, T, n_neurons)
        return predicted_activity, hidden_states

    def set_input_channel_weight(self, channel_idx, scale):
        """Multiplicatively re-scale the trained input weights for a single external input
        channel (used by the odor up/down-weighting sweep, step 4 of plan.md). Operates
        in-place on self.input_layer.weight (shape: hidden_size x n_input_channels)."""
        with torch.no_grad():
            self.input_layer.weight[:, channel_idx] *= scale
