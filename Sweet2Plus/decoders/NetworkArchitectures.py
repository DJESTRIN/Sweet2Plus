#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: NetworkArchitectures.py
Description: Contains neural network classes
Author: David Estrin
Version: 1.0
Date: 03-05-2025
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LSTMResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(LSTMResidualBlock, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(input_size, hidden_size)  # For matching input size to hidden size

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM output
        out = self.dropout(out)  # Apply dropout
        residual = self.linear(x)  # Residual connection
        return out + residual  # Add residual connection

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=46, hidden_size=256, num_layers=2, dropout=0.1):
        super(LSTMEncoder, self).__init__()
        self.residual_blocks = nn.ModuleList([
            LSTMResidualBlock(input_size if i == 0 else hidden_size, hidden_size, dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)  # Pass through residual blocks
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size=256, output_size=4, num_layers=2, dropout=0.1):
        super(LSTMDecoder, self).__init__()
        self.residual_blocks = nn.ModuleList([
            LSTMResidualBlock(hidden_size, hidden_size, dropout)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Ensure x is three-dimensional (batch_size, seq_length, hidden_size)
        # Here, we assume you want the last hidden state of the last layer
        for block in self.residual_blocks:
            x = block(x)  # Pass through residual blocks

        # If x is two-dimensional, you can expand it like this:
        if x.dim() == 2:  # If it's (batch_size, hidden_size)
            x = x.unsqueeze(1)  # Expand to (batch_size, 1, hidden_size)

        # Use only the last time step for the output
        return self.fc(x[:, -1, :])  # Output from the last time step


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size=46, hidden_size=256, output_size=4, num_layers=2, dropout=0.1):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, output_size, num_layers, dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_output = self.decoder(encoded)  # Use the final encoded representation
        return decoded_output

class NeuralNetworkFigures():
    def __init__(self,name):
        self.name = name
    
    def plot_learning_curve(self,data,label,output_file):
        plt.figure()
        plt.plot(np.asarray(data),label=label)
        plt.title(f"Current {label} results")
        plt.savefig(output_file)