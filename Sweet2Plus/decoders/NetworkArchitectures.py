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
import numpy as np
import ipdb
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class GCNTrialClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNTrialClassifier, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # Output size 5 (number of classes for each time point)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply convolution layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Global pooling to summarize the graph (if needed)
        x = global_mean_pool(x, batch)  # You may use other pooling methods depending on your task

        # Ensure hidden_dim matches 25 * 5 for reshaping
        assert x.shape[1] == 100, f"Expected 125 features, but got {x.shape[1]} features after pooling"

        # Reshape output to (batch_size, 25, 5)
        x = x.view(-1, 25, 4)  # Ensure shape is (batch_size, 25, 5)

        return x


class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.dropout(out) 


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=4, num_layers=1, dropout=0.7):
        super(LSTMEncoder, self).__init__()
        self.residual_blocks = nn.ModuleList([
            LSTMBlock(input_size if i == 0 else hidden_size, hidden_size, dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)  # Pass through residual blocks
        return x


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size=8, output_size=4, num_layers=1, dropout=0.7):
        super(LSTMDecoder, self).__init__()
        self.residual_blocks = nn.ModuleList([
            LSTMBlock(hidden_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for block in self.residual_blocks:
            x = block(x)  # Pass through residual blocks

        return self.fc(x[:, -1, :])  # Take the last time step output


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=4, output_size=4, num_layers=1, dropout=0.7):
        super(LSTMEncoderDecoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, output_size, num_layers, dropout)

    def forward(self, x):
        """
        x is expected to have shape (K, N, T) and needs to be transposed before processing.
        """
        x = x.transpose(1, 2)  # Convert (K, N, T) → (K, T, N) for LSTM
        encoded = self.encoder(x)
        decoded_output = self.decoder(encoded)
        return decoded_output
    
class LSTMDecoderSimple(nn.Module):
    def __init__(self, input_size, hidden_size=4, output_size=4, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape from (batch, seq_len, input_size) to (batch, input_size, seq_len)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply batch normalization on the output of the LSTM (we need to reshape it to 2D for BatchNorm1d)
        batch_size, seq_len, hidden_size = lstm_out.size()
        
        # Reshape to (batch_size * seq_len, hidden_size) to apply BatchNorm1d
        lstm_out_reshaped = lstm_out.contiguous().view(batch_size * seq_len, hidden_size)
        
        # Apply BatchNorm1d
        lstm_out_normalized = self.bn(lstm_out_reshaped)
        
        # Reshape it back to (batch_size, seq_len, hidden_size)
        lstm_out_normalized = lstm_out_normalized.view(batch_size, seq_len, hidden_size)
        
        # Return the output of the last time step using the fully connected layer
        return self.fc(lstm_out[:, -1, :])  # Use last time step


class NeuralNetworkFigures():
    def __init__(self, name):
        self.name = name
    
    def plot_learning_curve(self, data, label, output_file):
        plt.figure()
        plt.plot(np.asarray(data), label=label)
        plt.title(f"Current {label} results")
        plt.savefig(output_file)
    
    def plot_all_curves(self, metrics_dict, output_file):
        """
        Plots the loss, accuracy, and F1 score for training, validation, and testing (if present) in 3 subplots.
        
        metrics_dict: A dictionary containing training, validation, and testing metrics
        output_file: Path to save the resulting plot
        """
        try:

            # Create the subplots
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 subplots: loss, accuracy, and F1 score
            fig.tight_layout(pad=4.0)  # Adjust space between subplots
            
            # Plot loss
            axes[0].plot(metrics_dict['train_loss'], label='Train Loss', color='blue')
            axes[0].plot(metrics_dict['val_loss'], label='Validation Loss', color='green')
            if 'test_loss' in metrics_dict:
                axes[0].plot(metrics_dict['test_loss'], label='Test Loss', color='red')
            axes[0].set_title('Loss Curve')
            axes[0].set_ylabel('Loss')
            axes[0].legend()

            # Plot accuracy
            axes[1].plot(metrics_dict['train_accuracy'], label='Train Accuracy', color='blue')
            axes[1].plot(metrics_dict['val_accuracy'], label='Validation Accuracy', color='green')
            if 'test_accuracy' in metrics_dict:
                axes[1].plot(metrics_dict['test_accuracy'], label='Test Accuracy', color='red')
            axes[1].set_title('Accuracy Curve')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()

            # Plot F1 score
            axes[2].plot(metrics_dict['train_f1'], label='Train F1 Score', color='blue')
            axes[2].plot(metrics_dict['val_f1'], label='Validation F1 Score', color='green')
            if 'test_f1' in metrics_dict:
                axes[2].plot(metrics_dict['test_f1'], label='Test F1 Score', color='red')
            axes[2].set_title('F1 Score Curve')
            axes[2].set_ylabel('F1 Score')
            axes[2].legend()

            # Save the figure
            plt.savefig(output_file)
        except:
            print('Error with plotting results.... ')