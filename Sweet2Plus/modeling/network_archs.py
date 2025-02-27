#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: network_archs.py
Description: Model weights 
Author: David James Estrin
Version: 1.0
Date: 11-25-2024
"""
import torch
import torch.nn as nn
import ipdb

class SingleSampleLinearLayer(nn.Module):
    """ Each neuron has its own independent linear layer. """
    def __init__(self, sequence_length, num_neurons):
        super(SingleSampleLinearLayer, self).__init__()
        self.num_neurons = num_neurons
        # Create a linear layer for each neuron that maps from sequence_length to 1 output
        self.linear_layers = nn.ModuleList([
            nn.Linear(sequence_length, 1) 
            for _ in range(num_neurons)
        ])

    def forward(self, x):
        outputs = []
        for i in range(self.num_neurons):
            sample_input = x[:, i, :]  # Shape: (batch_size, sequence_length)
            prediction = self.linear_layers[i](sample_input)  # Output shape: (batch_size, 1)
            outputs.append(prediction)
        return torch.cat(outputs, dim=1)  # Shape: (batch_size, num_neurons)

class SingleSampleLinearNN(nn.Module):
    def __init__(self, sequence_length, num_neurons, output_size):
        super(SingleSampleLinearNN, self).__init__()
        self.singleLinearLayer = SingleSampleLinearLayer(sequence_length=sequence_length, 
                                                         num_neurons=num_neurons)
        self.final_fc = nn.Linear(num_neurons, output_size)  # Final FC layer across all neurons
    
    def forward(self, x):
        x = self.singleLinearLayer(x)  # Shape: (batch, num_neurons)
        x = self.final_fc(x)  # Final output shape: (batch, output_size)
        return x

class SingleSampleLSTMLayer(nn.Module):
    """ Each neuron has its own independent LSTM. """
    def __init__(self, sequence_length, num_neurons, hidden_size=64, num_layers=2):
        super(SingleSampleLSTMLayer, self).__init__()
        self.num_neurons = num_neurons
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) 
            for _ in range(num_neurons)
        ])
        self.fc = nn.Linear(sequence_length, 1)

    def forward(self, x):
        outputs = []
        for i in range(self.num_neurons):
            sample_input = x[:, i, :].unsqueeze(-1)  # Shape: (batch_size, sequence_length, 1)
            lstm_out, _ = self.lstm_layers[i](sample_input)  # Pass through LSTM
            last_output = lstm_out[:, :, -1]  # Take last time step output
            prediction = self.fc(last_output)
            outputs.append(prediction)
        return torch.cat(outputs, dim=1)  # Shape: (batch_size, num_neurons)

class SingleSampleNN(nn.Module):
    def __init__(self, sequence_length, hidden_size, num_layers, num_neurons, output_size):
        super(SingleSampleNN, self).__init__()
        self.singleLSTMlayer = SingleSampleLSTMLayer(sequence_length=sequence_length, 
                                                     hidden_size=hidden_size, 
                                                     num_layers=num_layers, 
                                                     num_neurons=num_neurons)
        self.final_fc = nn.Linear(num_neurons, output_size)  # Final FC layer across all neurons
    
    def forward(self, x):
        x = self.singleLSTMlayer(x)  # Shape: (batch, num_neurons)
        x = self.final_fc(x)  # Final output shape: (batch, output_size)
        return x

class DirectInputLayer(nn.Module):
    """ A Custom Layer where M inputs are directly input into N neurons,
      This is notably not a dense layer.  """
    def __init__(self, network_size, inputs_per_neuron,dropout=0.15,ns=0.1):
        super(DirectInputLayer, self).__init__()
        self.network_size = network_size
        self.inputs_per_neuron = inputs_per_neuron

        # Build layers
        self.d_layer1 = nn.Linear(inputs_per_neuron, 16, bias=True)
        self.d_layer2 = nn.Linear(16,8,bias=True)
        self.d_layer3 = nn.Linear(8,4,bias=True)
        self.d_layer4 = nn.Linear(4,1,bias=True)

        self.l_relu = torch.nn.LeakyReLU(negative_slope=ns)
        self.drop_out = torch.nn.Dropout(p=dropout)
    
    def forward(self,x):
        # Ensure tensor is contiguous before reshaping
        x = x.contiguous()  # Add this
        
        try:
            x = self.drop_out(self.l_relu(self.d_layer1(x)))
            x = self.drop_out(self.l_relu(self.d_layer2(x)))
            x = self.drop_out(self.l_relu(self.d_layer3(x)))
            x = self.drop_out(self.l_relu(self.d_layer4(x)))
        except:
            ipdb.set_trace()
            
        return x.squeeze(-1)

class SingleLayerNetwork(torch.nn.Module):
    """ A Single Layer network utalizing DirectInputLayer """
    def __init__(self, network_size=339,inputs_per_neuron=100,dropout=0.15):
        super(SingleLayerNetwork, self).__init__()
        # Set up layers for neural network
        self.direct_input_layer = DirectInputLayer(network_size, inputs_per_neuron)
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.1)
        self.drop1 = torch.nn.Dropout(p=dropout)
        self.dense_layer1 = torch.nn.Linear(network_size, network_size)  

    def forward(self,xoh):
        xoriginal=xoh
        xoh = self.direct_input_layer(xoh)
        xoh = self.relu1(xoh)
        xoh = self.drop1(xoh)
        xoh = self.dense_layer1(xoh)  
        return xoh
    
class SimpleSingleLayerNetwork(torch.nn.Module):
    """ A Single Layer network utalizing DirectInputLayer """
    def __init__(self, network_size=339,dropout=0.2):
        super(SimpleSingleLayerNetwork, self).__init__()
        # Set up layers for neural network
        self.dense_layer1 = torch.nn.Linear(network_size, network_size)  
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = torch.nn.Dropout(p=dropout)
        self.dense_layer2 = torch.nn.Linear(network_size, network_size)  

    def forward(self,xoh):
        xoh = self.dense_layer1(xoh)
        xoh = self.relu1(xoh)
        xoh = self.drop1(xoh)
        xoh = self.dense_layer2(xoh)
        return xoh
    
def weighted_mse_loss(output, target):
    # Initialize weights as ones
    weights = torch.ones_like(target)
    
    # Define weights for specific ranges
    weights = torch.where((target >= 0.9) & (target < 1.0), 20.0, weights)
    weights = torch.where((target >= 0.8) & (target < 0.9), 10.0, weights)
    weights = torch.where((target >= 0.7) & (target < 0.8), 9.0, weights)
    weights = torch.where((target >= 0.6) & (target < 0.7), 8.0, weights)
    weights = torch.where((target >= 0.5) & (target < 0.6), 7.0, weights)
    weights = torch.where((target >= 0.4) & (target < 0.5), 6.0, weights)
    weights = torch.where((target >= 0.3) & (target < 0.4), 5.0, weights)
    weights = torch.where((target >= 0.2) & (target < 0.3), 4.0, weights)
    weights = torch.where((target >= 0.1) & (target < 0.2), 3.0, weights)
    weights = torch.where((target >= 0.0) & (target < 0.1), 9.0, weights)
    weights = torch.where((target < 0.0), 1.0, weights)
    
    # Compute weighted MSE loss
    loss = weights * (output - target)**2
    return torch.mean(loss)

if __name__=='__main__':
    # Example of how to use the custom layer
    input_size = 10   # Number of features per time series
    hidden_size = 5   # Number of hidden units in LSTM
    num_layers = 1    # Number of LSTM layers
    num_samples = 50   # Number of samples (N)
    sequence_length = 20  # Length of the time series sequence
    output_size = 1

    # Random input data: (batch_size, sequence_length, num_samples, input_size)
    time_series_data = torch.randn(64, sequence_length, num_samples, input_size, output_size)  # Batch size = 5

    # Instantiate the custom layer
    custom_lstm_layer = SingleSampleNN(input_size, hidden_size, num_layers, num_samples)

    # Forward pass through the custom layer
    output = custom_lstm_layer(time_series_data)
    print(output.shape)  # Output shape will be (5, 3, 1) if num_samples = 3
    ipdb.set_trace()