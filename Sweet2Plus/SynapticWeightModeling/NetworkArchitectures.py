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
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

if __name__ == '__main__':
    # Create the full model instance
    sequence_length = 8
    num_neurons = 5
    
    G = nx.DiGraph()

    num_neurons = 5  # Example number of neurons
    sequence_length = 10  # Example sequence length

    # Add nodes and edges to the graph
    for j in range(num_neurons):
        # Create LSTM nodes for each time step (layer)
        for i in range(sequence_length):
            G.add_node(f"t={i+1}-{j+1}", layer="t", neuron=f"Neuron {j+1} Time {i+1}")

        # Connect each LSTM node to the corresponding neuron in the output layer (no mixing)
        G.add_node(f"Neuron t summary {j+1}", layer="FC", neuron=f"Neuron t summary {j+1}")  # Add FC layer for each neuron
        for i in range(sequence_length):
            G.add_edge(f"t={i+1}-{j+1}", f"Neuron t summary {j+1}")  # Connect LSTM nodes to their respective FC neuron

    # Add a new fully connected (FC) layer to the right of the current FC layer
    for j in range(num_neurons):
        G.add_node(f"Predicted Activity {j+1}", layer="FC2", neuron=f"FC2 Neuron {j+1}")  # New FC layer neurons
        for k in range(num_neurons):
            # Connect each neuron from the first FC layer to each new FC neuron
            G.add_edge(f"Neuron t summary {k+1}", f"Predicted Activity {j+1}", weight=f"W{k+1}-{j+1}")  # Label the edge with weight name

    # Define custom positions for each node
    positions = {}

    # t nodes in a grid layout, each row for a neuron, each column for time steps
    y_step = 1  # Step size for t neurons along the y-axis, adjust to make them closer
    for j in range(num_neurons):
        for i in range(sequence_length):
            # Set x to be the time step (i), and y to be the row for the neuron (j)
            positions[f"t={i+1}-{j+1}"] = (i+9, -(j * y_step))  # Position each t node by its time and neuron

    # FC layers positioned to the right of all LSTM nodes (at x = sequence_length)
    y_offset = 0.5  # Small offset to move the FC neurons down
    for j in range(num_neurons):
        positions[f"Neuron t summary {j+1}"] = (sequence_length + 10, -j - y_offset)  # Slight downward shift for FC neurons

    # FC2 layer positioned further to the right of FC layers (at x = sequence_length + 4)
    y_offset_fc2 = 0.5  # Small offset to move the FC2 neurons down
    for j in range(num_neurons):
        positions[f"Predicted Activity {j+1}"] = (sequence_length + 13, -j - y_offset_fc2)  # Slight downward shift for FC2 neurons

    # Color the nodes by layer
    node_colors = []
    for node in G.nodes:
        layer = G.nodes[node].get('layer')
        if layer == "t":
            node_colors.append('#4A6D89')  # Soft Purple
        elif layer == "FC":
            node_colors.append('#2980B9')  # Bright Teal
        elif layer == "FC2":
            node_colors.append('#7FB3D5') 

    # Create custom labels using f-strings (subset numbers in the node labels)
    labels = {}
    for node in G.nodes:
        if "t=" in node:
            # Label the LSTM nodes as t=i
            i = node.split('=')[1].split('-')[0]
            labels[node] = f"Neuron \u0394F \n at t={i}"  # This will be "t=1", "t=2", ..., "t=10"
        elif "Neuron" in node:
            # Label the FC nodes as Neuron J
            j = node.split(' ')[-1]
            labels[node] = f"Neuron {j} \n \u0394F summary"  # This will be "Neuron 1", "Neuron 2", ..., "Neuron 5"
        elif "Predicted Activity" in node:
            # Label the FC2 nodes (could be adjusted similarly if needed)
            j = node.split(" ")[-1]
            labels[node] = f"Predicted Activity {j}"
        else:
            labels[node] = node  # Keep other labels as is

    # Visualize the graph using NetworkX and Matplotlib
    plt.figure(figsize=(14, 8))
    # Set smaller node size for LSTM nodes (adjusting node size for LSTM layer)
    node_sizes = [4000 if 't=' in node else 9000 for node in G.nodes]  # Smaller nodes for LSTM
    nx.draw(G, pos=positions, with_labels=True, labels=labels, node_color=node_colors, node_size=node_sizes, font_size=10, font_weight='bold', edge_color='gray')   

    # Add arrows manually after the graph is drawn
    for j in range(num_neurons):
        # Position of the FC2 neuron and the corresponding position for the arrow
        fc2_pos = positions[f"Predicted Activity {j+1}"]
        
        # Draw arrows from FC2 to the right (horizontal direction)
        arrow2 = FancyArrowPatch(fc2_pos, (fc2_pos[0] + 1.8, fc2_pos[1]), mutation_scale=15, color='gray', arrowstyle='->')
        plt.gca().add_patch(arrow2)

    plt.savefig('simpleNN.jpg')