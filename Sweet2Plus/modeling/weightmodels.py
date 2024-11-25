#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: weightmodels.py
Description: Model weights 
Author: David James Estrin
Version: 1.0
Date: 11-25-2024
"""

"""
purpose: We are interested in determining how synaptic weights change across stress. 
Validations:
(1) For this shallow network (N input => N output), can it predict neural activity with good MSE.

Questions:
(1) Is there a difference in weight values across stress and non-stress groups on day 30 vs day 1. 
(2) If there is a difference in weight values, is there a feature regarding these neuron's activities that make them distinct?
(3) What is the distribution of theta weights across input neurons? Do certain neurons have more low theta weights than others? 
    Or does every neuron have at least one high theta weight
    (3) Can low or high theta inputs be 
(4) How is bias effected during all this for each input neuron?

# Psuedocode
(1) Arrange data
    timepoint x neurons.. network is trained on only one mouse at a time... New network for new mouse... 
    neural acitivity Y[t] = leakyReLu((W*X[t-1])+b)
    Maybe it should only be one neuron at a time. Cannot use previous neuron's activity ?
(2) Build NN
(3) Train neural network
(4) Plotting code to show predicted vs reality for randomly sampled neurons
(5) Plot weights across neurons across conditions ...
"""
# Import dependencies
import argparse
from Sweet2Plus.core.SaveLoadObjs import LoadObj
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import ipdb

# Custom functions and classes
def zscore_data_from_obj(obj_file_oh):
    objoh = LoadObj(FullPath=obj_file_oh)
    return objoh.recording

class datawrangler(torch.utils.data.Dataset):
    def __init__(self,z_scored_neural_data,test_size=0.2,random_state=43):
        self.zdata=z_scored_neural_data
        self.test_size=test_size
        self.random_state=random_state

        # Default methods
        self.X,self.y=self.rearrange_data()
    
    def __call__(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.split()
        return X_train, y_train, X_val, y_val, X_test, y_test

    def rearrange_data(self):
        X,y=[],[]
        for t1,t2 in zip(self.zdata[:-1],self.zdata[1:]):
            X.append(t1), y.append(t2)
        return X, y
    
    def split(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.random_state)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=self.test_size,random_state=self.random_state)
        return X_train, y_train, X_val, y_val, X_test, y_test

class SingleLayerNetwork(torch.nn.Module):
    def __init__(self, network_size):
        super(SingleLayerNetwork, self).__init__()
        self.lin = torch.nn.Linear(in_features=network_size, out_features=network_size)  
        self.relu=torch.nn.ReLU()
    
    def forward(self,xoh):
        return self.relu(self.lin(xoh))

def training(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda"):
    # Move the model to the specified device
    model.to(device)

    # Store loss history
    loss_history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_loss += loss.item() * inputs.size(0)

        # Average training loss
        train_loss /= len(train_loader.dataset)
        loss_history['train'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, targets)

                # Accumulate validation loss
                val_loss += loss.item() * inputs.size(0)

        # Average validation loss
        val_loss /= len(val_loader.dataset)
        loss_history['val'].append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, loss_history

def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--s2p_object_file',type=str,help='Directory to a custom s2p object')
    args = parser.parse_args()
    return args

def main():
    # Parse command line inputs
    args = cli_parser() 

    # Get neural data from file and seperate into X_train, y_train, etc
    zscore_data = zscore_data_from_obj(args.s2p_object_file) # Get z score data from file
    arranged_data = datawrangler(z_scored_neural_data=zscore_data)
    X_train, y_train, X_val, y_val, X_test, y_test = arranged_data()

    # Put data into torch formatting
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Build model
    number_of_neurons = len(X_train) # Need to fix this
    model_oh = SingleLayerNetwork(network_size=number_of_neurons)

    # Set up criterion and optimizers
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_oh.parameters(), lr=0.001)

    # Run training
    trained_model, history = training(model_oh, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda")



if __name__=='__main__':
    main()