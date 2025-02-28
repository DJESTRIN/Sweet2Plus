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
from Sweet2Plus.modeling.network_archs import SingleSampleNN, weighted_mse_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
import random
import json 
import ipdb

# Custom functions and classes 
def zscore_data_from_obj(obj_file_oh):
    """ Load in ztrace data from Sweet2Plus obj file """
    # Pull z scored neuronal trace data from object and load into numpy array
    objoh = LoadObj(FullPath=obj_file_oh)
    return np.asarray(objoh.ztraces).T

class datawrangler(torch.utils.data.Dataset):
    """ datawrangler 
    Puts neuronal activity data into a torch dataset format    
    """
    def __init__(self,z_scored_neural_data,test_size=0.2,random_state=43,X_points=100):
        self.zdata=z_scored_neural_data
        self.test_size=test_size
        self.random_state=random_state
        self.X_points=X_points
        self.normalize_data()
        self.X,self.y=self.rearrange_data()

    def __call__(self):
        X_train, y_train, X_test, y_test = self.split()
        return X_train, y_train, X_test, y_test
    
    def normalize_data(self):
        zdatanorm=[]
        for trace in self.zdata.T:
            neuron_data = (trace-trace.min())/(trace.max()-trace.min())
            zdatanorm.append(neuron_data)
        self.zdata=np.array(zdatanorm).T

    def rearrange_data(self):
        X, y = [], []
        # Iterate over the data with a sliding window
        for i in range(len(self.zdata) - (self.X_points + 1)):
            # Select X_points for input and the subsequent value for output
            stackoh = np.vstack(self.zdata[i:(i + self.X_points)])
            target = self.zdata[i + self.X_points + 1]
            X.append(stackoh)
            y.append(target)
        return X, y
        
    def get_original_data(self):
        return self.X, self.y
    
    def split(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.random_state)
        return X_train, y_train, X_test, y_test

class Education():
    """ Builds pipeline needed to train network """
    def __init__(self, data, model=[], num_data_points=300, epochs=100, print_training=True,
                 print_training_epoch=1, plot_neurons=False, run_study=False, device='cuda', 
                 study_trials = 50, learning_rate=0.001, weight_decay = 0, drop_directory=r'C:\Users\listo\Sweet2Plus\my_figs'):
        # Set initial attributes
        self.data = data
        self.model = model
        self.num_data_points = num_data_points
        self.device = device
        self.epochs = epochs
        self.print_training_epoch = print_training_epoch
        self.print_training = print_training
        self.plot_neurons = plot_neurons
        self.neuron_drop_directory = os.path.join(drop_directory,r'traces/')
        self.drop_directory=drop_directory
        self.run_study = run_study
        self.learning_rate=learning_rate
        self.study_trials = study_trials
        self.weight_decay = weight_decay

    def __call__(self):
        if self.run_study:
            print("Running a hyperparameter study via optuna ... ")
            print(" Relax, this might take a while ... ")
            self.epochs = 20 # Set to a low number just for study

            # Run optuna study and get best hyperparameters
            self.hypertuning_study()
            
            # run training
            print("Converting wrangled data into torch format ... ")
            X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(self.num_data_points)
            train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)
            
            print("Setting up loss function, optimizer and scheduler ... ")
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
 
            print("Starting training on model ... ")
            model, loss_history = self.train(self.model, train_loader, val_loader, self.criterion, self.optimizer, self.scheduler)
            
            print('Plotting loss results')
            self.plot_learning_curve(data=loss_history, filename=os.path.join(self.drop_directory,'best_hyper_loss_curve.jpg'))
            
            # Run testing and get correlation between signals
            print('Testing model ....')
            corroh = self.test(model_oh, original_loader)
            print(f'The average correlation of prediction to real data for all neurons is {corroh}')

            # Save model for later use
            print('Saving model to file ...')
            torch.save(model_oh, os.path.join(self.drop_directory,"best_tuned_model.pth"))
        
        else:
            print('Running single model with given learning rates and Xpoints')
            print('Setting up data structure into torch dataset and model parameters')
            # Set up datasets
            X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(num_data_points=self.num_data_points)
            train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)

            # Build model            
            model_oh = SingleSampleNN(sequence_length = np.array(X_train).shape[1], 
                                  hidden_size = 64, 
                                  num_layers = 3, 
                                  num_neurons = np.array(X_train).shape[2], 
                                  output_size = np.array(X_train).shape[2])
            
            # Set criterion and loss
            criterion_oh = nn.MSELoss()
            self.optimizer = optim.Adam(model_oh.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            # Run training and get loss history
            print('Training model ....')
            trained_model, history = self.train(model_oh, train_loader, val_loader, criterion_oh, self.optimizer, self.scheduler)

            print('Plotting loss results')
            self.plot_learning_curve(data=history, filename=os.path.join(self.drop_directory,"FinalLossCurve.jpg"))
            
            # Run testing and get correlation between signals
            print('Testing model ....')
            corroh = self.test(model_oh, original_loader)
            print(f'The average correlation of prediction to real data for all neurons is {corroh}')

            # Save model for later use
            print('Saving model to file ...')
            torch.save(model_oh, os.path.join(self.drop_directory,"solo_model.pth"))
        return 

    def get_data(self,num_data_points):
        arranged_data = datawrangler(z_scored_neural_data=self.data,X_points=num_data_points)
        X_train, y_train, X_test, y_test = arranged_data()
        X_original, y_original=arranged_data.get_original_data()
        return X_train, y_train, X_test, y_test, X_original, y_original
    
    def convert_to_torch_loader(self,X_train, y_train, X_test, y_test, X_original, y_original):
        # Put data into torch formatting
        X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32)
        y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32)
        xo = torch.tensor(np.asarray(X_original), dtype=torch.float32)
        yo = torch.tensor(np.asarray(y_original), dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        od = TensorDataset(xo, yo)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        original_loader = DataLoader(od, batch_size=1, shuffle=False)
        return train_loader, val_loader, original_loader
    
    def train(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        """ Run training on given model """
        model.to(self.device)
        loss_history = {'train': [], 'val': []}

        # Train across epochs
        for epoch in range(self.epochs):
            model.train()

            # Create empty variables 
            total_train_loss = 0.0
            num_batches_train = 0
            for inputs, targets in train_loader:
                inputs = inputs.transpose(1, 2).contiguous() # Reconfigure input shape
                inputs, targets = inputs.to(self.device), targets.to(self.device) # Send data to gpu
                
                # Run training
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Calculate total loss
                total_train_loss += loss.item()  
                num_batches_train += 1 

                # Backprop 
                loss.backward()
                optimizer.step()
            
            # Get current lr
            current_lr = optimizer.param_groups[0]['lr']

            # Update learning rate via scheduler
            avg_train_loss = total_train_loss / num_batches_train
            scheduler.step(avg_train_loss)

            # Update training loss
            loss_history['train'].append(avg_train_loss)

            # Validation phase
            model.eval()
            total_validation_loss = 0.0
            num_batches_validation = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.transpose(1, 2).contiguous() # Reconfigure input shape
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    total_validation_loss += loss.item()  
                    num_batches_validation += 1 

            avg_validation_loss = total_validation_loss / num_batches_validation
            loss_history['val'].append(avg_validation_loss)

            if self.print_training:
                if epoch%self.print_training_epoch==0:
                    print(f"Epoch {epoch}/{self.epochs}, LR: {current_lr}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_validation_loss:.4f}")
                    self.plot_learning_curve(data = loss_history, filename=os.path.join(self.drop_directory,"current_training_loss_results.jpg"))
       
        return model, loss_history

    def test(self, modeloh, test_loader):
        real_data=[]
        predict_data=[]
        for inputs, targets in test_loader:
            inputs = inputs.transpose(1, 2).contiguous() # Reconfigure input shape
            inputs = inputs.to(self.device)
            outputs = modeloh(inputs)
            outputs_numpy = outputs.cpu().detach().numpy()
            targets=targets.detach().numpy()
            real_data.append(targets)
            predict_data.append(outputs_numpy)

        real_data=np.squeeze(np.asarray(real_data)).T
        predict_data=np.squeeze(np.asarray(predict_data)).T
        
        all_correlations=[]
        for r,p in zip(real_data,predict_data):
            all_correlations.append(np.corrcoef(r, p)[0, 1])
        correlation = np.asarray(all_correlations).mean()

        if self.plot_neurons:
            for neuron_id in range(len(real_data)):
                # Generate output file path
                filepath = os.path.join(self.neuron_drop_directory,f"neuron{neuron_id}.jpg")

                # Create figure and save
                plt.figure(figsize=(10,10))
                plt.plot(real_data[neuron_id,:], linewidth=3, alpha=0.5, label='Real Output')
                plt.plot(predict_data[neuron_id,:], linewidth=3, alpha=0.5, label='Model Output')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Normalized dF')
                plt.savefig(filepath)
                plt.close()
        
        return correlation
    
    def plot_learning_curve(self,data,filename):
        plt.figure()
        plt.plot(data['train'])
        plt.plot(data['val'])
        plt.savefig(filename)
        plt.close()

    def hypertuning_study(self):
        """Utalize Optuna to run a study to find best hyperparameters """
        def objective(trial):
            # Hyperparameter suggestions
            learning_rate = trial.suggest_loguniform('learning_rate', 0.0001, 0.01)
            weight_decay = trial.suggest_float('weight_decay', 0, 1e-2)
            num_data_points = trial.suggest_int('num_data_points', 2, 300)
            hidden_suggested = trial.suggest_int('hidden_suggested', 4, 128)
            layers_suggested = trial.suggest_int('layers_suggested', 1, 5)

            # Get neural data from file and seperate into X_train, y_train, etc
            X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(num_data_points=num_data_points)
            train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)
            
            # Build model
            model_oh = SingleSampleNN(sequence_length = np.array(X_train).shape[1], 
                                  hidden_size = hidden_suggested, 
                                  num_layers = layers_suggested, 
                                  num_neurons = np.array(X_train).shape[2], 
                                  output_size = np.array(X_train).shape[2])

            # Set up criterion and optimizers
            criterion_oh = nn.MSELoss()
            optimizer = optim.Adam(model_oh.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            # Run training and get loss history
            trained_model, history = self.train(model_oh, train_loader, val_loader, criterion_oh, optimizer, scheduler)

            # Run testing and get correlation between signals
            corroh = self.test(model_oh, original_loader)

            # Maximize correlation
            return corroh

        # Run Optuna study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.study_trials, n_jobs=-1)

        best_hyperparameters = self.study.best_params
        self.learning_rate = best_hyperparameters['learning_rate']
        self.weight_decay = best_hyperparameters['weight_decay']
        self.hidden_suggested = best_hyperparameters['hidden_suggested']
        self.layers_suggested = best_hyperparameters['layers_suggested']
        self.num_data_points = best_hyperparameters['num_data_points']

        # Save hyperparameter data to a file
        best_hyperparameters = {"learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "hidden_suggested": self.hidden_suggested,
                "layers_suggested": self.layers_suggested,
                "num_data_points":self.num_data_points}

        with open(os.path.join(self.drop_directory,"best_hyperparameters.json"), "w") as f:
            json.dump(best_hyperparameters, f)

        print(f"The best hyperparmeters are: \n learning rate: {self.learning_rate} \n weight decay: {self.weight_decay} \n datapoints: {self.num_data_points}, hidden: {self.hidden_suggested}, \n layers: {self.layers_suggested}")
        return 

class capture_model_weights():
    def __init__(self, torch_file, drop_directory):
        # Set up attributes
        self.torch_file = torch_file
        self.drop_directory = drop_directory

        # Load in model and set to evaludation
        model = torch.load(self.torch_file)
        model.eval()

        # Get all weights
        self.all_weights = model.state_dict()

        # Get weights and biases of the fully connected layer
        self.connected_weights = model.final_fc.weight 
        self.connected_bias = model.final_fc.bias  

    def generate_heatmap(self):
        plt.figure(figsize=(20,20))
        plt.imshow(self.connected_weights, vmax=0.7, vmin=-0.7,cmap="seismic",interpolation='none')
        plt.colorbar()
        plt.xlabel('Neurons')
        plt.ylabel('Neurons')
        plt.title('Weight values')
        plt.savefig(os.path.join(self.drop_directory,"weightheatmap.jpg"),dpi=300, bbox_inches='tight')
        plt.close()

def synthetic_data(model, torch_data_loader, pseudo_time_points=4500):
    """ synthetic_data -- Generate synthetic data from neurons in dataset
    Inputs
        model -- a torch model that was already trained
        torch_data_loader -- a dataset converted to torch formating
        pseudo_time_points -- number of synthetic data points to be generated. 
    
    Outputs
        synthetic_data_oh -- outputs a M by N array containing synthetic data generated by model. This is a numpy array
    """
    random_index = random.randit(0,len(torch_data_loader))
    random_data = torch_data_loader[random_index]
    random_index = random.randit(0,len(random_data))
    activity_series = torch_data_loader[random_index]

    synthetic_data_oh = []
    for idx in range(pseudo_time_points):
        output = model(activity_series) 

        # Update activity series data
        activity_series = activity_series[:-1,:]
        activity_series[len(activity_series)+1,:] = output 

        # Append output to pseudo_activity_data
        synthetic_data_oh.append(output)
    
    # Convert data to numpy
    synthetic_data_oh = np.asarray(synthetic_data_oh)
    return synthetic_data_oh

def cli_parser():
    # Get command line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument('--s2p_object_file',type=str,help='Directory to a custom s2p object')
    parser.add_argument('--hypertuning_study',action='store_true',help='Run hyperparameter study with optuna')
    parser.add_argument('--full_model_path',type=str,help='Directory to pre-trained model')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = cli_parser()
    zdataoh = zscore_data_from_obj(args.s2p_object_file)
    objoh = Education(zdataoh, model=None, plot_neurons=True, run_study=args.hypertuning_study)
    result_oh = objoh()
