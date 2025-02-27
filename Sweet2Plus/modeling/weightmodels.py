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
import ipdb

""" Data wrangling and set up functions and classes """
def zscore_data_from_obj(obj_file_oh):
    # Pull z scored neuronal trace data from object and load into numpy array
    objoh = LoadObj(FullPath=obj_file_oh)
    return np.asarray(objoh.ztraces).T

def plot_loss(training_loss,validation_loss, full_output_file):
    """ Quickly plots training and validation loss for monitoring """
    if len(training_loss)<3:
        return
    else:
        plt.figure(figsize=(10,10))
        plt.plot(training_loss)
        plt.plot(validation_loss)
        plt.savefig(full_output_file)

class datawrangler(torch.utils.data.Dataset):
    """ Wrangles data from z score neuronal trace data into the correct format for ANN """
    def __init__(self,z_scored_neural_data,test_size=0.2,random_state=43,X_points=100,cheat_mode=False):
        self.zdata=z_scored_neural_data
        self.test_size=test_size
        self.random_state=random_state
        self.X_points=X_points
        self.cheat_mode=cheat_mode

        # Default methods
        self.normalize_data()
        if self.cheat_mode:
            self.X,self.y=self.get_cheating_data()
        else:
            self.X,self.y=self.rearrange_data()
    
    def normalize_data(self):
        zdatanorm=[]
        for trace in self.zdata.T:
            neuron_data = (trace-trace.min())/(trace.max()-trace.min())
            zdatanorm.append(neuron_data)
        self.zdata=np.array(zdatanorm).T

    def __call__(self):
        X_train, y_train, X_test, y_test = self.split()
        return X_train, y_train, X_test, y_test

    def rearrange_data(self):
        X, y = [], []
        
        # Iterate over the data with a sliding window
        for i in range(len(self.zdata) - (self.X_points + 1)):
            # Select X_points for input and the subsequent value for output
            stackoh = np.vstack(self.zdata[i:i + self.X_points])
            target = self.zdata[i + self.X_points + 1]

            X.append(stackoh)
            y.append(target)
        
        return X, y
    
    def get_cheating_data(self):
        # Set X and y to the exact same value
        X, y = [], []
        for i in range(len(self.zdata) - (self.X_points + 1)):
            stackoh = self.zdata[i + self.X_points] # Set X and Y to the exact same value
            target = self.zdata[i + self.X_points + 1]

            X.append(stackoh)
            y.append(target)
        return X, y
        
    def get_original_data(self):
        return self.X, self.y
    
    def split(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=self.test_size,random_state=self.random_state)
        return X_train, y_train, X_test, y_test

""" Primary analysis pipeline for training, testing and debugging ANN """
class weightmodel_pipeline():
    """ Builds pipeline needed to train network """
    def __init__(self, data, model=[], num_data_points=100, epochs=100, print_training=True,
                 print_training_epoch=1, plot_neurons=False, run_study=False, cheat_mode=False, 
                 device='cuda',learning_rate=0.002, weight_decay=1e-4, 
                 neuron_fig_path=r'C:\Users\listo\Sweet2Plus\my_figs\neuron_predictions',
                 main_fig_path=r'C:\Users\listo\Sweet2Plus\my_figs'):
        """
        Inputs:

        """
        # Set initial attributes
        self.data = data
        self.model = model
        self.num_data_points = num_data_points
        self.device = device
        self.epochs = epochs
        self.print_training_epoch = print_training_epoch
        self.print_training = print_training
        self.plot_neurons = plot_neurons
        self.neuron_fig_path = neuron_fig_path
        self.main_fig_path=main_fig_path
        self.run_study = run_study
        self.cheat_mode = cheat_mode
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay

    def __call__(self):
        """ General protocol for pipeline """
        # if not self.model and not self.run_study:
        #     user_answer=input('No model was given and run_study is false, \
        #                       would you like to run a study (yes or no)? \
        #                       If no, please re-run with model.','yes','no')

        #     if user_answer=='yes':
        #         self.run_study=True
        #     else:
        #         print('Without a model included and no study chosen, there is nothing to run. Terminating code... :(')
        #         return
        if self.model is None:
            self.simple_run()

        # Determine if NN should be run via a study to find best hyperparmeters
        if self.run_study:
            print("Running a hyperparameter study via optuna ... ")
            print(" Relax, this might take a while ... ")
           
            # Run optuna study and get best hyperparameters
            self.hypertuning_study()
            
            # run training
            print("Converting wrangled data into torch format ... ")
            X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(self.num_data_points)
            train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)
            
            print("Setting up loss function, optimizer and scheduler ... ")
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            optimizer = optim.Adam(model.parameters(), lr=0.001)  # Example optimizer
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

            print("Starting training on model ... ")
            model, loss_history = self.train(self.model, train_loader, val_loader, self.criterion, self.optimizer, self.scheduler)
            
            print('Plotting loss results')
            self.plot_learning_curve(data=loss_history, filename=self.main_fig_path)
            
            print('Testing model ...')
            coroh = self.test(self.model, original_loader)

            print(f'There was a correlation of {coroh} for this models predicted vs real data')
            return 
        
        # Run a regular model
        if not self.run_study:
            if self.cheat_mode:
                print("Running a model in cheat mode! Results should be nearly perfrect")

            print("Converting wrangled data into torch format ... ")
            X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(self.num_data_points)
            train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)
            
            print("Setting up loss function, optimizer and scheduler ... ")
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

            print("Starting training on model ... ")
            model, loss_history = self.train(self.model, train_loader, val_loader, self.criterion, self.optimizer, self.scheduler)
            
            print('Plotting loss results')
            self.plot_learning_curve(data=loss_history, filename=self.main_fig_path)
            
            print('Testing model ...')
            coroh = self.test(self.model, original_loader)

            print(f'There was a correlation of {coroh} for this models predicted vs real data')
            return

    def simple_run(self):
        learning_rate = 0.001
        # weight_decay = 1e-4
        # dropout_rate = 0.1
        numberXpoints = 300

        # Get neural data from file and seperate into X_train, y_train, etc
        X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(num_data_points=numberXpoints)
        train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)
        
        # Build model
        model_oh = SingleSampleNN(sequence_length = np.array(X_train).shape[1], 
                                  hidden_size = 64, 
                                  num_layers = 3, 
                                  num_neurons = np.array(X_train).shape[2], 
                                  output_size = np.array(X_train).shape[2])

        # Set up criterion and optimizers
        criterion_oh = nn.MSELoss()
        self.optimizer = optim.Adam(model_oh.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Run training and get loss history
        trained_model, history = self.train(model_oh, train_loader, val_loader, criterion_oh, self.optimizer, self.scheduler)

        # Run testing and get correlation between signals
        corroh = self.test(model_oh, original_loader)
        ipdb.set_trace()

        weights = model_oh.state_dict()
        for param_name, param_tensor in weights.items():
            print(f"Layer: {param_name} | Shape: {param_tensor.shape}")


    def get_data(self,num_data_points):
        arranged_data = datawrangler(z_scored_neural_data=self.data,X_points=num_data_points,cheat_mode=self.cheat_mode)
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
        # Send model to gpu
        model.to(self.device)

        # Record training and validation loss
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
                    plot_loss(loss_history['train'],loss_history['val'], os.path.join(self.main_fig_path,"current_training_loss_results.jpg"))

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
                plt.figure(figsize=(10,10))
                plt.plot(real_data[neuron_id,:], linewidth=3, alpha=0.5, label='Real Output')
                plt.plot(predict_data[neuron_id,:], linewidth=3, alpha=0.5, label='Model Output')
                plt.xlabel('time')
                plt.ylabel('Z_F')
                filepath = os.path.join(self.neuron_fig_path,f"neuron{neuron_id}.jpg")
                plt.savefig(filepath)
                plt.close()
        
        return correlation
    
    def generate_pseudo_data(self, modeloh, test_loader, pseudo_frames = 4500):
        ipdb.set_trace()
        random_index = random.randit(0,len(test_loader))
        random_data = test_loader[random_index]
        random_index = random.randit(0,len(random_data))
        activity_series = test_loader[random_index]

        pseudo_activity_data = []
        for idx in range(pseudo_frames):
            output = self.modeloh(activity_series) 

            # Update activity series data
            activity_series = activity_series[:-1,:]
            activity_series[len(activity_series)+1,:] = output 

            # Append output to pseudo_activity_data
            pseudo_activity_data.append(output)

        ipdb.set_trace()


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
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-1)
            weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
            dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
            numberXpoints = trial.suggest_int('numberXpoints', 2, 300)

            # Get neural data from file and seperate into X_train, y_train, etc
            X_train, y_train, X_test, y_test, X_original, y_original = self.get_data(num_data_points=numberXpoints)
            train_loader, val_loader, original_loader = self.convert_to_torch_loader(X_train, y_train, X_test, y_test, X_original, y_original)
            
            # Build model
            model_oh = SingleSampleNN(input_size=np.array(X_train).shape[2],inputs_per_neuron=np.array(X_train).shape[1],dropout=dropout_rate)

            # Set up criterion and optimizers
            criterion_oh = nn.MSELoss()
            optimizer = optim.Adam(model_oh.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            # Run training and get loss history
            trained_model, history = self.train(model_oh, train_loader, val_loader, criterion_oh, optimizer, scheduler)

            # Run testing and get correlation between signals
            corroh = self.test(model_oh, original_loader)

            # Maximize correlation
            return corroh

        # Run Optuna study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=30, n_jobs=1)

        best_hyperparameters = self.study.best_params
        self.learning_rate = best_hyperparameters['learning_rate']
        self.weight_decay = best_hyperparameters['weight_decay']
        self.dropout_rate = best_hyperparameters['dropout_rate']
        self.num_data_points = best_hyperparameters['numberXpoints']
        print(f"The best hyperparmeters are, \
               lr: {self.learning_rate}, \
               wd {self.weight_decay}, \
               drop: {self.dropout_rate}, dataponts: {self.num_data_points}")
        return 

def cli_parser():
    # Get command line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument('--s2p_object_file',type=str,help='Directory to a custom s2p object')
    parser.add_argument('--cheat_mode',action='store_true',help='When included, model will be given easy data that it should be able to get 100 accuracy')
    parser.add_argument('--hypertuning_study',action='store_true',help='Run hyperparameter study with optuna')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = cli_parser()
    zdataoh = zscore_data_from_obj(args.s2p_object_file)
    objoh = weightmodel_pipeline(zdataoh, model=None, plot_neurons=True, run_study=False, cheat_mode=False)
    ans=objoh()