#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: coders.py
Description: Contains classes for decoding neuronal activity. 
Author: David Estrin
Version: 1.0
Date: 12-06-2024

Analysis 1:
    What are we trying to test? -- We are trying to determine what percent of neurons encode information regarding each stimulus...
    What is the hypothesis? -- I hypothesize that most neurons encode information regarding the TMT stimulus 
    How should we graph the data? -- We should graph the trial type in x axis and decoder accuracy +/- sem in y axis. 
        We should also seperate the data based on Session and Group. 
    
Analysis 2:
    What are we trying to test? -- We are trying to determine whether clustered neurons do in fact encode different aspects of task
    What is the hypothesis? -- I hypothesize that different clusters activity decode better/worse certain stimuli than others
    How should we graph the data? -- 

"""
from Sweet2Plus.statistics.heatmaps import heatmap
from Sweet2Plus.statistics.coefficient_clustering import cli_parser, gather_data
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import optuna
import ipdb

class format_data(heatmap):
    def __init__(self, drop_directory, neuronal_activity, behavioral_timestamps, neuron_info, 
                 trial_list=['Vanilla', 'PeanutButter', 'Water', 'FoxUrine'],
                 normalize_neural_activity=False, regression_type='ridge', 
                 hyp_batch_size=64, preprocessed = None):
        super().__init__(drop_directory, neuronal_activity, behavioral_timestamps, neuron_info,
                         trial_list, normalize_neural_activity, regression_type)
        
        self.batch_size = hyp_batch_size
        self.preprocessed = preprocessed
        ipdb.set_trace()

    def __call__(self):
        if self.preprocessed:
            # Load data from .npy files
            ipdb.set_trace()
            X_path, y_path = self.preprocessed
            self.X_original = np.load(X_path)
            self.y_one_hot = np.load(y_path)
        else:
            super().__call__()

        X_train, X_test, y_train, y_test = self.clean_and_split_data()
        self.torch_loader(X_train, X_test, y_train, y_test)

    def clean_and_split_data(self):
        # Shuffle the data
        indices = np.arange(self.X_original.shape[0])
        np.random.shuffle(indices)
        self.X, y_one_hot = self.X_original[indices], self.y_one_hot[indices]

        # Convert one hot to arg max
        self.y = np.argmax(y_one_hot, axis=1)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def torch_loader(self, X_train, X_test, y_train, y_test):
        """ Put numpy arrays into torch's data loader format """
        X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))

        training_dataset = TensorDataset(X_train.float(), y_train.long())
        testing_dataset = TensorDataset(X_test.float(), y_test.long())

        self.train_loader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(testing_dataset, batch_size=self.batch_size)
        
class NeuronalActivity_Encoder_Decoder(nn.Module):
    def __init__(self, input_size, hyp_middle_dimension, output_size):
        super(NeuronalActivity_Encoder_Decoder, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(input_size, 32), 
                                     nn.ReLU(), 
                                     nn.Linear(32, hyp_middle_dimension))
        
        self.decoder = nn.Sequential(nn.Linear(hyp_middle_dimension, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, output_size))

    def forward(self, x):
        return self.decoder(self.encoder(x))

class NeuralNetworkFigures():
    def __init__(self,name):
        self.name = name
    
    def plot_learning_curve(self,data,label,outputfile):
        plt.figure()
        plt.plot(np.asarray(data),label=label)
        plt.title(f"Current {label} results")
        plt.savefig(outputfile)

class Education:
    def __init__(self, data_obj, neural_network_obj, figures_obj, hyp_total_epochs = 100, hyp_learning_rate = 0.001, hyp_gamma=0.5, show_results = 10):
        self.total_epochs = hyp_total_epochs
        self.learning_rate = hyp_learning_rate
        self.gamma = hyp_gamma
        self.show_results = show_results
        self.model_oh = neural_network_obj 
        self.train_loader = data_obj.train_loader 
        self.test_loader = data_obj.test_loader  
        self.drop_directory = os.path.join(data_obj.drop_directory,"neural_network_encoder_results/")

        # Generate path if not real
        if not os.path.exists(self.drop_directory):
            os.makedirs(self.drop_directory, exist_ok=True)

        self.training_loss = []
        self.training_f1 = []
        self.testing_f1 = None
        self.figures = figures_obj

    def __call__(self):
        average_train_f1 = self.training()
        average_test_f1 = self.testing()
        return average_train_f1, average_test_f1

    def training(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_oh.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)  

        for epoch in range(self.total_epochs):
            self.model_oh.train()  # Set model to training mode
            total_loss = 0
            all_preds = []
            all_labels = []

            for batch_X, batch_y in self.train_loader:
                optimizer.zero_grad()
                res = self.model_oh(batch_X)  # Forward pass
                loss = criterion(res, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                scheduler.step()

                total_loss += loss.item()
                all_preds.extend(torch.argmax(res, dim=1).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

            # Calculate average loss and F1 score for the epoch
            avg_loss = total_loss / len(self.train_loader)
            avg_f1 = f1_score(all_labels, all_preds, average='macro')
            
            # Store stats for graphing
            self.training_loss.append(avg_loss)
            self.training_f1.append(avg_f1)

            # Print stats every M epochs
            if (epoch + 1) % self.show_results == 0:
                print(f"Epoch {epoch + 1}/{self.total_epochs}, Loss: {avg_loss:.4f}, F1 Score: {avg_f1:.4f}")
                self.figures.plot_learning_curve(data = self.training_loss, 
                                                 label = "training_loss", 
                                                 output_file = os.path.join(self.drop_directory,"training_loss.jpg"))
                
                self.figures.plot_learning_curve(data = self.training_f1, 
                                                 label = "training_f1",
                                                 output_file = os.path.join(self.drop_directory,"training_f1.jpg"))
        return np.mean(self.training_f1)

    def testing(self):
        self.model_oh.eval()  # Set model to evaluation mode
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                res = self.model_oh(batch_X)  # Forward pass
                all_preds.extend(torch.argmax(res, dim=1).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # Calculate final F1 score
        self.testing_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Final Testing F1 Score: {self.testing_f1:.4f}")

        return self.testing_f1
    
def hyperparameter_search_wrapper(data_directory, drop_directory, ntrials=1000):
    """ Utalizes optuna to find the best hyperparmeter results """
    def objective(trial):
        hyp_learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
        hyp_gamma = trial.suggest_float("gamma", 0.1, 1.0, log=False)  # Typically between 0 and 1, no log scale
        hyp_middle_dimension = trial.suggest_int("middle_dimension", 16, 512, log=True)  # Log scale for larger ranges
        hype_batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
        hyp_total_epochs = trial.suggest_int("total_epochs", 10, 200)  # Reasonable range for training epochs


        # Preprocess and structure the data
        neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
        data_obj_oh = format_data(drop_directory=drop_directory,
                            neuronal_activity=neuronal_activity,
                            behavioral_timestamps=behavioral_timestamps,
                            neuron_info=neuron_info, 
                            hyp_batch_size = hype_batch_size)

        # Build neural network model
        current_model_oh = NeuronalActivity_Encoder_Decoder(input_size=46, hyp_middle_dimension = hyp_middle_dimension, output_size=4)

        # Build figures object
        figures_object_oh = NeuralNetworkFigures()

        # Build education pipeline
        education_obj = Education(data_obj = data_obj_oh,
                neural_network_obj = current_model_oh, 
                figures_obj = figures_object_oh,
                hyp_total_epochs = hyp_total_epochs, 
                hyp_learning_rate = hyp_learning_rate, 
                hyp_gamma = hyp_gamma, 
                show_results = int(np.round(hyp_total_epochs/10)))
        print(f"Results will be showed every {int(np.round(hyp_total_epochs/10))} epochs")
        
        average_train_f1 , average_test_f1 = education_obj()

        return average_test_f1
    
    study = optuna.create_study(direction="maximize")  # We want to minimize the metric
    study.optimize(objective, n_trials=ntrials, n_jobs=-1, show_progress_bar=True)  # Run optimization for 100 trials

    # Get the best parameters from the optimization
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best value: {best_trial.value}")
    print(f"Best parameters: {best_trial.params}")
    ipdb.set_trace()

if __name__=='__main__':
    # Get data paths
    data_directory, drop_directory = cli_parser()

    # Hard coded for now
    tune_hyperparameters = False

    # Search hyper parmeter space for best model or just run a single model
    if tune_hyperparameters:
        hyperparameter_search_wrapper(data_directory=data_directory,drop_directory=drop_directory)
    
    else:
        # Preprocess and structure the data
        # neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
        preprossesed_oh = (os.path.join(drop_directory,"X_original.npy"),os.path.join(drop_directory,"y_one_hot.npy"))
        data_obj_oh = format_data(drop_directory=drop_directory,
                            neuronal_activity=None,
                            behavioral_timestamps=None,
                            neuron_info=None,
                            preprocessed=preprossesed_oh)

        # Build neural network model
        ipdb.set_trace()
        # Need to set up automatic inputs and outputs for the model
        current_model_oh = NeuronalActivity_Encoder_Decoder(input_size=46, hyp_middle_dimension = 16, output_size=4)

        # Build figures object
        figures_object_oh = NeuralNetworkFigures()

        # Build education pipeline
        education_obj = Education(data_obj = data_obj_oh,
                neural_network_obj = current_model_oh, 
                figures_obj = figures_object_oh)
            
        education_obj()
