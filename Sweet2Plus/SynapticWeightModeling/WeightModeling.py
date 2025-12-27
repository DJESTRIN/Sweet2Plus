#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: weightmodels.py
Description: The purpose of this code is to study the merrit of synthetic weights developed by a NN which are used to map 
    prior neuron activity to current neuron activity. Weights will be studied w.r.t various experimental conditions to determine
    whether certain conditions modify weight values. Additionally, weights will be analyzed to determine whether there is a correlation
    between neuronal activity and weight distribution. 
Author: David James Estrin
Version: 2.0
Date: 02-27-2025
"""

# Import dependencies
import argparse
from Sweet2Plus.core.SaveLoadObjs import LoadObj
from Sweet2Plus.SynapticWeightModeling.NetworkArchitectures import SingleSampleNN, weighted_mse_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import optuna
import os
import scipy
import random
import json 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Custom functions and classes 
def zscore_data_from_obj(obj_file_oh):
    """ Load in ztrace data from Sweet2Plus obj file """
    # Pull z scored neuronal trace data from object and load into numpy array
    objoh = LoadObj(FullPath=obj_file_oh)
    return np.asarray(objoh.ztraces).T

class datawrangler(torch.utils.data.Dataset):
    """
    Prepares neural time-series data into (X, y) pairs.
    """

    def __init__(self, z_scored_neural_data, X_points=100):
        self.zdata = z_scored_neural_data
        self.X_points = X_points
        self.normalize_data()
        self.X, self.y = self.rearrange_data()

    def normalize_data(self):
        zdatanorm = []
        for trace in self.zdata.T:
            neuron_data = (trace - trace.min()) / (trace.max() - trace.min())
            zdatanorm.append(neuron_data)
        self.zdata = np.array(zdatanorm).T

    def rearrange_data(self):
        X, y = [], []
        for i in range(len(self.zdata) - (self.X_points + 1)):
            X.append(self.zdata[i:i + self.X_points])
            y.append(self.zdata[i + self.X_points + 1])
        return np.array(X), np.array(y)

    def __call__(self):
        return self.X, self.y


class Education():
    """ Builds pipeline needed to train network """
    def __init__(self, data, model=[], num_data_points=300, epochs=100, print_training=True,
                 print_training_epoch=1, plot_neurons=False, run_study=False, device='cuda', 
                 study_trials = 50, learning_rate=0.001, weight_decay = 0, default_hidden = 64, 
                 default_layers = 3, drop_directory=r'C:\Users\listo\Sweet2Plus\my_figs',drop_filename='filename'):
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
        self.hidden_suggested = default_hidden 
        self.layers_suggested = default_layers
        self.drop_filename = drop_filename

    def __call__(self):

        self.torch_file = os.path.join(self.drop_directory,"solo_model.pth")
        if not os.path.isfile(self.torch_file):
            # Load in best parameters if file exists
            best_params_file = os.path.join(self.drop_directory,"best_hyperparameters.json")
            if os.path.isfile(best_params_file):
                print('Best parameters found from previous study. Loading parameters and skipping study...')
                with open(best_params_file, "r") as f:
                    best_hyperparameters = json.load(f)
                    self.learning_rate = best_hyperparameters['learning_rate']
                    self.weight_decay = best_hyperparameters['weight_decay']
                    self.hidden_suggested = best_hyperparameters['hidden_suggested']
                    self.layers_suggested = best_hyperparameters['layers_suggested']
                    self.num_data_points = best_hyperparameters['num_data_points']
                    self.run_study = False

            if self.run_study:
                print("Running a hyperparameter study via optuna ... ")
                print(" Relax, this might take a while ... ")

                # Temporarily change the number of epochs
                original_epochs = self.epochs
                self.epochs = 10

                # Run optuna study and get best hyperparameters
                self.hypertuning_study()
                
                # Set epochs back to original value
                self.epochs = original_epochs
                
                # run training
                print("Converting wrangled data into torch format ... ")
                X_train, y_train, X_val, y_val, X_test, y_test = self.get_data(num_data_points=self.num_data_points)
                train_loader, val_loader, test_loader = self.convert_to_torch_loader(X_train, y_train, X_val, y_val, X_test, y_test)

                # Build model           
                model_oh = SingleSampleNN(sequence_length = np.array(X_train).shape[1], 
                                    hidden_size = self.hidden_suggested, 
                                    num_layers = self.layers_suggested, 
                                    num_neurons = np.array(X_train).shape[2], 
                                    output_size = np.array(X_train).shape[2])

                print("Setting up loss function, optimizer and scheduler ... ")
                self.criterion = nn.MSELoss()
                self.optimizer = optim.Adam(model_oh.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
                print("Starting training on model ... ")
                self.trained_model, loss_history = self.train(model_oh, train_loader, val_loader, self.criterion, self.optimizer, self.scheduler)
                
                print('Plotting loss results')
                self.plot_learning_curve(data=loss_history, filename=os.path.join(self.drop_directory,'best_hyper_loss_curve.jpg'))
                
                # Run testing and get correlation between signals
                print('Testing model ....')
                self.metric_results = self.test(model_oh, test_loader)
                csv_path = os.path.join(self.drop_directory, f"{self.drop_filename}_metrics.csv")
                metrics_df = pd.DataFrame.from_dict(
                {k: [v] if not isinstance(v, (list, tuple, np.ndarray)) else v
                for k, v in self.metric_results.items()},
                orient="columns")
                metrics_df.to_csv(csv_path, index=False)

                # Save model for later use
                print('Saving model to file ...')
                self.torch_file = os.path.join(self.drop_directory,"solo_model.pth")
                torch.save(model_oh, self.torch_file)
            
            else:
                print('Running single model with given learning rates and Xpoints')
                print('Setting up data structure into torch dataset and model parameters')
    
                # Set up datasets
                X_train, y_train, X_val, y_val, X_test, y_test = self.get_data(num_data_points=self.num_data_points)
                train_loader, val_loader, test_loader = self.convert_to_torch_loader(X_train, y_train, X_val, y_val, X_test, y_test)


                # Build model            
                model_oh = SingleSampleNN(sequence_length = np.array(X_train).shape[1], 
                                    hidden_size = self.hidden_suggested, 
                                    num_layers = self.layers_suggested, 
                                    num_neurons = np.array(X_train).shape[2], 
                                    output_size = np.array(X_train).shape[2])
                
                # Set criterion and loss
                criterion_oh = nn.MSELoss()
                self.optimizer = optim.Adam(model_oh.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

                # Run training and get loss history
                print('Training model ....')
                self.trained_model, history = self.train(model_oh, train_loader, val_loader, criterion_oh, self.optimizer, self.scheduler)

                print('Plotting loss results')
                self.plot_learning_curve(data=history, filename=os.path.join(self.drop_directory,"FinalLossCurve.jpg"))
                
                # Run testing and get correlation between signals
                print('Testing model ....')
                self.metric_results = self.test(model_oh, test_loader)
                csv_path = os.path.join(self.drop_directory, f"{self.drop_filename}_metrics.csv")
                metrics_df = pd.DataFrame.from_dict(
                {k: [v] if not isinstance(v, (list, tuple, np.ndarray)) else v
                for k, v in self.metric_results.items()},
                orient="columns")
                metrics_df.to_csv(csv_path, index=False)

                # Save model for later use
                print('Saving model to file ...')
                self.torch_file = os.path.join(self.drop_directory,"solo_model.pth")
                torch.save(model_oh, self.torch_file)
            return 
        
        else:
            print('Pretrained model was found!!!')
            # Load in best parameters if file exists
            best_params_file = os.path.join(self.drop_directory,"best_hyperparameters.json")
            if os.path.isfile(best_params_file):
                print('Best parameters found from previous study. Loading parameters and skipping study...')
                with open(best_params_file, "r") as f:
                    best_hyperparameters = json.load(f)
                    self.learning_rate = best_hyperparameters['learning_rate']
                    self.weight_decay = best_hyperparameters['weight_decay']
                    self.hidden_suggested = best_hyperparameters['hidden_suggested']
                    self.layers_suggested = best_hyperparameters['layers_suggested']
                    self.num_data_points = best_hyperparameters['num_data_points']
                    self.run_study = False

            # Get data
            X_train, y_train, X_val, y_val, X_test, y_test = self.get_data(num_data_points=self.num_data_points)
            train_loader, val_loader, test_loader = self.convert_to_torch_loader(X_train, y_train, X_val, y_val, X_test, y_test)

            # Load in the trained model        
            model_oh = torch.load(self.torch_file, map_location=torch.device(self.device))
            model_oh.eval()
                
            # Run testing and get correlation between signals
            print('Testing model ....')
            self.metric_results = self.test(model_oh, test_loader)
            csv_path = os.path.join(self.drop_directory, f"{self.drop_filename}_metrics.csv")
            metrics_df = pd.DataFrame.from_dict(
                {k: [v] if not isinstance(v, (list, tuple, np.ndarray)) else v
                for k, v in self.metric_results.items()},
                orient="columns")
            metrics_df.to_csv(csv_path, index=False)
            return 

    def get_data(self, num_data_points, train_frac=0.7, val_frac=0.15):
        data = datawrangler(
            z_scored_neural_data=self.data,
            X_points=num_data_points
        )

        X, y = data()  # clean and unambiguous

        N = len(X)
        idx = np.random.permutation(N)

        n_train = int(train_frac * N)
        n_val = int(val_frac * N)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        return X_train, y_train, X_val, y_val, X_test, y_test


    
    def convert_to_torch_loader(self, X_train, y_train, X_val, y_val, X_test, y_test):
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )

        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader

    
    def train(self, model, train_loader, val_loader, criterion, optimizer, scheduler,plot_mode=True):
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
                    
                    # Plot current training and validation loss history
                    if plot_mode:
                        self.plot_learning_curve(data = loss_history, filename=os.path.join(self.drop_directory,"current_training_loss_results.jpg"))
       
        return model, loss_history

    def test(self, modeloh, test_loader):
        modeloh.eval()

        real_data = []
        predict_data = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.transpose(1, 2).contiguous().to(self.device)
                outputs = modeloh(inputs)

                real_data.append(targets.cpu().numpy())
                predict_data.append(outputs.cpu().numpy())

        # Convert to arrays
        real_data = np.squeeze(np.asarray(real_data))
        predict_data = np.squeeze(np.asarray(predict_data))

        n_neurons = real_data.shape[1]

        corrs, r2s, rmses, maes = [], [], [], []

        for n in range(n_neurons):
            real = real_data[:, n, :].reshape(-1)
            pred = predict_data[:, n, :].reshape(-1)

            if np.std(real) == 0:
                continue

            corrs.append(np.corrcoef(real, pred)[0, 1])
            r2s.append(r2_score(real, pred))
            rmses.append(np.sqrt(mean_squared_error(real, pred)))
            maes.append(mean_absolute_error(real, pred))

        # Aggregate results
        metrics = {
            "corr_mean": np.nanmean(corrs),
            "corr_std": np.nanstd(corrs),
            "r2_mean": np.nanmean(r2s),
            "rmse_mean": np.nanmean(rmses),
            "mae_mean": np.nanmean(maes),
            "per_neuron": {
                "corr": corrs,
                "r2": r2s,
                "rmse": rmses,
                "mae": maes
            }
        }

        if self.plot_neurons:
            for neuron_id in range(n_neurons):
                filepath = os.path.join(
                    self.neuron_drop_directory,
                    f"neuron{neuron_id}.jpg"
                )

                plt.figure(figsize=(10, 6))
                plt.plot(real_data[neuron_id, :], label="Real", alpha=0.7)
                plt.plot(predict_data[neuron_id, :], label="Predicted", alpha=0.7)
                plt.xlabel("Time")
                plt.ylabel("Normalized dF")
                plt.legend()
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()

        return metrics

    
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

            # Set plotting to false for study, because too many results and waste of time
            self.plot_neurons = False

            # Get neural data from file and seperate into X_train, y_train, etc
            X_train, y_train, X_val, y_val, X_test, y_test = self.get_data(num_data_points=self.num_data_points)
            train_loader, val_loader, test_loader = self.convert_to_torch_loader(X_train, y_train, X_val, y_val, X_test, y_test)
            
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
            trained_model, history = self.train(model_oh, train_loader, val_loader, criterion_oh, optimizer, scheduler,plot_mode=False)

            # Run testing and get correlation between signals
            predicted_validation_loss = self.predicted_loss(epochs_oh=range(len(history['val'])),loss_values_oh=history['val'])
            print(f'The predicted validation loss at epoch 100 is {predicted_validation_loss}')
            return predicted_validation_loss

        # Run Optuna study
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.study_trials, n_jobs=3)

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
    
    def loss_decay(self,x,a,b):
        return a * np.exp(-b * x)

    def predicted_loss(self,epochs_oh,loss_values_oh):
        """ Try to estimate loss"""
        epochs_oh = np.array(epochs_oh)
        loss_values_oh = np.array(loss_values_oh)
        popt, _ = scipy.optimize.curve_fit(self.loss_decay, epochs_oh, loss_values_oh, maxfev=10000)
        predicted_loss = self.loss_decay(100, *popt)
        return predicted_loss

class capture_model_weights():
    """ capture_model_weights
    Takes synaptic weights from model and performs basic analyses on it.
    Converts data into a dataframe for later use. 
    """
    def __init__(self, torch_file, drop_directory, activity_data, correlation_result):
        # Set up attributes
        self.torch_file = torch_file
        self.drop_directory = drop_directory
        self.activity_data = activity_data
        self.correlation_result = correlation_result

        # Load in model and set to evaludation
        model = torch.load(self.torch_file)
        model.eval()

        # Get all weights
        self.all_weights = model.state_dict()

        # Get weights and biases of the fully connected layer
        self.connected_weights = model.final_fc.weight 
        self.connected_bias = model.final_fc.bias  

    def __call__(self):
        # Strip file information
        self.strip_filename()

        # Put data into a dataframe
        self.to_dataframes()
        self.generate_heatmap()

    def strip_filename(self):
        _,metadata = self.drop_directory.split('working_file')
        _,subdata = metadata.split('24-')
        subdata,_ = subdata.split('R')
        _,day,self.cage,self.mouse,_ = subdata.split('_')
        _,self.day=day.split('-')
        
        self.suid = f'{self.cage}_{self.mouse}'

        if 'control' in self.drop_directory:
            self.group = 'control'
        else:
            self.group = 'cort'

    def to_dataframes(self):
        """ Creates two dataframes
            one containing only weights, 
            The other containing weights and neuronal activity """
        # Create a dataframe in long format of just weight data
        self.weights_df = pd.DataFrame(self.connected_weights.cpu().detach().numpy())
        self.weights_df = self.weights_df.reset_index().melt(id_vars='index', var_name='neuron_id_2', value_name='synaptic_weight')
        self.weights_df.rename(columns={'index': 'neuron_id_1'}, inplace=True)
        self.weights_df["suid"] = self.suid
        self.weights_df["day"] = self.day
        self.weights_df["group"] = self.group
        self.weights_df.to_csv(os.path.join(self.drop_directory,'weights_df.csv'))

        # Create a dataframe that contains weight and activity data
        weight_df_oh = pd.DataFrame(self.connected_weights.cpu().detach().numpy())
        activity_df = pd.DataFrame(self.activity_data.T)
        self.weight_activity_df = pd.concat([weight_df_oh, activity_df], axis=1)
        activity_df_names = [f"t{i}" for i in range(activity_df.shape[1])] 
        self.weight_activity_df.columns = list(weight_df_oh.columns) + activity_df_names
        self.weight_activity_df.to_csv(os.path.join(self.drop_directory,'weight_activity_df.csv'))
        return 

    def generate_heatmap(self):
        plt.figure(figsize=(20,20))
        plt.imshow(self.connected_weights.cpu().detach().numpy(), vmax=0.7, vmin=-0.7,cmap="seismic",interpolation='none')
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
    parser.add_argument('--best_parameters_path',type=str,help='Directory to json file containing best hyperparameters')
    parser.add_argument('--drop_directory',type=str,help='Directory to model outputs')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # Parse cli inputs
    args = cli_parser()

    # Get neuronal data
    zdataoh = zscore_data_from_obj(args.s2p_object_file)
    s2pobject_filename = os.path.splitext(os.path.basename(args.s2p_object_file))[0]

    # make sure drop_dir has traces path
    if not os.path.exists(os.path.join(args.drop_directory,r'traces')):
        os.mkdir(os.path.join(args.drop_directory,r'traces'))

    # Build, train and test best model
    objoh = Education(zdataoh, model=None, plot_neurons=True, run_study=args.hypertuning_study, drop_directory=args.drop_directory,drop_filename=s2pobject_filename)
    objoh()

    # Pull model weight data, generate basic graphs, save to dataframe
    capture_obj = capture_model_weights(torch_file = objoh.torch_file, drop_directory = objoh.drop_directory, 
                                        activity_data = zdataoh, correlation_result = objoh.corroh)
    capture_obj()

    # objoh = capture_model_weights(torch_file=r'C:\Users\listo\Sweet2Plus\my_figs\weight_modeling_test_output\solo_model.pth',
    #                       drop_directory=r'C:\Users\listo\Sweet2Plus\my_figs\weight_modeling_test_output', 
    #                       activity_data=None, correlation_result=None)
    objoh.generate_heatmap()

    # C:\Users\listo\tmt_experiment_2024_working_file\C4620083_cohort-1_M1_cort\day_7\24-3-25_day-7_C4620083_M1_R1\objfile.json
    # C:\Users\listo\weightmodelingtest