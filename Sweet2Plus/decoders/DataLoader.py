#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: DataLoader.py
Description:
Author: David Estrin
Version: 1.0
Date: 12-06-2024 
"""

from Sweet2Plus.statistics.heatmaps import heatmap
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os 
import torch.nn.utils.rnn as rnn
import ipdb

def add_gaussian_noise(X, y, noise_factor_range=(0.01, 0.05), final_size=None):
    """
    Adds Gaussian noise to the input features (X) only. The noise factor is randomly selected from 
    the given range. If a final size is provided, the dataset is resized by adding augmented samples.

    Args:
        X (np.array): Input features dataset of shape (num_samples, seq_len, features).
        y (np.array): Target dataset of shape (num_samples, target_dim).
        noise_factor_range (tuple): Tuple of (min_noise_factor, max_noise_factor) for random noise.
        final_size (int, optional): The final number of samples in the dataset. If None, the dataset size remains unchanged.
    
    Returns:
        tuple: Augmented X and y datasets with added noise and possibly resized.
    """
    augmented_X = []
    augmented_y = []

    # Loop over each sample in the dataset and add noise to X
    for sample_X, sample_y in zip(X, y):
        noise_factor = np.random.uniform(noise_factor_range[0], noise_factor_range[1])
        noise = np.random.normal(0, noise_factor, sample_X.shape)
        augmented_sample_X = sample_X + noise
        
        augmented_X.append(augmented_sample_X)
        augmented_y.append(sample_y)  # Target remains unchanged

    # Convert augmented X and y back to numpy arrays
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)

    # If final_size is specified, resize the dataset by duplicating samples
    if final_size is not None:
        if final_size > len(augmented_X):
            repeat_factor = final_size // len(augmented_X)
            augmented_X = np.tile(augmented_X, (repeat_factor, 1, 1))[:final_size]
            augmented_y = np.tile(augmented_y, (repeat_factor, 1))[:final_size]
        elif final_size < len(augmented_X):
            augmented_X = augmented_X[:final_size]
            augmented_y = augmented_y[:final_size]

    return augmented_X, augmented_y

def sampling(X,y):
    """ Sampling -- 
     Takes data, finds smallest class and makes sure the data is equally sampled across all classes """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_count = np.max(class_counts)
    minority_class_index = np.argmin(class_counts) 
    minority_class = unique_classes[minority_class_index]
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    n_minority_samples = X_minority.shape[0]
    n_upsampled = max_count - n_minority_samples 
    indices = np.random.choice(n_minority_samples, n_upsampled, replace=True)
    X_upsampled = np.vstack([X_minority, X_minority[indices]])
    y_upsampled = np.hstack([y_minority, y_minority[indices]])
    X_balanced = np.vstack([X[y != minority_class], X_upsampled])
    y_balanced = np.hstack([y[y != minority_class], y_upsampled])
    return X_balanced, y_balanced

def zero_samples(X):
    X_zeroed = X - np.tile(X[:,0],(X.shape[1],1)).T
    return X_zeroed

class circuit_format_data():
    """ Format data into samples per circuit per trial """
    def __init__(self,drop_directory, neuronal_activity, behavioral_timestamps, neuron_info, 
                 trial_list=['Vanilla', 'PeanutButter', 'Water', 'FoxUrine'], batch_size=8):
        self.drop_directory = drop_directory
        self.neuronal_activity = neuronal_activity
        self.behavioral_timestamps = behavioral_timestamps
        self.neuron_info = neuron_info
        self.trial_list = trial_list
        self.batch_size = batch_size
    
    def __call__(self):
        # Get circuit data
        self.set_formatting()

        # Split into training and testing 
        X_train, X_test, y_train, y_test = self.clean_and_split_data()

        # Convert data to torch loaders
        self.torch_loader(X_train, X_test, y_train, y_test)

    def set_formatting(self,prewindow=int(10),postwindow=int(15)):
        self.X = []
        self.y = []
        for recording_oh, all_timestamps in zip(self.neuronal_activity,self.behavioral_timestamps):
            for trialname,timestamps in enumerate(all_timestamps):
                timestamps = np.asarray(timestamps, dtype=int)
                timestamps = timestamps[(timestamps - prewindow >= 0) & (timestamps + postwindow < recording_oh.shape[1])]
                # This line should also zero the data at the begining
                self.X.extend([(recording_oh[:, t - prewindow : t + postwindow + 1] - recording_oh[:, t - prewindow : t - prewindow + 5].mean(axis=1, keepdims=True) ) for t in timestamps])
                self.y.extend([trialname for t in timestamps])
        
        num_classes = len(self.trial_list)
        self.y_one_hot = np.eye(num_classes)[self.y]

    def clean_and_split_data(self):
        
        #self.X, self.y = add_gaussian_noise(X=self.X, y=self.y, final_size=10000)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def torch_loader(self, X_train, X_test, y_train, y_test):
        """ Put numpy arrays into torch's data loader format """
        X_train = [torch.tensor(x, dtype=torch.float32) for x in X_train]  # Convert to tensors
        X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]

        X_train = rnn.pad_sequence(X_train, batch_first=True)  # Ensures uniform shape
        X_test = rnn.pad_sequence(X_test, batch_first=True)

        X_train = np.array(X_train, dtype=np.float32)  # Convert lists to NumPy arrays first
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        self.X_train, self.X_test, self.y_train, self.y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))

        self.training_dataset = TensorDataset(self.X_train.float(), self.y_train.long())
        self.testing_dataset = TensorDataset(self.X_test.float(), self.y_test.long())

        self.train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testing_dataset, batch_size=self.batch_size, shuffle=True)

class format_data(heatmap):
    def __init__(self, drop_directory, neuronal_activity, behavioral_timestamps, neuron_info, 
                 trial_list=['Vanilla', 'PeanutButter', 'Water', 'FoxUrine'],
                 normalize_neural_activity=False, regression_type='ridge', 
                 hyp_batch_size=64, preprocessed = None, percentage_ds = 1 ):
        super().__init__(drop_directory, neuronal_activity, behavioral_timestamps, neuron_info,
                         trial_list, normalize_neural_activity, regression_type)
        
        self.batch_size = hyp_batch_size
        self.preprocessed = preprocessed
        self.percentage_ds = percentage_ds

    def __call__(self):
        if self.preprocessed:
            # Load data from .npy files
            X_path, y_path = self.preprocessed
            self.X_original = np.load(X_path)
            self.y_one_hot = np.load(y_path)
            num_samples = int(len(self.X_original) * self.percentage_ds)
            selected_indices = np.random.choice(len(self.X_original), size=num_samples, replace=False)
            self.X_original = self.X_original[selected_indices]
            self.y_one_hot  = self.y_one_hot[selected_indices]
            self.normalize_for_neural_network()
            self.quick_plot()
        else:
            super().__call__()

        X_train, X_test, y_train, y_test = self.clean_and_split_data()
        self.torch_loader(X_train, X_test, y_train, y_test)

    def normalize_for_neural_network(self):
        """
        Normalize the data
        """
        for k,row in enumerate(self.X_original):
            self.X_original[k] = (row-np.mean(row,axis=0))/(np.std(row,axis=0) + 1e-8) + 1e-8

    def quick_plot(self):
        plt.figure()
        maxes = np.argmax(self.y_one_hot,axis=1)
        for type,trial_name in zip(np.unique(maxes),self.trial_list):
            current_data = self.X_original[np.where(maxes==type)]
            average_current_data = np.nanmean(current_data,axis=0)
            plt.plot(average_current_data,label=trial_name)
        plt.savefig(os.path.join(self.drop_directory,"plotofavXdata.jpg"))

    def clean_and_split_data(self):
        # Shuffle the data
        indices = np.arange(self.X_original.shape[0])
        np.random.shuffle(indices)
        self.X, y_one_hot = self.X_original[indices], self.y_one_hot[indices]

        # Convert one hot to arg max
        self.y = np.argmax(y_one_hot, axis=1)

        # Oversample minority classes that are too small
        self.X, self.y = sampling(X=self.X, y=self.y)
        
        # Zero the data by first point
        self.X = zero_samples(self.X)

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