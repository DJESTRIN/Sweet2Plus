#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: coders.py
Description: Contains classes for decoding neuronal activity. 
Author: David Estrin
Version: 1.0
Date: 12-06-2024
"""
from Sweet2Plus.statistics.coefficient_clustering import cli_parser, gather_data
from Sweet2Plus.decoders.NetworkArchitectures import GCNTrialClassifier, NeuralNetworkFigures
import ipdb
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# Create a nice table for printing epoch and loader results
console = Console()

class GraphDataset(Dataset):
    def __init__(self, graph_list):
        super(GraphDataset, self).__init__()
        self.graphs = graph_list  # List of PyG Data objects

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]  # Return a single graph

class OrganizeGraphs:
    def __init__(self, neural_activity, behavioral_timestamps, neuron_info,
                 drop_directory, window_size=25, window_offset = 3, edge_threshold=0.5, 
                 trial_length=10, batch_size = 32):
        self.neural_activity = neural_activity
        self.behavioral_timestamps = behavioral_timestamps
        self.neuron_info = neuron_info
        self.drop_directory = drop_directory
        self.window_size = window_size
        self.window_offset = window_offset
        self.edge_threshold = edge_threshold
        self.trial_length = trial_length
        self.batch_size = batch_size

    def __call__(self):
        print('Generating all graphs ... ')
        self.GenerateGraphs()
       
        print('Splitting all graphs and putting into torch formats... ')
        self.train_graphs, self.test_graphs = train_test_split(self.all_graphs, test_size=0.2, random_state=42)
        self.train_graphs, self.val_graphs = train_test_split(self.train_graphs, test_size=0.2, random_state=42)

        # Convert to torch dataset       
        self.train_dataset = GraphDataset(self.train_graphs)
        self.val_dataset = GraphDataset(self.val_graphs)
        self.test_dataset = GraphDataset(self.test_graphs)

        for i, graph in enumerate(self.train_dataset ):
            print(f"Graph {i}: {graph.x.shape}")
            print(f"Graph {i}: {graph.y.shape}")
            if i>10:
                break

        # Conver to torch data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        for batch in self.train_loader:
            print(f"Batch num_graphs: {batch.num_graphs}")  # Should match batch_size
            print(f"Total nodes in batch: {batch.x.shape[0]}")  # Should be ~batch_size * avg_nodes
            print(f"Batch tensor shape: {batch.batch.shape}")  # Should match batch.x.shape[0]
            print(f"Batch tensor shape: {batch.y.shape}") 
            break

    def GenerateGraphs(self):
        self.all_graphs = []
        for recording,behoh in tqdm(zip(self.neural_activity,self.behavioral_timestamps),total=len(self.neural_activity)):
            windows_oh = self.SlidingWindow(recording_oh=recording,beh_oh=behoh)

            for (activity,output) in windows_oh:
                graph_oh = self.create_graph(window_oh=activity,labels_oh=output)
                self.all_graphs.append(graph_oh)
        
    def SlidingWindow(self, recording_oh, beh_oh):
        # Convert the behavior list to onehot array. 
        num_trial_types = len(beh_oh)
        onehot_vector = np.zeros((recording_oh.shape[1], num_trial_types + 1))  # +1 for ITI class

        for trial_type, timestamps in enumerate(beh_oh):
            for t in timestamps:
                onehot_vector[int(t):int(t+self.trial_length), int(trial_type)] = 1

        # Mark ITI indices where no other trial type is active
        iti_indices = np.where(onehot_vector[:, :-1].sum(axis=1) == 0)[0]
        onehot_vector[iti_indices, -1] = 1  # Set ITI class where no other activity is present

        # Create windows
        windows = []
        for start in range(0, recording_oh.shape[1] - self.window_size + 1, self.window_offset):
            end = start + self.window_size
            windowed_data = recording_oh[:, start:end]  # (window_size, N)
            windowed_labels = onehot_vector[start:end, :]  # (window_size, num_classes)

            # Remove windows where the 5th ITI column is 1 and ensure there's activity in other columns
            if not np.all(windowed_labels[:, -1] == 1):  # If ITI class is not the only active class
                # Filter out the ITI class (the last column) and keep only the trial types (first 4 columns)
                filtered_labels = windowed_labels[:, :-1]  # Only keep the first 4 trial types
                windows.append((windowed_data, filtered_labels))

        return windows

    def create_graph(self, window_oh, labels_oh):
        window_oh = torch.tensor(window_oh, dtype=torch.float32)  # Convert NumPy array to Tensor
        labels_oh = torch.tensor(labels_oh, dtype=torch.float32)  # Ensure labels are also Tensors

        N, t = window_oh.shape
        corr_matrix = torch.corrcoef(window_oh)  # Calculate the correlation matrix (N, N)
        edge_index = (corr_matrix.abs() > self.edge_threshold).nonzero(as_tuple=False).T  # Edge index based on correlation matrix

        # Create the batch tensor: Assign the same batch ID to all nodes in the current graph
        batch = torch.zeros(N, dtype=torch.long)  # Each node in this graph belongs to batch 0
        # If you have multiple graphs, you can increment batch for each graph in a batch

        # Create PyTorch Geometric Data object
        graph = Data(
            x=window_oh,  # Node features (N, t)
            edge_index=edge_index,  # Graph connectivity
            y=labels_oh,  # Output labels (t,1)
            batch=batch  # Batch assignment (N,)
        )
        return graph

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Pass tensor of shape (num_classes,)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[torch.argmax(targets.long(),axis=1)]  # Apply per-class weights, assuming `targets` is a tensor of class indices
            focal_loss *= alpha_t  # Element-wise multiplication

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class Education:
    def __init__(self, data_obj, neural_network_obj, device, figures_obj, 
                 hyp_total_epochs = 100, hyp_learning_rate = 0.001, weight_decay=5e-4):
        # Model 
        self.model_oh = neural_network_obj 

        # Hyperparameter information
        self.total_epochs = hyp_total_epochs
        self.learning_rate = hyp_learning_rate
        self.weight_decay = weight_decay

        # Update the loss function to use the class weights
        self.optimizer = torch.optim.AdamW(self.model_oh.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',         # Reduce LR when loss decreases
            patience=2,         # Lower patience to reduce faster (default is 10)
            factor=0.2,         # Reduce LR more aggressively (default is 0.1)
            threshold=1e-4,     # Detect small improvements
            threshold_mode='abs',  # Change based on absolute loss improvement
            verbose=True        # Print when LR is reduced
        )

        # Data information
        self.device = device
        self.train_loader = data_obj.train_loader 
        self.test_loader = data_obj.test_loader  
        self.val_loader = data_obj.val_loader
        self.drop_directory = os.path.join(data_obj.drop_directory,"neural_network_encoder_results/")
      
        # Update loss function to include class weights
        print('Grabbing class weights from training loader...')
        all_labels = []
        for batch in tqdm(self.train_loader):
            labels = batch.y
            labels = torch.argmax(labels, dim=-1)  # Convert one-hot to class indices
            all_labels.append(labels)

        all_labels = torch.cat(all_labels, dim=0)

        # Count occurrences of each class
        class_counts = np.bincount(all_labels.cpu().numpy(), minlength=4)

        # Prevent division by zero
        class_counts = np.maximum(class_counts, 1)

        # Compute class weights
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()  # Normalize weights
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        print(f'Class weights: {class_weights}')

        #self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)

        # Generate path if not real
        if not os.path.exists(self.drop_directory):
            os.makedirs(self.drop_directory, exist_ok=True)

        # Empty lists to drop data
        self.training_loss = []
        self.training_f1 = []
        self.testing_f1_all = []
        self.testing_f1 = None

        # Generate figures every training or testing loop
        self.figures = figures_obj

        # Run training and testing
        self.model_oh.to(self.device)
        self.training()
        self.testing()

    def training(self):
        # Initialize a dictionary to store metrics
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'train_f1': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'train_f1_per_class': [],
            'val_f1_per_class': []
        }
        
        for epoch in range(self.total_epochs):
            self.model_oh.train()  # Set model to training mode
            running_loss = 0.0
            all_labels = []
            all_preds = []

            train_loader = tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.total_epochs}] Training", leave=False)
            for data in train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model_oh(data)
                ipdb.set_trace()
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), data.y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=-1)  # Shape: (batch_size, T)
                real_labels = torch.argmax(data.y.reshape(preds.shape[0], preds.shape[1], 4), dim=-1)
                all_labels.append(real_labels.cpu().numpy())  
                all_preds.append(preds.cpu().numpy())

                train_loader.set_postfix(loss=loss.item())  # Show live loss update in tqdm

            # Calculate training loss
            avg_train_loss = running_loss / len(self.train_loader)
            all_labels = np.concatenate(all_labels, axis=0)
            all_preds = np.concatenate(all_preds, axis=0)
            train_accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())  
            train_f1_per_class = f1_score(all_labels.flatten(), all_preds.flatten(), average=None)
            train_f1 = np.mean(train_f1_per_class)  # Macro average

            self.metrics['train_loss'].append(avg_train_loss)
            self.metrics['train_accuracy'].append(train_accuracy)
            self.metrics['train_f1'].append(train_f1)
            self.metrics['train_f1_per_class'].append(train_f1_per_class.tolist())

            # Validation step
            self.model_oh.eval()  
            val_loss = 0.0
            val_labels = []
            val_preds = []

            val_loader = tqdm(self.val_loader, desc=f"[Epoch {epoch+1}/{self.total_epochs}] Validation", leave=False)
            with torch.no_grad():  
                for data in val_loader:
                    data = data.to(self.device)
                    outputs = self.model_oh(data)
                    loss = self.criterion(outputs.view(-1, outputs.size(-1)), data.y)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=-1)  
                    real_labels = torch.argmax(data.y.reshape(preds.shape[0], preds.shape[1], 4), dim=-1)
                    val_labels.append(real_labels.cpu().numpy())
                    val_preds.append(preds.cpu().numpy())

                    val_loader.set_postfix(loss=loss.item())  # Show live loss update

            avg_val_loss = val_loss / len(self.val_loader)
            
            # Update scheduler
            self.scheduler.step(avg_val_loss)
            val_labels = np.concatenate(val_labels, axis=0)
            val_preds = np.concatenate(val_preds, axis=0)
            val_accuracy = accuracy_score(val_labels.flatten(), val_preds.flatten())
            val_f1_per_class = f1_score(val_labels.flatten(), val_preds.flatten(), average=None)
            val_f1 = np.mean(val_f1_per_class)  # Macro average

            self.metrics['val_loss'].append(avg_val_loss)
            self.metrics['val_accuracy'].append(val_accuracy)
            self.metrics['val_f1'].append(val_f1)
            self.metrics['val_f1_per_class'].append(val_f1_per_class.tolist())

            # Class Distribution Stats
            train_pred_counts = np.bincount(all_preds.flatten(), minlength=4)
            val_pred_counts = np.bincount(val_preds.flatten(), minlength=4)
            train_pred_percent = train_pred_counts / train_pred_counts.sum() * 100
            val_pred_percent = val_pred_counts / val_pred_counts.sum() * 100

            # Get current lr
            current_lr = self.optimizer.param_groups[0]['lr']

            # Display Epoch Metrics with Rich Table
            table = Table(title=f"Epoch {epoch+1}/{self.total_epochs} Summary")
            table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
            table.add_column("Train", justify="center", style="green")
            table.add_column("Validation", justify="center", style="red")

            table.add_row("Loss", f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}")
            table.add_row("Accuracy", f"{train_accuracy:.4f}", f"{val_accuracy:.4f}")
            table.add_row("F1 Score", f"{train_f1:.4f}", f"{val_f1:.4f}")
            table.add_row("Learning Rate", f"{current_lr:.6f}", "-") 

            # Add per-class F1 scores
            for i in range(len(train_f1_per_class)):
                table.add_row(f"F1 Score (Class {i})", f"{train_f1_per_class[i]:.4f}", f"{val_f1_per_class[i]:.4f}")

            # Add class distribution
            for i in range(4):
                table.add_row(f"Class {i} % Predictions", f"{train_pred_percent[i]:.2f}%", f"{val_pred_percent[i]:.2f}%")

            console.print(table)

            self.figures.plot_all_curves(metrics_dict=self.metrics, output_file=os.path.join(self.drop_directory, 'GNN_metrics.jpg'))

    def testing(self):
        # Set model to evaluation mode
        self.model_oh.eval()
        
        # Initialize variables for test metrics
        test_loss = 0.0
        test_labels = []
        test_preds = []

        with torch.no_grad():  # Disable gradient calculation for testing
            for data in self.test_loader:
                data = data.to(self.device)
                outputs = self.model_oh(data)
                loss = self.criterion(outputs.view(-1,outputs.size(-1)), data.y)  # Assuming 5 trial types
                test_loss += loss.item()
                preds = torch.argmax(outputs, dim=-1)  # Get predictions
                real_labels = torch.argmax(data.y.reshape(preds.shape[0],preds.shape[1],4), dim=-1) 
                test_labels.append(real_labels.cpu().numpy())  # Store true labels
                test_preds.append(preds.cpu().numpy())  # Store predicted labels

        # Calculate average test loss
        avg_test_loss = test_loss / len(self.test_loader)
        test_labels = np.concatenate(test_labels, axis=0)
        test_preds = np.concatenate(test_preds, axis=0)

        # Calculate accuracy and F1 score
        test_accuracy = accuracy_score(test_labels.flatten(), test_preds.flatten())
        test_f1 = f1_score(test_labels.flatten(), test_preds.flatten(), average='weighted')

        # Save test metrics to dictionary
        self.metrics['test_loss'] = avg_test_loss
        self.metrics['test_accuracy'] = test_accuracy
        self.metrics['test_f1'] = test_f1

        # Print the test metrics
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}")
        self.figures.plot_all_curves(metrics_dict=self.metrics, 
                                    output_file=os.path.join(self.drop_directory,'GNN_metrics.jpg'))

def main():
    # Get cli inputs and gather data into a graph dataset
    data_directory, drop_directory = cli_parser()
    neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
    dataloaders = OrganizeGraphs(neuronal_activity, behavioral_timestamps, neuron_info, drop_directory = drop_directory)
    dataloaders()

    # Determine GPU device
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        deviceoh = torch.device('cuda')
        print(f"CUDA gpu is available. There are {torch.cuda.device_count()} gpu(s) on this \n {device_name} device.")
    else:
        raise RuntimeError("No cuda gpu was found, therefore code has ended. This code needs a gpu...")

    # Generate the GNN model
    GCNmodel = GCNTrialClassifier(input_dim=25, hidden_dim=125, output_dim=4)
    FigObj = NeuralNetworkFigures(name='GNNdata')

    # Educate GNN
    GNN_education = Education(data_obj = dataloaders, neural_network_obj=GCNmodel, 
                    device=deviceoh, figures_obj=FigObj, 
                    hyp_total_epochs = 100, hyp_learning_rate = 0.001, weight_decay=1e-2)

if __name__=='__main__':
    main()