#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: Calcium_analysis.py
Description:  
Author: David Estrin & Kenneth Johnson
Version: 1.0
Date: 03-20-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
import ipdb
import seaborn as sns
from scipy.stats import sem

def GetNumpyArray(parent_directory):
    Fnp = np.load(os.path.join(parent_directory, 'F.npy'))
    ICnp = np.load(os.path.join(parent_directory, 'iscell.npy'))
    return Fnp, ICnp

def FilterCells(Fnp, ICnp):
    return Fnp[ICnp[:,0].astype(bool),:]

def sliding_zscore_2d(arr, window_size=50):
    def sliding_zscore_1d(row):
        if len(row) < window_size:
            raise ValueError("Window size must be smaller than or equal to the row length.")

        mean_values = np.convolve(row, np.ones(window_size)/window_size, mode='valid')
        squared_row = row**2
        mean_squared = np.convolve(squared_row, np.ones(window_size)/window_size, mode='valid')
        std_values = np.sqrt(mean_squared - mean_values**2)  # std = sqrt(E[X^2] - E[X]^2)

        # Normalize each valid window
        normalized = (row[window_size//2: -(window_size//2)] - mean_values[:-1]) / std_values[:-1]

        # Handle edges by padding with NaN
        padded_result = np.full_like(row, np.nan, dtype=np.float64)
        padded_result[window_size//2: -(window_size//2)] = normalized

        return padded_result

    return np.apply_along_axis(sliding_zscore_1d, axis=1, arr=arr)

def plot_random_rows(data, num_rows=25, grid_size=(5, 5), seed=None):
    if seed is not None:
        np.random.seed(seed)

    M, N = data.shape
    if M < num_rows:
        raise ValueError("Not enough rows in the data to sample.")

    selected_rows = np.random.choice(M, num_rows, replace=False)

    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for ax, row_idx in zip(axes.flat, selected_rows):
        ax.plot(data[row_idx], color='black')
        ax.set_title(f"Row {row_idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(drop_directory,'randomly_pulled_neurons.jpg'))

def GetStats(Fnp, drop_directory, timepoints = [(0,738),(739,1439),(1440,-1)]):
    # Parse timepoints
    prestim_t = timepoints[0]
    stim_t = timepoints[1]
    poststim_t = timepoints[2]
    
    # Get all AUC data 
    AUCs = []
    for neuron in Fnp:
        prestim = np.trapz(neuron[prestim_t[0]:prestim_t[1]])/(prestim_t[1] - prestim_t[0])
        stim = np.trapz(neuron[stim_t[0]:stim_t[1]])/(stim_t[1] - stim_t[0])
        poststim = np.trapz(neuron[poststim_t[0]:poststim_t[1]])/(poststim_t[1] - poststim_t[0])
        AUCs.append([prestim, stim, poststim])
    
    # Convert back to numpy array
    AUCs = np.asarray(AUCs)
    
    # Get average trace
    Trace = np.mean(Fnp, axis=1)
    return AUCs, Trace

def GeneratePlots(AUCs, Trace,  drop_directory):

    # Generate average trace for all neurons. 
    sns.set_theme(style="whitegrid")  
    plt.figure(figsize=(10, 10))
    sns.lineplot(x=range(len(Trace)), y=Trace)
    plt.xlabel("Time")
    plt.ylabel("DF")
    plt.savefig(os.path.join(drop_directory,'average_of_traces.jpg'))

    # Generate mean +/- sem and jitter plot of AUC data
    means = np.mean(AUCs, axis=0)
    sems = sem(AUCs, axis=0)
    colors = ["black", "red", "black"]
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="whitegrid")
    x_positions = np.arange(3)  

    for i in range(3):
        sns.stripplot(x=np.full(len(AUCs), i), y=AUCs[:, i], jitter=True, color=colors[i], alpha=0.5)

    for i in range(3):
        plt.errorbar(x_positions[i], means[i], yerr=sems[i], fmt='o', color=colors[i], capsize=5, markersize=8, label="Mean ± SEM" if i == 0 else "")

    # Formatting
    plt.xticks(x_positions, ["Pre-stimulation", "Stimulation", "Post-Stimulation"])  
    plt.ylabel("Values")
    plt.legend()

    ax_inset = plt.axes([0.55, 0.55, 0.35, 0.35])  # [left, bottom, width, height]
    for i in range(3):
        ax_inset.errorbar(x_positions[i], means[i], yerr=sems[i], fmt='o', color=colors[i], capsize=5, markersize=8)

    # Inset formatting
    ax_inset.set_xticks(x_positions)
    ax_inset.set_xticklabels(["Pre", "Stim", "Post"], fontsize=8)
    ax_inset.grid(True)
    plt.savefig(os.path.join(drop_directory,'AUC_mean_sem_jitter.jpg'))

if __name__=='__main__':
    # Get parent directory and create drop directory for dataframe and images 
    parent_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\plane0" 
    drop_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\analysis" 
    os.makedirs(drop_directory, exist_ok=True)

    # Run current file through analysis. 
    FnpOH, ICnpOH = GetNumpyArray(parent_directory=parent_directory)
    FnpOH = FilterCells(FnpOH, ICnpOH)
    FnpOH = sliding_zscore_2d(FnpOH)
    plot_random_rows(data=FnpOH, num_rows=25, grid_size=(5, 5), seed=339)
    AUCsOH, TracesOH = GetStats(FnpOH, drop_directory)
    GeneratePlots(AUCsOH, TracesOH, drop_directory)

