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

def GetNumpyArray(parent_directory):
    Fnp = np.load(os.path.join(parent_directory, 'F.npy'))
    ICnp = np.load(os.path.join(parent_directory, 'iscell.npy'))
    return Fnp, ICnp

def FilterCells(Fnp, ICnp):
    return Fnp[ICnp[:,0].astype(bool),:]

def GetStats(Fnp, drop_directory, timepoints = [(0,738),(739,1439),(1440,-1)]):
    # Parse timepoints
    prestim_t = timepoints[0]
    stim_t = timepoints[1]
    poststim_t = timepoints[2]
    
    # Get all AUC data 
    AUCs = []
    for neuron in Fnp:
        prestim = np.trapz(neuron[prestim_t[0]:prestim_t[1]])
        stim = np.trapz(neuron[stim_t[0]:stim_t[1]])
        poststim = np.trapz(neuron[poststim_t[0]:poststim_t[1]])
        AUCs.append([prestim, stim, poststim])
    
    # Convert back to numpy array
    AUCs = np.asarray(AUCs)
    
    # Get average trace
    Trace = np.mean(Fnp, axis=1)
    return AUCs, Trace

def GeneratePlots(AUCs, Trace,  drop_directory):

    time = np.linspace(0, len(Trace))  
    sns.set_theme(style="whitegrid")  # Set theme for better visualization
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=time, y=Trace)

    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("DF")

    # Show the plot
    plt.savefig('examplt.jpg')
    ipdb.set_trace()

if __name__=='__main__':
    # Get parent directory and create drop directory for dataframe and images 
    parent_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\plane0" 
    drop_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\analysis" 
    os.makedirs(drop_directory, exist_ok=True)

    # Run current file through analysis. 
    FnpOH, ICnpOH = GetNumpyArray(parent_directory=parent_directory)
    FnpOH = FilterCells(FnpOH, ICnpOH)
    AUCsOH, TracesOH = GetStats(FnpOH, drop_directory)
    GeneratePlots(AUCsOH, TracesOH, drop_directory)

