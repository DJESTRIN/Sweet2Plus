#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: deconvolution.py
Description: This code takes 2P calcium data for neurons and runs oasis suite2P deconvolve on them. 
Author: David Estrin
Version: 1.0
Date: 03-06-2026
"""
from suite2p.extraction import dcnv
import pickle
import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt


def save(dropdirectory, data, filename):
    # Save collected data into a pickle file
    os.makedirs(dropdirectory, exist_ok=True)
    filepath = os.path.join(dropdirectory, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load(dropdirectory, filename):
    filepath = os.path.join(dropdirectory, filename)
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def deconvolve(activity_data, tau=1.0, fs=1.47):
    max_len = max(len(n) for n in activity_data)
    F = np.zeros((len(activity_data), max_len))
    for i, neuron in enumerate(activity_data):
        F[i, :len(neuron)] = neuron  

    deconv_all = dcnv.oasis(F, 1, tau, fs)
    output = [deconv_all[i, :len(activity_data[i])] for i in range(len(activity_data))]
    return output

def plot_deconvolved(deconv_traces,save_path, n_neurons=5, zoom_frames=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    total_neurons = len(deconv_traces)
    n_neurons = min(n_neurons, total_neurons)
    
    # pick random neurons
    chosen_idx = np.random.choice(total_neurons, n_neurons, replace=False)
    
    plt.figure(figsize=(12, 2 * n_neurons))
    
    for i, idx in enumerate(chosen_idx, 1):
        trace = deconv_traces[idx][:zoom_frames]  # zoomed portion
        plt.subplot(n_neurons, 1, i)
        plt.plot(trace, color='blue')
        plt.title(f'Neuron {idx} (first {zoom_frames} frames)')
        plt.xlabel('Frame')
        plt.ylabel('Deconv')
        plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

if __name__=='__main__':
    dropdirectory = r"C:\Users\listo\tmt_experiment_2024_working_file\engelhardglm_results"
    filename = f'trans_act.pkl'
    data = load(dropdirectory, filename)
    output_data = deconvolve(activity_data = data)
    plot_deconvolved(deconv_traces=output_data,save_path=os.path.join(dropdirectory,'neurons.jpg'))
    save(dropdirectory=dropdirectory, data=output_data, filename='deconvolved_act.pkl')
