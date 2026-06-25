#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: glmsummary.py
Description: Analysis of beta weights from Engelhard et al., 2019 based glm. 
Author: David James Estrin
Version: 1.1
Date: 03-27-2026

Current to do list:
 - write and read results to temp files. 
 - Write code for seperation of neurons by beta weight classifications
 - Analysis of functional connectivity and neuronal activity wrt to beta_weight classification, stress group, and day
"""

# Set up CLI for input directory
# Read in gz files into a dataframe
# Generate figures
# Generate summary statistics
# Run analyses? 
import argparse
import glob, os
import gzip
import pickle
import ipdb
import pandas as pd
import numpy as np

def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_directory',type=str,help='Parent directory where g zipped data is located')
    args=parser.parse_args()
    return args.input_directory

class collect():
    def __init__(self,input_path):
        self.input_path = input_path

    def load_results(self, search_string = None):
        """ Results are save to temp pickle files to lower use of RAM during fit. 
        Here we load results from pickled files after fit for analyses."""

        # Default search string when not provided
        if search_string is None:
            search_string = self.input_path + r'/*.pkl.gz'

        # Find model output files and load them in to common list attribute
        model_files = glob.glob(search_string)
        self.model_results = []
        for filename in model_files:
            with gzip.open(filename, "rb") as f:
                information,_ = (os.path.basename(filename)).split('.pk')
                day,cage,mouse,group,neuronid = information.split('_')
                self.model_results.append([pickle.load(f),day,cage,mouse,group,neuronid])
    
    def generate_dataframe(self):
        rows = []

        for neuron_data in self.model_results:
            data, day, cage, mouse, group, neuronid = neuron_data

            for perm_idx, j in enumerate(data):
                betas = np.array(j['betas'])      # (200,)
                pvals = np.array(j['pvalues'])    # (200,)
                r2 = float(np.array(j['r2']).flatten()[0])  # scalar
                perm_type = j['type']             # scalar or label

                # sanity check
                if betas.ndim != 1:
                    print("Unexpected shape:", betas.shape)
                    continue

                n_betas = betas.shape[0]

                rows.append(pd.DataFrame({
                    'day': day,
                    'cage': cage,
                    'mouse': mouse,
                    'group': group,
                    'neuronid': neuronid,
                    'permutation': perm_type,
                    'perm_idx': perm_idx,
                    'beta_idx': np.arange(n_betas),
                    'beta': betas,
                    'pval': pvals,
                    'r2': r2   # repeated automatically
                }))

        return pd.concat(rows, ignore_index=True)

def proc():
    input_directory = cli_parser()
    collection_obj = collect(input_path=input_directory)
    collection_obj.load_results()
    dataframe = collection_obj.generate_dataframe()
    ipdb.set_trace()

if __name__=='__main__':
    proc()



# def betaweight_collection(self):
    #     # generate a final long dataframe
    #     # NeuronID 

    #     # Place holder for where we grab information regarding each neuron's beta weight and put into a dataset
    #     print('getting_beta_weights')
    #     new_list = []
    #     grouped_neurons = defaultdict(list)
    #     for neuron in self.linearmodel_results:
    #         info_tuple = tuple(neuron['info'])  # day, cage, mouse, group
    #         grouped_neurons[info_tuple].append(neuron)

    #     for info_tuple, neurons in grouped_neurons.items():
    #         for neuron in neurons:
    #             betas = neuron['betas']  # numpy array of shape (202,)
    #             betas_trimmed = betas[1:-1]  # now length 200
    #             groups = np.split(betas_trimmed, 4)
    #             max_abs_betas = [np.max(np.abs(g)) for g in groups]
    #             new_list.append({
    #                 'max_abs_betas': max_abs_betas,
    #                 'type': neuron['type']
    #             })

    #     # 
    #     os.makedirs(self.dropdir, exist_ok=True)

    #     # For each of 4 events
    #     for event_idx in range(4):
    #         plt.figure(figsize=(6,4))
            
    #         # permutation values
    #         perm_values = [n['max_abs_betas'][event_idx] 
    #                     for n in new_list if 'permutation' in n['type']]
    #         plt.hist(perm_values, bins=30, alpha=0.7, color='blue', label='Permutation')
            
    #         # real values
    #         real_values = [n['max_abs_betas'][event_idx] 
    #                     for n in new_list if 'real' in n['type']]
    #         plt.scatter(real_values, [0]*len(real_values), color='red', zorder=10, label='Real')
            
    #         plt.title(f'Event {event_idx+1} Max Abs Beta')
    #         plt.xlabel('Max Abs Beta')
    #         plt.ylabel('Count')
    #         plt.legend()
            
    #         # Save figure
    #         save_path = os.path.join(self.dropdir, f'event{event_idx+1}_max_abs_beta.png')
    #         plt.savefig(save_path)
    #         plt.close()