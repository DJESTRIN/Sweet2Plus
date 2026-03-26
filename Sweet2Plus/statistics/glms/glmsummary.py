#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: engelhardglm.py
Description: Here we attempt to use a nearly identical glm as Engelhard et al., 2019 for neuronal encoding. We also run validation analyses. 
Author: David James Estrin
Version: 1.1
Date: 03-05-2026

Current to do list:
 - write and read results to temp files. 
 - Write code for seperation of neurons by beta weight classifications
 - Analysis of functional connectivity and neuronal activity wrt to beta_weight classification, stress group, and day
"""

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


    
    # def load_results(self, search_string = None):
    #     """ Results are save to temp pickle files to lower use of RAM during fit. 
    #     Here we load results from pickled files after fit for analyses."""
    #     # Default search string when not provided
    #     if search_string is None:
    #         search_string = self.dropdir + r'/temp/*.pkl.gz'
        
    #     # Find model output files and load them in to common list attribute
    #     model_files = glob.glob(search_string)
    #     self.model_results = []
    #     for filename in model_files:
    #         with gzip.open(filename, "rb") as f:
    #             information,_ = (os.path.basename(filename)).split('.pk')
    #             day,cage,mouse,group,neuronid = information.split('_')
    #             self.model_results.append([pickle.load(f),day,cage,mouse,group,neuronid])