#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: heatmaps.py
Description: Generates heatmap data based on trial types provided. Inherits classes from coefficient_cluster because these functions and
    objects organize the data already. 
Author: David Estrin
Version: 1.0
Date: 12-06-2024
"""
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from Sweet2Plus.statistics.coefficient_clustering import regression_coeffecient_pca_clustering, gather_data, cli_parser
import ipdb 

class heatmap(regression_coeffecient_pca_clustering):
    def __call__(self):
        # Inherited methods
        self.normalize_activity()
        self.timestamps_to_one_hot_array()

        # New methods
        self.gather_data_by_trial()
        self.plot_data_by_trial()

    def gather_data_by_trial(self,preceding_frames=20,post_stim_frames=26):
        self.preceding_frames = preceding_frames
        self.post_stim_frames = post_stim_frames
        self.preceding_seconds = self.preceding_frames/1.315
        self.post_stim_seconds = self.post_stim_frames/1.315

        self.all_neural_peth_data = []
        for k,subject_recording_activity in enumerate(self.neuronal_activity):
            subject_recording_activity = subject_recording_activity.T
            one_hot_oh = self.behavior_ts_onehot[k]
            
            # Get all indexes for time stamps in image time
            timestamps_oh = []
            for trial in one_hot_oh.T:
                
                trial_ts = []
                for index in range(len(trial)-1):
                    if trial[index]==0 and trial[index+1]==1:
                        trial_ts.append(index)
                
                timestamps_oh.append(trial_ts)

            # Get all neural data for each timestamp
            self.all_neural_trial_ts = []
            for trial in timestamps_oh:
                trial_neural_all_ts = []
                for ts in trial:
                    data_oh = subject_recording_activity[ts-preceding_frames:ts+post_stim_frames]
                    trial_neural_all_ts.append(data_oh)
                
                self.all_neural_trial_ts.append(trial_neural_all_ts)
            self.all_neural_peth_data.append(self.all_neural_trial_ts)

        # Regroup data by averages
        all_subject_avs_by_trial = []
        for subject_peth_oh in self.all_neural_peth_data:
            all_avs_by_trial = []
            for trial in subject_peth_oh:
                # Calculate the average neural activity across neurons for all time stamps for given trial
                all_av_neu_for_trial = [ts.mean(axis=1) for ts in trial]

                # Filter out any trials that might not have the correct number of timestamps 
                shapes = [array.shape for array in all_av_neu_for_trial]
                shape_counts = Counter(shapes)
                most_common_shape = shape_counts.most_common(1)[0][0]  # The most common shape
                all_av_neu_for_trial = [array for array in all_av_neu_for_trial if array.shape == most_common_shape]

                all_av_neu_for_trial = np.array(all_av_neu_for_trial)
                trial_mean = all_av_neu_for_trial.mean(axis=0)
                all_avs_by_trial.append(trial_mean)
            
            all_avs_by_trial = np.array(all_avs_by_trial)
            all_subject_avs_by_trial.append(all_avs_by_trial)
        all_subject_avs_by_trial = np.array(all_subject_avs_by_trial)
        self.all_avs = all_subject_avs_by_trial.mean(axis=0)
        self.all_sds = all_subject_avs_by_trial.std(axis=0)
        self.N = all_subject_avs_by_trial.shape[0]
        self.all_subject_avs_by_trial = all_subject_avs_by_trial

        # Regroup data by AUCs
        all_subject_auc_by_trial = []  # Store AUCs for all subjects
        for subject_peth_oh in self.all_neural_peth_data: # Loop over subjects
            
            all_auc_by_trial = [] # Collect average auc by trial for current subject
            for trial_type in subject_peth_oh: # Loop over trial types
                auc_by_trial = []
                for ts in trial_type: # Loop over time stamps in trial type
                    
                    # Calculate AUC
                    try:
                        cropped_data = ts[20:30,:]
                        neuron_aucs = [np.trapz(row_oh) for row_oh in cropped_data.T]
                        trial_mean_auc = np.array(neuron_aucs).mean()
                    
                    except:
                        trial_mean_auc = np.nan

                    auc_by_trial.append(trial_mean_auc) # Collects each trial auc
                all_auc_by_trial.append(auc_by_trial)

            avs_by_trial = np.array([np.array(trial_auc_oh).mean() for trial_auc_oh in all_auc_by_trial])
            all_subject_auc_by_trial.append(avs_by_trial)
        
        all_subject_auc_by_trial = np.array(all_subject_auc_by_trial) 

        # Pull out attributes 
        self.auc_avs = all_subject_auc_by_trial.mean(axis=0)
        self.auc_N = all_subject_auc_by_trial.shape[0]
        self.auc_ses = all_subject_auc_by_trial.std(axis=0)/np.sqrt(self.auc_N)
        self.all_subject_auc_by_trial = all_subject_auc_by_trial

    def plot_data_by_trial(self):
        # Set up x axis
        time = np.arange(self.all_avs.shape[1])
        custom_labels = np.linspace(-1 * self.preceding_seconds, self.post_stim_seconds, len(time))
        colors = plt.cm.tab10(np.arange(len(self.trial_list))) 

        # Generate plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})  # Adjust relative sizes

        # AVerage activity plot
        for row, sd, labeloh, coloroh in zip(self.all_avs, self.all_sds, self.trial_list, colors):
            row = row - row[0]  # Rescale plot
            se = sd / np.sqrt(self.N)
            axes[0].plot(custom_labels, row, label=labeloh, color = coloroh)
            axes[0].fill_between(custom_labels, row - se, row + se, alpha=0.2, color = coloroh)

        axes[0].axvline(x=0, color='black', linestyle='--')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Average Normalized DF \n Across Subjects and Neurons')
        axes[0].legend(loc="best")
        axes[0].grid(True)

        # AUC bar plot
        axes[1].bar(self.trial_list, self.auc_avs, yerr=self.auc_ses, capsize=5, alpha=0.8, edgecolor='black', color = colors)
        axes[1].set_ylabel('AUC')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.drop_directory, "average_acitvity_auc.jpg"))

    def generate_singular_neuronal_onehot(self):
        """ 
        
        output 
            individual neuronal activities for a trial (46, 1) floats -----> trial answer as one hot [0 , 0 , 1, 0]
        """
        all_trial_neuronal_data = []
        all_trial_results = []
        for subject_peth_oh in self.all_neural_peth_data:
            for k,trial_data,trial_name in enumerate(zip(subject_peth_oh,self.trial_list)):
                
                for single_neuron_data in enumerate(trial_data):
                    result_oh = np.zeros(shape=(1,4))
                    if single_neuron_data.shape[0]<46:
                        ipdb.set_trace()
                        continue

                    else:
                        ipdb.set_trace()
                        all_trial_neuronal_data.append(single_neuron_data)
                        result_oh[:,k] = 1
                        all_trial_results.append(result_oh)

        ipdb.set_trace()




if __name__=='__main__':
    data_directory, drop_directory = cli_parser()
    neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
    heatmap_obj = heatmap(drop_directory=drop_directory,
                         neuronal_activity=neuronal_activity,
                         behavioral_timestamps=behavioral_timestamps,
                         neuron_info=neuron_info)
    heatmap_obj()

    print('Finished generating heatmaps...')
