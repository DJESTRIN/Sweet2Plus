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
from collections import Counter
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
                try:
                    all_av_neu_for_trial = [ts.mean(axis=1) for ts in trial]
                    shapes = [array.shape for array in all_av_neu_for_trial]
                    shape_counts = Counter(shapes)
                    most_common_shape = shape_counts.most_common(1)[0][0]  # The most common shape
                    all_av_neu_for_trial = [array for array in all_av_neu_for_trial if array.shape == most_common_shape]

                except:
                    ipdb.set_trace()
                    
                trial_mean = all_av_neu_for_trial.mean(axis=0)
                all_avs_by_trial.append(trial_mean)

            try:
                all_avs_by_trial = np.array(all_avs_by_trial)
            except:
                ipdb.set_trace()

            all_subject_avs_by_trial.append(all_avs_by_trial)
        ipdb.set_trace()
        all_subject_avs_by_trial = np.array(all_subject_avs_by_trial)

    def plot_data_by_trial(self):
        """ """
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
