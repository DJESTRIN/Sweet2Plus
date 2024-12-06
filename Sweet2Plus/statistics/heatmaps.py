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

from Sweet2Plus.statistics.coefficient_clustering import regression_coeffecient_pca_clustering, gather_data, cli_parser
import ipdb

def heatmap(regression_coeffecient_pca_clustering):
    def __call__(self):
        # Inherited methods
        self.normalize_activity()
        self.timestamps_to_one_hot_array()

        # New methods
        self.gather_data_by_trial()
        self.plot_data_by_trial()

    def gather_data_by_trial(self):
        ipdb.set_trace()

    def plot_data_by_trial(self):
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
