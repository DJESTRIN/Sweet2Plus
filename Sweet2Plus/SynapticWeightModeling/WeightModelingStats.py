#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: WeightModelingStats.py
Description: 
Author: David James Estrin
Version: 2.0
Date: 03-03-2025
"""

"""
Questions to investigate:
    (1) Is there a general difference in weight values across interaction of group*day 
        (a) What is the distribution of weights across input neurons? 
             - Do certain neurons have more low theta weights than others? 
             - Does every neuron have at least one high theta weight?

    (2) Are there specific groups of neurons with weight values that change across group*day
        (a) Additionally, is there a feature regarding these neuron's activities that make them distinct?

    (3) How are bias values modified across experimental conditions?
        (a) If bias values are modified, what might be the biological significance?
"""
from Sweet2Plus.statistics.mixedmodel import mixedmodels
import numpy as np
import ipdb

class weightdata():
    """ Gathers and organizes all weight data into several tables """
    def __init__(self,parent_directory):
        self.parent_directory = parent_directory
        self.weight_df = []

class HyperparameterAnalysis(weightdata):
    # Generate figures regarding best hyperparemeters for all models
        # Spageti plot
    def __init__():
        ipdb.set_trace()

class weightclustering(weightdata):
    def __init__(self):
        ipdb.set_trace()

    def pca_weights(self):
        ipdb.set_trace()
    
    def cluster_pca_outputs(self):
        ipdb.set_trace()

    def analyze_cluster_activity_by_trialtype(self):
        ipdb.set_trace()

    def determine_cluster_distributions(self):
        ipdb.set_trace()
        # Are certain clusters from certain groups, days, responses to trial type

class weightfrequentiststats(weightdata):
    def __init__(self):
        ipdb.set_trace()

    def __call__(self):
        super().__call__(self)
        
        weight_group_day_df = self.gather_data()
        self.plot_data(df = weight_group_day_df)

    def gather_data(self,grouping_labels=['suid','group','day']):
        result = self.weight_df.groupby(grouping_labels).agg(weight_mean = ('weights',lambda x: np.mean(np.vstack(x),axis=0)),
                                             weight_se = ('weights',lambda x: np.std(np.vstack(x),axis=0)/np.sqrt(len(np.vstack(x)))))
        return result

    def plot_data(self, df):
        ipdb.set_trace()

    def model_data():
        mixedmodels()


# Generate plots and stats on average +/- sem stats for weights across conditions
    # What should they be averaged across?