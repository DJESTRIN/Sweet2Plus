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

# Generate figures regarding best hyperparemeters for all models
    # Spageti plot

# Generate a figure for all weights across all neurons
    # PCA weights to a common number of weights

# Run cluster on weights for all neurons on PCA output
    # Plot pca results with respect to trial activity to see if certain cell types have specific activity changes

# Generate plots and stats on average +/- sem stats for weights across conditions
    # What should they be averaged across?