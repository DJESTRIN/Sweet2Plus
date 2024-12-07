#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: decoders.py
Description: Contains classes for decoding neuronal activity. 
Author: David Estrin
Version: 1.0
Date: 12-06-2024

Analysis 1:
    What are we trying to test? -- We are trying to determine what percent of neurons encode information regarding each stimulus...
    What is the hypothesis? -- I hypothesize that most neurons encode information regarding the TMT stimulus 
    How should we graph the data? -- We should graph the trial type in x axis and decoder accuracy +/- sem in y axis. 
        We should also seperate the data based on Session and Group. 
    
Analysis 2:
    What are we trying to test? -- We are trying to determine whether clustered neurons do in fact encode different aspects of task
    What is the hypothesis? -- I hypothesize that different clusters activity decode better/worse certain stimuli than others
    How should we graph the data? -- 

"""

class format_data:
    def __init__(self, neuronal_activity=None, behavior_timestamps=None, behavior_timestamps_onehot=None, subgroups=None):

    def onehot_to_y(self):

    def activity_to_X(self):

    def clean_and_split_data(self):

    def support_vector_machine
    # Use argmax to find the index of the non-zero element in each row


class svm:

class neural_network:

class statistics_and_graphics:
    """ Generates statistics """

linear_vector = np.argmax(one_hot_matrix, axis=1)

# Handle cases where all values are zero (e.g., inter-trial intervals)
linear_vector[np.all(one_hot_matrix == 0, axis=1)] = -1