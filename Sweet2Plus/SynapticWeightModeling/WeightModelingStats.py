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
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import os, glob
from scipy.stats import ttest_ind
from statsmodels.stats.anova import anova_lm
import ipdb

class weightdata():
    """ Gathers and organizes all weight data into several tables """
    def __init__(self,parent_directory):
        self.parent_directory = parent_directory
        self.weights = pd.DataFrame()
        self.to_table()
        #self.quick_stats()
    
    def to_table(self):
        search_string = os.path.join(self.parent_directory,'**/*weights_df*.csv')
        found_files = glob.glob(search_string, recursive=True)

        for file in found_files:
            temp = pd.read_csv(file, index_col=False)
            self.weights = pd.concat([self.weights,temp],ignore_index=True)
        
        self.weights = self.weights.drop(self.weights.columns[0], axis=1)
        self.weights['abs_synaptic_weight'] = np.abs(self.weights['synaptic_weight'])
        self.weights['neuron_id_2'] = (
            self.weights['suid'].astype(str) + "_" + 
            self.weights['day'].astype(str) + "_" + 
            self.weights['group'].astype(str) + "_" + 
            self.weights['neuron_id_2'].astype(str))

        # Convert columns to categorical variables
        self.weights["group"] = self.weights["group"].astype("category")
        # self.weights["day"] = self.weights["day"].astype("category")
        self.weights["suid"] = self.weights["suid"].astype("category")
        self.weights["neuron_id_1"] = self.weights["neuron_id_1"].astype("category")
        self.weights["neuron_id_2"] = self.weights["neuron_id_2"].astype("category")
        # self.weights["connection_id"] = self.weights["connection_id"].astype("category")

    def quick_stats(self):
        print('creating model...')
        full_model = smf.mixedlm("abs_synaptic_weight ~ group * day", self.weights, groups=self.weights["suid"], re_formula="1", vc_formula={"connectionid": "1"}).fit()
        reduced_model = smf.mixedlm("abs_synaptic_weight ~ group + day", self.weights, groups=self.weights["suid"], re_formula="1", vc_formula={"connectionid": "1"}).fit()
        ipdb.set_trace()
        print(anova_lm(reduced_model, full_model))

        

        ipdb.set_trace()


class graphics(weightdata):
    def __call__(self):
        self.average_sem()

    def average_sem(self):
        running_average = self.weights.groupby(['day', 'suid', 'group','neuron_id_2'])['abs_synaptic_weight'].mean().reset_index()
        running_average = running_average.groupby(['suid','group','day'])['abs_synaptic_weight'].mean().reset_index()
        running_average2 = running_average
        running_average = running_average.groupby(['group','day']).agg( mean=('abs_synaptic_weight', 'mean'),
                                                          std_error=('abs_synaptic_weight', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))))
        running_average = running_average.reset_index()
        groups = running_average['group']
        means = running_average['mean']
        errors = running_average['std_error']
        df = pd.DataFrame({'Group': groups, 'Mean': means, 'StdError': errors})

        sns.set_style("whitegrid")  # Automatically adds dashed horizontal lines
        plt.figure(figsize=(6, 4))
        ax = sns.barplot(data=df, x='Group', y='Mean', ci=None, palette=['green', 'red'])
        for i, (mean, error) in enumerate(zip(means, errors)):
            plt.errorbar(i, mean, yerr=error, fmt='none', capsize=5, color='black')
        plt.xlabel('Group')
        plt.ylabel('Average of Synaptic Weight')
        plt.ylim(0.08,0.1)
        plt.tight_layout()
        plt.savefig('group_av.jpg')




        # Calculate the baseline std
        ipdb.set_trace()
        baseline = self.weights[self.weights['day'] == 0].groupby(['suid', 'group', 'neuron_id_2'])['abs_synaptic_weight'].mean().reset_index()
        baseline.rename(columns={'abs_synaptic_weight': 'baseline'}, inplace=True)

        # Merge baseline data back into the weights dataframe
        self.weights = self.weights.merge(baseline, on=['suid', 'group', 'neuron_id_2'], how='left')

        # Calculate the baseline std
        baseline_std = self.weights[self.weights['day'] == 0].groupby(['suid', 'group', 'neuron_id_2'])['abs_synaptic_weight'].std().reset_index()
        baseline_std.rename(columns={'abs_synaptic_weight': 'std_baseline'}, inplace=True)

        # Merge baseline std back into the weights dataframe
        self.weights = self.weights.merge(baseline_std, on=['suid', 'group', 'neuron_id_2'], how='left')

        # Step 2: Normalize the synaptic weights by subtracting the baseline and dividing by the std_baseline
        self.weights['normalized_weight'] = (self.weights['abs_synaptic_weight'] - self.weights['baseline']) / self.weights['std_baseline']

        # Step 3: Group by suid, calculate mean normalized weight
        running_average = self.weights.groupby(['day', 'suid', 'group', 'neuron_id_2'])['synaptic_weight'].mean().reset_index()
        running_average = running_average.dropna(subset=['synaptic_weight'])
        running_average = running_average.groupby(['day', 'suid', 'group'])['synaptic_weight'].mean().reset_index()
        # Step 4: Group by day and group and calculate the mean and std error over suid
        running_average = running_average.groupby(['day', 'group']).agg(
            mean=('normalized_weight', lambda x: np.nanmean(x)),
            std_error=('normalized_weight', lambda x: np.nanstd(x, ddof=1) / np.sqrt(len(x)))
        ).reset_index()

        running_average['std_error'] = pd.to_numeric(running_average['std_error'], errors='coerce')
        ipdb.set_trace()
        # Step 5: Faceting using Seaborn
        # Create a Seaborn FacetGrid, facet by 'day' column
        g = sns.FacetGrid(running_average, col="day", hue="group", height=6, aspect=1.2, margin_titles=True)

        # Step 6: Plot the data (barplot for each facet)
        g.map(sns.barplot, 'group', 'mean', yerr='std_error', capsize=5, width=0.3)

        # Add labels and titles
        g.set_axis_labels("Group", "Average Normalized Synaptic Weight")
        g.set_titles(col_template="Day {col_name}")
        g.add_legend(title="Group")

        # Adjust the layout to avoid overlap
        plt.tight_layout()

        # Step 7: Save and show the plot
        plt.savefig('group_day_normalized_faceted.jpg')
        


if __name__ == '__main__':
    graphobh = graphics(parent_directory=r'C:\Users\listo\SynapticWeightModeling')
    graphobh()