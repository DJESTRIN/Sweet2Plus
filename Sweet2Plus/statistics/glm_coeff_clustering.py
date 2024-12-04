#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: glm_w_clustering.py
Description: Performs a glm wiht clustering. 
Author: Puja Parekh, Jesse Keminsky
Version: 1.0
Date: 11-25-2024

Note: This code will likely be soon removed from repo because it is not utalized. 
"""

# Import depenencies
import glob, os
import numpy as np
import scipy.io as sio
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as kmeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Set up default matplotlib plot settings
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 24})

class glm_coeffecient_pca_clustering:
    def __init__(self, neuronal_activity,behavioral_timestamps,normalize_neural_activity=False):
        self.neuronal_activity=neuronal_activity
        self.normalize_neural_activity=normalize_neural_activity
        #neural_activity = StandardScaler().fit_transform(neural_activity.T).T

    def timestamps_to_one_hot_array(self):
        a=2

    def normalize_activity(self):
        print("Normalizing Neuronal Activity for each neuron via z-score ....")
        if self.normalize_neural_activity:
            for idx,neuron_activity in self.neuronal_activity:
                self.neuronal_activity[idx]=(neuron_activity-np.mean(neuron_activity))/np.std(neuron_activity)
        
    def ols_regression(self):
        """ Individually run's OLS regression on each neuron in dataset """
        # Create empty arrays to place tvalues, pvalues and coeffs
        self.tvals = np.ndarray((self.neural_activity.shape[0], self.preds.shape[1]))
        self.pvals = np.ndarray((self.neural_activity.shape[0], self.preds.shape[1]))
        self.coeffs = np.ndarray((self.neural_activity.shape[0], self.preds.shape[1]))

        # Loop over neurons in dataset, fit neural activity to predictors, get tvalues, coeffs, and pvalues
        for neuron_idx, neuron in enumerate(self.neural_activity):
            ols_results = OLS(neuron, self.preds).fit()
            self.tvals[neuron_idx] = ols_results.tvalues
            self.coeffs[neuron_idx] = ols_results.params
            self.pvals[neuron_idx] = ols_results.pvalues
    
    def principal_component_analysis(self,values_to_be_clustered, max_clusters=20):

        # Dimension reduction via PCA
        pca_results = PCA(n_components=min(values_to_be_clustered.shape)).fit_transform(values_to_be_clustered)
        cluster_range = range(2,max_clusters)

        # Determine best number of clusters unbiased via silhouette scores
        silhouette_scores = np.zeros(len(cluster_range))
        for idx, number_clusters in enumerate(cluster_range):
            kmeans_results = kmeans(n_clusters=number_clusters, max_iter=1000).fit(pca_results)
            labels = kmeans_results.labels_
            silhouette_scores[idx] = silhouette_score(values_to_be_clustered,labels)
        lowest_sil = silhouette_scores.argmax()
        final_cluster_number = list(cluster_range)[lowest_sil]

        # Perform final kmeans clustering via correct number of clusters
        final_clusters = kmeans(n_clusters=final_cluster_number).fit(pca_results)
        final_labels = final_clusters.labels_


        # Sort values for plotting?

    def plot_cluster_results(self,values_to_be_plotted,plot_label='Coeffecients'):

        # Generate max and minimum bounding values
        bound = max([np.percentile(values_to_be_plotted,99),np.abs(np.percentile(values_to_be_plotted,1))])
        max_val = bound
        min_val = bound * -1

        plt.figure(figsize=[12,14])
        plt.imshow(values_to_be_plotted, aspect='auto', cmap='YlGnBu', vmin=min_val, vmax=max_val, origin='lower', interpolation='none')
        plt.xticks(np.arange(values_to_be_plotted.shape[1]), self.predictor_labels, rotation=-45, ha='left')
        plt.xlabel('Predictors', fontsize=34)
        plt.ylabel('Neurons', fontsize=34)
        cbar = plt.colorbar()
        cbar.set_label(plot_label)
        plt.tight_layout()
        plt.savefig(os.path.join(self.drop_directory,f"{plot_label}_clustering.jpg"))
        plt.close()
        
        # TO DO: Add in hlines for nicer plotting between trials
        # plt.hlines(hlines, xmin=-.5, xmax=vals.shape[1]-.5, color='k')

    def __call__(self):
        a=1
       
if __name__=='__main__':
    a=1
