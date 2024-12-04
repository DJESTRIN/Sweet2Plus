#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: coefficient_clustering.py
Description: Performs clustering on regression coefficients. Following clustering, statistics and other interesting models can be developed
    to determine cluster's biological relevance. 
Author: David Estrin
Version: 1.0
Date: 12-03-2024

Note: Portions of code are based on code from Drs. Puja Parekh & Jesse Keminsky, Parekh et al., 2024 
"""

# Import depenencies
import argparse
import glob, os
import numpy as np
from Sweet2Plus.core.SaveLoadObjs import LoadObj
from statsmodels.regression.linear_model import OLS
from sklearn.cluster import KMeans as kmeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import ipdb

# Set up default matplotlib plot settings
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 24})

class regression_coeffecient_pca_clustering:
    def __init__(self, drop_directory, neuronal_activity, behavioral_timestamps, neuron_info, trial_list=['Vanilla','PeanutButter','Water','FoxUrine'],
                 normalize_neural_activity=False,regression_type='ridge'):
        """ regression_coeffecient_pca_clustering -- The primary point of this class is to:
          (1) reshape neural activity and behavioral timestamps 
          (2) regress activity onto the timestamps
          (3) grab coeffecients from regression for each neuron
          (4) Perform PCA demension reduction on coeffecients
          (5) Cluster neurons based on coeffecients 

        Inputs:
            drop_directory -- A path where results will be saved into. 
            neuronal_activity -- A list of numpy array (m x n) that includes each neuron's activity
            behavioral_timestamps -- A list of numpy arrays (m x n) that includes corresponding timestamps for behaviors for
                corresponding neuron in neuronal_activity list
            neuron_info -- A list of lists containg information regarding each neurons mouse, cage, session, etc. 
            normalize_neural_activity -- (True or False) Whether to z-score neuronal activity for each neuron. 
            regression_type -- Whether to run ridge or OLS regression

        Outputs (see drop_directory):
            cluster_results.jpg -- A heatmap image of results
            cluster_results.csv -- A dataframe containing each neuron's clustering results
        """
        self.drop_directory=drop_directory
        self.neuronal_activity = neuronal_activity
        self.behavioral_timestamps = behavioral_timestamps
        self.neuron_info = neuron_info
        self.trial_list = trial_list
        self.normalize_neural_activity = normalize_neural_activity
        
        # Set regression type
        self.regression_type = regression_type
        if self.regression_type!='ridge' and self.regression_type!='OLS':
            raise("regression_type must equal 'ridge' or 'OLS'. regression_type is currently incorrectly set...")
        
        #neural_activity = StandardScaler().fit_transform(neural_activity.T).T

    def timestamps_to_one_hot_array(self):
        beh_timestamp_onehots=[]
        for activity_oh,beh_oh in zip(self.neuronal_activity,self.behavioral_timestamps):
            one_hot_oh = np.zeros((activity_oh.shape[1],len(beh_oh)))

            for idx,beh in enumerate(beh_oh):
                for ts in beh:
                    one_hot_oh[int(ts),idx]=1

            beh_timestamp_onehots.append(one_hot_oh)
        self.behavior_ts_onehot=beh_timestamp_onehots

    def normalize_activity(self):
        if self.normalize_neural_activity:
            print("Normalizing Neuronal Activity for each neuron via z-score ....")
            for idx,neuron_activity in self.neuronal_activity:
                self.neuronal_activity[idx]=(neuron_activity-np.mean(neuron_activity))/np.std(neuron_activity)
        
    def ols_regression(self):
        """ Individually run's OLS regression on each neuron in dataset """
        # Create empty arrays to place coeffs
        self.coeffs = np.ndarray((self.neural_activity.shape[0], self.preds.shape[1]))

        # Loop over neurons in dataset, fit neural activity to predictors,  coeffs
        for neuron_idx, neuron in enumerate(self.neural_activity):
            ols_results = OLS(neuron, self.preds).fit()
            self.coeffs[neuron_idx] = ols_results.params

    def ridge_regression(self):
        """ Individually run's ridge regression on each neuron in dataset """
        self.all_coeffs=[]
        for recording_activity,recording_beh in zip(self.neuronal_activity,self.behavior_ts_onehot):
            # Create empty arrays to place coeffs
            recording_coeffs = np.zeros((recording_activity.shape[0], recording_beh.shape[1]))

            # Loop over neurons in dataset, fit neural activity to predictors, get tvalues, coeffs, and pvalues
            for neuron_idx, neuron in enumerate(recording_activity):
                ridge_results = Ridge(alpha=1.0).fit(neuron.reshape(-1,1),recording_beh)
                recording_coeffs[neuron_idx] = ridge_results.coef_.squeeze(-1)
            
            self.all_coeffs.append(recording_coeffs)
    
    def principal_component_analysis(self,values_to_be_clustered, max_clusters=20):
        # Convert list of lists to numpy array
        self.values_to_be_clustered=np.concatenate(values_to_be_clustered,axis=0)

        # Dimension reduction via PCA
        pca_results = PCA(n_components=min(self.values_to_be_clustered.shape)).fit_transform(self.values_to_be_clustered)
        cluster_range = range(2,max_clusters)

        # Determine best number of clusters unbiased via silhouette scores
        silhouette_scores = np.zeros(len(cluster_range))
        for idx, number_clusters in enumerate(cluster_range):
            if idx%5==0:
                print(f'Calculating silhouette score for {number_clusters} clusters')
            kmeans_results = kmeans(n_clusters=number_clusters, max_iter=1000).fit(pca_results)
            labels = kmeans_results.labels_
            silhouette_scores[idx] = silhouette_score(self.values_to_be_clustered,labels)
        lowest_sil = silhouette_scores.argmax()
        final_cluster_number = list(cluster_range)[lowest_sil]

        # Perform final kmeans clustering via correct number of clusters
        final_clusters = kmeans(n_clusters=final_cluster_number).fit(pca_results)
        final_labels = final_clusters.labels_

        # Sort data for easy visualization
        sort_indices = np.argsort(final_labels)
        self.sorted_values_to_be_clustered = self.values_to_be_clustered[sort_indices,:]
        self.sorted_final_labels = final_labels[sort_indices]

    def plot_cluster_results(self,plot_label='Coeffecients'):

        # Create horizontal lines
        hlines = [x_idx-0.5 for x_idx, x in enumerate(self.sorted_final_labels[1:]) if x!=self.sorted_final_labels[x_idx]]
        hlines.append(len(self.sorted_final_labels)-0.5)

        # Generate max and minimum bounding values
        bound = max([np.percentile(self.sorted_values_to_be_clustered,99),np.abs(np.percentile(self.sorted_values_to_be_clustered,1))])
        max_val = bound
        min_val = bound * -1

        plt.figure(figsize=[12,14])
        plt.imshow(self.sorted_values_to_be_clustered, aspect='auto', cmap='YlGnBu', vmin=min_val, vmax=max_val, origin='lower', interpolation='none')
        plt.xticks(np.arange(self.values_to_be_clustered.shape[1]), self.trial_list, rotation=-45, ha='left')
        plt.xlabel('Task Stimuli', fontsize=34)
        plt.ylabel('Neurons', fontsize=34)
        plt.hlines(hlines, xmin=-.5, xmax=self.values_to_be_clustered.shape[1]-.5, color='k',linewidth=2,linestyle='--')
        cbar = plt.colorbar()
        cbar.set_label(plot_label)
        plt.tight_layout()
        plt.savefig(os.path.join(self.drop_directory,f"{plot_label}_clustering.jpg"))
        plt.close()

    def __call__(self):
        # Clean data
        self.normalize_activity()
        self.timestamps_to_one_hot_array()

        # Run regression
        if self.regression_type=='ridge':
            self.ridge_regression()
        elif self.regression_type=='ols':
            self.ols_regression()

        # Run PCA and clustering
        self.principal_component_analysis(values_to_be_clustered=self.all_coeffs)

        # Plot clustering results
        self.plot_cluster_results(plot_label=f'{self.regression_type} regression coeffecients')

# class map_clusters_to_activity:
#     a=1
#     # Plot clusters across subjects, sessions and groups to make sure they are randomly distributed... Clusters arent holding info on mice
#     # Plot average +/- neuronal activity  for each trial type with respect to cluster to determine whether there are obvious differences
#     # 

# class svm_neuronal_activity:
#     # Take behavioral time stamps and neuronal activity and get svm deconding results
#     a=1

# class svm_based_on_cluster:
#     # Do the same as above but divide by cluster
#     a=1

def gather_data(parent_data_directory,file_indicator='obj'):
    """ Gather all data into lists from parent directory """
    # Get full path to object files
    objfiles=glob.glob(os.path.join(parent_data_directory,f'**/{file_indicator}*.json'),recursive=True)

    # Grab relevant data from files and create lists
    neuronal_activity=[]
    behavioral_timestamps=[]
    neuron_info=[]
    for file in objfiles:
        objoh=LoadObj(FullPath=file)
        neuronal_activity.append(objoh.ztraces)
        behavioral_timestamps.append(objoh.all_evts_imagetime)
        neuron_info.append([objoh.day, objoh.cage, objoh.mouse, objoh.group])
    
    return neuronal_activity, behavioral_timestamps, neuron_info

def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,help='Parent directory where data is located')
    parser.add_argument('--drop_directory',type=str,help='where results are saved to')
    args=parser.parse_args()
    return args.data_directory, args.drop_directory

if __name__=='__main__':
    data_directory, drop_directory = cli_parser()
    neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory)
    regressobj = regression_coeffecient_pca_clustering(drop_directory=drop_directory,
                                                       neuronal_activity=neuronal_activity,
                                                       behavioral_timestamps=behavioral_timestamps,
                                                       neuron_info=neuron_info)
    regressobj()

    ipdb.set_trace()
