#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: UnsupervisedLearning.py
Description: Performs clustering on neuronal activity. Dimensionality reduction via PCA and then clustering via Kmeans (for now...)
Author: David Estrin
Version: 1.0
Date: 03-12-2025
"""

# Import depenencies
import argparse
import glob, os
import numpy as np
from Sweet2Plus.core.SaveLoadObjs import LoadObj
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import ipdb
import tqdm
import pandas as pd
import seaborn as sns

# Set up default matplotlib plot settings
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['axes.linewidth'] = 2 

class DimensionalityReduction:
    def __init__(self,neuronal_activity,behavioral_timestamps,neuron_info,components=None):
        self.neuronal_activity=neuronal_activity
        self.behavioral_timestamps=behavioral_timestamps
        self.neuron_info=neuron_info
        self.components = components
    
    def __call__(self):
        self.reformat_data() # put data into a format for pca
        self.pca() # Run pca

    def reformat_data(self):
        self.activitydata = []
        min_columns = min(arrayoh.shape[1] for arrayoh in self.neuronal_activity)
        for arrayoh in self.neuronal_activity:
            self.activitydata.append(arrayoh[:, :min_columns])

        self.activitydata  = np.vstack( self.activitydata)

    def pca(self):
        if self.components is None:
            pca = PCA()
            pca.fit(self.activitydata)

            # Calculate the cumulative explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            # Plot the elbow plot
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', color='b')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Elbow Plot for PCA')
            plt.axvline(x=4, linestyle='--', color='r')  # Example: the elbow is at 4 components (you can adjust this)
            plt.savefig('pcaelbowplot.jpg')
        else:
            scaler = StandardScaler()
            self.activitydata_scaled = scaler.fit_transform(self.activitydata)
            pca = PCA(n_components=self.components)
            self.pca_data = pca.fit_transform(self.activitydata_scaled)

            # Generate figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            x = self.pca_data[:, 0]  # PC1
            y = self.pca_data[:, 1]  # PC2
            z = self.pca_data[:, 2]  # PC3
            ax.scatter(x, y, z, c='blue', marker='o', alpha=0.1, s=10)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('3D Scatter Plot of PCA with 500 Components')
            plt.savefig('pca500comp.jpg')

class Clustering(DimensionalityReduction):
    def __call__(self):
        super().__call__()
        self.dbscanit()
        # bestn = self.ncluster_search()
    
    def ncluster_search(self,minn=2,maxn=20):

        sil_scores = []

        # Loop over different numbers of clusters
        for n_clusters in range(minn, maxn + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(self.pca_data)  # Fit K-Means on PCA data
            cluster_labels = kmeans.labels_  # Get the cluster labels
            silhouette_avg = silhouette_score(self.pca_data, cluster_labels)
            sil_scores.append(silhouette_avg)

        # Plot silhouette scores for each number of clusters
        plt.figure(figsize=(8, 6))
        plt.plot(range(minn, maxn + 1), sil_scores, marker='o')
        plt.title('Silhouette Score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.savefig('nclustersearch.jpg')

        best_n_clusters = np.argmax(sil_scores) + minn
        return best_n_clusters
    
    def dbscanit(self):
        print('Running DBSCAN ...')
        dbscan = DBSCAN(eps=0.3, min_samples=10)
        dbscan_labels = dbscan.fit_predict(self.pca_data)

        # Remove noise points (where label == -1)
        ipdb.set_trace()
        mask = dbscan_labels != -1
        pca_data_filtered = self.pca_data[mask]
        dbscan_labels_filtered = dbscan_labels[mask]

        # Calculate silhouette score
        sil_score = silhouette_score(pca_data_filtered, dbscan_labels_filtered)
        print(f'The sil score is {sil_score}')
        
        # Assuming pca_data is 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pca_data[:, 0], self.pca_data[:, 1], self.pca_data[:, 2], c=dbscan_labels, cmap='viridis', s=50)
        ax.set_title("DBSCAN Clustering Results")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        plt.savefig('dbscanresults.jpg')
        ipdb.set_trace()

def gather_data(parent_data_directory,drop_directory,file_indicator='obj'):
    """ Gather all data into lists from parent directory """
    # Get full path to object files
    objfiles=glob.glob(os.path.join(parent_data_directory,f'**/{file_indicator}*.json'),recursive=True)
 
    # Grab relevant data from files and create lists
    neuronal_activity=[]
    behavioral_timestamps=[]
    neuron_info = pd.DataFrame(columns=['day', 'cage', 'mouse', 'group'])
    for file in tqdm.tqdm(objfiles):
        objoh=LoadObj(FullPath=file)
        neuronal_activity.append(objoh.ztraces)
        behavioral_timestamps.append(objoh.all_evts_imagetime)
        repeated_info = np.tile([objoh.day, objoh.cage, objoh.mouse, objoh.group], objoh.ztraces.shape[0]) 
        repeated_info = repeated_info.reshape(objoh.ztraces.shape[0], 4)
        repeated_info_df = pd.DataFrame(repeated_info, columns=['day', 'cage', 'mouse', 'group'])
        neuron_info = pd.concat([neuron_info, repeated_info_df], ignore_index=True)

    return neuronal_activity, behavioral_timestamps, neuron_info

def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,help='Parent directory where data is located')
    parser.add_argument('--drop_directory',type=str,help='where results are saved to')
    args=parser.parse_args()
    return args.data_directory, args.drop_directory

if __name__=='__main__':
    data_directory, drop_directory = cli_parser()
    neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
    objoh = Clustering(neuronal_activity,behavioral_timestamps,neuron_info,components=500)
    objoh()

    print('Finished PCA clustering...')
