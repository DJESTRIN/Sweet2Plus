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
from Sweet2Plus.core.SaveLoadObjs import SaveObj, LoadObj, SaveList, OpenList
from statsmodels.regression.linear_model import OLS
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
import ipdb
import tqdm
import pandas as pd
import seaborn as sns
from itertools import combinations


# Set up default matplotlib plot settings
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['axes.linewidth'] = 2 

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
        beh_timestamp_onehots = []

        for activity_oh, beh_oh in zip(self.neuronal_activity, self.behavioral_timestamps):
            num_timepoints = activity_oh.shape[1]
            num_behaviors = len(beh_oh)

            # Create the base one-hot encoding
            one_hot_oh = np.zeros((num_timepoints, num_behaviors), dtype=int)

            for idx, beh in enumerate(beh_oh):
                for ts in beh:
                    ts = int(ts)  # Ensure it's an integer
                    if 0 <= ts < num_timepoints:  # Check to ensure ts is within bounds
                        # Set a 1 for the range from ts to ts+10
                        end_ts = min(ts + 20, num_timepoints)  # Ensure not to go beyond num_timepoints
                        one_hot_oh[ts:end_ts, idx] = 1

            # Generate combination columns
            all_combinations = []
            for r in range(2, num_behaviors + 1):  # 2-way to N-way combinations
                all_combinations.extend(combinations(range(num_behaviors), r))

            # Expand one-hot matrix to include combinations
            extended_one_hot = np.zeros((num_timepoints, num_behaviors + len(all_combinations)), dtype=int)
            extended_one_hot[:, :num_behaviors] = one_hot_oh  # Copy original one-hot

            # Fill combination columns
            for comb_idx, comb in enumerate(all_combinations):
                new_col_idx = num_behaviors + comb_idx  # Column index in extended_one_hot
                extended_one_hot[:, new_col_idx] = np.any(one_hot_oh[:, comb], axis=1)

            beh_timestamp_onehots.append(extended_one_hot)

        self.behavior_ts_onehot = beh_timestamp_onehots


    def normalize_activity(self):
        if self.normalize_neural_activity:
            print("Normalizing Neuronal Activity for each neuron via z-score ....")
            for idx,neuron_activity in self.neuronal_activity:
                self.neuronal_activity[idx]=(neuron_activity-np.mean(neuron_activity))/np.std(neuron_activity)
        
    def ols_regression(self):
        """ Individually run's OLS regression on each neuron in dataset """
        self.all_coeffs=[]
        for recording_activity,recording_beh in zip(self.neuronal_activity,self.behavior_ts_onehot):
            # Create empty arrays to place coeffs
            recording_coeffs = np.zeros((recording_activity.shape[0], recording_beh.shape[1]))

            # Loop over neurons in dataset, fit neural activity to predictors, get tvalues, coeffs, and pvalues
            for neuron_idx, neuron in enumerate(recording_activity):
                ols_results = OLS(neuron.reshape(-1,1),recording_beh).fit()
                recording_coeffs[neuron_idx] = ols_results.params
  
            self.all_coeffs.append(recording_coeffs)

    def calculate_auc_and_behavior_comb(self, filtered_neuron_data, filtered_behavior_data):
        def condense_array(arr):
            condensed_rows = [row[np.insert(np.diff(row) != 0, 0, True)] for row in arr]
            return np.vstack(condensed_rows)
        
        def binary_to_index(arr):
            """Converts each row of a binary 4-column array to a unique index (1-15)."""
            return np.dot(arr, [8, 4, 2, 1]) - 1  # Convert binary to decimal (1-15 index)

        def convert_to_15_columns(arr):
            """Converts Mx4 binary array to Mx15 one-hot encoded array."""
            indices = binary_to_index(arr)  # Get indices (0-14)
            result = np.zeros((arr.shape[0], 15), dtype=int)  # Initialize 15-column array
            result[np.arange(arr.shape[0]), indices] = 1  # Set 1s at the right positions
            return result

        behshortened = filtered_behavior_data[:,0:4] 

        trial_boolean = np.sum(filtered_behavior_data[:,0:4],axis=1)
        trial_ones = np.where(trial_boolean == 1)[0] 

        trial_starts = [trial_ones[0]]
        trial_stops = []
        for i in range(0, len(trial_ones)-1):
            if (trial_ones[i+1] - trial_ones[i])>1:  # Gap between 1s indicates end of trial
                trial_stops.append(trial_ones[i])
                if i != (len(trial_ones)-1):
                    trial_starts.append(trial_ones[i+1])

            if i == len(trial_ones)-2:
                trial_stops.append(trial_ones[-1])

        AUCs=[]
        for start,stop in zip(trial_starts,trial_stops):
            AUCs.append(np.trapz(filtered_neuron_data[start:stop]))
        AUC = np.asarray(AUCs)

        trial_order = []
        previous_trial = None

        for row in behshortened:
            if np.all(row == 0):  # Skip zero rows (invalid trials)
                previous_trial = row 
                continue
            if previous_trial is None or not np.array_equal(row, previous_trial):
                trial_order.append(np.argmax(row))  # Store index of active column
                previous_trial = row  # Update previous trial
            else:
                previous_trial = row 

        beh_condensed = np.eye(4)[trial_order]  # Reconstruct one-hot encoding
        col_combinations = [list(combinations(range(beh_condensed.shape[1]), r)) for r in range(1, beh_condensed.shape[1] + 1)]
        col_combinations = [combo for sublist in col_combinations for combo in sublist]  # Flatten list
        beh_condensed = np.array([np.any(beh_condensed[:, list(combo)], axis=1) for combo in col_combinations]).T

        return AUC, beh_condensed

    def ridge_regression(self):
        """ Individually run's ridge regression on each neuron in dataset """
        self.all_coeffs=[]
        self.all_r2s = []
        for recording_activity,recording_beh in tqdm.tqdm(zip(self.neuronal_activity,self.behavior_ts_onehot),total=len(self.behavior_ts_onehot)):
            # Create empty arrays to place coeffs
            recording_coeffs = np.zeros((recording_activity.shape[0], recording_beh.shape[1]))

            # Loop over neurons in dataset, fit neural activity to predictors, get tvalues, coeffs, and pvalues
            for neuron_idx, neuron in enumerate(recording_activity):
                
                auc_oh, behavior_oh = self.calculate_auc_and_behavior_comb(filtered_neuron_data=neuron,
                                                                          filtered_behavior_data=recording_beh)
                behavior_oh = behavior_oh.astype(int)
                
                # Standardize the predictor variable
                scaler = StandardScaler()
                auc_oh_scaled = scaler.fit_transform(auc_oh.reshape(-1, 1))

                # Create polynomial features (up to 3rd order interactions)
                poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
                auc_oh_poly = poly.fit_transform(auc_oh_scaled)  # Expands feature set

                # Fit Ridge Regression with polynomial features
                ridge_model = Ridge(alpha=1.0)
                ridge_results = ridge_model.fit(auc_oh_poly, behavior_oh)

                # Compute R^2 on training data
                r2_score = ridge_model.score(auc_oh_poly, behavior_oh.astype(int))
                print(f"Training R^2: {r2_score}")

                # Cross-validation to check for overfitting
                cv_scores = cross_val_score(ridge_model, auc_oh_poly, behavior_oh.astype(int), cv=5, scoring='r2')

                # Print mean and standard deviation of cross-validated R^2
                print(f"Cross-validated R^2: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

                self.all_r2s.append(r2_score)
                recording_coeffs[neuron_idx] = ridge_results.coef_.squeeze(-1)
            
            self.all_coeffs.append(recording_coeffs)
        ipdb.set_trace()
    
    def principal_component_analysis(self,values_to_be_clustered, max_clusters=20):
        # Convert list of lists to numpy array
        self.values_to_be_clustered=np.concatenate(values_to_be_clustered,axis=0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.values_to_be_clustered)

        # Perform t-SNE
        tsne = TSNE(n_components=3, random_state=42)  # You can also set n_components=3 for 3D
        X_tsne = tsne.fit_transform(X_scaled)

        # Plot the results (2D plot for now)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c='blue', marker='o', alpha=0.7)
        ax.set_xlabel('tsne 1')
        ax.set_ylabel('tsne 2')
        ax.set_zlabel('tsne 3')
        ax.set_title('tsne - 3D Plot')
        plt.savefig('tsne_results.jpg')
        plt.close()


        ipdb.set_trace()

        # Generate pca plot 
        pca = PCA()
        X_pca = pca.fit_transform(self.values_to_be_clustered)

        # Plot cumulative explained variance
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid()
        plt.savefig('cumulativevariance.jpg')
        plt.close()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='blue', marker='o', alpha=0.7)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('PCA - 3D Plot')
        plt.savefig('pca_results.jpg')
        plt.close()

        # Dimension reduction via PCA
        pca_results = PCA(n_components=3).fit_transform(self.values_to_be_clustered)
        cluster_range = range(2,max_clusters)

        # Determine best number of clusters unbiased via silhouette scores
        silhouette_scores = np.zeros(len(cluster_range))
        for idx, number_clusters in enumerate(cluster_range):
            if number_clusters%5==0:
                print(f'Calculating silhouette score for {number_clusters} clusters')
            labels = GaussianMixture(n_components=number_clusters, random_state=42).fit_predict(pca_results)
            #kmeans_results = kmeans(n_clusters=number_clusters, max_iter=1000).fit(pca_results)
            #labels = kmeans_results.labels_
            silhouette_scores[idx] = silhouette_score(pca_results,labels)
            print(f'The sil score is {silhouette_scores[idx]}')
        lowest_sil = silhouette_scores.argmax()
        final_cluster_number = list(cluster_range)[lowest_sil]
        print(f'The final cluster number is {final_cluster_number} clusters with a silhouette score of {lowest_sil}.')
        ipdb.set_trace()

        # Perform final kmeans clustering via correct number of clusters
        final_clusters = kmeans(n_clusters=final_cluster_number).fit(pca_results)
        final_labels = final_clusters.labels_

        # Sort data for easy visualization
        sort_indices = np.argsort(final_labels)
        self.sorted_values_to_be_clustered = self.values_to_be_clustered[sort_indices,:]
        self.sorted_final_labels = final_labels[sort_indices]
        self.sorted_neuron_info = self.neuron_info.iloc[sort_indices]
        self.sort_indices=sort_indices

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
        elif self.regression_type=='OLS':
            self.ols_regression()
        
        ipdb.set_trace()

        # Run PCA and clustering
        self.principal_component_analysis(values_to_be_clustered=self.all_coeffs)

        # Plot clustering results
        self.plot_cluster_results(plot_label=f'{self.regression_type} regression coeffecients')

class map_clusters_to_activity(regression_coeffecient_pca_clustering):
    def __call__(self):
        super().__call__()
        # Plot distribution of neurons in cluster
        self.distribution_of_neurons_in_clusters()

        # Plot trial activity by cluster
        self.get_activity_by_cluser()
        self.plot_activity_by_cluser()
        self.plot_heat_maps()

    def distribution_of_neurons_in_clusters(self,columns = ['day', 'cage', 'mouse', 'group']):
        """ Generate plot of average number of neurons in each cluster w.r.t group and day
         The primary purpose of this method is to analyze how the number of neurons in each group change as a 
         function of day. In other words, are there more or less number of neurons per group during a given session? """
        # Build a dataframe for plotting cluster info by session and group
        self.sorted_neuron_info['subjectid'] = self.sorted_neuron_info['mouse'].astype(str) + "_" + self.sorted_neuron_info['cage'].astype(str)
        self.sorted_neuron_info = self.sorted_neuron_info.drop(columns=['mouse', 'cage'])
        cluster_df = pd.DataFrame(self.sorted_final_labels, columns=['cluster'])
        cluster_info_df = pd.concat([self.sorted_neuron_info, cluster_df], axis=1)
        cluster_counts = cluster_info_df.groupby(['group', 'day', 'subjectid', 'cluster']).size().reset_index(name='count')
        subject_totals = cluster_counts.groupby('subjectid')['count'].transform('sum')
        cluster_counts['count'] = cluster_counts['count'] / subject_totals
        plot_data = cluster_counts.groupby(['group', 'day', 'cluster']).agg(mean_count=('count', 'mean'),sem_count=('count', 'sem')).reset_index()
        plot_data['day'] = pd.to_numeric(plot_data['day'], errors='coerce')

        # Generate plot
        g = sns.catplot(
            data=plot_data,
            x='day',
            y='mean_count',
            hue='cluster',
            col='group',
            kind='bar',
            errorbar=None,  # Manual error bars
            palette='Set2',
            height=5,
            aspect=1.2
        )

        # Customize plot
        g.set_axis_labels('Session', 'Normalized # Neurons')
        g.set_titles('Group: {col_name}')
        g.set(ylim=(0, None))
        g.figure.suptitle('Distribution of Cluster Values by Group and Session')
        plt.legend(
            title='cluster',
            bbox_to_anchor=(1.05, 1),  # Position to the right of the plot
            loc='upper left',          # Anchor at the upper left of the legend box
            borderaxespad=0
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.drop_directory,"distribution_of_clusters.jpg"))
        plt.close()

    def get_activity_by_cluser(self, preceding_frames=20,post_stim_frames=26):
        """Plot average +/- neuronal activity  for each trial type with respect to cluster to 
            determine whether there are differences"""

        self.preceding_frames = preceding_frames
        self.post_stim_frames = post_stim_frames
        self.preceding_seconds = self.preceding_frames/1.315
        self.post_stim_seconds = self.post_stim_frames/1.315

        # Stack neuronal data
        self.activity_stack = np.concat(self.neuronal_activity,axis=0)
        self.activity_stack_sort = self.activity_stack[self.sort_indices]

        # Build behavioral array
        van_trials=[]
        pb_trials=[]
        wat_trials=[]
        tmt_trials=[]

        for behoh,neuoh in zip(self.behavioral_timestamps,self.neuronal_activity):
            van,pb,wat,tmt=behoh
            van_trials.append(np.tile(van, (neuoh.shape[0],1)))
            pb_trials.append(np.tile(pb, (neuoh.shape[0],1)))
            wat_trials.append(np.tile(tmt, (neuoh.shape[0],1)))
            tmt_trials.append(np.tile(wat, (neuoh.shape[0],1)))

        # Reformat to a single array
        target_shape = van_trials[0].shape
        adjusted_arrays = [arr[:, :target_shape[1]] for arr in van_trials]
        van_trials=np.concat(adjusted_arrays,axis=0)
        van_trials=van_trials[self.sort_indices]

        target_shape = pb_trials[0].shape
        adjusted_arrays = [arr[:, :target_shape[1]] for arr in pb_trials]
        pb_trials=np.concat(adjusted_arrays,axis=0)
        pb_trials=pb_trials[self.sort_indices]

        target_shape = wat_trials[0].shape
        adjusted_arrays = [arr[:, :target_shape[1]] for arr in wat_trials]
        wat_trials=np.concat(adjusted_arrays,axis=0)
        wat_trials=wat_trials[self.sort_indices]

        target_shape = tmt_trials[0].shape
        adjusted_arrays = [arr[:, :target_shape[1]] for arr in tmt_trials]
        tmt_trials=np.concat(adjusted_arrays,axis=0)
        tmt_trials=tmt_trials[self.sort_indices]
        all_trials=[van_trials,pb_trials,wat_trials,tmt_trials]

        # Loop over trials and clusters to get averages
        data_list=[]
        heat_map_list=[]
        for trial,trial_names in zip(all_trials,['vanilla','peanutbutter','water','tmt']):
            for cluster_id in np.unique(self.sorted_final_labels):
                
                current_cluster_neurons=self.activity_stack_sort[np.where(self.sorted_final_labels==cluster_id)]
                current_cluster_timestamps=trial[np.where(self.sorted_final_labels==cluster_id)]

                all_neuron_average_activity=[]
                for neuron,timestamps in zip(current_cluster_neurons,current_cluster_timestamps):
                    average_neuron_activity=[]
                    for time in timestamps:
                        act_oh = neuron[int(np.round(time-self.preceding_frames)):int(np.round(time+post_stim_frames))]
                        act_oh = act_oh - np.mean(act_oh[0]) # Subtract by average of baseline values for scaling. 
                        average_neuron_activity.append(act_oh) # ~ 10 seconds before and after each stimulus

                    average_neuron_activity=np.asarray(average_neuron_activity).mean(axis=0)
                    all_neuron_average_activity.append(average_neuron_activity)

                all_neuron_error_activity = np.std(np.asarray(all_neuron_average_activity), axis=0, ddof=1) / np.sqrt(np.asarray(all_neuron_average_activity).shape[1])
                heat_map_list.append([trial_names,cluster_id,all_neuron_average_activity])
                all_neuron_average_activity=np.asarray(all_neuron_average_activity).mean(axis=0)
                data_list.append([trial_names,cluster_id,all_neuron_average_activity,all_neuron_error_activity])
        
        # Convert data to a DataFrame for easier grouping
        self.activity_by_cluster_df = pd.DataFrame(data_list, columns=['Trial', 'Cluster', 'Average', 'Error'])
        self.heat_map_by_cluster = heat_map_list

    def plot_activity_by_cluser(self):
        # Unique groups and trials
        groups = self.activity_by_cluster_df['Cluster'].unique()
        trials = self.activity_by_cluster_df['Trial'].unique()
        colors = cm.get_cmap('Greens', (len(groups) + 4))

        # Create a grid of subplots
        fig, axes = plt.subplots(len(groups), len(trials), figsize=(20, 20), sharex=True, sharey=True)

        for i, group in enumerate(groups):
            for j, trial in enumerate(trials):
                # Get the data for the current group and trial
                subset = self.activity_by_cluster_df[(self.activity_by_cluster_df['Cluster'] == group) & (self.activity_by_cluster_df['Trial'] == trial)]
                ax = axes[i, j]

                # If data exists for this group-trial combination
                if not subset.empty:
                    color = colors((i+2) / (len(groups) + 4))
                    avg_activity = subset['Average'].values[0]
                    error_activity = subset['Error'].values[0]
                    time = np.linspace(-1*self.preceding_seconds, self.post_stim_seconds, len(avg_activity))

                    # Plot the line and error ribbon
                    ax.plot(time, avg_activity, label='Average Activity', color=color)
                    ax.fill_between(
                        time,
                        avg_activity - error_activity,
                        avg_activity + error_activity,
                        color=color,
                        alpha=0.6,
                        label='Error'
                    )

                    ax.axvline(x=-1, color='black', linestyle='--', label='t=0')
                    ax.grid(False)
                    ax.spines['top'].set_visible(False) 
                    ax.spines['right'].set_visible(False) 

                # Set titles and labels
                #ax.set_title(f"Cluster {group}, {trial}")
                if i == len(groups) - 1:
                    ax.set_xlabel("Time")
                if j == 0:
                    ax.set_ylabel("Activity")
                
                if i == 0:  # Add trial name to the first row
                    ax.set_title(trial)

        # Adjust layout and add a legend
        fig.tight_layout()
        plt.savefig(os.path.join(self.drop_directory,"activity_by_cluster_trial.jpg"))
        plt.close()

    def plot_heat_maps(self):
        # Seperate data into lists
        trials=[]
        clusters=[]
        heatmaps=[]
        for trial, clusterid, heatmap in self.heat_map_by_cluster:
            trials.append(trial)
            clusters.append(clusterid)
            heatmaps.append(heatmap)
        trials=np.array(trials)
        clusters=np.array(clusters)

        nrows=np.unique(trials)
        ncols=np.unique(clusters)

        fig, axes = plt.subplots(len(nrows), len(ncols), figsize=(20, 20), sharex=True, sharey=True)
        for i, cluster in enumerate(ncols):
            for j, trial in enumerate(nrows):
                heatmap_oh = heatmaps[i+j]

                # Sort array from largest activity to least
                heatmap_oh = np.asarray(heatmap_oh)
                row_averages = np.mean(heatmap_oh, axis=1)
                sorted_indices = np.argsort(row_averages)[::-1]
                heatmap_oh = heatmap_oh[sorted_indices]

                # Plot array as image
                ax = axes[i, j]
                #norm = matplotlib.colors.Normalize(vmin=np.min(heatmap_oh), vmax=np.max(heatmap_oh))
                ax.imshow(heatmap_oh, cmap='jet', interpolation='nearest')
                ax.set_aspect('auto')
                ax.axvline(x=4, color='black', linestyle='--', label='t=0')
                ax.grid(False)
                ax.spines['top'].set_visible(False) 
                ax.spines['right'].set_visible(False) 

                if i == len(ncols) - 1:
                    ax.set_xlabel("Time")

                if j == 0:
                    ax.set_ylabel("Activity")
                
                if i == 0:  # Add trial name to the first row
                    ax.set_title(trial)
                
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.drop_directory,"activity_by_cluster_trialheatmap.jpg"))
        plt.close()


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
    regressobj = map_clusters_to_activity(drop_directory=drop_directory,
                                                       neuronal_activity=neuronal_activity,
                                                       behavioral_timestamps=behavioral_timestamps,
                                                       neuron_info=neuron_info)
    regressobj()

    print('Finished coefficient clustering...')
