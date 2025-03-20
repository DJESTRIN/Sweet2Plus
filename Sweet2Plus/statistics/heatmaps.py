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
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from Sweet2Plus.statistics.coefficient_clustering import regression_coeffecient_pca_clustering, gather_data, cli_parser
import pandas as pd
import tqdm 
import ipdb 

class heatmap(regression_coeffecient_pca_clustering):
    def __call__(self):
        # Inherited methods
        self.normalize_activity()
        self.timestamps_to_one_hot_array()

        # Gather data from each trial and put into tables
        self.gather_data()
        self.data_to_dataframe()
        self.violin_plot()
        df_group_trial = self.data_to_table(groupby_labels = ['group', 'trialtype'])
        df_day_trial = self.data_to_table(groupby_labels = ['day', 'trialtype'])
        df_group_trial_day = self.data_to_table(groupby_labels = ['group', 'trialtype','day'])

        # # Plot results of dataframes
        self.plot_data(result_df=df_group_trial, plot_orders=['group'], output_filename='AverageGroupTrial.jpg')
        self.plot_data(result_df=df_day_trial, plot_orders=['day'], output_filename='AverageDayTrial.jpg')
        self.plot_data(result_df=df_group_trial_day, plot_orders = ['group', 'day'], output_filename='AverageGroupTrialDay.jpg')
 
        # self.plot_data_by_trial()

        # Save one hot vectors for decoder or ML models
        # self.generate_singular_neuronal_onehot()
        # self.generate_circuit_neuronal_onehot()

    def gather_data(self,preceding_frames=20,post_stim_frames=26, baseline_start=10, baseline_stop=20):
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
                    data_oh = data_oh - data_oh[baseline_start:baseline_stop,:].mean(axis=0) # Zero the data by baseline period 
                    trial_neural_all_ts.append(data_oh)
                
                self.all_neural_trial_ts.append(trial_neural_all_ts)
            self.all_neural_peth_data.append(self.all_neural_trial_ts)

        # Regroup data by averages
        all_subject_avs_by_trial = []
        for subject_peth_oh in self.all_neural_peth_data:
            all_avs_by_trial = []
            for trial in subject_peth_oh:
                # Calculate the average neural activity across neurons for all time stamps for given trial
                all_av_neu_for_trial = [ts.mean(axis=1) for ts in trial]

                # Filter out any trials that might not have the correct number of timestamps 
                shapes = [array.shape for array in all_av_neu_for_trial]
                shape_counts = Counter(shapes)
                most_common_shape = shape_counts.most_common(1)[0][0]  # The most common shape
                all_av_neu_for_trial = [array for array in all_av_neu_for_trial if array.shape == most_common_shape]

                all_av_neu_for_trial = np.array(all_av_neu_for_trial)
                trial_mean = all_av_neu_for_trial.mean(axis=0)
                all_avs_by_trial.append(trial_mean)
            
            all_avs_by_trial = np.array(all_avs_by_trial)
            all_subject_avs_by_trial.append(all_avs_by_trial)

        all_subject_avs_by_trial = np.array(all_subject_avs_by_trial)
        self.all_avs = all_subject_avs_by_trial.mean(axis=0)
        self.all_sds = all_subject_avs_by_trial.std(axis=0)
        self.N = all_subject_avs_by_trial.shape[0]
        self.all_subject_avs_by_trial = all_subject_avs_by_trial

        # Regroup data by AUCs
        all_subject_auc_by_trial = []  # Store AUCs for all subjects
        for subject_peth_oh in self.all_neural_peth_data: # Loop over subjects
            
            all_auc_by_trial = [] # Collect average auc by trial for current subject
            for trial_type in subject_peth_oh: # Loop over trial types
                auc_by_trial = []
                for ts in trial_type: # Loop over time stamps in trial type
                    
                    # Calculate AUC
                    try:
                        cropped_data = ts[20:30,:]
                        neuron_aucs = [np.trapz(row_oh) for row_oh in cropped_data.T]
                        trial_mean_auc = np.array(neuron_aucs).mean()
                    
                    except:
                        trial_mean_auc = np.nan

                    auc_by_trial.append(trial_mean_auc) # Collects each trial auc
                all_auc_by_trial.append(auc_by_trial)

            avs_by_trial = np.array([np.array(trial_auc_oh).mean() for trial_auc_oh in all_auc_by_trial])
            all_subject_auc_by_trial.append(avs_by_trial)
        
        all_subject_auc_by_trial = np.array(all_subject_auc_by_trial) 

        # Pull out attributes 
        self.auc_avs = all_subject_auc_by_trial.mean(axis=0)
        self.auc_N = all_subject_auc_by_trial.shape[0]
        self.auc_ses = all_subject_auc_by_trial.std(axis=0)/np.sqrt(self.auc_N)
        self.all_subject_auc_by_trial = all_subject_auc_by_trial

    def data_to_dataframe(self):
        """ Generate Data Frame for modeling (lmm, glm, etc) """
        
        # Gather neuron info data
        neuron_info_df = pd.DataFrame(self.neuron_info)
        subject_sessions = neuron_info_df.drop_duplicates()

        # Define columns
        columns = ['suid', 'neuid', 'group', 'day', 'trialtype', 'trialid', 'period', 'auc']
        
        # Use a list to collect data (much faster than repeated DataFrame concatenation)
        data_list = []

        for subject_peth_data_oh, info_oh in tqdm.tqdm(zip(self.all_neural_peth_data, subject_sessions.itertuples()), total=len(self.all_neural_peth_data)):
            
            # Precompute reusable info
            suid = f"{info_oh.cage}{info_oh.mouse}"
            group = info_oh.group
            day = info_oh.day

            for trial_type, trial_name in zip(subject_peth_data_oh, self.trial_list):
                for trial_id, trial_number in enumerate(trial_type):
                    auc_values = np.trapz(trial_number[:, 10:20], axis=1), np.trapz(trial_number[:, 20:30], axis=1)

                    for id_oh, (baseline_auc, event_auc) in enumerate(zip(*auc_values)):
                        neuid = f"{suid}_day{day}_neu{id_oh}"
                        trial_id_str = str(trial_id)
                        
                        # Append baseline and event data to the list
                        data_list.append([suid, neuid, group, day, trial_name, trial_id_str, 'baseline', baseline_auc])
                        data_list.append([suid, neuid, group, day, trial_name, trial_id_str, 'event', event_auc])
        
        # Convert list to DataFrame in one go (much faster than appending)
        self.final_dataframe = pd.DataFrame(data_list, columns=columns)

        # Save data to a CSV file
        self.final_dataframe.to_csv(os.path.join(self.drop_directory, "auc_dataset.csv"), index=False)

    def plot_data_by_trial(self):
        # Set up x axis
        time = np.arange(self.all_avs.shape[1])
        custom_labels = np.linspace(-1 * self.preceding_seconds, self.post_stim_seconds, len(time))
        colors = plt.cm.tab10(np.arange(len(self.trial_list))) 

        # Generate plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})  # Adjust relative sizes

        # Average activity plot
        for row, sd, labeloh, coloroh in zip(self.all_avs, self.all_sds, self.trial_list, colors):
            row = row - row[0]  # Rescale plot
            se = sd / np.sqrt(self.N)
            axes[0].plot(custom_labels, row, label=labeloh, color = coloroh)
            axes[0].fill_between(custom_labels, row - se, row + se, alpha=0.2, color = coloroh)

        axes[0].axvline(x=0, color='black', linestyle='--')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Average Normalized DF \n Across Subjects and Neurons')
        axes[0].legend(loc="best")
        axes[0].grid(True)

        # AUC bar plot
        axes[1].bar(self.trial_list, self.auc_avs, yerr=self.auc_ses, capsize=5, alpha=0.8, edgecolor='black', color = colors)
        axes[1].set_ylabel('AUC')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.drop_directory, "average_acitvity_auc.jpg"))

    def data_to_table(self, groupby_labels = ['group', 'trialtype'], 
                      columns = ['suid', 'day', 'group', 'trialtype', 'time_series','baseline_auc','event_auc'], 
                      auc_ts = [(5, 20), (20, 35)]):
    
        """ Converts data into a table format via pandas which is easier for plotting  """
        # Create empty dataframes
        neuron_info_df = pd.DataFrame(self.neuron_info)
        subject_sessions = neuron_info_df.drop_duplicates()
        group_df = pd.DataFrame(columns=columns)

        # Loop over neuronal activity and neuronal data
        for sub_session,neuron_info_oh in zip(self.all_subject_avs_by_trial,subject_sessions.itertuples(index=False)):
            day, cage, mouse, group = neuron_info_oh # Parse info 
            suid = cage + mouse

            # Put data into dataframe
            current_rows = []
            for trial,time_series_data in enumerate(sub_session):
                AUCs_oh = []
                for auc_timestamps in auc_ts:
                    AUCs_oh.append(np.trapz(time_series_data[auc_timestamps[0]:auc_timestamps[1]]))
                new_row = pd.DataFrame({'suid':[suid], 'day':[day], 'group': [group], 'trialtype': [trial], 
                                        'time_series': [time_series_data],'baseline_auc': [AUCs_oh[0]], 'event_auc':[AUCs_oh[1]]})
                current_rows.append(new_row)
            
            group_df = pd.concat([group_df] + current_rows, ignore_index=True)

        # Average the data in dataframe by given groupby_labels
        result = group_df.groupby(groupby_labels).agg(time_series_mean = ('time_series',lambda x: np.mean(np.vstack(x),axis=0)),
                                             time_series_se = ('time_series',lambda x: np.std(np.vstack(x),axis=0)/np.sqrt(len(np.vstack(x)))),
                                             baseline_auc_mean = ('baseline_auc', lambda x: np.mean(x)),
                                             baseline_auc_se = ('baseline_auc', lambda x:(np.std(x)/np.sqrt(len(x)))),
                                             event_auc_mean = ('event_auc', lambda x: np.mean(x)),
                                             event_auc_se = ('event_auc', lambda x: (np.std(x)/np.sqrt(len(x)))),
                                             event_n_auc = ('event_auc', lambda x: len(x))).reset_index()
        return result

    def plot_data(self, result_df, plot_orders=['group', 'day'], output_filename='plot_data.jpg',x_crop=[-10,20]):
        """Plot result dataframe data average and errors in an N x M grid based on plot_orders."""
        # Get time for x-axis
        time = np.arange(self.all_avs.shape[1])
        custom_labels = np.linspace(-1 * self.preceding_seconds, self.post_stim_seconds, len(time))

        # Extract unique values for grouping variables
        unique_groups = np.sort(result_df[plot_orders[0]].unique())  # e.g., 'group'
        N = len(unique_groups)  # Rows

        if len(plot_orders) > 1 and plot_orders[1] in result_df.columns:
            unique_days = np.sort(pd.to_numeric(result_df[plot_orders[1]].unique()))  # e.g., 'day'
            M = len(unique_days)  # Columns
        else:
            unique_days = [None]  # If no second plot order, treat as single column
            M = 1

        # Create figure
        fig, axes = plt.subplots(N, M, figsize=(5 * M, 4 * N), sharey=True, sharex=True)
        
        # Ensure axes is always a 2D array
        if N == 1:
            axes = np.expand_dims(axes, axis=0)
        if M == 1:
            axes = np.expand_dims(axes, axis=1)

        # Define colors for different trial types
        colors = ['#E0C68D', '#D19A6A', '#A4D7E1', '#E74C3C']  # Modify as needed

        # Loop through groups and days to plot data
        for i, group in enumerate(unique_groups):  # Row-wise (Groups)
            for j, day in enumerate(unique_days):  # Column-wise (Days)
                ax = axes[i, j]
                print(f"Processing group: {group}, day: {day}")

                # Filter dataframe
                group_df = result_df[result_df[plot_orders[0]] == group]
                if day is not None:
                    value_df = group_df[group_df[plot_orders[1]] == str(day)]
                else:
                    value_df = group_df

                if value_df.empty:
                    print(f"Empty DataFrame for group '{group}' and day '{day}', skipping...")
                    continue

                # Plot each trial type
                auc_mean_oh, auc_se_oh = [], []
                for indx, (row, trial_label) in enumerate(zip(value_df.itertuples(index=False), self.trial_list)):
                    ax.plot(custom_labels, row.time_series_mean, label=trial_label, color=colors[indx])
                    ax.fill_between(custom_labels, row.time_series_mean - row.time_series_se,
                                    row.time_series_mean + row.time_series_se, alpha=0.2, color=colors[indx])
                    auc_mean_oh.append(row.event_auc_mean)
                    auc_se_oh.append(row.event_auc_se)

                if auc_mean_oh:
                    # Add inset bar plot
                    ax_inset = inset_axes(ax, width="30%", height="50%", 
                      bbox_to_anchor=(0.12, 0.3, 0.5, 0.7),  # Adjust (x, y) position
                      bbox_transform=ax.transAxes, 
                      loc='upper left')
                    ax_inset.bar(self.trial_list, auc_mean_oh, yerr=auc_se_oh, capsize=5, alpha=0.7, color=colors)
                    ax_inset.set_ylabel('AUC')
                    ax_inset.set_xticklabels([])

                # Set titles
                if day is not None:
                    ax.set_title(f'{group} - {day}')
                else:
                    ax.set_title(f'{group}')

                ax.grid(True)
                ax.axvline(x=0, linestyle='--', color='black', linewidth=1)
                ax.set_xlim(x_crop[0], x_crop[1])

                # Axis labels
                if j == 0:
                    ax.set_ylabel('Average Normalized DF \n Across Subjects and Neurons')
                if i == N - 1:
                    ax.set_xlabel('Time')

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Remove duplicates
        fig.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.drop_directory, output_filename), bbox_inches='tight')

    def violin_plot(self):
        colors = ['red', 'blue']
        palette_dict = {group: color for group, color in zip(self.final_dataframe['group'].unique(), colors)}


        # Create FacetGrid to separate by 'day'
        g = sns.FacetGrid(self.final_dataframe, col="day", col_wrap=2, height=5, sharey=True)  # 2 plots per row
        g.map_dataframe(sns.violinplot, x="trialtype", y="auc", hue="group", split=True, dodge=True, palette=palette_dict)

        g.set_axis_labels("Trial Type", "AUC")
        g.set_titles("Day {col_name}")  # Label each plot with Day
        g.add_legend(title="Group")

        plt.xticks(rotation=45)  
        plt.savefig(os.path.join(self.drop_directory, "violin_all_data_day.jpg"))
        ipdb.set_trace()

    def generate_singular_neuronal_onehot(self):
        """ 
        output 
            individual neuronal activities for a trial (46, 1) floats -----> trial answer as one hot [0 , 0 , 1, 0]
        """
        all_trial_neuronal_data = []
        all_trial_results = []
        for subject_peth_oh in self.all_neural_peth_data:
            for k,trial_data in enumerate(subject_peth_oh):
                for neuron_data in trial_data:
                    if neuron_data.shape[0]<46:
                        continue

                    else:
                        for single_neuron_data in neuron_data.T:
                            all_trial_neuronal_data.append(single_neuron_data)
                            result_oh = np.zeros(shape=(1,4))
                            result_oh[:,k] = 1
                            all_trial_results.append(result_oh)

        all_trial_results = np.array(all_trial_results)
        all_trial_results = np.squeeze(all_trial_results)
        all_trial_neuronal_data = np.array(all_trial_neuronal_data)
        
        self.X_original = all_trial_neuronal_data
        self.y_one_hot = all_trial_results

        # Save results to numpy file
        np.save(file = os.path.join(self.drop_directory, "X_original.npy"), arr=self.X_original, allow_pickle=True)
        np.save(file = os.path.join(self.drop_directory, "y_one_hot.npy"), arr=self.y_one_hot, allow_pickle=True)
        print('Data was saved!')

    def generate_circuit_neuronal_onehot(self):
        """ generate circuit neuronal onehot 
            This method takes previously calculated data from heatmap class and puts it in a 
            46 x M x J array. Where M is variable number of neurons in the circuit (or image)
            and J is the number of samples in total. A second dataset is save as a onehot, indicating
            current trial class. 
        """
        total_skipped = 0
        trial_ensemble_neuronal_data = []
        trial_types = []
        for subject_peth_oh in self.all_neural_peth_data:
            for k, trial_data in enumerate(subject_peth_oh):
                for ensemble_neuron_data in trial_data:
                    if ensemble_neuron_data.shape[0]<46:
                        total_skipped+=1
                        continue

                    else:
                        trial_ensemble_neuronal_data.append(ensemble_neuron_data)
                        result_oh = np.zeros(shape=(1,4))
                        result_oh[:,k] = 1
                        trial_types.append(result_oh)

        trial_types = np.squeeze(np.array(trial_types))
        trial_ensemble_neuronal_data = np.array(trial_ensemble_neuronal_data)
        
        self.X_circuit_original = trial_ensemble_neuronal_data
        self.y_circuit_one_hot = trial_types

        # Save results to numpy file
        np.save(file = os.path.join(self.drop_directory, "X_circuit_original.npy"), arr=self.X_circuit_original, allow_pickle=True)
        np.save(file = os.path.join(self.drop_directory, "y_circuit_one_hot.npy"), arr=self.y_circuit_one_hot, allow_pickle=True)
        print('Circuit data was saved!')

if __name__=='__main__':
    data_directory, drop_directory = cli_parser()
    neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
    heatmap_obj = heatmap(drop_directory=drop_directory,
                         neuronal_activity=neuronal_activity,
                         behavioral_timestamps=behavioral_timestamps,
                         neuron_info=neuron_info)
    heatmap_obj()

    print('Finished generating heatmaps...')
