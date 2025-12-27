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
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ipdb
import pandas as pd
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import time
from scipy.stats import ttest_ind
from scipy.stats import t
from venn import venn
from scipy.stats import sem
import seaborn as sns

# Set up default matplotlib plot settings
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['axes.linewidth'] = 2 

class circuit_regression:
    def __init__(self, drop_directory, neuronal_activity, behavioral_timestamps, neuron_info, trial_list=['Vanilla','PeanutButter','Water','FoxUrine'],
                 normalize_neural_activity=False, regression_type='ridge'):
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
    
    @staticmethod
    def keep_balanced_zeros_before_ones(X, y):
        keep_indices = set()
        n_rows, n_cols = y.shape

        for col in range(n_cols):
            y_col = y[:, col]
            i = 0
            while i < n_rows:
                if y_col[i] == 1:
                    # Start of a 1s series
                    start = i
                    while i < n_rows and y_col[i] == 1:
                        i += 1
                    end = i  # exclusive
                    length = end - start

                    # Keep 1s
                    keep_indices.update(range(start, end))

                    # Keep same number of zeros before the series
                    pre_start = max(start - length, 0)
                    keep_indices.update(range(pre_start, start))
                else:
                    i += 1

        keep_indices = np.array(sorted(keep_indices))
        return X[keep_indices], y[keep_indices]

    @staticmethod
    def process_recording(recording_idx, recording_activity, recording_beh, n_bootstraps=20):
        """ Process a single recording at a time. For parrallel use. """
        # Get start time
        start_time = time.time()

        # Lists for statistics
        accuracies = []
        f1_scores = []
        r_s = []
        trialnum = []

        # Clean up data into ML format
        X = recording_activity.T
        y = recording_beh[:, :]
        X, y = circuit_regression.keep_balanced_zeros_before_ones(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        all_boot_coefs = {}

        for i in range(y_train.shape[1]):

            # Boot strap Beta Weight values
            coefs = []
            for _ in range(n_bootstraps):
                indices = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices, i]

                model = LogisticRegression(
                    penalty='elasticnet',
                    l1_ratio=0.5,
                    class_weight='balanced',
                    solver='saga',
                    max_iter=10000)
                model.fit(X_boot, y_boot)
                coefs.append(model.coef_.flatten())

            coefs = np.stack(coefs)
            all_boot_coefs[i] = coefs

            # Get model stats
            model = LogisticRegression(
                penalty='elasticnet',
                l1_ratio=0.5,
                class_weight='balanced',
                solver='saga',
                max_iter=10000)
            model.fit(X_train, y_train[:, i])

            probabilities_train = model.predict_proba(X_train)[:, 1]
            probabilities_test = model.predict_proba(X_test)[:, 1]

            r2_tjur = probabilities_train[y_train[:, i] == 1].mean() - probabilities_train[y_train[:, i] == 0].mean()
            predicted_test = (probabilities_test >= 0.5).astype(int)
            accuracy = accuracy_score(y_test[:, i], predicted_test)
            f1 = f1_score(y_test[:, i], predicted_test)
            
            accuracies.append(accuracy)
            f1_scores.append(f1)
            r_s.append(r2_tjur)
            trialnum.append(i)

        results_df = pd.DataFrame({
            'Recording': [recording_idx] * len(trialnum),
            'Behavior': trialnum,
            'Accuracy': accuracies,
            'F1 Score': f1_scores,
            'Rs': r_s,
        })

        summary_frames = []
        for beh_idx, coefs in all_boot_coefs.items():
            mean_weights = np.mean(coefs, axis=0)
            lower_ci = np.percentile(coefs, 2.5, axis=0)
            upper_ci = np.percentile(coefs, 97.5, axis=0)
            df = pd.DataFrame({
                'Recording': recording_idx,
                'Neuron': np.arange(coefs.shape[1]),
                'Behavior': f'Behavior_{beh_idx}',
                'Mean_Beta': mean_weights,
                'CI_lower': lower_ci,
                'CI_upper': upper_ci
            })
            summary_frames.append(df)

        neuron_df = pd.concat(summary_frames, ignore_index=True)
        elapsed_time = time.time() - start_time

        return results_df, neuron_df, recording_idx, elapsed_time

    def run_glm(self, coef_file='allcoefs.csv', model_file='allmodels.csv'):
        coef_filename = os.path.join(self.drop_directory, coef_file)
        model_filename = os.path.join(self.drop_directory, model_file)

        if os.path.isfile(coef_filename) and os.path.isfile(model_filename):
            model_results_df = pd.read_csv(model_filename)
            neuron_coefficients_df = pd.read_csv(coef_filename)
        
        else:
            total_tasks = len(self.neuronal_activity)
            with tqdm_joblib(tqdm(desc="Processing recordings", total=total_tasks)):

                results = Parallel(n_jobs=-1)(
                    delayed(circuit_regression.process_recording)(idx, rec, beh)
                    for idx, (rec, beh) in enumerate(zip(self.neuronal_activity, self.behavior_ts_onehot))
                )

            # Save data to csv
            model_results, neuron_coefficients, indxoh, elapsedtime = zip(*results)
            model_results_df = pd.concat(model_results, ignore_index=True)
            neuron_coefficients_df = pd.concat(neuron_coefficients, ignore_index=True)
            model_results_df.to_csv(model_filename, index=False)
            neuron_coefficients_df.to_csv(coef_filename, index=False)

        # Gather decoder results
        self.subject_info = self.neuron_info.drop_duplicates(subset=['day', 'cage', 'mouse', 'group'])
        self.subject_info  = self.subject_info.reset_index(drop=True)
        subject_info_indexed = self.subject_info.copy()
        subject_info_indexed['Recording'] = subject_info_indexed.index
        self.decoder_results = model_results_df.merge(subject_info_indexed, on='Recording', how='left')
        self.decoder_results.to_csv(os.path.join(self.drop_directory,'decoder_results.csv'))

        # Gather beta weight results
        self.beta_results = neuron_coefficients_df.merge(subject_info_indexed, on='Recording', how='left')
        self.beta_results['nuid'] = (self.beta_results['cage'].astype(str) + '_' + self.beta_results['mouse'].astype(str) + '_' + self.beta_results['day'].astype(str)+ '_' + self.beta_results['Neuron'].astype(str))
        self.beta_results['suid'] = (self.beta_results['cage'].astype(str) + '_' + self.beta_results['mouse'].astype(str))

        df = self.beta_results
        allowed_behaviors = [f'Behavior_{i}' for i in range(4)]
        df = df[df['Behavior'].isin(allowed_behaviors)].copy()
        df['Behavior_num'] = df['Behavior'].str.extract(r'Behavior_(\d)').astype(int)
        df['df'] = 19
        df['t_crit'] = t.ppf(0.975, df['df'])
        df['SE'] = (df['CI_upper'] - df['CI_lower']) / (2 * df['t_crit'])
        df['t_value'] = df['Mean_Beta'] / df['SE']
        df['p_value'] = 2 * t.sf(abs(df['t_value']), df['df'])
        df['sig'] = (df['p_value'] < 0.05).astype(int)
        df_final = df[['nuid', 'suid', 'group','day','Behavior_num', 'Mean_Beta', 'CI_lower', 'CI_upper',
                    't_value', 'p_value', 'sig']]
        self.beta_results_filtered = df_final
        self.beta_results_filtered.to_csv(os.path.join(self.drop_directory,'beta_filtered.csv'))

        for (name,i) in zip(['water','peanut','vanilla','TMT'],range(4)):
            df = self.beta_results_filtered
            neurons_behavior0 = df[df['Behavior_num'] == i]
            total_neurons = neurons_behavior0['nuid'].nunique()
            sig_neurons = neurons_behavior0[neurons_behavior0['sig'] == 1]['nuid'].nunique()
            percent_sig = (sig_neurons / total_neurons) * 100
            print(f"Percent of significant neurons for {name}: {percent_sig:.2f}%")

    def cor_peth(self):
        cor_PETH_list=[]
        for activity, behaviors in zip(self.neuronal_activity,self.behavioral_timestamps):
            cor_PETH_list_oh=[]
            for behavior in behaviors:
                window_pre = 10
                window_post = 20
                window_size = window_pre + window_post + 1  # 21
                relative_times = np.arange(-window_pre, window_post + 1)

                n_neurons = activity.shape[0]
                n_trials = len(behavior)

                # Create an array to collect aligned data: shape (n_neurons, n_trials, 21)
                aligned = []

                for t in behavior:
                    start = int(t - window_pre)
                    end = int(t + window_post + 1)
                    if start < 0 or end > activity.shape[1]:
                        continue
                    aligned.append(activity[:, start:end])  # shape (n_neurons, 21)

                aligned = np.stack(aligned, axis=1)  # shape: (n_neurons, n_valid_trials, 21)

                # Output: avg correlation per neuron at each relative timepoint
                avg_corrs = np.full((n_neurons, window_size), np.nan)

                # Loop over each relative timepoint
                for i in range(window_size):
                    data = aligned[:, :, i]  # shape: (n_neurons, n_trials) at this timepoint
                    corr_matrix = np.corrcoef(data)
                    np.fill_diagonal(corr_matrix, np.nan)
                    avg_corrs[:, i] = np.nanmean(corr_matrix, axis=1)

         
                baseline = avg_corrs[:, 0:(window_pre-1)].mean(axis=1, keepdims=True)
                avg_corrs_corrected = avg_corrs - baseline
                cor_PETH_list_oh.append(avg_corrs_corrected)
            cor_PETH_list.append(cor_PETH_list_oh)

        all_values = [ neuron for recording in cor_PETH_list for behavior in recording for neuron in behavior]
        self.beta_results_filtered['correlation_peth']=all_values
        self.beta_results_filtered['correlation_peth'] = self.beta_results_filtered['correlation_peth'].apply(lambda x: ','.join(map(str, x)))
        self.beta_results_filtered.to_csv(os.path.join(self.drop_directory,'betacorrelation.csv'), index=False)

        all_times = [[] for _ in range(4)]

        for recording in cor_PETH_list:
            for i in range(4):
                mean_trace = recording[i].mean(axis=0)  
                all_times[i].append(mean_trace)

        all_times = [np.array(traces) for traces in all_times]
        all_times_mean = np.array([traces.mean(axis=0) for traces in all_times])
        all_times_sem = np.array([sem(traces, axis=0) for traces in all_times])

        n_timepoints = all_times_mean.shape[1]
        x = np.arange(n_timepoints) - 10  
        colors = ['blue', 'green', 'red', 'purple']
        labels = ['Water', 'Peanut', 'Vanilla', 'TMT']

        plt.figure(figsize=(10, 5))

        for i in range(4):
            mean = all_times_mean[i]
            error = all_times_sem[i]
            plt.plot(x, mean, label=labels[i], color=colors[i])
            plt.fill_between(x, mean - error, mean + error, color=colors[i], alpha=0.3)

        plt.axvline(0, color='k', linestyle='--', label='Aligned Event (t=5)')
        plt.xlabel('Time relative to event (frames)')
        plt.ylabel('Mean Correlation ± SEM')
        plt.title('PETH of Correlation for Each Element')
        plt.legend()
        plt.tight_layout()
        plt.savefig('average_corr_peth_per_trial.jpg')



    def gen_venn_diagram_beta(self):
        # Generate overall vendiagram
        df = self.beta_results_filtered
        sig_neuron_sets = {}
        behavior_map = {
            0: 'water',
            1: 'peanut',
            2: 'vanilla',
            3: 'TMT'
        }
        for behavior_num, name in behavior_map.items():
            subset = df[(df['Behavior_num'] == behavior_num) & (df['sig'] == 1)]
            sig_neuron_sets[name] = set(subset['nuid'].unique())

        venn(sig_neuron_sets)
        plt.title("Overlap of Significant Neurons Across Behaviors")
        plt.savefig('Vendiagram.jpg')

        # Plot vendiagram wrt day and group
        df = self.beta_results_filtered.copy()

        # Set up behavior labels
        behavior_map = {
            0: 'water',
            1: 'peanut',
            2: 'vanilla',
            3: 'TMT'
        }

        # Unique values for subplotting
        groups = df['group'].unique()
        days = sorted(df['day'].unique())
        n_rows = len(groups)
        n_cols = len(days)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        # If only one row/col, ensure axes is still 2D
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]

        # Loop through all group × day combinations
        for i, group in enumerate(groups):
            for j, day in enumerate(days):
                ax = axes[i][j]
                subset_df = df[(df['group'] == group) & (df['day'] == day)]

                # Get sig sets per behavior
                sig_neuron_sets = {}
                for behavior_num, name in behavior_map.items():
                    subset = subset_df[(subset_df['Behavior_num'] == behavior_num) & (subset_df['sig'] == 1)]
                    sig_neuron_sets[name] = set(subset['nuid'].unique())

                # Plot the Venn
                if sum(len(v) > 0 for v in sig_neuron_sets.values()) >= 2:
                    venn(sig_neuron_sets, ax=ax)
                else:
                    ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                    ax.set_axis_off()

                ax.set_title(f"Group: {group}, Day: {day}")

        plt.tight_layout()
        plt.savefig("Venn_Group_Day.jpg", dpi=300)

        df = self.beta_results_filtered.copy()
        # Behavior mapping
        behavior_map = {0: 'water', 1: 'peanut', 2: 'vanilla', 3: 'TMT'}

        groups = df['group'].unique()
        days = sorted(df['day'].unique())

        n_rows = 2  # Adjust depending on your groups
        n_cols = 4  # Adjust depending on your days

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)

        for i, group in enumerate(groups):
            for j, day in enumerate(days):
                ax = axes[i, j]
                
                # Filter df for current group and day
                subset_df = df[(df['group'] == group) & (df['day'] == day)]
                
                # Get total neurons for percentage base
                total_neurons = subset_df['nuid'].nunique()
                if total_neurons == 0:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.axis('off')
                    continue
                
                # Get sets of neurons for each behavior where sig==1
                sets = []
                labels = []
                for bnum in range(4):
                    neuron_set = set(subset_df[(subset_df['Behavior_num'] == bnum) & (subset_df['sig'] == 1)]['nuid'])
                    sets.append(neuron_set)
                    labels.append(behavior_map[bnum])
                
                # Check if at least two sets have data, else skip
                if sum(len(s) > 0 for s in sets) < 2:
                    ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
                    ax.axis('off')
                    continue
                
                # Compute subset sizes for venn4 - this expects a dict with keys being binary strings of membership
                # e.g. '1000' = set1 only, '1100' = set1 & set2, etc.
                # We'll do this by hand:
                from itertools import product
                
                # 16 subsets for 4 sets, keys as strings representing membership (1 = in set, 0 = not)
                subset_sizes = {}
                sets_list = sets
                
                all_neurons = set.union(*sets_list)
                for comb in product([0,1], repeat=4):
                    # select neurons in exactly those sets:
                    included_sets = [sets_list[idx] for idx, val in enumerate(comb) if val==1]
                    excluded_sets = [sets_list[idx] for idx, val in enumerate(comb) if val==0]
                    
                    if included_sets:
                        # Intersect all included sets
                        intersect = set.intersection(*included_sets)
                    else:
                        intersect = all_neurons
                    
                    if excluded_sets:
                        # Remove neurons in any excluded sets
                        exclude = set.union(*excluded_sets)
                        intersect = intersect - exclude
                    
                    # Count intersection size
                    subset_sizes[''.join(map(str, comb))] = len(intersect)
                
                # Convert counts to percentages (round 1 decimal)
                subset_percents = {k: round(100*v/total_neurons,1) for k,v in subset_sizes.items()}
                
                # matplotlib_venn expects subset sizes as tuple in order:
                # 12 elements representing sizes for venn4 diagram subsets, order:
                # (1000, 0100, 1100, 0010, 1010, 0110, 1110, 0001, 1001, 0101, 1101, 0011, 1011, 0111, 1111, 0000)
                # We map from our keys:
                venn_order = ['1000','0100','1100','0010','1010','0110','1110',
                            '0001','1001','0101','1101','0011','1011','0111','1111','0000']
                
                sizes_ordered = tuple(subset_sizes.get(k, 0) for k in venn_order)
                perc_ordered = tuple(subset_percents.get(k, 0) for k in venn_order)

                # Create the venn diagram
                v = venn4(subsets=sizes_ordered[:15], set_labels=labels, ax=ax)
                
                # Replace counts with percentages
                for subset in v.subset_labels:
                    if subset:
                        idx = v.subset_labels.index(subset)
                        # Map label to percentage from perc_ordered
                        # Note: v.subset_labels are ordered as in venn_order[:15]
                        perc = perc_ordered[idx]
                        subset.set_text(f"{perc}%")
                
                ax.set_title(f"Group: {group}, Day: {day}")
                
        plt.tight_layout()


    def plot_decoder_results(self):
        self.decoder_results['suid'] = (self.decoder_results['cage'].astype(str) + '_' + self.decoder_results['mouse'].astype(str))
        filtered = self.decoder_results[self.decoder_results['Behavior'].isin([0, 1, 2, 3])]
        per_suid = (filtered.groupby(['group', 'day', 'Behavior', 'suid'])['F1 Score'].mean().reset_index())
        day0_avg_per_group = (
            per_suid[per_suid['day'] == '0']
            .groupby('group')['F1 Score']
            .mean()
            .to_dict())
        per_suid['F1_norm'] = per_suid.apply(
            lambda row: row['F1 Score'] / day0_avg_per_group.get(row['group'], float('nan')),
            axis=1)
        day30_data = per_suid[(per_suid['day'] == '30') & (per_suid['Behavior'].isin([0,1,2,3]))]
        summary = (per_suid.groupby(['group', 'day', 'Behavior'])['F1_norm'].agg(['mean', 'sem']).reset_index())
        summary['day'] = summary['day'].astype(int)
        summary['day'] = pd.Categorical(summary['day'], categories=[0, 7, 14, 30], ordered=True)
        

        g = sns.FacetGrid(
            summary,
            col='group',
            height=5,
            aspect=1.3,
            sharey=True
        )

        g.map_dataframe(
            sns.lineplot,
            x='day',
            y='mean',
            hue='Behavior',
            marker='o',
            err_style='bars',
            ci=None
        )
        for ax, (_, group_data) in zip(g.axes.flat, summary.groupby('group')):
            for _, row in group_data.iterrows():
                ax.errorbar(
                    x=row['day'],
                    y=row['mean'],
                    yerr=row['sem'],
                    fmt='none',
                    color='gray',
                    capsize=3
                )

        g.set_axis_labels("Day", "Mean F1 Score")
        g.add_legend(title="Behavior")
        plt.subplots_adjust(top=0.85)
        plt.savefig('F1Decoderresults.jpg')


        # List to store results
        ttest_results = []

        for behavior in [0,1,2,3]:
            # Data for this behavior
            behavior_data = day30_data[day30_data['Behavior'] == behavior]
            
            # F1 scores for each group (assuming groups are named, e.g., 'A' and 'B')
            group_names = behavior_data['group'].unique()
            if len(group_names) < 2:
                print(f"Not enough groups for behavior {behavior} to run t-test.")
                continue
            
            group1_scores = behavior_data[behavior_data['group'] == group_names[0]]['F1_norm']
            group2_scores = behavior_data[behavior_data['group'] == group_names[1]]['F1_norm']
            
            # Run t-test (independent samples)
            t_stat, p_val = ttest_ind(
                group1_scores,
                group2_scores,       
                nan_policy='omit',
                alternative='less',
                equal_var=False
            )
            
            ttest_results.append({
                'Behavior': behavior,
                'Group1': group_names[0],
                'Group2': group_names[1],
                't-statistic': t_stat,
                'p-value': p_val
            })

        # Convert results to DataFrame for easier viewing
        results_df = pd.DataFrame(ttest_results)

    def beta_weight_validation(self):
        # Calculate the trace for each neuron for each trial
        all_curves=[]
        for activity,timestamps in zip(self.neuronal_activity, self.behavioral_timestamps):
            for ts in timestamps:
                curvesoh = [activity[:,int(tsj-10):int(tsj+15)] for tsj in ts]
                try:
                    stacked = np.stack(curvesoh, axis=0)
                    average_trace_per_neuron = np.mean(stacked, axis=0)
                except:
                    stacked = np.stack(curvesoh[:-1], axis=0)
                    average_trace_per_neuron = np.mean(stacked, axis=0)
                all_curves.append(average_trace_per_neuron)

        all_curves = np.concatenate(all_curves)
        all_curves_list = [row for row in all_curves]
        self.beta_results_filtered['trace'] = all_curves_list

        def categorize_beta(row):
            if row['sig'] == 1:
                if row['Mean_Beta'] > 0:
                    return 'pos_sig'
                elif row['Mean_Beta'] < 0:
                    return 'neg_sig'
                else:
                    return 'zero_sig'  # just in case
            else:
                return 'non_sig'

        df = self.beta_results_filtered.copy()
        df['beta_cat'] = df.apply(categorize_beta, axis=1)

        def avg_trace(traces):
            stacked = np.vstack(traces)
            return stacked.mean(axis=0)

        ipdb.set_trace()
        avg_traces_df = (
            df.groupby(['Behavior_num', 'group', 'beta_cat'])['trace']
            .apply(lambda traces: np.mean(np.vstack(traces.tolist()),axis=0))
            .reset_index()
        )

        behaviors_to_plot = [0, 1, 2, 3]
        groups = sorted(avg_traces_df['group'].unique())
        n_rows = len(groups)
        n_cols = len(behaviors_to_plot)
        trace_length = len(avg_traces_df['trace'].iloc[0])
        x = range(trace_length)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True, sharex=True)

        for i, group in enumerate(groups):
            for j, behavior_num in enumerate(behaviors_to_plot):
                ax = axes[i, j] if n_rows > 1 else axes[j]  # Handle single row case
                subset = df[(df['group'] == group) & (df['Behavior_num'] == behavior_num)]
                
                if subset.empty:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_axis_off()
                    continue
                
                for beta_cat in subset['beta_cat'].unique():
                    traces = np.vstack(subset[subset['beta_cat'] == beta_cat]['trace'].tolist())
                    mean_trace = traces.mean(axis=0)
                    sem_trace = traces.std(axis=0) / np.sqrt(traces.shape[0])
                    
                    ax.plot(x, mean_trace, label=beta_cat)
                    ax.fill_between(x, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3)
                
                if i == 0:
                    ax.set_title(f"Behavior {behavior_num}")
                if j == 0:
                    ax.set_ylabel(f"{group}")
                
                ax.set_xlabel("Time")
                ax.legend(fontsize='small')

        plt.tight_layout()
        plt.savefig('beta_activity_sem.jpg')      

        # Convert data to long format
        # rows = []
        # for idx, row in df.iterrows():
        #     trace = row['trace']
        #     for t, val in enumerate(trace):
        #         rows.append({
        #             't': t,
        #             'normF': val,
        #             **{col: row[col] for col in df.columns if col != 'trace'}
        #         })

        # # Step 2: Convert to DataFrame
        # long_df = pd.DataFrame(rows)
        # long_df.to_csv(os.path.join(self.drop_directory,'longdftracebeta.csv'))
    
    def beta_weight_functional_connectivity(self):
        # Add whole session trace to dataframe for each neuron
        all_traces=[]
        for activity in self.neuronal_activity:
            all_traces.append(activity)

        df=self.beta_results_filtered.copy()
        all_traces_list = [row for recording in all_traces for row in recording]
        df['whole_session_trace'] = all_traces_list*4
        df['recording_name_behavior'] = (df['group'].astype(str) + df['day'].astype(str) + df['suid'].astype(str) + df['Behavior_num'].astype(str))

        def correlate_traces(df):
            traces = df['whole_session_trace'].to_numpy()
            traces_matrix = np.stack(traces)
            corr_matrix = np.corrcoef(traces_matrix)
            df = df.copy()
            df['corrs'] = [corr_matrix[i, :].tolist() for i in range(corr_matrix.shape[0])]
            return df

        processed_dfs = []  # <-- list to collect all processed subsets

        for name, group_df in df.groupby('recording_name_behavior'):
            group_sig0 = group_df[group_df['sig'] == 0]
            group_sig1_pos = group_df[(group_df['sig'] == 1) & (group_df['Mean_Beta'] > 0)]
            group_sig1_neg = group_df[(group_df['sig'] == 1) & (group_df['Mean_Beta'] <= 0)]

            if len(group_sig0) > 1:
                group_sig0 = correlate_traces(group_sig0)
                processed_dfs.append(group_sig0)

            if len(group_sig1_pos) > 1:
                group_sig1_pos = correlate_traces(group_sig1_pos)
                processed_dfs.append(group_sig1_pos)

            if len(group_sig1_neg) > 1:
                group_sig1_neg = correlate_traces(group_sig1_neg)
                processed_dfs.append(group_sig1_neg)

        df_processed = pd.concat(processed_dfs, ignore_index=True)
        df_processed = df_processed.sort_values(by=['recording_name_behavior', 'nuid']).reset_index(drop=True)
        
        df_processed['Mean_Beta_sign'] = np.where(df_processed['Mean_Beta'] > 0, 'pos', 'neg')
        df_processed['sig_beta_group'] = df_processed['sig'].astype(str) + '_' + df_processed['Mean_Beta_sign']
        df_processed['cor'] = df_processed['corrs'].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        summary_df = (
            df_processed
            .groupby(['group', 'day', 'Behavior_num', 'sig_beta_group'])
            .agg(
                cor_mean=('cor', 'mean'),
                cor_sem=('cor', lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)
            )
            .reset_index()
        )
        summary_df['day'] = pd.to_numeric(summary_df['day'], errors='coerce')
        day0_means = (
            summary_df[summary_df['day'] == 0]
            .set_index(['group', 'Behavior_num', 'sig_beta_group'])['cor_mean'])

        def get_day0_mean(row):
            key = (row['group'], row['Behavior_num'], row['sig_beta_group'])
            return day0_means.get(key, np.nan)

        summary_df['cor_mean_norm'] = summary_df.apply(lambda row: row['cor_mean'] / get_day0_mean(row) if get_day0_mean(row) else np.nan, axis=1)
        summary_df['cor_sem_norm'] = summary_df.apply(lambda row: row['cor_sem'] / get_day0_mean(row) if get_day0_mean(row) else np.nan,axis=1)
        summary_df['Behavior_num'] = summary_df['Behavior_num'].astype(str)

        sns.set(style='whitegrid')

        g = sns.FacetGrid(
            summary_df,
            row='group',
            col='Behavior_num',
            margin_titles=True,
            sharey=True,
            sharex=True,
            height=4,
            aspect=1.5
        )

        g.map_dataframe(
            sns.lineplot,
            x='day',
            y='cor_mean_norm',
            hue='sig_beta_group',
            style='sig_beta_group',
            markers=True,
            err_style='bars',
            ci=None  # We plot SEM manually below
        )

        # Manually add error bars for SEM (normalized SEM is roughly SEM / day0_mean)
        for ax, (_, subdf) in zip(g.axes.flat, summary_df.groupby(['group', 'Behavior_num'])):
            for label, group_df in subdf.groupby('sig_beta_group'):
                # Compute normalized SEM (sem_norm = sem / day0_mean)
                day0_val = day0_means.get((group_df['group'].iloc[0], group_df['Behavior_num'].iloc[0], label), np.nan)
                if not np.isnan(day0_val) and day0_val != 0:
                    sem_norm = group_df['cor_sem'] / day0_val
                else:
                    sem_norm = group_df['cor_sem']  # fallback, no normalization if day0 missing
                
                ax.errorbar(
                    group_df['day'],
                    group_df['cor_mean_norm'],
                    yerr=sem_norm,
                    fmt='o',
                    capsize=3,
                    label=label
                )

        g.add_legend(title='sig + Mean_Beta sign')
        g.set_axis_labels("Day", "Normalized Avg Correlation")
        g.set_titles(row_template="{row_name}", col_template="Behavior {col_name}")
        plt.tight_layout()
        plt.savefig('correlationbysubgroup.jpg')

        df_sig = summary_df[(summary_df['sig_beta_group'] != '0_neg') & (summary_df['sig_beta_group'] != '0_pos')].copy()
        df_sig['beta_sign'] = df_sig['sig_beta_group'].apply(lambda x: 'pos' if 'pos' in x else ('neg' if 'neg' in x else np.nan))

        g = sns.FacetGrid(
            df_sig,
            row='beta_sign',          # Separate rows for Positive / Negative
            col='Behavior_num',
            margin_titles=True,
            sharey=True,
            sharex=True,
            height=4,
            aspect=1.5
        )

        g.map_dataframe(
            sns.lineplot,
            x='day',
            y='cor_mean_norm',
            hue='group',
            style='group',
            markers=True,
            err_style='bars',
            ci=None
        )

        # Manually add SEM error bars on each subplot
        for ax, ((beta_sign, behavior_num), subdf) in zip(g.axes.flat, df_sig.groupby(['beta_sign', 'Behavior_num'])):
            for group, group_df in subdf.groupby('group'):
                ax.errorbar(
                    group_df['day'],
                    group_df['cor_mean_norm'],
                    yerr=group_df['cor_sem_norm'],
                    fmt='none',  # no marker, only error bars
                    capsize=3,
                    label=None,
                    ecolor='black',
                    alpha=0.7
                )

        g.add_legend(title='Group')
        g.set_axis_labels("Day", "Normalized Avg Correlation")
        g.set_titles(row_template="{row_name}", col_template="Behavior {col_name}")
        plt.tight_layout()
        plt.savefig('correlationbysubbeta.jpg')

        # Analyze decoder results to correlation
        decoder_copy = self.decoder_results.copy()
        decoder_copy = decoder_copy[decoder_copy['Behavior'].astype(int).isin([0, 1, 2, 3])]
        decoder_copy['recording_id'] = decoder_copy['suid'].astype(str) + decoder_copy['day'].astype(str) + decoder_copy['Behavior'].astype(str) 
        decoder_copy = decoder_copy.drop(['Recording', 'cage','mouse'], axis=1)
        decoder_copy = decoder_copy.rename(columns={'Behavior': 'behavior', 'Accuracy':'accuracy','F1 Score':'f1','Rs':'r2'})

        df_processed = df_processed.rename(columns={'Behavior_num': 'behavior', 'Mean_Beta_Sign':'sign'})
        df_processed['recording_id'] = df_processed['suid'].astype(str) + df_processed['day'].astype(str) + df_processed['behavior'].astype(str) 

        merged_df = pd.merge(df_processed, decoder_copy, on=['recording_id','suid','group','day','behavior'])
        agg_cols = ['F1', 'cor', 'accuracy', 'r2']  # replace with your actual column names
        summary_df = merged_df.groupby(['recording_id','behavior','group'])[agg_cols].mean().reset_index()

        sns.lmplot(
            data=summary_df,  # replace with your actual DataFrame
            x='cor', y='F1',
            hue='group',
            col='behavior',  # or 'Behavior_num' if you used that
            height=4, aspect=1.2,
            scatter_kws={'alpha': 0.6},
            line_kws={'color': 'black'}
        )
        plt.suptitle("Decoder F1 vs Neuronal Correlation (per odor)", y=1.02)
        plt.tight_layout()

        from scipy.stats import spearmanr

        for odor in summary_df['behavior'].unique():
            sub = summary_df[summary_df['behavior'] == odor]
            rrr, ppp = spearmanr(sub['cor'], sub['F1'])
            print(f"{odor}: Spearman r = {rrr:.2f}, p = {ppp:.3f}")















        def categorize_beta(row):
            if row['sig'] == 1:
                if row['Mean_Beta'] > 0:
                    return 'positive'
                elif row['Mean_Beta'] < 0:
                    return 'negative'
            return 'non_sig'
        
        all_cors=[]
        for activity,timestamps in zip(self.neuronal_activity, self.behavioral_timestamps):
            for i, ts in enumerate(timestamps):
                corr_matrix = np.corrcoef(activity)
                all_cors.append(corr_matrix)
        
        all_cors_list = [row for dataset in all_cors for row in dataset]
        self.beta_results_corrs = self.beta_results_filtered.copy()
        self.beta_results_corrs['beta_significance'] = self.beta_results_corrs.apply(categorize_beta, axis=1)
        self.beta_results_corrs['correlations'] = all_cors_list
        self.beta_results_corrs = self.beta_results_corrs.drop(columns=['trace'])
        self.beta_results_corrs = self.beta_results_corrs[self.beta_results_corrs['correlations'].apply(lambda x: not np.isnan(x).all())]
        df = self.beta_results_corrs.copy()
        cor_lengths = df['correlations'].apply(len)
        df_repeated = df.loc[df.index.repeat(cor_lengths)].reset_index(drop=True)
        all_corrs = np.concatenate(df['correlations'].values)
        df_repeated['cor'] = all_corrs
        df_repeated['neuron_index'] = np.concatenate([np.arange(n) for n in cor_lengths])
        df_repeated = df_repeated.rename(columns={'neuron_index': 'neuron_comp'})
        self.beta_results_corrs_long = df_repeated
        self.beta_results_corrs_long = self.beta_results_corrs_long.drop(columns=['correlations'])
        self.beta_results_corrs_long.to_csv(os.path.join(self.drop_directory,'betaresultslong.csv'))

        df = self.beta_results_corrs_long.copy()

        # Step 1: Average cor per (day, group, suid, nuid, beta_significance)
        grouped = (
            df.groupby(['day', 'group', 'suid', 'nuid', 'beta_significance'])['cor']
            .mean()
            .reset_index()
        )

        # Step 2: Get Day 0 means per (group, beta_significance) for normalization
        baseline = (
            grouped[grouped['day'] == '0']
            .groupby(['group', 'beta_significance'])['cor']
            .mean()
            .reset_index()
            .rename(columns={'cor': 'day0_mean'})
        )

        # Step 3: Merge baseline back and normalize
        grouped_norm = grouped.merge(baseline, on=['group', 'beta_significance'], how='left')
        grouped_norm['cor_norm'] = grouped_norm['cor'] / grouped_norm['day0_mean']

        # Step 4: Compute mean and SEM of normalized cor per (day, group, beta_significance)
        summary = (
            grouped_norm.groupby(['day', 'group', 'beta_significance'])['cor_norm']
            .agg(['mean', sem])
            .reset_index()
            .rename(columns={'mean': 'cor_mean', 'sem': 'cor_sem'})
        )

        # Step 5: Plot using FacetGrid
        g = sns.FacetGrid(
            summary,
            row='group',         # facet by group (rows)
            col='day',           # facet by day (columns)
            hue='group',         # color by group
            height=4,            # optional tweak for smaller subplots
            aspect=1,
            palette='Set2',
            sharey=False         # allow different y-axis scales per facet
        )

        def facet_errorplot(data, **kwargs):
            x_labels = data['beta_significance'].unique()
            x_map = {label: i for i, label in enumerate(x_labels)}
            for group in data['group'].unique():
                group_data = data[data['group'] == group]
                xs = [x_map[val] for val in group_data['beta_significance']]
                plt.errorbar(
                    xs,
                    group_data['cor_mean'],
                    yerr=group_data['cor_sem'],
                    fmt='o',
                    label=group,
                    capsize=4,
                    markersize=6,
                    elinewidth=1.5,
                    color=kwargs.get('color')
                )
            plt.xticks(ticks=range(len(x_labels)), labels=x_labels)
            plt.xlabel('Beta Significance')
            plt.ylabel('Normalized Mean Correlation ± SEM')

        g.map_dataframe(facet_errorplot)
        g.add_legend(title='Group')
        g.set_titles(col_template='Day {col_name}')
        plt.tight_layout()
        plt.savefig('normalized_facet_correlations_by_day2.jpg')




    def __call__(self):
        # Clean data
        self.normalize_activity()
        self.timestamps_to_one_hot_array()
        # self.run_glm()
        # self.cor_peth()
        # self.gen_venn_diagram_beta()
        # self.plot_decoder_results()
        self.beta_weight_validation()
        self.beta_weight_functional_connectivity()

def gather_data(parent_data_directory,drop_directory,file_indicator='obj'):
    """ Gather all data into lists from parent directory """
    # Get full path to object files
    objfiles=glob.glob(os.path.join(parent_data_directory,f'**/{file_indicator}*.json'),recursive=True)
 
    # Grab relevant data from files and create lists
    neuronal_activity=[]
    behavioral_timestamps=[]
    neuron_info = pd.DataFrame(columns=['day', 'cage', 'mouse', 'group'])
    for file in tqdm(objfiles):
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
    regressobj = circuit_regression(drop_directory=drop_directory,
                                                       neuronal_activity=neuronal_activity,
                                                       behavioral_timestamps=behavioral_timestamps,
                                                       neuron_info=neuron_info)
    regressobj()

    print('Finished coefficient clustering...')
