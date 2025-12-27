#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: state_dynamics.py
Description: Compare vector angles and euclidiean distances of states wrt to group x day
Author: David Estrin
Version: 1.0
Date: 11-06-2025
"""
import argparse
import ipdb
from Sweet2Plus.core.SaveLoadObjs import SaveObj, LoadObj, SaveList, OpenList, gather_data
import numpy as np
import pandas as pd
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class StateDynamics:
    def __init__(self, neuronal_activity, behavioral_timestamps, neuron_info, fps=1.3, baseline=300):
        self.neuronal_activity = neuronal_activity
        self.behavioral_timestamps = behavioral_timestamps
        self.neuron_info = neuron_info
        self.fps=fps
        self.baseline_width = baseline

    def __call__(self):
        self.get_AUC()
        self.get_euclidien_distance()
        self.get_vector_angle()

        # merge dataframes
        self.result_dataframe = pd.merge(self.euclid_df,
                                         self.angle_df,
                                         on=['suid', 'day', 'group', 'odor1', 'odor2'],
                                         how='outer')


    def get_AUC(self):
        """ Calculate the AUC for baseline, water, peanut, vanilla and TMT odors """
        def local_dataframe(AUCs, Odors):
            n_neurons = len(AUCs[0])
            for arr in AUCs:
                if len(arr) != n_neurons:
                    raise ValueError("All AUC arrays must have same length")
            df = pd.DataFrame({odor: AUCs[i] for i, odor in enumerate(Odors)})
            df['Neuron'] = np.arange(1, n_neurons + 1)
            return df

        # Calculate AUC for each neuron in dataset for each odor and baseline
        df = pd.DataFrame()  # Final dataframe
        for subject_odors, subject_neurons in tqdm.tqdm(zip(self.behavioral_timestamps, self.neuronal_activity)):
            AUCs=[] # AUCs for subject's neurons will be temporarily stored here
            Odors = []  # Names of odors are temporarily stored here
            for odor_timestamps,odor_names in zip(subject_odors,['water','peanut','vanilla','tmt']):
                window_size = round(10 / self.fps) 
                valid_ts = [ts for ts in odor_timestamps if ts + window_size <= subject_neurons.shape[1]]
                try:
                    windows = np.array([subject_neurons[:, int(ts):int(ts+window_size)] for ts in valid_ts])  
                except:
                    ipdb.set_trace()
                windows = np.swapaxes(windows, 0, 1)
                aucs = np.trapz(windows, axis=2)  
                AUCs_oh_mean = aucs.mean(axis=1) 
                AUCs.append(AUCs_oh_mean)
                Odors.append(odor_names)

            # Add baseline period AUC (0-5 minutes before first trial)
            first_odor = min([t for sublist in subject_odors for t in sublist])
            baseline_auc = np.trapz(subject_neurons[:, int(first_odor-round(self.baseline_width/self.fps)):int(first_odor-5)]) # Baseline period is from -baselinewidth to first odor - 5 frames
            AUCs.append(baseline_auc)
            Odors.append('baseline')

            # Append data to dataframe
            df_subj = local_dataframe(AUCs, Odors)
            df = pd.concat([df, df_subj], ignore_index=True)
        
        # Set dataframe as attribute
        self.df = pd.concat([df, self.neuron_info], axis=1)
        self.df['NeuronID'] = self.neuron_info.index

    def get_euclidien_distance(self):
        """ Build dataframe containing euclidian distance values by subject """
        self.df['suid'] = self.df['day'].astype(str) + '_' + self.df['mouse'].astype(str) + '_' + self.df['cage'].astype(str)
        odors = ['water', 'peanut', 'vanilla', 'tmt']

        results = []
        for suid, sub_df in self.df.groupby('suid'):
            neuron_matrix = sub_df[odors].values  

            # Take subject-level info
            day = sub_df['day'].iloc[0]
            group = sub_df['group'].iloc[0]

            for i in range(len(odors)):
                for j in range(i + 1, len(odors)):
                    odor1 = odors[i]
                    odor2 = odors[j]
                    vec1 = neuron_matrix[:, i]
                    vec2 = neuron_matrix[:, j]
                    dist = np.linalg.norm(vec1 - vec2)
                    results.append({
                        'suid': suid,
                        'day': day,
                        'group': group,
                        'odor1': odor1,
                        'odor2': odor2,
                        'euclidean_distance': dist
                    })

        self.euclid_df = pd.DataFrame(results)

    def get_vector_angle(self):
        """ Build dataframe containing vector angle values by subject """
        self.df['suid'] = self.df['day'].astype(str) + '_' + self.df['mouse'].astype(str) + '_' + self.df['cage'].astype(str)
        odors = ['water', 'peanut', 'vanilla', 'tmt']
        baseline_col = 'baseline'  # Column used as origin

        results_angle = []
        for suid, sub_df in self.df.groupby('suid'):
            neuron_matrix = sub_df[odors].values
            baseline_vector = sub_df[baseline_col].values  
            day = sub_df['day'].iloc[0]
            group = sub_df['group'].iloc[0]

            for i in range(len(odors)):
                for j in range(i + 1, len(odors)):
                    odor1 = odors[i]
                    odor2 = odors[j]
                    vec1 = neuron_matrix[:, i] - baseline_vector  
                    vec2 = neuron_matrix[:, j] - baseline_vector
                    dot_product = np.dot(vec1, vec2)
                    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    
                    if norm_product == 0:
                        angle = np.nan  
                    else:
                        cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
                        angle = np.arccos(cos_theta)
                    
                    results_angle.append({
                        'suid': suid,
                        'day': day,
                        'group': group,
                        'odor1': odor1,
                        'odor2': odor2,
                        'angle_rad': angle
                    })

        self.angle_df = pd.DataFrame(results_angle)


class StateStatistics:
    def __init__(self, dataframe):
        self.df = dataframe

    def graph_euclid_distance_summary(self):
        """ Generate summary graph of Euclidean distance normalized to baseline day 0 """
        df_norm = self.df.copy()
        df_norm['odor_comparison'] = df_norm['odor1'] + ' vs ' + df_norm['odor2']
        subject_baseline = (
            df_norm[df_norm['day'] == '0']
            .groupby(['suid', 'odor_comparison'])['euclidean_distance']
            .mean()
        )
        group_baseline = (
            df_norm[df_norm['day'] == '0']
            .groupby(['group', 'odor_comparison'])['euclidean_distance']
            .mean()
        )
        df_norm['baseline_value'] = df_norm.apply(
            lambda r: subject_baseline.get((r['suid'], r['odor_comparison'])),
            axis=1
        )
        df_norm['baseline_value'] = df_norm.apply(
            lambda r: group_baseline.get((r['group'], r['odor_comparison']))
                    if pd.isna(r['baseline_value'])
                    else r['baseline_value'],
            axis=1
        )
        df_norm['euclid_norm'] = df_norm['euclidean_distance'] / df_norm['baseline_value']
        ipdb.set_trace()

        summary_df = df_norm.groupby(['group', 'day', 'odor1', 'odor2']).agg(
            mean_distance=('euclid_norm', 'mean'),
            sem_distance=('euclid_norm', lambda x: np.std(x, ddof=1)/np.sqrt(len(x)))
        ).reset_index()
        summary_df['odor_pair'] = summary_df['odor1'] + ' vs ' + summary_df['odor2']

        groups = summary_df['group'].unique()
        odor_pairs = summary_df['odor_pair'].unique()
        n = len(groups)

        day_order = ['0', '7', '14', '30']

        fig, axes = plt.subplots(1, n, figsize=(6*n, 6), sharey=True)

        if n == 1:
            axes = [axes]

        for ax, group in zip(axes, groups):
            df_g = summary_df[summary_df['group'] == group].copy()
            # Ensure days are ordered correctly
            df_g['day'] = pd.Categorical(df_g['day'], categories=day_order, ordered=True)
            df_g = df_g.sort_values('day')

            sns.barplot(
                data=df_g,
                x='day',
                y='mean_distance',
                hue='odor_pair',
                ci=None,
                ax=ax,
                legend=False
            )

            n_hue = len(odor_pairs)
            total_width = 0.8
            bar_width = total_width / n_hue

            for i, row in df_g.iterrows():
                day_index = day_order.index(str(row['day']))
                hue_index = list(odor_pairs).index(row['odor_pair'])
                xpos = day_index - total_width/2 + bar_width/2 + hue_index*bar_width

                ax.errorbar(
                    xpos,
                    row['mean_distance'],
                    yerr=row['sem_distance'],
                    fmt='none',
                    color='black',
                    capsize=5
                )

            ax.set_title(f'Group: {group}')
            ax.set_xlabel('Day')
            ax.set_ylabel('Normalized Euclidean Distance')
            ax.set_xticks(range(len(day_order)))
            ax.set_xticklabels(day_order)

        # Put legends outside
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='center right',
            bbox_to_anchor=(1.05, 0.5),
            title='Odor Pair'
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig('euclid_state_changes.jpg')


    def graph_vector_angle_summary(self):
        " Generate summary graph of vector angle  "
        df_norm = self.df.copy()
        df_norm['odor_comparison'] = df_norm['odor1'] + ' vs ' + df_norm['odor2']
        subject_baseline = (
            df_norm[df_norm['day'] == '0']
            .groupby(['suid', 'odor_comparison'])['angle_rad']
            .mean()
        )
        group_baseline = (
            df_norm[df_norm['day'] == '0']
            .groupby(['group', 'odor_comparison'])['angle_rad']
            .mean()
        )
        df_norm['baseline_value'] = df_norm.apply(
            lambda r: subject_baseline.get((r['suid'], r['odor_comparison'])),
            axis=1
        )
        df_norm['baseline_value'] = df_norm.apply(
            lambda r: group_baseline.get((r['group'], r['odor_comparison']))
                    if pd.isna(r['baseline_value'])
                    else r['baseline_value'],
            axis=1
        )
        df_norm['angle_rad_norm'] = df_norm['angle_rad'] / df_norm['baseline_value']

        ipdb.set_trace()

        summary_df = df_norm.groupby(['group', 'day', 'odor1', 'odor2']).agg(
            mean_ang=('angle_rad_norm', 'mean'),
            sem_ang=('angle_rad_norm', lambda x: np.std(x, ddof=1)/np.sqrt(len(x)))
        ).reset_index()
        summary_df['odor_pair'] = summary_df['odor1'] + ' vs ' + summary_df['odor2']

        groups = summary_df['group'].unique()
        odor_pairs = summary_df['odor_pair'].unique()
        n = len(groups)

        day_order = ['0', '7', '14', '30']

        fig, axes = plt.subplots(1, n, figsize=(6*n, 6), sharey=True)

        if n == 1:
            axes = [axes]

        for ax, group in zip(axes, groups):
            df_g = summary_df[summary_df['group'] == group].copy()
            # Ensure days are ordered correctly
            df_g['day'] = pd.Categorical(df_g['day'], categories=day_order, ordered=True)
            df_g = df_g.sort_values('day')

            sns.barplot(
                data=df_g,
                x='day',
                y='mean_ang',
                hue='odor_pair',
                ci=None,
                ax=ax,
                legend=False
            )

            n_hue = len(odor_pairs)
            total_width = 0.8
            bar_width = total_width / n_hue

            for i, row in df_g.iterrows():
                day_index = day_order.index(str(row['day']))
                hue_index = list(odor_pairs).index(row['odor_pair'])
                xpos = day_index - total_width/2 + bar_width/2 + hue_index*bar_width

                ax.errorbar(
                    xpos,
                    row['mean_ang'],
                    yerr=row['sem_ang'],
                    fmt='none',
                    color='black',
                    capsize=5
                )

            ax.set_title(f'Group: {group}')
            ax.set_xlabel('Day')
            ax.set_ylabel('Normalized Vector Angle')
            ax.set_xticks(range(len(day_order)))
            ax.set_xticklabels(day_order)

        # Put legends outside
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='center right',
            bbox_to_anchor=(1.05, 0.5),
            title='Odor Pair'
        )

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        plt.savefig('vector_angle_state_changes.jpg')


def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,help='Parent directory where data is located')
    parser.add_argument('--drop_directory',type=str,help='where results are saved to')
    args=parser.parse_args()
    return args.data_directory, args.drop_directory

if __name__=='__main__':
    data_directory, drop_directory = cli_parser()
    neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)
    States_oh = StateDynamics(neuronal_activity, behavioral_timestamps, neuron_info)
    States_oh()
    summary_stats = StateStatistics(dataframe=States_oh.result_dataframe)
    summary_stats.graph_euclid_distance_summary()
    summary_stats.graph_vector_angle_summary()