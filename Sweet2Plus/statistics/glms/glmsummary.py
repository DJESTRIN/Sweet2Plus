#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: glmsummary.py
Description: Analysis of beta weights from Engelhard et al., 2019 based glm. 
Author: David James Estrin
Version: 1.1
Date: 03-27-2026

Current to do list:
 - write and read results to temp files. 
 - Write code for seperation of neurons by beta weight classifications
 - Analysis of functional connectivity and neuronal activity wrt to beta_weight classification, stress group, and day
"""

# Set up CLI for input directory
# Read in gz files into a dataframe
# Generate figures
# Generate summary statistics
# Run analyses? 
import argparse
import glob, os
import gzip
import pickle
import pandas as pd
import numpy as np

def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_directory',type=str,help='Parent directory where g zipped data is located')
    parser.add_argument('--output_file',type=str,default='engelhard_stimulus_summary.csv',
                         help='Where the per-neuron per-stimulus summary CSV is saved')
    parser.add_argument('--number_events',type=int,default=4,help='Number of distinct stimulus/odor events fit by the GLM')
    parser.add_argument('--number_bases_spline',type=int,default=50,help='Number of B-spline basis functions per event used by engelhardglm')
    args=parser.parse_args()
    return args.input_directory, args.output_file, args.number_events, args.number_bases_spline

# Canonical odor order established in Sweet2Plus/core/behavior.py (quick_timestamps' `trials` list):
# index 0=Vanilla, 1=PeanutButter, 2=Water, 3=FoxUrine(TMT). This must match circuit_coefficient_clustering.py's
# behavior_map so that engelhardglm's per-stimulus summary can be joined against circuit_regression's beta_filtered.csv.
DEFAULT_STIMULUS_NAMES = ['vanilla', 'peanut', 'water', 'TMT']

class collect():
    def __init__(self,input_path, number_events=4, number_bases_spline=50, stimulus_names=None):
        self.input_path = input_path
        self.number_events = number_events
        self.number_bases_spline = number_bases_spline
        self.stimulus_names = stimulus_names if stimulus_names is not None else DEFAULT_STIMULUS_NAMES

    def load_results(self, search_string = None):
        """ Results are save to temp pickle files to lower use of RAM during fit. 
        Here we load results from pickled files after fit for analyses."""

        # Default search string when not provided
        if search_string is None:
            search_string = self.input_path + r'/*.pkl.gz'

        # Find model output files and load them in to common list attribute
        model_files = glob.glob(search_string)
        self.model_results = []
        for filename in model_files:
            with gzip.open(filename, "rb") as f:
                information,_ = (os.path.basename(filename)).split('.pk')
                day,cage,mouse,group,neuronid = information.split('_')
                # Filenames are written as f'D{day}_C{cage}_M{mouse}_G{group}_N{neuron_number}' in
                # engelhardglm.fit(); strip the single-letter prefix so values match neuron_info's
                # raw day/cage/mouse/group/neuron values used elsewhere (e.g. circuit_regression's nuid).
                day, cage, mouse, group, neuronid = day[1:], cage[1:], mouse[1:], group[1:], neuronid[1:]
                self.model_results.append([pickle.load(f),day,cage,mouse,group,neuronid])
    
    def generate_dataframe(self):
        rows = []

        for neuron_data in self.model_results:
            data, day, cage, mouse, group, neuronid = neuron_data

            for perm_idx, j in enumerate(data):
                betas = np.array(j['betas'])      # (200,)
                pvals = np.array(j['pvalues'])    # (200,)
                r2 = float(np.array(j['r2']).flatten()[0])  # scalar
                perm_type = j['type']             # scalar or label

                # sanity check
                if betas.ndim != 1:
                    print("Unexpected shape:", betas.shape)
                    continue

                n_betas = betas.shape[0]

                rows.append(pd.DataFrame({
                    'day': day,
                    'cage': cage,
                    'mouse': mouse,
                    'group': group,
                    'neuronid': neuronid,
                    'permutation': perm_type,
                    'perm_idx': perm_idx,
                    'beta_idx': np.arange(n_betas),
                    'beta': betas,
                    'pval': pvals,
                    'r2': r2   # repeated automatically
                }))

        return pd.concat(rows, ignore_index=True)

    def generate_stimulus_summary(self):
        """ Collapse each neuron's per-spline-basis GLM betas into a single encoding-strength weight
        (and empirical p-value) per stimulus, matching circuit_regression's decoder output granularity
        so the two models' neuron/stimulus weights can be directly compared.

        For each neuron:
          - The real-data betas (excluding the intercept added by sm.add_constant) are split into
            `number_events` contiguous blocks of `number_bases_spline` spline-basis coefficients each,
            one block per stimulus (assumes engelhardglm was fit with interactions=False, the default).
          - The summary weight per stimulus is max(|beta|) across that stimulus's spline basis, i.e.
            the strongest single-timepoint encoding effect for that odor.
          - The empirical p-value compares that real summary weight against the same summary statistic
            computed from the circular-lag permutation fits (the null distribution), following the
            circular-lag permutation procedure already run in engelhardglm.linearmodel.

        Returns a tidy dataframe with one row per neuron per stimulus:
        nuid, day, cage, mouse, group, neuronid, stimulus, stimulus_idx, weight, signed_weight, p_value, sig.
        `nuid` is built as cage_mouse_day_neuronid to match circuit_regression's beta_results['nuid'].
        `weight` is unsigned (max |beta|); `signed_weight` retains sign for direction comparisons.
        """
        n_expected = self.number_events * self.number_bases_spline
        rows = []

        for neuron_data in self.model_results:
            data, day, cage, mouse, group, neuronid = neuron_data

            real_entry = next((j for j in data if j.get('type') == 'real'), None)
            if real_entry is None:
                continue

            # engelhardglm.include_interactions() always prepends a redundant constant column via
            # sm.add_constant(X) (named "X0" in the formula), on top of which smf.glm's formula
            # interface ("Y ~ X0 + X1 + ...") adds its *own* "Intercept" term. So result.params is
            # ordered [Intercept, X0(const), X1, X2, ...] -- drop both leading terms to reach the
            # actual number_events x number_bases_spline spline-basis coefficients.
            real_betas = np.asarray(real_entry['betas'])[2:]
            if real_betas.shape[0] != n_expected:
                print(f"Skipping neuron {neuronid} (day={day}, cage={cage}, mouse={mouse}): "
                      f"expected {n_expected} spline-basis betas (number_events x number_bases_spline), "
                      f"got {real_betas.shape[0]}. Check number_events/number_bases_spline or interactions setting.")
                continue

            real_groups = np.split(real_betas, self.number_events)
            real_summary = [np.max(np.abs(g)) for g in real_groups]
            # Signed value at the spline basis with the largest magnitude, kept for sign/direction
            # comparisons against circuit_regression's signed Mean_Beta (max(|beta|) itself has no sign).
            real_signed = [g[np.argmax(np.abs(g))] for g in real_groups]

            perm_summaries = []
            for j in data:
                perm_type = j.get('type')
                if not (isinstance(perm_type, str) and perm_type.startswith('permutation')):
                    continue
                perm_betas = np.asarray(j['betas'])[2:]
                if perm_betas.shape[0] != n_expected:
                    continue
                perm_groups = np.split(perm_betas, self.number_events)
                perm_summaries.append([np.max(np.abs(g)) for g in perm_groups])
            perm_summaries = np.array(perm_summaries) if perm_summaries else np.zeros((0, self.number_events))

            nuid = f"{cage}_{mouse}_{day}_{neuronid}"
            for ev_idx in range(self.number_events):
                stim_name = self.stimulus_names[ev_idx] if ev_idx < len(self.stimulus_names) else f'Behavior_{ev_idx}'
                weight = real_summary[ev_idx]

                if perm_summaries.shape[0] > 0:
                    null_dist = perm_summaries[:, ev_idx]
                    # Empirical (permutation) p-value: fraction of null summaries at least as extreme as observed.
                    p_value = (np.sum(null_dist >= weight) + 1) / (len(null_dist) + 1)
                else:
                    p_value = np.nan

                rows.append({
                    'nuid': nuid,
                    'day': day, 'cage': cage, 'mouse': mouse, 'group': group, 'neuronid': neuronid,
                    'stimulus': stim_name, 'stimulus_idx': ev_idx,
                    'weight': weight, 'signed_weight': real_signed[ev_idx], 'p_value': p_value,
                    'sig': int(p_value < 0.05) if not np.isnan(p_value) else 0,
                })

        return pd.DataFrame(rows)

    def save_stimulus_summary(self, output_path='engelhard_stimulus_summary.csv'):
        df = self.generate_stimulus_summary()
        df.to_csv(output_path, index=False)
        return df

def proc():
    input_directory, output_file, number_events, number_bases_spline = cli_parser()
    collection_obj = collect(input_path=input_directory, number_events=number_events, number_bases_spline=number_bases_spline)
    collection_obj.load_results()
    summary_df = collection_obj.save_stimulus_summary(output_path=output_file)
    print(f"Saved per-neuron per-stimulus GLM summary ({len(summary_df)} rows) to {output_file}")

if __name__=='__main__':
    proc()