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

import math
import glob, os, sys
import warnings
import collections
from random import shuffle

import dill
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import scipy.io as sio
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as db
from sklearn.metrics import silhouette_score as ss
from sklearn.decomposition import PCA


matplotlib.rc('font', family='sans-serif')
matplotlib.rc('font', serif='Arial')
matplotlib.rcParams.update({'font.size': 24})


def main():


    reg_res = {'f13': {'effort3': []}, 'f15': {'effort3': []}, 'f16': {'effort3': []}, 'm17': {'effort3': []}, 'm21': {'effort3': []}, 'm58': {'effort3': []}, 'f60': {'effort3': []}, 'f64': {'effort3': []}, 'm76': {'effort3': []}, 'm78': {'effort3': []}, 'f82': {'effort3': []}, 'f822': {'effort3': []}}

    #reg_res = {'m49': {'effort3': []}, 'm68': {'effort3': []}, 'm70': {'effort3': []}, 'm96': {'effort3': []}, 'm99': {'effort3': []}, 'm702': {'effort3': []}, 'f71': {'effort3': []}, 'f75': {'effort3': []}, 'f104': {'effort3': []}, 'f712': {'effort3': []}}

    #reg_res = {'f13': {'CNSDS1': []}, 'f15': {'CNSDS1': []}, 'm17': {'CNSDS1': []}, 'm21': {'CNSDS1': []}, 'm212': {'CNSDS1': []}, 'm58': {'CNSDS1': []}, 'f64': {'CNSDS1': []}, 'm76': {'CNSDS1': []}, 'f82': {'CNSDS1': []}, 'f822': {'CNSDS1': []}}

    #reg_res = {'m27': {'effort3': []}, 'm28': {'effort3': []}, 'm29': {'effort3': []}, 'm30': {'effort3': []}, 'f45': {'effort3': []}, 'm86': {'effort3': []}, 'm862': {'effort3': []}, 'm90': {'effort3': []}, 'f95': {'effort3': []}}

    #reg_res = {'m27': {'CNSDS1': []}, 'm29': {'CNSDS1': []}, 'm30': {'CNSDS1': []}, 'm272': {'CNSDS1': []}, 'f45': {'CNSDS1': []}, 'm86': {'CNSDS1': []}, 'm862': {'CNSDS1': []}, 'm90': {'CNSDS1': []}, 'm902': {'CNSDS1': []}, 'f95': {'CNSDS1': []}}

    #reg_res = {'m49': {'CNSDS1': []}, 'm70': {'CNSDS1': []}, 'm702': {'CNSDS1': []}, 'f71': {'CNSDS1': []}, 'f75': {'CNSDS1': []}}

    mice = sorted(reg_res.keys())
    sess_types = ['effort3']
    var_name = 'denoise'
    pred_names = ['CS-', 'CS+', 'CS- LE cue', 'CS+ LE cue', 'CS- HE cue', 'CS+ HE cue',    #0-5 (indices for pie.py)
                  'CS- LE TI', 'CS+ LE TI', 'CS- HE TI', 'CS+ HE TI',   #6-9
                  'CS- LE early reward', 'CS+ LE early reward', 'CS- HE early reward', 'CS+ HE early reward',   #10-13
                  'CS- LE late reward', 'CS+ LE late reward', 'CS- HE late reward', 'CS+ HE late reward',   #14-17
                  'lickport returns', 'Lick Bouts']

    for sess_type in sess_types:
        all_neuron_tvals = []
        all_neuron_coeffs = []
        all_neuron_sig_idx = []
        if sess_type in ['3td1', '3tlate']:
            continue
            # no_eff = 1
            # pred_names[2] = 'Late Cue'
            # pred_names.remove('Effort')
            # print(pred_names)
        else:
            no_eff = 0
        for mouse in mice:

            neural_activity, preds = parse_input_files(sess_type, mouse, var_name, pred_names, no_eff)

            tvals, coeffs, pvals = run_regression(neural_activity, preds)

            
            sig_idx = np.reshape(multipletests(pvals.flatten(), method='bonferroni')[0], pvals.shape)

            reg_res[mouse][sess_type].extend([coeffs, tvals, sig_idx])
            
            all_neuron_tvals.append(tvals)
            all_neuron_coeffs.append(coeffs)
            all_neuron_sig_idx.append(sig_idx)

            
            plot_order = [0,2,6,10,14,4,8,12,16,1,3,7,11,15,5,9,13,17,18,19]
            plot_reg_res(tvals[:,plot_order], 't-values', sess_type, [pred_names[i] for i in plot_order], mouse=mouse, sig_idx=sig_idx[:,plot_order])
            plot_reg_res(coeffs[:,plot_order], 'coefficients', sess_type, [pred_names[i] for i in plot_order], mouse=mouse)

            print(mouse, sess_type, 'done')

        
        map_all = np.concatenate(all_neuron_tvals, axis=0)
        all_neuron_sig_idx = np.concatenate(all_neuron_sig_idx, axis=0)
        plot_reg_res(map_all[:,plot_order], 't-values', sess_type, [pred_names[i] for i in plot_order], sig_idx=all_neuron_sig_idx[:,plot_order])

        map_all = np.concatenate(all_neuron_coeffs, axis=0)
        plot_reg_res(map_all[:,plot_order], 'coefficients', sess_type, [pred_names[i] for i in plot_order])


    with open('/Users/pujaparekh/Desktop/analysis_pickles/PFC_NAc/effort3/reg_res.db', 'wb') as ofile:
        dill.dump(reg_res, ofile)


def parse_input_files(sess_type, mouse, var_name, pred_names, no_eff):


    na = '/Users/pujaparekh/Desktop/dfof_denoise_decon_files/PFC_NAc/DC/' + mouse + '/' + mouse + '_' + sess_type + '_DC_allneurons.mat'
    ff_dir = '/Users/pujaparekh/Desktop/frames_files/' + mouse + '/' + mouse + '_' + sess_type + '_frames'
    lick_dir = '/Users/pujaparekh/Desktop/behavior_files/' + mouse + '/' + 'licks/' + mouse + '_' + sess_type + '_licks'
    
    os.chdir(ff_dir)
    ffs = sorted(list(glob.glob('*.txt')))
    loaded_mat = sio.loadmat(na)
    keys = list(loaded_mat.keys())

    behav_frames = []
    for ff in ffs:
        with open(ff, 'r') as frames_file:
            behav_frames.append([line.split()[1] for line in frames_file.readlines()])
            print(len(behav_frames[-1]), ff)

    print(mouse, sess_type, 'good')

    os.chdir(lick_dir)
    lick_file = list(glob.glob('*' + sess_type + '*.txt'))
    print(lick_file[0])
    with open(lick_file[0], 'r') as lf:
        lick_trials = [line.split()[3:-2] if 'ITI:' in line.split() else line.split()[3:] for line in lf.readlines()]

    num_files = len(ffs)
    nas = [np.array(loaded_mat[var_name + str(x)]) if x > 1 else np.array(loaded_mat[var_name]) for x in np.arange(1,num_files+1)]
    
    neural_activity = np.concatenate(nas, axis=1)
    behavior = np.concatenate(behav_frames)
    print(neural_activity.shape, behavior.shape)

    preds = np.zeros((len(behavior), len(pred_names)))
    trial_starts = []

    num_reward_frames = 0
    num_cue_frames = 0
    prev_frame = ''
    trial_type = None

    for frame_idx, frame in enumerate(behavior[:-1]):
        if frame != prev_frame:
            if frame != behavior[frame_idx+1]:
                frame = behavior[frame_idx+1]
            if frame in ['25','26','27','28'] and (prev_frame == '24' or frame_idx==0):
                trial_starts.append((frame_idx, frame))
            if frame == '24' and trial_type in [2,3]:
                preds[frame_idx:frame_idx+30,18] = 1

        if frame in ['25','26','27','28']:
            trial_type = ['25','26','27','28'].index(frame)
            num_cue_frames += 1
            if num_cue_frames < 30:
                if frame == '25' or frame == '27':
                    preds[frame_idx, 0] = 1
                else:
                    preds[frame_idx, 1] = 1
            else:
                preds[frame_idx,2+trial_type] = 1

        else:
            num_cue_frames = 0

            if frame == '30'and trial_type is not None:
                preds[frame_idx, 6+trial_type] = 1
            elif frame == '29' and trial_type is not None:
                num_reward_frames += 1
                if num_reward_frames < 75:
                    preds[frame_idx, 10+trial_type] = 1
                else:
                    preds[frame_idx, 14+trial_type] = 1
            else:
                num_reward_frames = 0
        prev_frame = frame

    trial_idx = 0
    lick_trial_len = len(lick_trials)
    while trial_idx < lick_trial_len:
        trial = lick_trials[trial_idx]
        trial_type = int(trial[0])

        if trial_starts[trial_idx][1] in ['25', '27']:
            rew_type = 1
        else:
            rew_type = 2
        
        if no_eff == 0 and trial_starts[trial_idx][1] in ['27', '28']:
            rew_type += 2

        if trial_starts[trial_idx][0]+34 < preds.shape[0]:
            # if preds[trial_starts[trial_idx]+34, 3] == 1 and no_eff == 0:
            #     rew_type += 2
            
            if trial_type != rew_type:
                raise TypeError(str(trial_idx) + ' ' + str(trial) + ' ' + str(rew_type))
                # del(lick_trials[trial_idx])
                # trial = lick_trials[trial_idx]
                # lick_trial_len -= 1

            
            lick_bout_starts = []
            for lick_idx, lick in enumerate(trial[1:-1]):
                if int(lick) > 8000:
                    if len(lick_bout_starts) == 0:
                        if int(trial[lick_idx+2]) - int(lick) < 200:
                            lick_bout_starts.append(int(lick))
                    else:
                        if (int(lick) - lick_bout_starts[-1]) >= 200:
                            if int(trial[lick_idx+2]) - int(lick) < 200:
                                lick_bout_starts.append(int(lick))
            for lick in lick_bout_starts:
                frame_num = int(lick *.03)
                preds[(trial_starts[trial_idx][0] + frame_num):(trial_starts[trial_idx][0] + frame_num + 7), -1] = 1
        
        trial_idx += 1

    if preds.shape[0] == neural_activity.shape[1]-1:
        neural_activity = neural_activity[:,1:]

    for neuron_idx, neuron in enumerate(neural_activity):
        mean = np.mean(neuron)
        std = np.std(neuron)
        neural_activity[neuron_idx] = (neuron - mean) / std


    return neural_activity, preds


def run_regression(neural_activity, preds):

    tvals = np.ndarray((neural_activity.shape[0], preds.shape[1]))
    pvals = np.ndarray((neural_activity.shape[0], preds.shape[1]))
    coeffs = np.ndarray((neural_activity.shape[0], preds.shape[1]))
    neural_activity = StandardScaler().fit_transform(neural_activity.T).T
    for neuron_idx, neuron in enumerate(neural_activity):
        ols_results = OLS(neuron, preds).fit()
        tvals[neuron_idx] = ols_results.tvalues
        coeffs[neuron_idx] = ols_results.params
        pvals[neuron_idx] = ols_results.pvalues
    
    return tvals, coeffs, pvals


def plot_reg_res(vals, val_type, sess_type, pred_names, mouse='', sig_idx=None):


    if sig_idx is not None:
        sig_idx = [True if row.any() else False for row in sig_idx]
        vals = vals[sig_idx,:]

    pca_vals = PCA(n_components=min(vals.shape)).fit_transform(vals)

    clusts_to_consider = range(2,min(20,vals.shape[0]))
    sil_scores = np.zeros(len(clusts_to_consider))
    for clust_idx, num_clust in enumerate(clusts_to_consider):
        clust = db(n_clusters=num_clust, max_iter=1000).fit(pca_vals)
        labels = clust.labels_
        sil_scores[clust_idx] = ss(vals,labels)
    lowest_sil = sil_scores.argmax()
    num_clust = list(clusts_to_consider)[lowest_sil]


    clust = db(n_clusters=num_clust).fit(pca_vals)
    labels = clust.labels_
    sort_indices = np.argsort(labels)
    sorted_vals = vals[sort_indices,:]
    sorted_labels = labels[sort_indices]
    hlines = [x_idx-0.5 for x_idx, x in enumerate(sorted_labels[1:]) if x!=sorted_labels[x_idx]]
    hlines.append(len(sorted_labels)-0.5)

    bound = max([np.percentile(sorted_vals,99),np.abs(np.percentile(sorted_vals,1))])
    max_val = bound
    min_val = bound * -1
    
    plt.figure(figsize=[12,14])
    plt.imshow(sorted_vals, aspect='auto', cmap='YlGnBu', vmin=min_val, vmax=max_val, origin='lower', interpolation='none')
    plt.xticks(np.arange(vals.shape[1]), pred_names[:vals.shape[1]], rotation=-45, ha='left')
    plt.xlabel('Predictors', fontsize=34)
    plt.ylabel('Neurons', fontsize=34)
    # plt.yticks(np.arange(6), [str(x) for x in range(1,7)])
    plt.hlines(hlines, xmin=-.5, xmax=vals.shape[1]-.5, color='k')
    cbar = plt.colorbar()
    cbar.set_label(val_type)
    #plt.title('Regression ' + val_type.capitalize(), fontsize=30)
    # if mouse != '':
    #     mouse += '/'
    plt.tight_layout()

    plt.savefig('/Users/pujaparekh/Desktop/regression/PFC_NAc/' + mouse + sess_type + '_' + val_type + '.png')
    plt.close()


if __name__ == "__main__":
    main()
