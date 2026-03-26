#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: engelhardglm.py
Description: Here we attempt to use a nearly identical glm as Engelhard et al., 2019 for neuronal encoding. We also run validation analyses. 
Author: David James Estrin
Version: 1.1
Date: 03-05-2026
"""
# import Sweet2Plus code
from Sweet2Plus.core.SaveLoadObjs import gather_data  # Custom function which basically loads in all of the activity, event, and info data

# import libraries
import argparse
import os, glob
import shutil
import pickle
import tqdm
import time
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.interpolate import BSpline
from itertools import combinations
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.model_selection import GridSearchCV

# Misc. libraries
from joblib import Parallel, delayed
import ipdb

# Custom classes and functions
class currate_data(object):
    # Gathers data, putting it into correct format for glm analysis
    def __init__(self,  neuronal_activity, behavioral_timestamps, neuron_info, dropdirectory, tsoffset_frames = 10):
        self.act = neuronal_activity
        self.ts = behavioral_timestamps
        self.info = neuron_info
        self.dropdirectory = dropdirectory # Directory where charts and data will be saved
        self.tsoffset_frames = tsoffset_frames # The number of frames from odor onset to offset in 2P recording. 
    
    def __call__(self):
        self.trans_act, self.trans_ts, self.trans_info = self.reshape()

    def reshape(self):
        # Loop over activity list (which is activity for all neurons)
        trans_act = []
        trans_ts = []
        for recording,timestamps in tqdm.tqdm(zip(self.act,self.ts),total=len(self.act)):

            # Convert timestamps
            trans_timestamps = np.zeros((recording.shape[1],len(timestamps)))
            for jj,k in enumerate(timestamps):
                numbers = np.array(k)

                # Take out trials accidently too close to the end of recording
                numbers = np.array([x for x in numbers if x <= (recording.shape[1]-10)])
                offsets = np.arange(self.tsoffset_frames)  # A parameter
                indices = numbers[:, None] + offsets  
                indices = indices.flatten().astype(int)    
                trans_timestamps[indices,jj] = 1

            for neuron in recording:
                trans_act.append(neuron)
                trans_ts.append(trans_timestamps)

        return  trans_act, trans_ts, self.info

    def save(self):
        # Save collected data into a pickle file
        os.makedirs(self.dropdirectory, exist_ok=True)
        attributes = {
            "trans_act.pkl": self.trans_act,
            "trans_ts.pkl": self.trans_ts,
            "trans_info.pkl": self.trans_info,}

        for filename, data in attributes.items():
            filepath = os.path.join(self.dropdirectory, filename)
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

    def load(self):
        # Load collected data from a pickle file
        attributes = {
            "trans_act.pkl": "trans_act", # replaced for deconvolved trace
            "trans_ts.pkl": "trans_ts",
            "trans_info.pkl": "trans_info",}

        # Load each file and assign as attribute
        for filename, attr_name in attributes.items():
            filepath = os.path.join(self.dropdirectory, filename)
            with open(filepath, "rb") as f:
                setattr(self, attr_name, pickle.load(f))

class engelhardglm(object):
    def __init__(self, activity, timestamps, info, dropdir, post_event_frames = 20, model_type = 'glm', selected_family_glm = None, 
                 scaling_parameter = 1, spline_duration=483, number_bases_spline=50, polynom_degree=2, batch_size=20, interactions = False, 
                 graphics=False, graphing_rate = 10, auto_regressor=False, delete_previous_results = True, small_sample=False, 
                 start_neuron=None, stop_neuron=None):
        # General data attributes
        self.activity = activity
        self.timestamps = timestamps
        self.info = info
        self.dropdir = dropdir

        # Data parameters
        self.post_event_frames = post_event_frames # The number of frames to be included after an event onset for fitting.
        self.scaling_parameter = scaling_parameter # Maybe delete? Scales the predictors by some factor to attempt to match activity range

        # Hyper parameters
        self.spline_duration = spline_duration
        self.number_bases_spline = number_bases_spline
        self.polynom_degree = polynom_degree
        self.small_sample = small_sample
        self.batch_size = batch_size

        # Model attributes
        self.interactions = interactions # Will linear model include interactions of events which are main effects (Ex. Y ~ Ev1 + Ev2 + Ev1*Ev2)
        self.auto_regressor = auto_regressor # 
        model_selction = {'lm', 'lmm', 'glm'}
        if model_type not in model_selction:
            raise ValueError(f"model_type must be one of {model_selction}, got '{model_type}'")
        self.model_type = model_type
        self.selected_family_glm = selected_family_glm

        # Graphing attributes
        self.graphics = graphics
        self.graphing_rate = graphing_rate # graphs every nth neuron where n is the number you set
        if small_sample:
            self.graphing_rate = 1
        self.graphics_path_predictions = f"{self.dropdir}/neuron_predictions/"
        self.graphics_path_metrics = f"{self.dropdir}/neuron_general_metrics/"

        # Clear previous graphics 
        if delete_previous_results:
            if os.path.exists(self.graphics_path_predictions):
                shutil.rmtree(self.graphics_path_predictions)

            if os.path.exists(self.graphics_path_metrics):
                shutil.rmtree(self.graphics_path_metrics)
            
        os.makedirs(self.graphics_path_predictions, exist_ok=True)
        os.makedirs(self.graphics_path_metrics, exist_ok=True)

    def __call__(self):
        self.inspectdependent()
        self.fit()

    def inspectdependent(self):
        flat_array = np.concatenate(self.activity)
        sample_size = int(len(flat_array) * 0.1)
        flat_sample = np.random.choice(flat_array, size=sample_size, replace=False)
        
        # Plot distribution of activity
        plt.hist(flat_sample, bins=1000, edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Flattened Array')
        plt.savefig(os.path.join(self.graphics_path_metrics, "histogram.png"))
        plt.close()
        
        # Suggest family wise error if glm is selected
        if self.selected_family_glm is None:
            if np.array_equal(np.unique(flat_array), [0, 1]):
                self.selected_family_glm = sm.families.Binomial()
            elif np.all(flat_array >= 0) and np.all(flat_array == np.floor(flat_array)):
                self.selected_family_glm = sm.families.Poisson()
            elif np.all(flat_array > 0):
                self.selected_family_glm = sm.families.Gamma()
            else:
                self.selected_family_glm = sm.families.Gaussian()

    def binarytospline(self):
        def spline_kernel(n_bases=7, duration=10, degree=4):
            t = np.arange(duration)
            n_knots = n_bases + degree + 1
            knots = np.linspace(0, duration, n_knots - 2 * degree)
            knots = np.concatenate([
                np.repeat(knots[0], degree),
                knots,
                np.repeat(knots[-1], degree)
            ])
            kernels = np.zeros((duration, n_bases))
            for i in range(n_bases):
                c = np.zeros(n_bases)
                c[i] = 1
                kernels[:, i] = BSpline(knots, c, degree)(t)
            return kernels 

        self.kernels = spline_kernel(n_bases=self.number_bases_spline, duration=self.spline_duration, degree=self.polynom_degree)

    def include_interactions(self, number_events, converted_events):
        if self.interactions:
            all_X = []
            for r in range(1, number_events + 1):  # 1-way to 4-way (or n_events)
                for comb in combinations(range(number_events), r):
                    X_comb = converted_events[comb[0]]
                    for idx in comb[1:]:
                        X_comb = X_comb * converted_events[idx]  # elementwise multiply
                    all_X.append(X_comb)

            X = np.hstack(all_X)
            X = sm.add_constant(X)
        else:
            X = np.hstack(converted_events)
            X = sm.add_constant(X)
        return X
    
    def linearmodel(self, X, Y,  info_oh, counter, random_intercept = None, permutations=500):
        """ Method: linear model 
        Generate linear model. Model can be linear model, linear mixed model or generalized linear model. The use of glmm is not a good option in python. 

        Inputs:
        X -- (numpy array) The event data converted into splines for predicting neuronal activity
        Y -- (numpy array) Neuronal activity which we are trying to predict
        info_oh -- (numpy array) information regarding the current neuron being fit with a linear model
        counter -- (int) current neuron we are analyzing in entire dataset. Used to plot a subset of data for inspection. 
        random_intercept -- (lst) Used for linear mixed model to account for repeated trials. 
        """
        # For permutation testing for glm
        def circular_time_lag(X):
            n = len(X)
            lag = np.random.randint(0, n)  # random lag
            X_lagged = pd.DataFrame(np.roll(X.values, shift=lag, axis=0), index=X.index,columns=X.columns)
            return X_lagged, lag

        results=[]

        match self.model_type:
            case "lm":
                model = sm.OLS(Y, X).fit()
                predicted = model.predict(X)

                # Save relevant data to list
                results.append({
                    "info": info_oh,
                    "betas": result.params,
                    "pvalues": result.pvalues,})

            case "lmm":
                df_oh = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
                df_oh['activity'] = Y
                df_oh['trial'] = random_intercept
                model = MixedLM(
                    endog=df_oh['activity'],
                    exog=df_oh.drop(columns=['activity','trial']),
                    groups=df_oh['trial'],
                    exog_re=np.ones((len(df_oh), 1)))
                
                # Attempt to use REML first
                try:
                    result = model.fit(reml=True)
                except:
                    result = model.fit(reml=False) 
                predicted = result.predict(df_oh.drop(columns=['activity','trial']))

                # Save relevant data to list
                results.append({
                    "info": info_oh,
                    "betas": result.params,
                    "pvalues": result.pvalues,})

            case "glm":
                df_glm = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

                # GLM on Circular Lag Permutation testing first
                for i in range(permutations):
                    X_perm, lag = circular_time_lag(df_glm)
                    
                    # Using formula interface for GLM
                    X_perm['Y'] = Y
                    model = smf.glm("Y ~ " + " + ".join(X_perm.columns[:-1]), data=X_perm, family=self.selected_family_glm)
                    result = model.fit()
                    predicted = result.predict(X_perm)

                    # Save relevant data to list
                    results.append({
                        "info": info_oh,
                        "betas": result.params,
                        "pvalues": result.pvalues,
                        'type':f'permutation_{str(i)}',
                        'r2':(np.corrcoef(Y, predicted)[0, 1])**2,})

                # GLM on Real Data 
                df_glm = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
                df_glm['Y'] = Y
                model = smf.glm("Y ~ " + " + ".join(df_glm.columns[:-1]), data=df_glm, family=self.selected_family_glm)
                result = model.fit()
                predicted = result.predict(df_glm)

                results.append({
                    "info": info_oh,
                    "betas": result.params,
                    "pvalues": result.pvalues,
                    'type':f'real',
                    'r2':(np.corrcoef(Y, predicted)[0, 1])**2,})

        # Save graphics of predicted and obsorved data
        if self.graphics:
            if counter % self.graphing_rate == 0:
                self._modeldiagnostics(glm_result=result, observed=Y, predictors=X, neuron_id=info_oh, neu_number=counter, predicted=predicted)
        
        return results 

    def fit(self):
        # Prepare selected trials if small_sample
        if getattr(self, "small_sample", False) and len(self.activity) > self.batch_size:
            selected_trials = set(np.random.choice(len(self.activity), size=self.batch_size, replace=False))
        
        elif self.start_neuron is not None and self.stop_neuron is not None:
            selected_trials = set(range(self.start_neuron, self.stop_neuron + 1))
        
        else:
            selected_trials = set(range(len(self.activity)))

        # Convert activity, timestamps and info to arrays and make sure same shape
        self.activity = np.array(self.activity)
        self.timestamps = np.array(self.timestamps)
        self.info = np.array(self.info)
        assert self.activity.shape[0] == self.timestamps.shape[0] == self.info.shape[0], \
            "Error, activity, timestamps and info do not match in shape."

        # Takes event data and makes a set of B-splines
        self.binarytospline()

        # Nested function for processing neurons in parallel
        def process_single_neuron(dropdirectory, counter, activity_oh, ts_oh, info_oh, neuron_number,kernels, scaling_parameter, graphics, 
                                  graphics_path_metrics,post_event_frames,linearmodel_func, include_interactions_func):
            
            # Save the results to a temp directory inside of dropdirectory
            def save_temp_results(fullfilename,results):
                with gzip.open(fullfilename, "wb") as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Get onset of trials 
            timestamps_len, number_odors = ts_oh.shape
            trial_segments = []
            for odor_idx in range(number_odors):
                odor_trace = ts_oh[:, odor_idx].copy()
                odor_onsets = np.where(np.diff(odor_trace, prepend=0) > 0)[0]
                odor_offsets = np.where(np.diff(odor_trace, append=0) < 0)[0]
                for onset, offset in zip(odor_onsets, odor_offsets):
                    trial_segments.append((onset, min(offset + post_event_frames, timestamps_len), odor_idx + 1))

            trial_segments.sort(key=lambda x: x[0])
            if len(trial_segments) == 0:
                return KeyError('No trials') 

            # Crop out beginning and end of recording where there are no trials. 
            trial_id_full = np.zeros(timestamps_len, dtype=int)
            for i, (start_seg, end_seg, trial_num) in enumerate(trial_segments):
                if i < len(trial_segments) - 1:
                    next_onset = trial_segments[i + 1][0]
                    trial_id_full[start_seg:next_onset] = trial_num
                else:
                    trial_id_full[start_seg:end_seg] = trial_num

            start = trial_segments[0][0]
            end = trial_segments[-1][1]
            activity_crop = activity_oh[start:end]
            ts_crop = ts_oh[start:end, :]
            trial_id = trial_id_full[start:end]
            timestamp_len, number_events = ts_crop.shape

            # Convert trial timestamps to splines using kernels, these are predictors for model
            converted_events = []
            for ev_oh in range(number_events):
                ev_trace = ts_crop[:, ev_oh]
                onset = np.diff(ev_trace, prepend=0)
                onset[onset < 0] = 0
                expanded_onset = np.zeros((timestamp_len, kernels.shape[1]))
                for k in range(kernels.shape[1]):
                    conv_oh = np.convolve(onset, kernels[:, k], mode='full')[:timestamp_len]
                    expanded_onset[:, k] = conv_oh
                converted_events.append(expanded_onset)

            # Normalize the spline predictors to a similar scale. 
            activity_std = np.std(activity_crop) * scaling_parameter
            normalized_events = []
            for X_i in converted_events:
                X_norm = np.zeros_like(X_i)
                for j in range(X_i.shape[1]):
                    col = X_i[:, j]
                    col_min, col_max = col.min(), col.max()
                    if col_max - col_min > 1e-8:
                        X_norm[:, j] = (col - col_min) / (col_max - col_min)
                X_norm *= activity_std
                normalized_events.append(X_norm)
            converted_events = normalized_events

            # Include interactions of events and add intercept. Default is to not include interactions
            X_oh = include_interactions_func(number_events, converted_events)
            model_result = linearmodel_func(X=X_oh, Y=activity_crop, info_oh=info_oh,
                                            counter=counter, random_intercept=trial_id)
            
            # Set filename and save model's results
            os.makedirs(dropdirectory+r'/temp/',exist_ok=True) # make sure directory is made
            filename = dropdirectory+r'/temp/' + f'D{info_oh[0]}_C{info_oh[1]}_M{info_oh[2]}_G{info_oh[3]}_N{str(neuron_number)}.pkl.gz'
            save_temp_results(filename,model_result)
            return 

        # Run Code in parrallel
        selected_trials = np.array(list(selected_trials)).astype(int)
        activity_subset = self.activity[selected_trials]
        timestamps_subset = self.timestamps[selected_trials]
        info_subset = self.info[selected_trials]

        Parallel(n_jobs=-1, backend="loky", verbose=10)(delayed(process_single_neuron)
                                                                  (self.dropdir, counter, act, ts, info,neuron_number,
                                                                   self.kernels, self.scaling_parameter,
                                                                   self.graphics, self.graphics_path_metrics,
                                                                   self.post_event_frames,
                                                                   self.linearmodel, self.include_interactions)
                                                                   for counter, (act, ts, info, neuron_number) in enumerate(zip(activity_subset, timestamps_subset, info_subset, selected_trials)))

    def _modeldiagnostics(self, glm_result, observed, predictors, neuron_id=None, neu_number=None, predicted=None):
        # Compute predicted values if not provided
        if predicted is None:
            predicted = glm_result.predict(predictors)

        # Metrics
        r = np.corrcoef(observed, predicted)[0, 1]  # Pearson correlation
        r2 = r**2
        rmse = np.sqrt(np.mean((observed - predicted)**2))  

        # Create figure
        plt.figure(figsize=(12, 4))
        plt.plot(observed, label="Observed", color='blue', alpha=0.7)
        plt.plot(predicted, label="Predicted", color='orange', alpha=0.7)
        title_str = f"{neuron_id[0]}_{neuron_id[1]}_{neuron_id[2]}_{neuron_id[3]}_{str(neu_number)}"
        plt.title(f"GLM Predicted vs Observed: {title_str}\nr={r:.3f}, R²={r2:.3f}, RMSE={rmse:.3f}")
        plt.xlabel("Timepoint")
        plt.ylabel("Activity")
        plt.legend()
        plt.tight_layout()

        # Save figure if dropdir exists
        if hasattr(self, "dropdir") and self.dropdir is not None:
            os.makedirs(f"{self.dropdir}/neuron_predictions/", exist_ok=True)
            filename = f"glm_neu_diagnostics_{title_str}.png" if neuron_id is not None else "glm_diagnostics.png"
            plt.savefig(os.path.join(self.graphics_path_predictions, filename))
            plt.close()

    def _optimize_hyper_params(self, n_iter=200, random_state=None):
        rng = np.random.default_rng(random_state)
        
        # Parameters
        spline_duration_range = (5, 500)    
        number_bases_options = np.arange(5, 51, 5) 
        polynom_degree_options = np.arange(2, 7)  

        best_median_r2 = -np.inf
        best_params = None
        for _ in tqdm.tqdm(range(n_iter)):
            try:
                sd = rng.uniform(*spline_duration_range)    
                nb = rng.choice(number_bases_options)       
                deg = rng.choice(polynom_degree_options)

                # run small batch through model with these parameters
                self.spline_duration = int(sd)
                self.number_bases_spline = int(nb)
                self.polynom_degree = int(deg)
                self.fit()

                # Compute median R^2
                median_r2 = np.median(np.hstack(self.r2))
                if median_r2 > best_median_r2:
                    best_median_r2 = median_r2
                    best_params = (sd, nb, deg)
            except:
                continue

        # Set best parameters
        self.spline_duration, self.number_bases_spline, self.polynom_degree = best_params
        
        # Save best parameters
        os.makedirs(self.dropdir, exist_ok=True)
        filepath = os.path.join(self.dropdir, "best_parameters.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(best_params, f)
        print(f"Best median R²: {best_median_r2:.4f} with params: {best_params} has been saved")

def cli_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,help='Parent directory where data is located')
    parser.add_argument('--drop_directory',type=str,help='where results are saved to')
    parser.add_argument('--data_provided',action='store_true',help='If data was already provided in correct format')
    parser.add_argument('--start_neuron',type=int, default=None, help='start neuron index')
    parser.add_argument('--stop_neuron',type=int, default=None, help='stop neuron index')
    args=parser.parse_args()
    return args.data_directory, args.drop_directory, args.data_provided, args.start_neuron, args.stop_neuron

def proc():
    # General procedure 
    data_directory, drop_directory, data_provided, start_neuron_indx, stop_neuron_indx = cli_parser()

    # If data was already processed and is in drop directory, skip this step
    if not data_provided:
        neuronal_activity, behavioral_timestamps, neuron_info = gather_data(parent_data_directory=data_directory,drop_directory=drop_directory)

        # Currate the data
        dataset = currate_data(neuronal_activity, behavioral_timestamps, neuron_info, drop_directory)
        dataset()
        dataset.save()
    else:
        dataset = currate_data([],[],[], drop_directory)
        dataset.load()

    if (start_neuron_indx  is not None) and (stop_neuron_indx is not None):
        print(f'Running neuron {start_neuron_indx} to {stop_neuron_indx}')
        glmobj = engelhardglm(activity=dataset.trans_act,
                              timestamps=dataset.trans_ts, 
                              info=dataset.trans_info, 
                              dropdir=drop_directory, 
                              graphics=True,
                              start_neuron=start_neuron_indx,
                              stop_neuron=stop_neuron_indx)
        glmobj()
    
    else:
        print('Running all neurons')
        glmobj = engelhardglm(activity=dataset.trans_act,timestamps=dataset.trans_ts, info=dataset.trans_info, dropdir=drop_directory, graphics=True)
        glmobj()
    
    #glmobj._optimize_hyper_params() # Run optimization to determine hyperparameters for model 
    return 

if __name__=='__main__':
    proc()




