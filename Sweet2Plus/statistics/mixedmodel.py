#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: mixedmodel.py
Description: Run GLM or LMM on neuronal data. 

Author: David Estrin
Version: 1.0
Date: 02-25-2025
"""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse 
import ipdb
import itertools

class mixedmodels():
    """ General class for building, running and evaluating mixed models """
    def __init__(self, drop_directory, dataframe, model_type='lmm', fixed_effects_formula = "Activity ~ Group * Session * Trialtype", 
                 random_effects='Subject',nested_effects='Neuron', multicompare_correction = 'fdr_bh',
                 verbose=True):
        """  For running mixed model statistics on neuronal data
        Inputs 
        drop_directory -- Where results will be saved. 
        dataframe -- A Pandas dataframe containing all rows of data. Ex, every row is individual neuron by trial 
        model_type
        fixed_effects_formula -- A String containing the layout of main effects and interactions
        random_effects -- Varaibles considered a random effect
        nested_effects -- Nest variables such as neuron that lie within the random effect of subject
        verbose -- Prints results to command line for quick viewing. 

        Outputs
        - A number of figures will be saved in drop directory
        - CSV files derived from pandas arrays containing relevant stats data such as multiple comparisons and effects
        """
        # Attributes
        self.drop_directory = drop_directory
        self.dataframe = dataframe
        self.model_type = model_type
        self.formula = fixed_effects_formula
        self.random_effects = random_effects
        self.nested_effects = nested_effects
        self.multicompare_correction = multicompare_correction
        self.verbose = verbose

        self.dataframe['group'] = self.dataframe['group'].astype('category').cat.codes
        self.dataframe['trialtype'] = self.dataframe['trialtype'].astype('category').cat.codes
        self.dataframe['period'] = self.dataframe['period'].astype('category').cat.codes
        self.dataframe['day'] = self.dataframe['day'].astype('category').cat.codes
        self.dataframe['suid'] = self.dataframe['suid'].astype('category').cat.codes
        self.dataframe['neuid'] = self.dataframe['neuid'].astype('category').cat.codes
        self.dataframe['trialid'] = self.dataframe['trialid'].astype('category').cat.codes
        self.dataframe['auc'] = self.dataframe['auc'].astype('float')
        # self.dataframe['auc_avg'] = self.dataframe.groupby(['suid','neuid','group', 'day', 'trialtype', 'period'])['auc'].transform('mean')
        # self.dataframe = self.dataframe[self.dataframe['period'] != 0].drop(columns=['period'])
        self.columns_order = ['auc', 'group', 'day', 'trialtype', 'period', 'trialid', 'suid', 'neuid']
        self.dataframe = self.dataframe[self.columns_order]
        assert self.model_type=='lmm'

    def __call__(self):
        """ General statistical protocol """
        self.data_distributions()
        self.generate_model()
        self.EMM()
        self.EMM_multiple_comparisons()
        self.residual_evaluation()
        ipdb.set_trace()

    def data_distributions(self):
        plt.figure(figsize=(7,5))
        sns.histplot(self.dataframe['auc'], bins=1000, kde=True)
        plt.axvline(0, color='red', linestyle='dashed')
        plt.xlabel("AUC")
        plt.ylabel("Denisty")
        plt.savefig(os.path.join(self.drop_directory,"KernelDensityAUC.jpg"))

    def generate_model(self):
        """ Build and run LinearMixed model based on attributes """
        print(f'Fitting full model with formula:{self.formula}')
        self.full_model = smf.mixedlm('auc ~ group * day * trialtype ',
                                 self.dataframe,
                                 groups=self.dataframe['suid'], 
                                 re_formula="1",
                                 vc_formula={'suid:neuid': '1'})
        
        self.full_model_result = self.full_model.fit()

        ipdb.set_trace()
        self.predictions = self.full_model_result.predict(self.dataframe.drop(columns='auc'))
        self.dataframe['predictions'] = self.predictions

    def EMM(self):
        """ Python does not have a package that does this, so I needed to code it """
        # Group by categorical variables and calculate the mean prediction for each group
        emmeans = self.dataframe.groupby(['group', 'day', 'trialtype', 'period'])['predictions'].mean().reset_index()
        ipdb.set_trace()
        std_devs = self.dataframe.groupby(['group', 'day', 'trialtype', 'period'])['predictions'].std().reset_index()
        group_sizes = self.dataframe.groupby(['group', 'day', 'trialtype', 'period']).size().reset_index(name='n')
        emmeans = emmeans.merge(std_devs, on=['group', 'day', 'trialtype', 'period'])
        self.emmeans = emmeans.merge(group_sizes, on=['group', 'day', 'trialtype', 'period'])

    def EMM_multiple_comparisons(self):
        # Generate all possible combinations
        combinations = list(itertools.product(self.emmeans['group'].unique(), 
                                            self.emmeans['day'].unique(), 
                                            self.emmeans['trialtype'].unique(), 
                                            self.emmeans['period'].unique()))

        # Function to check if two tuples differ in exactly one element
        def differs_by_one(a, b):
            diff_count = sum(x != y for x, y in zip(a, b))
            return diff_count == 1

        # Compare every combination with every other combination
        for combo1, combo2 in itertools.combinations(combinations, 2):
            if differs_by_one(combo1, combo2):
                mean1 = self.emmeans[(self.emmeans['group'] == combo1[0]) & 
                  (self.emmeans['day'] == combo1[1]) & 
                  (self.emmeans['trialtype'] == combo1[2]) & 
                  (self.emmeans['period'] == combo1[3])]['predictions_x']
                
                std1 = self.emmeans[(self.emmeans['group'] == combo1[0]) & 
                  (self.emmeans['day'] == combo1[1]) & 
                  (self.emmeans['trialtype'] == combo1[2]) & 
                  (self.emmeans['period'] == combo1[3])]['predictions_y']
        
                n1 = self.emmeans[(self.emmeans['group'] == combo1[0]) & 
                  (self.emmeans['day'] == combo1[1]) & 
                  (self.emmeans['trialtype'] == combo1[2]) & 
                  (self.emmeans['period'] == combo1[3])]['n']
                
                mean2 = self.emmeans[(self.emmeans['group'] == combo2[0]) & 
                  (self.emmeans['day'] == combo2[1]) & 
                  (self.emmeans['trialtype'] == combo2[2]) & 
                  (self.emmeans['period'] == combo2[3])]['predictions_x']
                
                std2 = self.emmeans[(self.emmeans['group'] == combo2[0]) & 
                  (self.emmeans['day'] == combo2[1]) & 
                  (self.emmeans['trialtype'] == combo2[2]) & 
                  (self.emmeans['period'] == combo2[3])]['predictions_y']
        
                n2 = self.emmeans[(self.emmeans['group'] == combo2[0]) & 
                  (self.emmeans['day'] == combo2[1]) & 
                  (self.emmeans['trialtype'] == combo2[2]) & 
                  (self.emmeans['period'] == combo2[3])]['n']
                ipdb.set_trace()

    def multiple_comparisons(self):
        """ Run multiple comparisons on significant interactions and/or main effects """
        factor_levels = [self.dataframe[f].unique() for f in ["group", "day", "trialtype", "period"]]
        all_comparisons = list(itertools.combinations(itertools.product(*factor_levels), 2))

        # Store p-values
        p_values = []
        t_stats = []
        df_stats = []
        comparisons = []

        for (a, b) in all_comparisons:
            subset = self.dataframe[
                (self.dataframe["group"] == a[0]) & (self.dataframe["day"] == a[1]) &
                (self.dataframe["trialtype"] == a[2]) & (self.dataframe["period"] == a[3])]
            
            subset_b = self.dataframe[
                (self.dataframe["group"] == b[0]) & (self.dataframe["day"] == b[1]) &
                (self.dataframe["trialtype"] == b[2]) & (self.dataframe["period"] == b[3])]
            
            # Run t-test
            t_stat, p_val, df_oh = sm.stats.ttest_ind(subset["auc"], subset_b["auc"])
            p_values.append(p_val)
            comparisons.append(f"{a} vs {b}")
            t_stats.append(t_stat)
            df_stats.append(df_oh)

        # Apply FDR correction (Benjamini-Hochberg)
        _, p_corrected, _, _ = multipletests(p_values, method="fdr_bh")

        # Show results
        for comp, t_val, df_oh, p_val, p_corr in zip(comparisons, t_stats, df_stats, p_values, p_corrected):
            print(f"{comp}: t({df_oh})={t_val:.4f}, p={p_val:.4f} -->  FDR-corrected p={p_corr:.4f}")

    def residual_evaluation(self):
        """ Generate common plots and stats for residuals to manaully evaluate model fit """
        # Pull residuals and fitted values
        residuals = self.dataframe['auc_avg'] - self.predictions
        fit_vals = self.predictions

        # Plot of residuals and 
        plt.figure(figsize=(7,5))
        sns.scatterplot(x=fit_vals, y=residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='dashed')
        plt.xlabel("Fitted Values")
        plt.ylabel("Deviance Residuals")
        plt.title("Residuals vs. Fitted Values")
        plt.savefig(os.path.join(self.drop_directory,"ResidualsVFitted_Check.jpg"))

        # Histogram of residuals
        plt.figure(figsize=(7,5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.xlabel("Deviance Residuals")
        plt.title("Histogram of Deviance Residuals")
        plt.savefig(os.path.join(self.drop_directory,"ResidualHistogram_Check.jpg"))

        # qqplot plot
        plt.figure(figsize=(7,5))
        sm.qqplot(residuals, line='45', fit=True)
        plt.title("Q-Q Plot of Residuals")
        plt.savefig(os.path.join(self.drop_directory,"ResidualQQplot_Check.jpg"))

        # Predictor vs residual plot
        plt.figure(figsize=(7,5))
        sns.scatterplot(x=self.dataframe["group"], y=residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='dashed')
        plt.xlabel("Predictor Variable")
        plt.ylabel("Deviance Residuals")
        plt.title("Residuals vs. Predictor")
        plt.savefig(os.path.join(self.drop_directory,"ResidualVpredictor_Check.jpg"))

        # Heteroskedacity plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        plt.figure(figsize=(7,5))
        sns.scatterplot(x=fit_vals, y=sqrt_abs_residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='dashed')
        plt.xlabel("Fitted Values")
        plt.ylabel("√|Deviance Residuals|")
        plt.title("Scale-Location Plot")
        plt.savefig(os.path.join(self.drop_directory,"Heterosckedacity_Check.jpg"))

class compare_models():
    def __init__(self, drop_directory, dataframe, dependent_variable, fixed_effects, random_effects, nested_effects,incorporate_random_slopes=False):
        self.drop_directory = drop_directory
        self.dataframe = dataframe
        self.dependent_variable = dependent_variable
        self.fixed_effects = fixed_effects
        self.random_effects = random_effects
        self.nested_effects = nested_effects
        self.incorporate_random_slopes = incorporate_random_slopes
    
    def __call__(self):
        self.all_models = self.get_all_models()
        
        all_aic_data = []
        for model_oh in self.all_models:
            if model_oh['groups'] is  None or model_oh['vc_formula'] is None:
                continue
            else:
                try:
                    current_model_oh = smf.mixedlm(model_oh['formula'], self.dataframe, groups=model_oh['groups'], 
                                            re_formula="1", vc_formula=model_oh['vc_formula'])
                    result_oh = current_model_oh.fit()
                    AIC_value = result_oh.aic
                except:
                    AIC_value = np.nan

                all_aic_data.append([model_oh, AIC_value])
        
        ipdb.set_trace()

    def get_all_models(self):
        model_specifications = []
        fixed_effect_combos = list(self.powerset(self.fixed_effects))[1:] 
        random_effect_combos = list(self.powerset(self.random_effects))

        nested_combos = []
        for re_combo in random_effect_combos:
            nested_structure = {}
            for re in re_combo:
                if re in nested_effects:
                    for nested in self.powerset(self.nested_effects[re]):
                        if nested: 
                            nested_structure[re] = "1"
            nested_combos.append(nested_structure)

        for fixed_combo, re_combo, nested_combo in product(fixed_effect_combos, random_effect_combos, nested_combos):
            fixed_part = " + ".join(fixed_combo)
            groups = re_combo[0] if re_combo else None
            random_parts = [f"(1|{re})" for re in re_combo] 
            random_part = " + ".join(random_parts) if random_parts else ""

            if random_part:
                formula = f"{self.dependent_variable} ~ {fixed_part} + {random_part}"
            else:
                formula = f"{self.dependent_variable} ~ {fixed_part}"

            model_specifications.append({
                "formula": formula,
                "groups": groups,
                "vc_formula": nested_combo if nested_combo else None})

        return model_specifications

    def powerset(self, itoh):
        s = list(itoh)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
def cli_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_path', type=str, required=True, help='A CSV file containing all stats data')
    parser.add_argument('--drop_directory', type=str, required=True, help='A directory where results will be saved')
    #parser.add_argument('--model_simulation', action='store_bool', required=False, help='Determine whether to run simulation to find best fit model from fixed effects')
    args = parser.parse_args()
    return args

def main(arguments):
    # Load in dataframe
    df = pd.read_csv(arguments.data_file_path)

    # Create mixed models
    all_model_obj = mixedmodels(drop_directory=arguments.drop_directory, dataframe=df, model_type='lmm', 
                                fixed_effects_formula = "auc_avg ~ group * day * trialtype * period", 
                                random_effects='suid',nested_effects='neuid', multicompare_correction = 'fdr_bh',
                                verbose=True)
    all_model_obj()

if __name__=='__main__':
    args = cli_parser()
    main(arguments=args)
    ipdb.set_trace()