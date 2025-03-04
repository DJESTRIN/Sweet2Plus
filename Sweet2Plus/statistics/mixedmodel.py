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
from itertools import chain, combinations, product
import ipdb
from statsmodels.stats.anova import anova_lm


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

        self.dataframe['auc'] = self.dataframe['auc'].astype('float16')

        assert self.model_type=='lmm'

    def __call__(self):
        """ General statistical protocol """
        self.generate_model()
        self.multiple_comparisons()
        self.residual_evaluation()

    def generate_model(self):
        """ Build and run LinearMixed model based on attributes """
        ipdb.set_trace()
        print(f'Fitting full model with formula:{self.formula}')
        self.full_model = smf.mixedlm(self.formula,
                                 self.dataframe,
                                 groups=self.dataframe[self.random_effects], 
                                 re_formula="1",
                                 vc_formula={self.nested_effects: "1"})
        self.full_model_result = self.full_model.fit()

        self.formula_reduced = 'auc~group+day+trialtype+period'
        print(f'Fitting reduced model with formula: {self.formula_reduced}')
        self.reduced_model = smf.mixedlm(self.formula_reduced,
                                 self.dataframe,
                                 groups=self.dataframe[self.random_effects], 
                                 re_formula="1",
                                 vc_formula={self.nested_effects: "1"})
        self.full_reduced_result = self.full_model.fit()

        # Calculate LRT
        lrt_stat_oh = 2 * (self.full_model_result.llf - self.full_reduced_result.llf) 
        df_diff_oh = (len(self.full_model_result.params) - len(self.full_reduced_result.params))
        p_value = stats.chisquare([lrt_stat_oh], df_diff_oh)[1]
        print(f"LRT Statistic: {lrt_stat_oh}, p-value: {p_value}")
        ipdb.set_trace()

        # Perform Type III ANOVA for F-values and p-values
        # anova_results = anova_lm(result, typ=3 )
        # self.effect_p_values = self.model_results.pvalues
        # ipdb.set_trace()
        # if self.verbose:
        #     print(self.model_results)
        #     print(self.effect_p_values)

    def multiple_comparisons(self):
        """ Run multiple comparisons on significant interactions and/or main effects """
        # Get p values
        rejected, pvals_corrected, _, _ = multipletests(self.effect_p_values, method=self.multicompare_correction)

        # Convert p value results to table and save to csv
        self.multiple_comparisons_results = pd.DataFrame({ "Effect": self.effect_p_values.index, "p_value": self.effect_p_values.values,
                                                          "FDR_corrected_p": pvals_corrected, "Significant (FDR < 0.05)": rejected})
        self.multiple_comparisons_results.to_csv(os.path.join(self.drop_directory,"multiple_comparison_results.csv"))

    def residual_evaluation(self):
        """ Generate common plots and stats for residuals to manaully evaluate model fit """
        # Pull residuals and fitted values
        residuals = self.model.resid_deviance
        fit_vals = self.model.fittedvalues

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
        sm.qqplot(residuals, line='45', fit=True)
        plt.title("Q-Q Plot of Residuals")
        plt.savefig(os.path.join(self.drop_directory,"ResidualQQplot_Check.jpg"))

        # Predictor vs residual plot
        sns.scatterplot(x=X["some_predictor"], y=residuals, alpha=0.6)
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

# class compare_models():
#     def __init__(self, drop_directory, dataframe, dependent_variable, fixed_effects, random_effects, nested_effects,incorporate_random_slopes=False):
#         self.drop_directory = drop_directory
#         self.dataframe = dataframe
#         self.dependent_variable = dependent_variable
#         self.fixed_effects = fixed_effects
#         self.random_effects = random_effects
#         self.nested_effects = nested_effects
#         self.incorporate_random_slopes = incorporate_random_slopes
    
#     def __call__(self):
#         self.all_models = self.get_all_models()
        
#         all_aic_data = []
#         for model_oh in self.all_models:
#             if model_oh['groups'] is  None or model_oh['vc_formula'] is None:
#                 continue
#             else:
#                 try:
#                     current_model_oh = smf.mixedlm(model_oh['formula'], self.dataframe, groups=model_oh['groups'], 
#                                             re_formula="1", vc_formula=model_oh['vc_formula'])
#                     result_oh = current_model_oh.fit()
#                     AIC_value = result_oh.aic
#                 except:
#                     AIC_value = np.nan

#                 all_aic_data.append([model_oh, AIC_value])
        
#         ipdb.set_trace()

#     def get_all_models(self):
#         model_specifications = []
#         fixed_effect_combos = list(self.powerset(self.fixed_effects))[1:] 
#         random_effect_combos = list(self.powerset(self.random_effects))

#         nested_combos = []
#         for re_combo in random_effect_combos:
#             nested_structure = {}
#             for re in re_combo:
#                 if re in nested_effects:
#                     for nested in self.powerset(self.nested_effects[re]):
#                         if nested: 
#                             nested_structure[re] = "1"
#             nested_combos.append(nested_structure)

#         for fixed_combo, re_combo, nested_combo in product(fixed_effect_combos, random_effect_combos, nested_combos):
#             fixed_part = " + ".join(fixed_combo)
#             groups = re_combo[0] if re_combo else None
#             random_parts = [f"(1|{re})" for re in re_combo] 
#             random_part = " + ".join(random_parts) if random_parts else ""

#             if random_part:
#                 formula = f"{self.dependent_variable} ~ {fixed_part} + {random_part}"
#             else:
#                 formula = f"{self.dependent_variable} ~ {fixed_part}"

#             model_specifications.append({
#                 "formula": formula,
#                 "groups": groups,
#                 "vc_formula": nested_combo if nested_combo else None})

#         return model_specifications

#     def powerset(self, itoh):
#         s = list(itoh)
#         return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    
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
                                fixed_effects_formula = "auc ~ group*day*trialtype*period", 
                                random_effects='suid',nested_effects='neuid', multicompare_correction = 'fdr_bh',
                                verbose=True)
    all_model_obj()

if __name__=='__main__':
    args = cli_parser()
    main(arguments=args)
    ipdb.set_trace()