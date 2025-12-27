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
    def __init__(self, drop_directory, dataframe, model_type='lmm', dependent_var_name='AUC',
                 fixed_effects_formula = "Activity ~ Group * Session * Trialtype", 
                 random_effects='Subject',nested_effects='Neuron', multicompare_correction = 'fdr_bh',
                 force_categorical_cols=None, verbose=True):
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
        self.dependent_var_name = dependent_var_name
        self.formula = fixed_effects_formula
        self.random_effects = random_effects
        self.nested_effects = nested_effects
        self.multicompare_correction = multicompare_correction
        self.verbose = verbose
        self.force_categorical_cols = force_categorical_cols
        assert self.model_type=='lmm'

    def set_types(self):
        """ Set each column as cateogorical or numeric """
        if self.force_categorical_cols is None:
            self.force_categorical_cols = []

        for col in self.dataframe.columns:
            if col in self.force_categorical_cols:
                self.dataframe[col] = self.dataframe[col].astype('category')

            elif pd.api.types.is_numeric_dtype(self.dataframe[col]):
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce')
            else:
                self.dataframe[col] = self.dataframe[col].astype('category')

    def __call__(self):
        """ General statistical protocol """
        # Force dataframe columns to categorical or numeric
        self.set_types()
        self.remove_outliers_iqr()

        # Plot data distribution
        self.data_distributions()
        ipdb.set_trace()

        # Run model and get emms
        self.generate_model()
        self.residual_evaluation()
        ipdb.set_trace()

        self.model_predictions()
        self.EMM()
        self.EMM_multiple_comparisons()

    def data_distributions(self):
        plt.figure(figsize=(7,5))
        sns.histplot(self.dataframe[self.dependent_var_name], bins=100, kde=True)
        plt.axvline(0, color='red', linestyle='dashed')
        plt.xlabel(self.dependent_var_name)
        plt.ylabel("Denisty")
        plt.savefig(os.path.join(self.drop_directory,f"KernelDensity{self.dependent_var_name}.jpg"))

    def remove_outliers_iqr(self):
        """Remove outliers from a DataFrame using the IQR method."""
        Q1 = self.dataframe[self.dependent_var_name].quantile(0.25)
        Q3 = self.dataframe[self.dependent_var_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.dataframe = self.dataframe[~((self.dataframe[self.dependent_var_name] < lower_bound) | (self.dataframe[self.dependent_var_name] > upper_bound)).any(axis=1)]

    def generate_model(self):
        """ Build and run LinearMixed model based on attributes """
        print(f'Fitting full model with formula:{self.formula}')
        self.full_model = smf.mixedlm(self.formula,
                                 self.dataframe,
                                 groups=self.dataframe[self.random_effects], 
                                 re_formula="1",
                                 vc_formula={f'{self.random_effects}:{self.nested_effects}': '1'})
        
        self.full_model_result = self.full_model.fit()
        self.params = self.full_model_result.params

    def model_predictions(self):

        # Extract model coefficients
        intercept = self.full_model_result.params["Intercept"]
        group_effect = self.full_model_result.params.get("group[T.cort]", 0)
        cluster_effect = self.full_model_result.params.get("cluster[T.TMT-Responsive]", 0)
        interaction_effect = self.full_model_result.params.get("group[T.cort]:cluster[T.TMT-Responsive]", 0)
        cov_matrix = self.full_model_result.cov_params()

        conditions = [
            {"group": "control", "cluster": "Non-TMT", "Intercept": 1, "group_effect": 0, "cluster_effect": 0, "interaction": 0},
            {"group": "control", "cluster": "TMT-Responsive", "Intercept": 1, "group_effect": 0, "cluster_effect": 1, "interaction": 0},
            {"group": "cort", "cluster": "Non-TMT", "Intercept": 1, "group_effect": 1, "cluster_effect": 0, "interaction": 0},
            {"group": "cort", "cluster": "TMT-Responsive", "Intercept": 1, "group_effect": 1, "cluster_effect": 1, "interaction": 1},
        ]
        # Convert to DataFrame
        conditions_df = pd.DataFrame(conditions)

        # Compute Predicted Values
        conditions_df["predicted"] = (
            intercept +
            conditions_df["group_effect"] * group_effect +
            conditions_df["cluster_effect"] * cluster_effect +
            conditions_df["interaction"] * interaction_effect
        )

        # Compute standard error for each condition
        design_matrix = conditions_df[["Intercept", "group_effect", "cluster_effect", "interaction"]].values
        variance_estimates = np.diag(design_matrix @ cov_matrix.loc[["Intercept", "group[T.cort]", "cluster[T.TMT-Responsive]", "group[T.cort]:cluster[T.TMT-Responsive]"]] @ design_matrix.T)
        conditions_df["SE"] = np.sqrt(variance_estimates)

        # Compute confidence intervals (95% CI)
        df_resid = self.full_model_result.df_resid  # Residual degrees of freedom
        t_value = t.ppf(0.975, df_resid)  # Two-tailed critical t-value

        conditions_df["CI_lower"] = conditions_df["predicted"] - t_value * conditions_df["SE"]
        conditions_df["CI_upper"] = conditions_df["predicted"] + t_value * conditions_df["SE"]

        # Print results
        print(conditions_df[["group", "cluster", "predicted", "SE", "CI_lower", "CI_upper"]])

        # Get unique levels
        group_levels = self.dataframe['group'].unique()
        cluster_levels = self.dataframe['cluster'].unique()

        # Create all possible group-cluster combinations
        combinations = pd.DataFrame([(g, c) for g in group_levels for c in cluster_levels], columns=['group', 'cluster'])

        # Add an intercept column (needed for prediction)
        combinations['Intercept'] = 1

        # Create dummy variables but keep control over naming
        group_dummies = pd.get_dummies(combinations['group'], prefix='group', drop_first=True)
        cluster_dummies = pd.get_dummies(combinations['cluster'], prefix='cluster', drop_first=True)

        # Manually rename dummy variables to match statsmodels format
        group_dummies.columns = [f'group[T.{col}]' for col in group_dummies.columns]
        cluster_dummies.columns = [f'cluster[T.{col}]' for col in cluster_dummies.columns]

        # Merge everything
        combinations = pd.concat([combinations, group_dummies, cluster_dummies], axis=1)

        # Ensure all parameter names from model are in the dataframe
        for param in self.full_model_result.params.index:
            if param not in combinations.columns:
                combinations[param] = 0  # Set to 0 for missing terms (now it should work correctly!)

        # Compute EMMs using the fixed effects
        combinations['predicted'] = np.dot(combinations[self.full_model_result.params.index], self.full_model_result.params)

        # Debugging: Print to check if values change across conditions
        print(combinations[['group', 'cluster', 'predicted']])
        ipdb.set_trace()


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
        fit_vals = self.full_model_result.fittedvalues  
        residuals = self.full_model_result.resid  

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