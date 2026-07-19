#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: engelhard_subpopulations.py
Description: Encoder-model analog of coefficient_clustering.py's
    map_clusters_to_activity.distribution_of_neurons_in_clusters() -- buckets neurons into
    weight-based subpopulations and plots the normalized proportion of neurons in each
    subpopulation across day/session, faceted by group.

    Unlike coefficient_clustering.py (which unsupervised-clusters neurons via KMeans on a
    multi-stimulus regression-coefficient profile), subpopulations here are assigned directly
    from each neuron's signed engelhardglm weight and permutation-based significance flag --
    mirroring circuit_coefficient_clustering.py's categorize_beta(), which does the same thing
    for the decoder's (logistic regression) Mean_Beta/sig columns. Using the same
    pos_sig/neg_sig/non_sig categorization scheme for both models keeps the two subpopulation
    analyses directly comparable.

Input: engelhard_stimulus_summary.csv, produced by glmsummary.py
    (columns: nuid, day, cage, mouse, group, neuronid, stimulus, stimulus_idx, weight,
     signed_weight, p_value, sig).

Author: David Estrin
Version: 1.0
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def categorize_weight(row, weight_col='signed_weight', sig_col='sig'):
    """ Bin a single neuron/stimulus row into a subpopulation from its signed weight and
    significance flag. Identical logic to circuit_coefficient_clustering.py's categorize_beta(),
    generalized to accept either the decoder's Mean_Beta/sig columns or the encoder's
    signed_weight/sig columns so both models define subpopulations the same way. """
    if row[sig_col] == 1:
        if row[weight_col] > 0:
            return 'pos_sig'
        elif row[weight_col] < 0:
            return 'neg_sig'
        else:
            return 'zero_sig'  # just in case
    return 'non_sig'


def distribution_of_neurons_in_subpopulations(df, weight_col='signed_weight', sig_col='sig',
                                               output_dir='.', subject_col=None, label_prefix='engelhard',
                                               model_label='Encoder'):
    """ Generate, per stimulus, a plot of the average normalized proportion of neurons in each
    weight-based subpopulation (pos_sig/neg_sig/non_sig) w.r.t. group and day.

    The primary purpose is to analyze how the proportion of neurons in each subpopulation
    changes as a function of day -- i.e. are there more or fewer model-defined
    stimulus-responsive neurons per group during a given session? Mirrors
    coefficient_clustering.py's distribution_of_neurons_in_clusters() normalization: for each
    subject, counts are normalized by that subject's total neuron-stimulus-day observations
    within the same stimulus (a neuron gets an independent subpopulation label per stimulus,
    since both engelhardglm and circuit_regression fit one weight per stimulus per neuron).

    subject_col -- column already holding a per-animal subject id (e.g. circuit_regression's
    'suid'). If None (the default, used for engelhardglm's per-recording 'cage'/'mouse' columns),
    subjectid is built as mouse_cage instead. label_prefix/model_label control output filenames
    and plot titles so decoder and encoder runs don't clobber each other's outputs.

    Returns (counts, plot_data) -- the per-subject-day counts/proportions and the group/day/
    subpopulation summary (mean +/- sem) used for plotting. Also writes both to CSV under
    output_dir and saves one bar plot per stimulus.
    """
    df = df.copy()
    df['subpopulation'] = df.apply(lambda row: categorize_weight(row, weight_col, sig_col), axis=1)
    if subject_col is not None:
        df['subjectid'] = df[subject_col].astype(str)
    else:
        df['subjectid'] = df['mouse'].astype(str) + "_" + df['cage'].astype(str)
    df['day'] = pd.to_numeric(df['day'], errors='coerce')

    counts = (
        df.groupby(['stimulus', 'group', 'day', 'subjectid', 'subpopulation'])
        .size()
        .reset_index(name='count')
    )
    subject_totals = counts.groupby(['stimulus', 'subjectid'])['count'].transform('sum')
    counts['proportion'] = counts['count'] / subject_totals

    plot_data = (
        counts.groupby(['stimulus', 'group', 'day', 'subpopulation'])
        .agg(mean_proportion=('proportion', 'mean'), sem_proportion=('proportion', 'sem'))
        .reset_index()
    )

    os.makedirs(output_dir, exist_ok=True)
    counts.to_csv(os.path.join(output_dir, f'{label_prefix}_subpopulation_counts.csv'), index=False)
    plot_data.to_csv(os.path.join(output_dir, f'{label_prefix}_subpopulation_proportions.csv'), index=False)

    for stim in sorted(plot_data['stimulus'].dropna().unique()):
        stim_data = plot_data[plot_data['stimulus'] == stim]

        g = sns.catplot(
            data=stim_data,
            x='day',
            y='mean_proportion',
            hue='subpopulation',
            col='group',
            kind='bar',
            errorbar=None,  # Manual error bars
            palette='Set2',
            height=5,
            aspect=1.2
        )

        g.set_axis_labels('Session', 'Normalized # Neurons')
        g.set_titles('Group: {col_name}')
        g.set(ylim=(0, None))
        g.figure.suptitle(f'{model_label} ({stim}): Distribution of Weight-Based Subpopulations by Group and Session')
        plt.legend(
            title='subpopulation',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{label_prefix}_subpopulation_distribution_{stim}.jpg'))
        plt.close()

    return counts, plot_data


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--encoder_csv', type=str, required=True,
                         help='Path to glmsummary.py output (engelhard_stimulus_summary.csv)')
    parser.add_argument('--output_dir', type=str, default='.',
                         help='Directory to write subpopulation CSVs/plots into')
    parser.add_argument('--weight_col', type=str, default='signed_weight',
                         help="Column holding the signed encoder weight (default: signed_weight)")
    parser.add_argument('--sig_col', type=str, default='sig',
                         help="Column holding the significance flag (default: sig)")
    args = parser.parse_args()
    return args


def proc():
    args = cli_parser()
    df = pd.read_csv(args.encoder_csv)
    counts, plot_data = distribution_of_neurons_in_subpopulations(
        df, weight_col=args.weight_col, sig_col=args.sig_col, output_dir=args.output_dir)
    print(f"Wrote {len(counts)} subpopulation count rows and "
          f"{plot_data['stimulus'].nunique()} stimulus plots to {args.output_dir}")


if __name__ == '__main__':
    proc()
