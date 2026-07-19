#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: decoder_subpopulations.py
Description: Decoder-model counterpart to engelhard_subpopulations.py -- buckets neurons into
    weight-based subpopulations (pos_sig/neg_sig/non_sig) from circuit_regression's decoder
    weights and plots the normalized proportion of neurons in each subpopulation across
    day/session, faceted by group.

    Uses the exact same categorize_weight()/distribution_of_neurons_in_subpopulations() logic
    as engelhard_subpopulations.py (imported from that module) so the decoder and encoder
    proportion-over-time curves are built identically and can be compared side by side --
    matching the categorize_beta() scheme already used inline in
    circuit_coefficient_clustering.py's beta_weight_validation(), just turned into its own
    reusable proportion-over-time analysis (that plot didn't previously exist for the decoder;
    the repo's only prior "proportion of neurons over time" plot,
    coefficient_clustering.py's distribution_of_neurons_in_clusters(), used an unrelated
    KMeans-clustering scheme over ridge/OLS coefficients, not the decoder's actual logistic
    regression weights).

Input: beta_filtered.csv, produced by circuit_regression (circuit_coefficient_clustering.py)
    (columns include: nuid, suid, group, day, Behavior_num, Mean_Beta, CI_lower, CI_upper,
     t_value, p_value, sig). Behavior_num is mapped to stimulus names using the same
     canonical odor order as glmsummary.py's DEFAULT_STIMULUS_NAMES (0=vanilla, 1=peanut,
     2=water, 3=TMT) so decoder and encoder stimulus labels line up.

Author: David Estrin
Version: 1.0
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engelhard_subpopulations import distribution_of_neurons_in_subpopulations

# Canonical odor order established in NeuroSweet/core/behavior.py (quick_timestamps' `trials`
# list) and mirrored in glmsummary.py's DEFAULT_STIMULUS_NAMES: index 0=Vanilla, 1=PeanutButter,
# 2=Water, 3=FoxUrine(TMT).
DEFAULT_BEHAVIOR_MAP = {0: 'vanilla', 1: 'peanut', 2: 'water', 3: 'TMT'}


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--decoder_csv', type=str, required=True,
                         help='Path to circuit_regression output (beta_filtered.csv)')
    parser.add_argument('--output_dir', type=str, default='.',
                         help='Directory to write subpopulation CSVs/plots into')
    parser.add_argument('--weight_col', type=str, default='Mean_Beta',
                         help="Column holding the signed decoder weight (default: Mean_Beta)")
    parser.add_argument('--sig_col', type=str, default='sig',
                         help="Column holding the significance flag (default: sig)")
    parser.add_argument('--subject_col', type=str, default='suid',
                         help="Column holding the per-animal subject id (default: suid)")
    args = parser.parse_args()
    return args


def proc():
    args = cli_parser()
    df = pd.read_csv(args.decoder_csv)

    if 'stimulus' not in df.columns:
        if 'Behavior_num' not in df.columns:
            raise ValueError("decoder_csv must have either a 'stimulus' column or a "
                              "'Behavior_num' column to map to stimulus names.")
        df['stimulus'] = df['Behavior_num'].map(DEFAULT_BEHAVIOR_MAP)

    counts, plot_data = distribution_of_neurons_in_subpopulations(
        df, weight_col=args.weight_col, sig_col=args.sig_col, output_dir=args.output_dir,
        subject_col=args.subject_col, label_prefix='decoder', model_label='Decoder')
    print(f"Wrote {len(counts)} subpopulation count rows and "
          f"{plot_data['stimulus'].nunique()} stimulus plots to {args.output_dir}")


if __name__ == '__main__':
    proc()
