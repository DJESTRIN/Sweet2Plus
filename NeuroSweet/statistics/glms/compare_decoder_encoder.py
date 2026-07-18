#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: compare_decoder_encoder.py
Description: Compares per-neuron, per-stimulus weights produced by two different modeling
    approaches used elsewhere in this repo:

    (1) circuit_regression (NeuroSweet/statistics/circuit_coefficient_clustering.py) -- a
        population DECODER. For each recording, a bootstrapped elastic-net LogisticRegression
        predicts which stimulus occurred from the joint activity of all recorded neurons. Each
        neuron gets a Mean_Beta (+ CI/p-value) per stimulus reflecting its *relative* contribution
        to a shared, regularized, multivariate classifier (correlated neurons compete for weight).

    (2) engelhardglm (NeuroSweet/statistics/glms/engelhardglm.py + glmsummary.py) -- a per-neuron
        ENCODER. Each neuron's activity is fit independently with a GLM against B-spline-expanded
        stimulus-onset kernels (following Engelhard et al., 2019). glmsummary.collect.generate_stimulus_summary
        collapses the spline-basis betas into one summary weight (+ permutation p-value) per neuron
        per stimulus, entirely independent of other neurons.

    Because the two weights are on different scales and have different statistical meaning (relative/
    competitive vs. independent/marginal), we compare them primarily via rank correlation, significance
    agreement and sign/direction concordance -- not raw magnitude equality.

Author: David James Estrin
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Canonical odor order (see NeuroSweet/core/behavior.py's `trials` list and
# circuit_coefficient_clustering.py's behavior_map): 0=Vanilla, 1=PeanutButter, 2=Water, 3=FoxUrine(TMT).
DEFAULT_BEHAVIOR_MAP = {0: 'vanilla', 1: 'peanut', 2: 'water', 3: 'TMT'}


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder_csv', type=str, required=True,
                         help='Path to circuit_regression beta_filtered.csv (population decoder output)')
    parser.add_argument('--encoder_csv', type=str, required=True,
                         help='Path to glmsummary engelhard_stimulus_summary.csv (per-neuron encoder output)')
    parser.add_argument('--output_dir', type=str, default='.',
                         help='Where the merged comparison CSV, report, and plots are saved')
    args = parser.parse_args()
    return args.decoder_csv, args.encoder_csv, args.output_dir


def load_and_align(decoder_csv, encoder_csv, behavior_map=None):
    """Load both CSVs and align them onto a common nuid + stimulus key."""
    behavior_map = behavior_map or DEFAULT_BEHAVIOR_MAP

    decoder_df = pd.read_csv(decoder_csv)
    encoder_df = pd.read_csv(encoder_csv)

    decoder_df = decoder_df.copy()
    decoder_df['stimulus'] = decoder_df['Behavior_num'].map(behavior_map)
    decoder_df['nuid'] = decoder_df['nuid'].astype(str)

    encoder_df = encoder_df.copy()
    encoder_df['nuid'] = encoder_df['nuid'].astype(str)

    merged = decoder_df.merge(
        encoder_df,
        on=['nuid', 'stimulus'],
        how='inner',
        suffixes=('_decoder', '_encoder'),
    )
    return decoder_df, encoder_df, merged


def compare(merged):
    """Compute rank correlation, significance agreement, and sign agreement, overall and per stimulus."""
    results = {}

    def _kappa(a, b):
        """Cohen's kappa for two binary (0/1) arrays; avoids adding a new dependency (sklearn is already
        used elsewhere in this repo, but we keep this self-contained for a lightweight comparison script)."""
        a = np.asarray(a).astype(int)
        b = np.asarray(b).astype(int)
        n = len(a)
        if n == 0:
            return np.nan
        po = np.mean(a == b)
        p_a1 = np.mean(a)
        p_b1 = np.mean(b)
        pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
        if pe == 1:
            return 1.0
        return (po - pe) / (1 - pe)

    def _summarize(df, label):
        n = len(df)
        if n == 0:
            return {
                'stimulus': label, 'n_neurons': 0, 'spearman_r': np.nan, 'spearman_p': np.nan,
                'kappa_sig': np.nan, 'pct_sig_both': np.nan, 'pct_sig_decoder_only': np.nan,
                'pct_sig_encoder_only': np.nan, 'pct_sig_neither': np.nan, 'sign_agreement_pct': np.nan,
            }

        rho, pval = spearmanr(df['Mean_Beta'].abs(), df['weight'])

        dec_sig = df['sig_decoder'].astype(int)
        enc_sig = df['sig_encoder'].astype(int)
        both = np.mean((dec_sig == 1) & (enc_sig == 1)) * 100
        dec_only = np.mean((dec_sig == 1) & (enc_sig == 0)) * 100
        enc_only = np.mean((dec_sig == 0) & (enc_sig == 1)) * 100
        neither = np.mean((dec_sig == 0) & (enc_sig == 0)) * 100
        kappa = _kappa(dec_sig, enc_sig)

        # Sign/direction agreement: only meaningful where both models flag a real (non-zero) effect.
        both_nonzero = df[(df['Mean_Beta'] != 0) & (df['signed_weight'] != 0)]
        if len(both_nonzero) > 0:
            sign_agree = np.mean(np.sign(both_nonzero['Mean_Beta']) == np.sign(both_nonzero['signed_weight'])) * 100
        else:
            sign_agree = np.nan

        return {
            'stimulus': label, 'n_neurons': n, 'spearman_r': rho, 'spearman_p': pval,
            'kappa_sig': kappa, 'pct_sig_both': both, 'pct_sig_decoder_only': dec_only,
            'pct_sig_encoder_only': enc_only, 'pct_sig_neither': neither, 'sign_agreement_pct': sign_agree,
        }

    summary_rows = [_summarize(merged, 'ALL')]
    for stim in sorted(merged['stimulus'].dropna().unique()):
        summary_rows.append(_summarize(merged[merged['stimulus'] == stim], stim))

    results['summary'] = pd.DataFrame(summary_rows)
    return results


def plot_comparison(merged, output_dir):
    stimuli = sorted(merged['stimulus'].dropna().unique())
    n = len(stimuli)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for ax, stim in zip(axes[0], stimuli):
        sub = merged[merged['stimulus'] == stim]
        both_sig = (sub['sig_decoder'] == 1) & (sub['sig_encoder'] == 1)
        ax.scatter(sub.loc[~both_sig, 'Mean_Beta'].abs(), sub.loc[~both_sig, 'weight'],
                   alpha=0.4, color='gray', label='Not both sig.')
        ax.scatter(sub.loc[both_sig, 'Mean_Beta'].abs(), sub.loc[both_sig, 'weight'],
                   alpha=0.8, color='crimson', label='Both sig.')
        ax.set_xlabel('|Decoder Mean_Beta|')
        ax.set_ylabel('Encoder weight (max |beta|)')
        ax.set_title(stim)
        ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'decoder_vs_encoder_scatter.png'), dpi=200)
    plt.close()


def write_report(summary_df, output_dir):
    lines = ["Decoder (circuit_regression) vs Encoder (engelhardglm) neuron/stimulus weight comparison", "=" * 90]
    for _, row in summary_df.iterrows():
        lines.append(
            f"\nStimulus: {row['stimulus']} (n={int(row['n_neurons'])})\n"
            f"  Spearman rank correlation (|decoder beta| vs encoder weight): "
            f"r={row['spearman_r']:.3f}, p={row['spearman_p']:.3g}\n" if row['n_neurons'] else
            f"\nStimulus: {row['stimulus']} (n=0) -- no overlapping neurons after merge.\n"
        )
        if row['n_neurons']:
            lines.append(
                f"  Significance agreement: kappa={row['kappa_sig']:.3f} | "
                f"both sig={row['pct_sig_both']:.1f}%, decoder-only={row['pct_sig_decoder_only']:.1f}%, "
                f"encoder-only={row['pct_sig_encoder_only']:.1f}%, neither={row['pct_sig_neither']:.1f}%\n"
                f"  Sign/direction agreement (where both non-zero): {row['sign_agreement_pct']:.1f}%\n"
            )
    lines.append(
        "\nInterpretation notes:\n"
        "- Low correlation/kappa does NOT necessarily mean either model is wrong: circuit_regression's\n"
        "  weights are relative/competitive (elastic-net splits credit among correlated neurons in a\n"
        "  shared population model), while engelhardglm's weights are independent per-neuron marginal\n"
        "  effects. A neuron can have a strong individual encoding effect but a small decoder weight if\n"
        "  a correlated neuron 'absorbs' credit in the joint model, and vice versa.\n"
        "- Differing temporal windows (short fixed window in circuit_regression vs. long spline-based\n"
        "  kernel in engelhardglm) and differing significance procedures (bootstrap CI/t-test vs.\n"
        "  circular-lag permutation) can also drive disagreement independent of true encoding strength."
    )
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'decoder_encoder_comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    return report_path


def main():
    decoder_csv, encoder_csv, output_dir = cli_parser()
    _, _, merged = load_and_align(decoder_csv, encoder_csv)
    if len(merged) == 0:
        raise ValueError(
            "No overlapping (nuid, stimulus) rows found between decoder and encoder outputs. "
            "Check that nuid formats match (cage_mouse_day_neuron) and that stimulus names/behavior_map align."
        )
    results = compare(merged)
    os.makedirs(output_dir, exist_ok=True)
    merged.to_csv(os.path.join(output_dir, 'decoder_encoder_merged.csv'), index=False)
    results['summary'].to_csv(os.path.join(output_dir, 'decoder_encoder_summary.csv'), index=False)
    plot_comparison(merged, output_dir)
    report_path = write_report(results['summary'], output_dir)
    print(f"Wrote comparison outputs to {output_dir} (see {report_path})")


if __name__ == '__main__':
    main()
