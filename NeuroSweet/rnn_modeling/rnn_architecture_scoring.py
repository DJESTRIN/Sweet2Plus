#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: rnn_architecture_scoring.py
Description: Defines the quantitative metric used to select the "winning" external-input
    architecture (unmixed / semi-mixed / fully-mixed) per plan.md step 3 -- i.e., which
    architecture's control-fit-vs-cort-fit divergence in RNN hidden-unit decoder/encoder
    statistics best reproduces the REAL control-vs-cort divergence documented in the original
    report (report.tex Findings 1, 2, 5).

Real-data reference targets (hardcoded from report.tex, NOT re-derived here -- see
report.tex for the underlying analysis):
  - Finding 1/2 direction: control mice accumulate MORE multi-odor (k>=2) coding neurons over
    days; cort mice accumulate MORE zero-odor (k==0) neurons instead. We operationalize this
    as: growth in P(k_decoder >= 2) should be POSITIVE for control and near-zero/negative for
    cort, across the fitted day range.
  - Finding 5 direction: noise correlation among odor-tuned neurons grows over days in control
    (~1.44x day-0 baseline) but not cort (~0.94x). We approximate this with the growth in
    mean pairwise correlation among encoder-tuned (k_encoder>0) hidden units.
  - Finding 1/2 weak-agreement magnitude: |Spearman rho| between decoder importance and
    encoder tuning strength should stay small (report: |r| <= 0.065), a sanity check that the
    RNN isn't trivially collapsing decoder and encoder into the same statistic.

For a given architecture, after fitting RNNs separately to control sessions and cort sessions
(across days, per plan.md step 3), this module computes a single scalar "match score" (lower
is better -- it is a distance from the ideal real-data-matching signature) combining:
  1. sign/magnitude match of P(k_decoder>=2) day-trend (control positive, cort ~flat/negative)
  2. sign/magnitude match of tuned-unit noise-correlation day-trend (control grows, cort flat)
  3. a penalty if decoder/encoder agreement |rho| is NOT small (>0.3), since that would
     indicate the RNN doesn't reproduce the real weak-agreement structure at all and any
     apparent trend match would not be meaningful.

This is intentionally a simple, transparent, inspectable scoring function (not a fitted model
itself) so the choice of "winning architecture" can be justified in plain language, consistent
with the earlier feedback that overly fancy framing of simple procedures should be avoided.
Author: David Estrin (GitHub Copilot CLI assisted)
Version: 1.0
"""
import numpy as np
import pandas as pd

# Real-data reference signs (from report.tex); see module docstring.
REAL_CONTROL_MULTIODOR_TREND_SIGN = +1   # control: P(k>=2) increases with day
REAL_CORT_MULTIODOR_TREND_SIGN = 0       # cort: ~flat or decreasing (treated as <= 0 target)
REAL_CONTROL_NOISECORR_TREND_SIGN = +1   # control: tuned-neuron noise corr grows (~1.44x day0)
REAL_CORT_NOISECORR_TREND_SIGN = 0       # cort: ~flat (~0.94x day0, i.e. no growth)
AGREEMENT_MAGNITUDE_CEILING = 0.3        # if |rho| exceeds this, penalize (real data: <=0.065)


def _day_trend_sign_and_slope(days, values):
    """OLS slope of `values` vs `days`; returns (sign, slope). Requires >=2 distinct days."""
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(np.unique(days)) < 2 or np.any(np.isnan(values)):
        return 0.0, 0.0
    slope, _intercept = np.polyfit(days, values, 1)
    return float(np.sign(slope)), float(slope)


def score_architecture(session_summaries):
    """
    session_summaries : list of dicts, one per (architecture, group, day, seed) RNN fit, each
        with keys: 'day' (int), 'group' ('control'/'cort'), 'p_k_decoder_ge2' (float in [0,1],
        fraction of hidden units with k_decoder>=2 that session/seed), 'tuned_noise_corr'
        (float, mean pairwise correlation among k_encoder>0 units that session/seed),
        'agreement_rho' (float, decoder/encoder Spearman rho that session/seed).
        Should already be averaged/aggregated across seeds per (group, day) before calling,
        OR pass all seeds and this function will average internally by (group, day).

    Returns dict: {'match_score': float (lower=better), 'components': {...}} -- see module
    docstring for how the three components are combined (unweighted sum of absolute
    deviations from the ideal real-data-matching signs/magnitudes).
    """
    df = pd.DataFrame(session_summaries)
    if df.empty:
        return {"match_score": np.inf, "components": {}, "note": "no session summaries provided"}

    agg = df.groupby(["group", "day"], as_index=False).agg(
        p_k_decoder_ge2=("p_k_decoder_ge2", "mean"),
        tuned_noise_corr=("tuned_noise_corr", "mean"),
        agreement_rho=("agreement_rho", "mean"),
    )

    components = {}

    for group, real_multiodor_sign, real_noisecorr_sign in [
        ("control", REAL_CONTROL_MULTIODOR_TREND_SIGN, REAL_CONTROL_NOISECORR_TREND_SIGN),
        ("cort", REAL_CORT_MULTIODOR_TREND_SIGN, REAL_CORT_NOISECORR_TREND_SIGN),
    ]:
        sub = agg[agg["group"] == group].sort_values("day")
        if len(sub) < 2:
            components[f"{group}_multiodor_trend_penalty"] = 1.0  # can't evaluate -> max penalty
            components[f"{group}_noisecorr_trend_penalty"] = 1.0
            continue

        m_sign, m_slope = _day_trend_sign_and_slope(sub["day"], sub["p_k_decoder_ge2"])
        n_sign, n_slope = _day_trend_sign_and_slope(sub["day"], sub["tuned_noise_corr"])

        if real_multiodor_sign > 0:
            # control target: positive trend. Penalty = 0 if positive, else |slope| scaled.
            components[f"{group}_multiodor_trend_penalty"] = 0.0 if m_sign > 0 else abs(m_slope) + 0.5
        else:
            # cort target: <=0 trend (flat or negative is fine; positive growth is the mismatch)
            components[f"{group}_multiodor_trend_penalty"] = max(0.0, m_slope) if m_sign > 0 else 0.0

        if real_noisecorr_sign > 0:
            components[f"{group}_noisecorr_trend_penalty"] = 0.0 if n_sign > 0 else abs(n_slope) + 0.5
        else:
            components[f"{group}_noisecorr_trend_penalty"] = max(0.0, n_slope) if n_sign > 0 else 0.0

    mean_abs_rho = df["agreement_rho"].abs().mean()
    components["agreement_magnitude_penalty"] = max(0.0, mean_abs_rho - AGREEMENT_MAGNITUDE_CEILING)

    match_score = sum(components.values())
    return {"match_score": float(match_score), "components": components}


def select_winning_architecture(architecture_scores):
    """architecture_scores: dict {architecture_name: score_architecture(...) output}.
    Returns the architecture name with the lowest 'match_score' (best real-data match)."""
    ranked = sorted(architecture_scores.items(), key=lambda kv: kv[1]["match_score"])
    return ranked[0][0], ranked
