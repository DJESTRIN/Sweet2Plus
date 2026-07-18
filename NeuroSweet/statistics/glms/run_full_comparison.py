#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_comparison.py -- Orchestrates a full, throttled run of both GLM pipelines
(circuit_regression decoder + engelhardglm encoder) on real recording data, followed by
compare_decoder_encoder.py, while deliberately avoiding monopolizing the machine's CPU.

CPU-containment strategy (see also the printed summary at the top of the log):
  1. `--n_jobs` caps the number of parallel worker processes joblib spawns for both models
     (default: a conservative quarter of logical cores, min 2) instead of the code's previous
     hardcoded n_jobs=-1 (all cores).
  2. BLAS/OpenMP thread-pool env vars (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
     NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS) are pinned to 1 *before numpy is imported*,
     so numpy/scipy/sklearn/statsmodels don't each additionally fan out to all cores inside every
     one of the n_jobs worker processes (a common source of CPU oversubscription that n_jobs alone
     does not prevent).
  3. The whole script is meant to be launched at BelowNormal OS process priority (see the
     accompanying PowerShell launcher), so interactive/foreground work on the machine is not
     starved even while this runs.

Bypasses the heavy suite2p/torch/optuna import chain pulled in by NeuroSweet.core.SaveLoadObjs
(only needed for raw-imaging-pipeline steps we don't use here) by stubbing that module and
reading the same obj*.json schema directly (see `lightweight_gather_data` below).
"""
import os

# Must happen before numpy/sklearn/statsmodels get imported anywhere in the process.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import glob
import json
import argparse
import time
import numpy as np
import pandas as pd

REPO_ROOT = r"C:\Users\listo\Sweet2Plus"
sys.path.insert(0, REPO_ROOT)

# Note: NeuroSweet.core.SaveLoadObjs (and the suite2p/torch/optuna import chain it pulls in via
# NeuroSweet.core.core) is imported lazily/optionally inside circuit_coefficient_clustering.py and
# engelhardglm.py, so importing those modules here -- and re-importing them fresh inside joblib's
# loky subprocess workers -- does not require those unrelated raw-imaging dependencies to be installed.
from NeuroSweet.statistics.circuit_coefficient_clustering import circuit_regression
from NeuroSweet.statistics.glms.engelhardglm import currate_data, engelhardglm
from NeuroSweet.statistics.glms.glmsummary import collect
from NeuroSweet.statistics.glms.compare_decoder_encoder import load_and_align, compare, plot_comparison, write_report


def lightweight_gather_data(parent_data_directory, file_indicator="obj"):
    """Reimplements NeuroSweet.core.SaveLoadObjs.gather_data's public contract (same return shape)
    by reading the obj*.json files' documented schema directly, without constructing the full
    corralative_activity object (and therefore without needing suite2p/torch/optuna installed).
    Schema indices per NeuroSweet/core/SaveLoadObjs.py's SaveObj/LoadObj: 9=ztraces,
    11=all_evts_imagetime, 19=day, 20=cage, 21=mouse, 22=group."""
    objfiles = glob.glob(os.path.join(parent_data_directory, f"**/{file_indicator}*.json"), recursive=True)
    print(f"Found {len(objfiles)} object files under {parent_data_directory}")

    neuronal_activity, behavioral_timestamps, neuron_info_frames = [], [], []
    for i, file in enumerate(objfiles):
        with open(file, "r") as f:
            big_list = json.load(f)
        ztraces = np.asarray(big_list[9])
        all_evts_imagetime = big_list[11]
        day, cage, mouse, group = big_list[19], big_list[20], big_list[21], big_list[22]

        neuronal_activity.append(ztraces)
        behavioral_timestamps.append(all_evts_imagetime)
        repeated_info = np.tile([day, cage, mouse, group], ztraces.shape[0]).reshape(ztraces.shape[0], 4)
        neuron_info_frames.append(pd.DataFrame(repeated_info, columns=["day", "cage", "mouse", "group"]))

        if (i + 1) % 10 == 0 or (i + 1) == len(objfiles):
            print(f"  loaded {i + 1}/{len(objfiles)}: {os.path.basename(file)} ({ztraces.shape[0]} neurons)")

    neuron_info = pd.concat(neuron_info_frames, ignore_index=True) if neuron_info_frames else \
        pd.DataFrame(columns=["day", "cage", "mouse", "group"])
    return neuronal_activity, behavioral_timestamps, neuron_info


def cli_parser():
    cpu_count = os.cpu_count() or 4
    default_n_jobs = max(2, cpu_count // 4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--drop_directory", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=default_n_jobs,
                         help=f"Max parallel workers per model (default {default_n_jobs} of {cpu_count} logical cores)")
    parser.add_argument("--number_bases_spline", type=int, default=50)
    parser.add_argument("--spline_duration", type=int, default=483)
    return parser.parse_args()


def main():
    args = cli_parser()
    os.makedirs(args.drop_directory, exist_ok=True)
    print(f"n_jobs={args.n_jobs} (of {os.cpu_count()} logical cores); "
          f"BLAS threads pinned to {os.environ.get('OMP_NUM_THREADS')}")

    t0 = time.time()
    neuronal_activity, behavioral_timestamps, neuron_info = lightweight_gather_data(args.data_directory)
    print(f"Data loaded: {len(neuronal_activity)} recordings, {len(neuron_info)} total neurons "
          f"({time.time() - t0:.1f}s)")

    # ---- Decoder: circuit_regression ----
    t0 = time.time()
    regressobj = circuit_regression(
        drop_directory=args.drop_directory,
        neuronal_activity=[a.copy() for a in neuronal_activity],
        behavioral_timestamps=behavioral_timestamps,
        neuron_info=neuron_info.copy(),
        normalize_neural_activity=False,
    )
    regressobj.normalize_activity()
    regressobj.timestamps_to_one_hot_array()
    regressobj.run_glm(coef_file="allcoefs.csv", model_file="allmodels.csv", n_jobs=args.n_jobs)
    decoder_csv = os.path.join(args.drop_directory, "beta_filtered.csv")
    print(f"Decoder done ({time.time() - t0:.1f}s) -> {decoder_csv}")

    # ---- Encoder: engelhardglm + glmsummary ----
    t0 = time.time()
    dataset = currate_data([a.copy() for a in neuronal_activity], behavioral_timestamps, neuron_info.copy(), args.drop_directory)
    dataset()
    dataset.save()

    glmobj = engelhardglm(
        activity=dataset.trans_act,
        timestamps=dataset.trans_ts,
        info=dataset.trans_info,
        dropdir=args.drop_directory,
        graphics=False,
        number_bases_spline=args.number_bases_spline,
        spline_duration=args.spline_duration,
        local_neuron_id=dataset.trans_local_id,
    )
    glmobj(n_jobs=args.n_jobs)
    temp_dir = os.path.join(args.drop_directory, "temp")
    print(f"Encoder per-neuron fits done ({time.time() - t0:.1f}s) -> {temp_dir}")

    t0 = time.time()
    collector = collect(input_path=temp_dir, number_events=4, number_bases_spline=args.number_bases_spline)
    collector.load_results()
    encoder_csv = os.path.join(args.drop_directory, "engelhard_stimulus_summary.csv")
    summary_df = collector.save_stimulus_summary(output_path=encoder_csv)
    print(f"Encoder summary built ({time.time() - t0:.1f}s): {len(summary_df)} rows -> {encoder_csv}")

    # ---- Comparison ----
    decoder_df, encoder_df, merged = load_and_align(decoder_csv, encoder_csv)
    print(f"Merged decoder/encoder rows: {len(merged)}")
    if len(merged) == 0:
        print("WARNING: no overlap between decoder and encoder nuid/stimulus -- skipping comparison.")
        return

    results = compare(merged)
    merged.to_csv(os.path.join(args.drop_directory, "decoder_encoder_merged.csv"), index=False)
    results["summary"].to_csv(os.path.join(args.drop_directory, "decoder_encoder_summary.csv"), index=False)
    plot_comparison(merged, args.drop_directory)
    report_path = write_report(results["summary"], args.drop_directory)
    print(f"Comparison written to {report_path}")
    print("\nALL DONE")


if __name__ == "__main__":
    main()
