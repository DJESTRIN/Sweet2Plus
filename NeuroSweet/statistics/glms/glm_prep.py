#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_prep.py -- Cluster-friendly "step 1" of the decoder/encoder GLM comparison pipeline.

Meant to run as a single SLURM job (see cluster_scripts/glm_prep.sh) before the encoder's
per-neuron fits are fanned out across a SLURM array (cluster_scripts/glm_encoder_array.sh).

Does three things, all fast relative to engelhardglm's per-neuron permutation fits:
  1. Loads the raw recordings once (lightweight_gather_data, borrowed from run_full_comparison.py --
     avoids the heavy suite2p/torch/optuna import chain).
  2. Runs the population DECODER (circuit_regression) to completion and writes beta_filtered.csv.
     This is cheap (a handful of bootstrapped elastic-net fits per recording) so there is no benefit
     to parallelizing it across a SLURM array.
  3. Builds and saves the per-neuron dataset engelhardglm needs (trans_act.pkl, trans_ts.pkl,
     trans_info.pkl, trans_local_id.pkl via currate_data), and writes glm_manifest.json recording
     the total neuron count -- so the array-submission wrapper knows how many array tasks to launch
     without re-loading/re-curating the data itself.

After this job completes, run each neuron chunk with:
    python -m NeuroSweet.statistics.glms.engelhardglm --drop_directory <dropdir> --data_provided \
        --start_neuron <start> --stop_neuron <stop>
(engelhardglm's --data_provided path loads the trans_*.pkl files written here instead of re-gathering
raw data), then aggregate + compare with glmsummary.collect and compare_decoder_encoder.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import json
import argparse
import time

REPO_ROOT = os.environ.get("NEUROSWEET_REPO_ROOT", r"C:\Users\listo\Sweet2Plus")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from NeuroSweet.statistics.circuit_coefficient_clustering import circuit_regression
from NeuroSweet.statistics.glms.engelhardglm import currate_data
from NeuroSweet.statistics.glms.run_full_comparison import lightweight_gather_data
from NeuroSweet.utils.parallel_helper import get_default_n_jobs


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--drop_directory", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=None,
                         help="Parallel workers for the decoder's bootstrapped fits "
                              "(default: SLURM_CPUS_PER_TASK, or all logical cores)")
    parser.add_argument("--skip_decoder", action="store_true",
                         help="Skip the circuit_regression decoder step (e.g. to re-run just the "
                              "encoder data curation after the decoder already finished)")
    return parser.parse_args()


def main():
    args = cli_parser()
    n_jobs = args.n_jobs if args.n_jobs is not None else get_default_n_jobs()
    os.makedirs(args.drop_directory, exist_ok=True)
    print(f"n_jobs={n_jobs}; BLAS threads pinned to {os.environ.get('OMP_NUM_THREADS')}")

    t0 = time.time()
    neuronal_activity, behavioral_timestamps, neuron_info = lightweight_gather_data(args.data_directory)
    n_recordings = len(neuronal_activity)
    n_neurons = int(sum(a.shape[0] for a in neuronal_activity))
    print(f"Data loaded: {n_recordings} recordings, {n_neurons} total neurons ({time.time() - t0:.1f}s)")

    if not args.skip_decoder:
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
        regressobj.run_glm(coef_file="allcoefs.csv", model_file="allmodels.csv", n_jobs=n_jobs)
        decoder_csv = os.path.join(args.drop_directory, "beta_filtered.csv")
        print(f"Decoder done ({time.time() - t0:.1f}s) -> {decoder_csv}")
    else:
        print("Skipping decoder step (--skip_decoder)")

    t0 = time.time()
    dataset = currate_data([a.copy() for a in neuronal_activity], behavioral_timestamps,
                            neuron_info.copy(), args.drop_directory)
    dataset()
    dataset.save()
    n_curated_neurons = len(dataset.trans_act)
    print(f"Encoder dataset curated + saved ({time.time() - t0:.1f}s): {n_curated_neurons} neurons "
          f"-> {args.drop_directory}/trans_*.pkl")

    manifest = {
        "n_recordings": n_recordings,
        "n_neurons": n_curated_neurons,
        "data_directory": os.path.abspath(args.data_directory),
        "drop_directory": os.path.abspath(args.drop_directory),
    }
    manifest_path = os.path.join(args.drop_directory, "glm_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest -> {manifest_path}: {manifest}")


if __name__ == "__main__":
    main()
