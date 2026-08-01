#!/bin/bash
# submit_rnn_pipeline.sh -- Orchestrates the full in-silico RNN pipeline on SLURM: for each of the
# 3 external-input architectures (unmixed / semi-mixed / fully-mixed), submits one SLURM array job
# with one array task per session (subject x day), then a final aggregation+scoring job that
# depends on all three arrays completing. Mirrors the existing submit_full_glm_pipeline.sh pattern
# in this folder.
#
# Usage:
#   ./submit_rnn_pipeline.sh <data_directory> <drop_directory> [seeds] [epochs] [max_hidden_size] \
#       [repo_root] [conda_env]
#
# Example:
#   ./submit_rnn_pipeline.sh /athena/listonlab/scratch/dje4001/.../tmt_experiment_2024_working_file \
#       /athena/listonlab/scratch/dje4001/.../rnn_drop "0 1 2" 150 128
#
# IMPORTANT -- verify before running on a real allocation:
#   - Check your cluster's MaxArraySize / QOS core limits before submitting (this dataset currently
#     has 88 sessions -- one array task each, x3 architectures = 264 total array tasks submitted
#     across 3 jobs).
#   - Confirm the conda env name (default "pytorch") has torch/rich/pandas/scikit-learn installed
#     matching the local prototype environment (see rnn_array.sh's OMP/BLAS thread pinning).
#   - Local benchmark (single desktop, hidden_size=339, 300 epochs): ~42s train time per
#     (session, seed); the bootstrapped decoder/encoder step adds a few more CPU-minutes per
#     (session, seed) on top of that -- rnn_array.sh's --time=02:00:00 default budgets generously
#     for multiple seeds run sequentially within one array task, but re-check actual per-task
#     wall time from the first array's .out logs before relying on this for a much larger dataset.
set -e

data_directory=$1
drop_directory=$2
seeds=${3:-"0 1 2"}
epochs=${4:-150}
max_hidden_size=${5:-128}
repo_root=${6:-"/home/dje4001/NeuroSweet"}
conda_env=${7:-"pytorch"}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$data_directory" ] || [ -z "$drop_directory" ]; then
    echo "Usage: $0 <data_directory> <drop_directory> [seeds] [epochs] [max_hidden_size] [repo_root] [conda_env]"
    exit 1
fi

mkdir -p "$drop_directory"

source ~/.bashrc
conda activate "$conda_env"
export NEUROSWEET_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:$PYTHONPATH"

echo "=== Step 0/2: writing session manifest (for array sizing) ==="
manifest="$drop_directory/rnn_session_manifest.txt"
n_sessions_line=$(python -m NeuroSweet.rnn_modeling.write_session_manifest \
    --data_directory "$data_directory" --out_manifest "$manifest" | grep -oE 'n_sessions=[0-9]+')
n_sessions=$(echo "$n_sessions_line" | grep -oE '[0-9]+')
if [ -z "$n_sessions" ] || [ "$n_sessions" -lt 1 ]; then
    echo "ERROR: could not determine session count from $data_directory" >&2
    exit 1
fi
last_task_idx=$(( n_sessions - 1 ))
echo "Total sessions: $n_sessions (array=0-$last_task_idx per architecture)"

echo "=== Step 1/2: submitting one array job per architecture (unmixed / semi-mixed / fully-mixed) ==="
job_ids=()
for architecture in unmixed semi-mixed fully-mixed; do
    jobid=$(sbatch --parsable --array=0-"$last_task_idx" "$script_dir/rnn_array.sh" \
        "$data_directory" "$drop_directory" "$architecture" "$seeds" "$epochs" "$max_hidden_size" \
        "$repo_root" "$conda_env")
    echo "  submitted $architecture array job: $jobid"
    job_ids+=("$jobid")
done

dependency_str=$(IFS=,; echo "afterok:${job_ids[*]}")
dependency_str=$(echo "$dependency_str" | sed 's/,/:/g; s/afterok:/afterok:/')

echo "=== Step 2/2: submitting aggregation+scoring job, dependency=$dependency_str ==="
aggregate_jobid=$(sbatch --parsable --dependency="$dependency_str" "$script_dir/rnn_aggregate.sh" \
    "$drop_directory" "$repo_root" "$conda_env")
echo "Submitted aggregation job: $aggregate_jobid"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Final architecture comparison (once $aggregate_jobid finishes) will be in: "
echo "  $drop_directory/rnn_architecture_comparison.json"
