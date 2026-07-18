#!/bin/bash
# submit_full_glm_pipeline.sh -- Orchestrates the full decoder/encoder GLM comparison pipeline on
# SLURM: prep (decoder + data curation) -> encoder array (one task per neuron chunk) -> summarize +
# compare. Mirrors the existing CAA.sh / parallel_CAA.sh submission pattern in this folder.
#
# Usage:
#   ./submit_full_glm_pipeline.sh <data_directory> <drop_directory> [chunk_size] [repo_root] [conda_env]
#
# Example:
#   ./submit_full_glm_pipeline.sh /athena/listonlab/scratch/dje4001/.../tmt_experiment_2024_working_file \
#       /athena/listonlab/scratch/dje4001/.../tmt_experiment_2024_working_file_GLM_drop 100
#
# IMPORTANT -- verify before running on a real allocation:
#   - Check your cluster's MaxArraySize and per-user/QOS core limits (sacctmgr show qos, or ask your
#     admin) before submitting -- a large neuron count / small chunk_size can produce a very large
#     array. Increase chunk_size (and glm_encoder_array.sh's --cpus-per-task/--time together) if your
#     cluster caps array size.
#   - Confirm the conda env name (default "sweet2p") and repo_root path match your cluster setup --
#     these scripts were written from local benchmarking and have NOT been tested on the actual
#     cluster yet.
#   - Each glm_encoder_array.sh task was benchmarked locally at ~5 min/neuron single-threaded (500
#     circular-lag permutations + 1 real GLM fit per neuron); tune --time accordingly once you've
#     confirmed per-neuron timing on the cluster's hardware.
set -e

data_directory=$1
drop_directory=$2
chunk_size=${3:-100}
repo_root=${4:-"/home/dje4001/NeuroSweet"}
conda_env=${5:-"sweet2p"}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$data_directory" ] || [ -z "$drop_directory" ]; then
    echo "Usage: $0 <data_directory> <drop_directory> [chunk_size] [repo_root] [conda_env]"
    exit 1
fi

mkdir -p "$drop_directory"

echo "=== Step 1/3: glm_prep.sh (decoder + data curation), blocking until complete ==="
sbatch --wait "$script_dir/glm_prep.sh" "$data_directory" "$drop_directory" "$repo_root" "$conda_env"

manifest="$drop_directory/glm_manifest.json"
if [ ! -f "$manifest" ]; then
    echo "ERROR: $manifest not found -- glm_prep.sh likely failed. Check glm_prep_*.err/.out." >&2
    exit 1
fi

n_neurons=$(grep -oE '"n_neurons":[[:space:]]*[0-9]+' "$manifest" | grep -oE '[0-9]+')
if [ -z "$n_neurons" ]; then
    echo "ERROR: could not parse n_neurons from $manifest" >&2
    exit 1
fi
echo "Total neurons to fit: $n_neurons"

n_array_tasks=$(( (n_neurons + chunk_size - 1) / chunk_size ))
last_task_idx=$(( n_array_tasks - 1 ))
echo "=== Step 2/3: glm_encoder_array.sh, array=0-$last_task_idx (chunk_size=$chunk_size) ==="

array_jobid=$(sbatch --parsable --array=0-"$last_task_idx" "$script_dir/glm_encoder_array.sh" \
    "$drop_directory" "$n_neurons" "$chunk_size" "$repo_root" "$conda_env")
echo "Submitted encoder array job: $array_jobid"

echo "=== Step 3/3: glm_summarize_compare.sh, queued with --dependency=afterok:$array_jobid ==="
compare_jobid=$(sbatch --parsable --dependency=afterok:"$array_jobid" "$script_dir/glm_summarize_compare.sh" \
    "$drop_directory" "$repo_root" "$conda_env")
echo "Submitted summarize+compare job: $compare_jobid"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Final comparison outputs (once $compare_jobid finishes) will be in: $drop_directory/comparison/"
