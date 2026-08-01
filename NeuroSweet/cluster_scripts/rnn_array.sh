#!/bin/bash
#SBATCH --job-name=rnn_mpfc                              # Job name
#SBATCH --output=./rnn_mpfc_%A_%a.out                     # Output file name (%A=array job id, %a=task id)
#SBATCH --error=./rnn_mpfc_%A_%a.err                       # Error file name
#SBATCH --ntasks=1                                        # Number of tasks per array element
#SBATCH --cpus-per-task=4                                 # Workers: 1 for the torch RNN train loop
                                                           # (small models, CPU-bound single-thread is
                                                           # fine) + headroom for the bootstrapped
                                                           # decoder's internal BLAS calls.
#SBATCH --mem=16G                                         # Memory per array element
#SBATCH --time=02:00:00                                   # Local benchmark after training-accuracy-audit
                                                           # fixes (GRU cell, mini-batch training, weight
                                                           # decay, early stopping): ~60s per (session,
                                                           # seed) with early stopping typically halting
                                                           # well before the 400-epoch cap, + a few
                                                           # CPU-min for the bootstrapped decoder/encoder
                                                           # step per (session, seed). Budget generously
                                                           # since larger sessions (up to ~1800 real
                                                           # neurons, capped at --max_hidden_size) and
                                                           # multiple seeds run sequentially per task.

# One SLURM array task = one session (one row of the session manifest written by
# write_session_manifest.py), training all requested seeds for ONE input architecture
# (unmixed / semi-mixed / fully-mixed -- submit_rnn_pipeline.sh launches one array PER
# architecture) and appending results to that architecture's results CSV.
#
# Must run *after* write_session_manifest.py has produced the manifest so
# --manifest_row_index maps consistently to the same session across all array tasks (the
# manifest itself is just a deterministic re-listing of the same data_directory via
# rnn_data.list_sessions, so this script re-derives the row directly rather than reading the
# manifest file -- kept here only for the array-size computation in the submit wrapper).
#
# Usage (normally invoked by submit_rnn_pipeline.sh, not run directly):
#   sbatch --array=0-<N_sessions-1> rnn_array.sh <data_directory> <drop_directory> <architecture> \
#       [seeds_space_separated_in_quotes] [epochs] [max_hidden_size] [repo_root] [conda_env]

data_directory=$1
drop_directory=$2
architecture=$3
seeds=${4:-"0 1 2"}
epochs=${5:-400}
max_hidden_size=${6:-128}
repo_root=${7:-"/home/dje4001/NeuroSweet"}
conda_env=${8:-"pytorch"}

if [ -z "$data_directory" ] || [ -z "$drop_directory" ] || [ -z "$architecture" ]; then
    echo "Usage: sbatch --array=0-<N-1> rnn_array.sh <data_directory> <drop_directory> <architecture> [seeds] [epochs] [max_hidden_size] [repo_root] [conda_env]"
    exit 1
fi

source ~/.bashrc
conda activate "$conda_env"

export NEUROSWEET_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:$PYTHONPATH"
# Keep each array task single-threaded for BLAS/OMP so SLURM_CPUS_PER_TASK isn't
# oversubscribed by numpy/scipy/sklearn spawning their own thread pools underneath the
# already-parallel array (mirrors the convention in glm_prep.py/run_full_comparison.py).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Array task $SLURM_ARRAY_TASK_ID: architecture=$architecture seeds=$seeds epochs=$epochs max_hidden_size=$max_hidden_size"

# Results CSV path is per-architecture and per-task, to avoid concurrent-write races across
# array tasks writing to the same file; submit_rnn_pipeline.sh's final aggregation step
# (rnn_aggregate_and_score.py) concatenates all per-task CSVs together afterward.
results_csv="$drop_directory/rnn_batch_results_${architecture}_task${SLURM_ARRAY_TASK_ID}.csv"

python -m NeuroSweet.rnn_modeling.rnn_batch_run \
    --data_directory "$data_directory" \
    --drop_directory "$drop_directory" \
    --architecture "$architecture" \
    --seeds $seeds \
    --epochs "$epochs" \
    --max_hidden_size "$max_hidden_size" \
    --manifest_row_index "$SLURM_ARRAY_TASK_ID" \
    --results_csv "$results_csv"
