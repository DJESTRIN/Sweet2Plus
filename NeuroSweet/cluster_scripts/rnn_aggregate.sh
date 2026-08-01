#!/bin/bash
#SBATCH --job-name=rnn_aggregate                          # Job name
#SBATCH --output=./rnn_aggregate_%j.out                    # Output file name
#SBATCH --error=./rnn_aggregate_%j.err                     # Error file name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Step 3/3 of the in-silico RNN pipeline: aggregates all 3 architectures' per-session-array-task
# result CSVs and scores each architecture against the real report's decoder/encoder divergence
# signature (see rnn_architecture_scoring.py), reporting the winning architecture.
#
# Usage: sbatch --dependency=afterok:<job1>:<job2>:<job3> rnn_aggregate.sh <drop_directory> [repo_root] [conda_env]

drop_directory=$1
repo_root=${2:-"/home/dje4001/NeuroSweet"}
conda_env=${3:-"pytorch"}

if [ -z "$drop_directory" ]; then
    echo "Usage: sbatch rnn_aggregate.sh <drop_directory> [repo_root] [conda_env]"
    exit 1
fi

source ~/.bashrc
conda activate "$conda_env"

export NEUROSWEET_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:$PYTHONPATH"

python -m NeuroSweet.rnn_modeling.rnn_aggregate_and_score \
    --drop_directory "$drop_directory" \
    --architectures unmixed semi-mixed fully-mixed
