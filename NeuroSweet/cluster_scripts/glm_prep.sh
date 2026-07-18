#!/bin/bash
#SBATCH --job-name=glm_prep                              # Job name
#SBATCH --output=./glm_prep_%j.out                       # Output file name
#SBATCH --error=./glm_prep_%j.err                        # Error file name
#SBATCH --ntasks=1                                       # Number of tasks
#SBATCH --cpus-per-task=8                                # Cores for the decoder's bootstrapped fits
#SBATCH --mem=64G                                        # Total memory for the job
#SBATCH --time=04:00:00                                  # Time limit -- decoder + data curation only,
                                                          # NOT the per-neuron encoder fits (see
                                                          # glm_encoder_array.sh). Adjust if the decoder's
                                                          # bootstrapped fits run long on your dataset.

# Step 1 of the decoder/encoder GLM comparison pipeline: loads raw recordings once, runs the cheap
# population decoder (circuit_regression) to completion, and curates + saves the per-neuron dataset
# that engelhardglm's SLURM array tasks (glm_encoder_array.sh) will load via --data_provided.
#
# Usage: sbatch glm_prep.sh <data_directory> <drop_directory> [repo_root] [conda_env]

data_directory=$1
drop_directory=$2
repo_root=${3:-"/home/dje4001/NeuroSweet"}
conda_env=${4:-"sweet2p"}

if [ -z "$data_directory" ] || [ -z "$drop_directory" ]; then
    echo "Usage: sbatch glm_prep.sh <data_directory> <drop_directory> [repo_root] [conda_env]"
    exit 1
fi

mkdir -p "$drop_directory"

source ~/.bashrc
conda activate "$conda_env"

export NEUROSWEET_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:$PYTHONPATH"

python -m NeuroSweet.statistics.glms.glm_prep \
    --data_directory "$data_directory" \
    --drop_directory "$drop_directory" \
    --n_jobs "$SLURM_CPUS_PER_TASK"
