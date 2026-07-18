#!/bin/bash
#SBATCH --job-name=glm_encoder                          # Job name
#SBATCH --output=./glm_encoder_%A_%a.out                 # Output file name (%A=array job id, %a=task id)
#SBATCH --error=./glm_encoder_%A_%a.err                  # Error file name
#SBATCH --ntasks=1                                       # Number of tasks per array element
#SBATCH --cpus-per-task=16                               # Workers for fitting neurons within this chunk
#SBATCH --mem=32G                                        # Memory per array element
#SBATCH --time=03:00:00                                  # Adjust alongside CHUNK_SIZE (see below)

# Step 2: fits engelhardglm's per-neuron GLM (500 circular-lag permutations + 1 real fit per neuron)
# across a SLURM array, one array task per contiguous chunk of neurons. Must run *after*
# glm_prep.sh has written <drop_directory>/trans_*.pkl and glm_manifest.json.
#
# Runtime note: an isolated single-fit benchmark on a 32-core desktop gave ~0.6s/fit x 501 fits/neuron
# ~= 5 min/neuron, but an actual end-to-end smoke test of this exact CLI path (synthetic data, 2000
# frames/neuron, 11 neurons total) measured ~7-9 CPU-min/neuron once patsy formula parsing, trial
# cropping, and spline convolution overhead are included -- use ~8 min/neuron as the planning number.
# With CPUS_PER_TASK workers splitting CHUNK_SIZE neurons within a task, wall time per array task is
# roughly ceil(CHUNK_SIZE / CPUS_PER_TASK) x 8 min (+overhead). Tune CHUNK_SIZE,
# --cpus-per-task, and --time together, and check your cluster's MaxArraySize / QOS core limits before
# submitting a large array (see submit_full_glm_pipeline.sh, which computes --array from the actual
# neuron count in glm_manifest.json).
#
# Usage (normally invoked by submit_full_glm_pipeline.sh, not run directly):
#   sbatch --array=0-<N-1> glm_encoder_array.sh <drop_directory> <total_neurons> [chunk_size] [repo_root] [conda_env]

drop_directory=$1
total_neurons=$2
chunk_size=${3:-100}
repo_root=${4:-"/home/dje4001/NeuroSweet"}
conda_env=${5:-"sweet2p"}

if [ -z "$drop_directory" ] || [ -z "$total_neurons" ]; then
    echo "Usage: sbatch --array=0-<N-1> glm_encoder_array.sh <drop_directory> <total_neurons> [chunk_size] [repo_root] [conda_env]"
    exit 1
fi

source ~/.bashrc
conda activate "$conda_env"

export NEUROSWEET_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:$PYTHONPATH"

start_neuron=$(( SLURM_ARRAY_TASK_ID * chunk_size ))
stop_neuron=$(( start_neuron + chunk_size - 1 ))
last_valid_index=$(( total_neurons - 1 ))
if [ "$stop_neuron" -gt "$last_valid_index" ]; then
    stop_neuron=$last_valid_index
fi
if [ "$start_neuron" -gt "$last_valid_index" ]; then
    echo "Array task $SLURM_ARRAY_TASK_ID: start_neuron $start_neuron is past the last valid neuron index "
    echo "$last_valid_index (total_neurons=$total_neurons) -- nothing to do, exiting cleanly."
    exit 0
fi

echo "Array task $SLURM_ARRAY_TASK_ID: neurons $start_neuron-$stop_neuron (chunk_size=$chunk_size, cpus=$SLURM_CPUS_PER_TASK)"

python -m NeuroSweet.statistics.glms.engelhardglm \
    --drop_directory "$drop_directory" \
    --data_provided \
    --start_neuron "$start_neuron" \
    --stop_neuron "$stop_neuron" \
    --n_jobs "$SLURM_CPUS_PER_TASK"
