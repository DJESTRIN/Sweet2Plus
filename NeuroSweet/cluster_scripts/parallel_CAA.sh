#!/bin/bash
# Usage: bash parallel_CAA.sh <root_data_folder> [folder_name_pattern] [sbatch_script] [repo_root] [conda_env]
# Login-node helper (no #SBATCH headers -- run with `bash`, not `sbatch`) that finds every
# folder matching folder_name_pattern under root_data_folder and submits one CAA.sh SLURM job
# per matched folder, forwarding repo_root/conda_env to each job. If conda_env is omitted, you
# will be prompted to type the anaconda environment name interactively (default: sweet2p).
# Get Inputs from command line
root_data_folder=${1:-"/athena/listonlab/scratch/dje4001/mdt02/tmt_experiment_2024_working_file/Animals"}
folder_name_pattern=${2:-"*day_*"}
sbatch_script=${3:-"/home/dje4001/NeuroSweet/NeuroSweet/cluster_scripts/CAA.sh"}
repo_root=${4:-"/home/dje4001/NeuroSweet"}
conda_env=$5

# If conda_env wasn't given as an argument, prompt for it interactively (falls back to
# "sweet2p" on empty input or when stdin isn't a terminal, e.g. non-interactive/CI use).
if [ -z "$conda_env" ]; then
    if [ -t 0 ]; then
        read -r -p "Anaconda environment name to activate for each CAA.sh job [sweet2p]: " conda_env
    fi
    conda_env=${conda_env:-sweet2p}
fi

# Warn (but don't block) if the named env doesn't appear to exist locally.
if command -v conda >/dev/null 2>&1; then
    if ! conda env list | awk '{print $1}' | grep -qx "$conda_env"; then
        echo "WARNING: conda env '$conda_env' not found in 'conda env list' -- each CAA.sh job may fail to activate it." >&2
    fi
fi

# Print out info about upcoming run
echo -e "Searching in $root_data_folder \nusing pattern $folder_name_pattern \nand will run sbatch script $sbatch_script \nwith repo_root=$repo_root conda_env=$conda_env"

# Use find to locate folders matching the pattern and submit them to sbatch
counter=0
for beh_folder in $(find "$root_data_folder" -type d -name "$folder_name_pattern"); do
    image_folder=$(find "$beh_folder" -type d -name '*_R*')

    if [ -n "$image_folder" ]; then
        echo "Submitted $beh_folder and $image_folder to sbatch."
        sbatch "$sbatch_script" "$beh_folder" "$image_folder" "$repo_root" "$conda_env"
        counter=$((counter + 1)) 
    fi
done

echo "Submitted a total of $counter jobs for $sbatch_script"