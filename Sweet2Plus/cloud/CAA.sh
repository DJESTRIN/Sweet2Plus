#!/bin/bash
#SBATCH --job-name=sweet2p                              # Job name
#SBATCH --output=./%j.out         # Output file name
#SBATCH --error=./%j.err       # Error file name
#SBATCH --ntasks=1                                      # Number of tasks
#SBATCH --cpus-per-task=4                               # Number of CPUs per task
#SBATCH --mem=60G                                       # Total memory for the job
#SBATCH --time=06:00:00                                 # Time limit (e.g., 1 hour)

# Get inputs from command line
beh_folder=$1
image_folder=$2

# Set up log_directory
# log_dir="/home/dje4001/slurm_logs/sweet2plus/"
# err_dir="/home/dje4001/slurm_errors/sweet2plus/"
# mkdir -p "$log_dir"
# mkdir -p "$err_dir"

# Load correct anaconda enviornment
source ~/.bashrc
conda activate sweet2p

# Run the Python script
python /home/dje4001/Sweet2Plus/Sweet2Plus/core/CorrelativeActivityAnalysis.py --data_directory $image_folder --beh_directory $beh_folder --single_subject_flag --force_redo
