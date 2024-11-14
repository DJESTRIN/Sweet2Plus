#!/bin/bash
#SBATCH --job-name=sweet2p               # Job name
#SBATCH --output=sweet2p_job.out         # Output file name
#SBATCH --error=sweet2p_job.err          # Error file name
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # Number of CPUs per task
#SBATCH --mem=20G                        # Total memory for the job
#SBATCH --time=01:00:00                  # Time limit (e.g., 1 hour)

# Get inputs from command line
subject_folder=$1

# Set up log_directory
log_dir="~/slurm_logs/sweet2plus/"
mkdir -p "$log_dir"

# Load correct anaconda enviornment
source ~/.bashrc
conda activate sweet2p

# Run the Python script
python ~/Sweet2Plus/Sweet2Plus/core/CorrelativeActivityAnalysis.py --data_directory $subject_folder --single_subject_flag
