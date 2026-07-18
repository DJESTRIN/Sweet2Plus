#!/bin/bash
#SBATCH --job-name=glm_compare                           # Job name
#SBATCH --output=./glm_compare_%j.out                    # Output file name
#SBATCH --error=./glm_compare_%j.err                      # Error file name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Step 3 (final): aggregates every neuron's per-stimulus encoder weight (glmsummary.collect) and
# compares it against the decoder's beta_filtered.csv (compare_decoder_encoder.py). Must run after
# ALL glm_encoder_array.sh tasks have completed (submit_full_glm_pipeline.sh submits this with
# --dependency=afterok:<array_job_id> automatically).
#
# Usage: sbatch glm_summarize_compare.sh <drop_directory> [repo_root] [conda_env]

drop_directory=$1
repo_root=${2:-"/home/dje4001/NeuroSweet"}
conda_env=${3:-"sweet2p"}

if [ -z "$drop_directory" ]; then
    echo "Usage: sbatch glm_summarize_compare.sh <drop_directory> [repo_root] [conda_env]"
    exit 1
fi

source ~/.bashrc
conda activate "$conda_env"

export NEUROSWEET_REPO_ROOT="$repo_root"
export PYTHONPATH="$repo_root:$PYTHONPATH"

encoder_csv="$drop_directory/engelhard_stimulus_summary.csv"
decoder_csv="$drop_directory/beta_filtered.csv"
comparison_dir="$drop_directory/comparison"

python -m NeuroSweet.statistics.glms.glmsummary \
    --input_directory "$drop_directory/temp" \
    --output_file "$encoder_csv"

python -m NeuroSweet.statistics.glms.compare_decoder_encoder \
    --decoder_csv "$decoder_csv" \
    --encoder_csv "$encoder_csv" \
    --output_dir "$comparison_dir"

echo "Done. See $comparison_dir/decoder_encoder_comparison_report.txt"
