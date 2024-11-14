#!/bin/bash
# Get Inputs from command line
root_data_folder=${1:-"/athena/listonlab/scratch/dje4001/mdt02/tmt_experiment_2024_working_file/Animals"}
folder_name_pattern=${2:-"*day_*"}
sbatch_script=${3:-"~/Sweet2Plus/Sweet2Plus/cloud/CCA.sh"}

# Print out info about upcoming run
echo -e "Searching in $root_data_folder \
    using pattern $folder_name_pattern \
    and will run sbatch script $sbatch_script"

# Use find to locate folders matching the pattern and submit them to sbatch
for beh_folder in $(find "$root_data_folder" -type d -name "$folder_name_pattern"); do
    image_folder=$(find "$beh_folder" -type d -name '*R*')
    #sbatch "$sbatch_script" "$beh_folder" "$image_folder"
    echo "Submitted $beh_folder and $image_folder to sbatch."
done