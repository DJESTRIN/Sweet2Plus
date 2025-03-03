#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: WeightModelingSpinup.py
Description: 
Author: David James Estrin
Version: 2.0
Date: 02-27-2025
"""
import argparse
import subprocess
import os, glob
import time

def parse_cli_inputs():
    """ Get parent directory containing all s2p objs """
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_input_directory')
    parser.add_argument('--slurm_output_directory')
    return parser.parse_args()

def build_wrap(basepath,filelist,droplist):
    output_wrap_calls = []
    for file, drop in zip(filelist,droplist):
        wrap_call_oh = f'python {basepath}/WeightModeling.py \
            --s2p_object_file {file} \
            --hypertuning_study \
            --drop_directory {drop}'
        # Append the current wrap string to list
        output_wrap_calls.append(wrap_call_oh)
    return output_wrap_calls

def call_slurm(full_wrap_calls, slurm_output_directory, job_name='SynapticWeightModeling', partition='scu-gpu', gpus=1, 
               memory=64, conda_environment='Sweet2p'):
    
    # Loop over all calls that were generate per s2p file
    all_job_ids = []
    for wrap_oh in full_wrap_calls:
        # Build command
        sbatch_command = f"sbatch --job-name={job_name} \
                            --partition={partition} \
                            --output={slurm_output_directory} \
                            --gres=gpu:{gpus} \
                            --mem={memory}GB \
                            --wrap='sourece ~/.bashrc && \
                                conda activate {conda_environment} && \
                                    {wrap_oh}'"

        # Run command
        result = subprocess.run(sbatch_command, capture_output=True, text=True, shell=True)

        # Print submission status
        if result.returncode == 0:
            print(f'Job ID: {result.stdout} was submitted successfully')
        else:
            print(f'Job ID: {result.stdout} FAILED to submit')
        
        all_job_ids.append(result.stdout)
    return all_job_ids

def check_jobs_status(job_ids):
    job_status = {}
    result = subprocess.run(['squeue', '--job', ','.join(job_ids), '--format', '%i %t'], 
                            capture_output=True, text=True)
    
    if result.returncode == 0:
        output = result.stdout
        # Parse the output
        for line in output.splitlines():
            # Skip the header line
            if line.startswith('JOBID'):
                continue
            job_id, status = line.split()
            job_status[job_id] = status
    else:
        print("Failed to query job statuses")
    
    return job_status

def wait_for_jobs_to_finish(job_ids, check_interval=300):
    # Keep checking job statuses until all jobs are finished
    while True:
        job_status = check_jobs_status(job_ids)
        unfinished_jobs = [job_id for job_id, status in job_status.items() if status not in ['CD', 'COMPLETED', 'FAILED']]
        
        if not unfinished_jobs:
            print("All jobs are finished.")
            break
        
        print(f"Jobs still running: {unfinished_jobs}. Checking again in {check_interval} seconds...")
        time.sleep(check_interval)

def main():
    args = parse_cli_inputs()

    # Get obj files
    search_path = os.path.join(args.parent_input_directory,'/**/*obj*.json')
    objfiles = glob.glob(search_path,recursive=True)

    # Generate output directory for each objfile
    drop_directories = []
    for file in objfiles:
        base = file[:-5] # remove .json
        drop_directory_oh = os.path.join(base,r'/weight_drop_dir/')
        if not os.path.exists(drop_directory_oh):
            os.makedirs(drop_directory_oh)
        drop_directories.append(drop_directory_oh)

    # Run all s2p through models and save weight dataframes
    output_wraps = build_wrap(basepath = os.path.abspath(__file__), filelist = objfiles, droplist = drop_directories)
    jobs_oh = call_slurm(full_wrap_calls=output_wraps)

    # Determine when jobs are finished
    wait_for_jobs_to_finish(job_ids=jobs_oh)

    # Run statistics code
    path_to_python_script = os.path.join(os.path.abspath(__file__),'WeightModelingStats.py')
    sbatch_command = "sbatch --job-name=StatsWeightModeling " \
                 "--output=output.log " \
                 "--ntasks=2 " \
                 "--cpus-per-task=4 " \
                 "--mem=128GB " \
                 "--wrap='source ~/.bashrc && conda activate {args.conda_environment} && python {path_to_python_script}'"
    result = subprocess.run(sbatch_command, capture_output=True, text=True, shell=True)
    
if __name__=='__main__':
    main()