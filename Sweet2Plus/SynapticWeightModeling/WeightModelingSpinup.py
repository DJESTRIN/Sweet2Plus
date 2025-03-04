#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: WeightModelingSpinup.py
Description: Run WeightModeling code via slurm on all files
Author: David James Estrin
Version: 1.0
Date: 02-27-2025
"""
import argparse
import subprocess
import os, glob
import time
import ipdb

def parse_cli_inputs():
    """ Get parent directory containing all s2p objs """
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_input_directory')
    parser.add_argument('--slurm_output_directory')
    return parser.parse_args()

class spinup:
    def __init__(self,parent_input_directory, slurm_output_directory, job_name='SynapticWeightModeling', 
                 partition='scu-gpu', gpus=1, memory=64, conda_environment='Sweet2p'):
        # Directory containing all s2p data objects
        self.parent_input_directory = parent_input_directory

        # SLURM cluster information
        self.slurm_output_directory = slurm_output_directory
        if not os.path.exists(self.slurm_output_directory):
            os.mkdir(self.slurm_output_directory)

        self.job_name = job_name
        self.partition = partition
        self.gpus = gpus
        self.memory = memory
        self.conda_environment = conda_environment

    def build_wrap(self, basepath,filelist,droplist):
        basepath, _ = basepath.split('WeightModelingSpinup')
        output_wrap_calls = []
        for file, drop in zip(filelist,droplist):
            wrap_call_oh = f'python {basepath}/WeightModeling.py \
                --s2p_object_file {file} \
                --hypertuning_study \
                --drop_directory {drop}'
            # Append the current wrap string to list
            output_wrap_calls.append(wrap_call_oh)
        return output_wrap_calls

    def call_slurm(self, full_wrap_calls):
        """ Builds and executes slurm commands via subprocess """
        # Loop over all calls that were generate per s2p file
        all_job_ids = []
        full_commands = []
        for wrap_oh in full_wrap_calls:
            # Build command
            sbatch_command = f"sbatch --job-name={self.job_name} \
                                --partition={self.partition} \
                                --output={self.slurm_output_directory}/output_%j.log \
                                --error={self.slurm_output_directory}/error_%j.log \
                                --gres=gpu:{self.gpus} \
                                --mem={self.memory}GB \
                                --wrap='sourece ~/.bashrc && \
                                    conda activate {self.conda_environment} && \
                                        {wrap_oh}'"
            full_commands.append(sbatch_command)

            # Run command
            result = subprocess.run(sbatch_command, capture_output=True, text=True, shell=True)

            # Print submission status
            if result.returncode == 0:
                print(f'Job ID: {result.stdout} was submitted successfully')
            else:
                print(f'Job ID: {result.stdout} FAILED to submit')
            
            all_job_ids.append(result.stdout)
        return all_job_ids, full_commands

    def check_jobs_status(self, job_ids):
        job_status = {}
        result = subprocess.run(['squeue', '--job', ','.join(job_ids), '--format', '%i %t'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            for line in output.splitlines():
                if line.startswith('JOBID'):
                    continue
                job_id, status = line.split()
                job_status[job_id] = status
        else:
            print("Failed to query job statuses")
        return job_status

    def wait_for_jobs_to_finish(self, job_ids, check_interval=300):
        while True:
            job_status = self.check_jobs_status(job_ids)
            unfinished_jobs = [job_id for job_id, status in job_status.items() if status not in ['CD', 'COMPLETED', 'FAILED']]
            
            if not unfinished_jobs:
                print("All jobs are finished.")
                break
            
            print(f"Jobs still running: {unfinished_jobs}. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)

    def run_final_stats(self):
        # Run statistics code
        path_to_python_script = os.path.join(os.path.abspath(__file__).split('WeighModelingSpinup')[0],'WeightModelingStats.py')
        run_stats_command = f"sbatch --job-name=StatsWeightModeling \
                        --output={self.slurm_output_directory}/output_%j.log \
                        --error={self.slurm_output_directory}/error_%j.log \
                        --ntasks=2 \
                        --cpus-per-task=4 \
                        --mem=128GB \
                        --wrap='source ~/.bashrc && conda activate {self.conda_environment} && python {path_to_python_script}'"
        result = subprocess.run(run_stats_command, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f'Stats job ({result.stdout}) was submitted successfully')
        else:
            print(f'Stats job ({result.stdout}) FAILED to submit')
        return
    
    def gather_files(self):
        # Get obj files
        search_path = os.path.join(self.parent_input_directory,r'**/objfile.json')
        objfiles = glob.glob(search_path,recursive=True)

        # Generate output directory for each objfile
        drop_directories = []
        for file in objfiles:
            base , _ = file.split('objfile')
            drop_directory_oh = os.path.join(base,r'weight_drop_dir/')
            if not os.path.exists(drop_directory_oh):
                os.makedirs(drop_directory_oh)
            drop_directories.append(drop_directory_oh)
        return objfiles, drop_directories

    def __call__(self):
        # Get obj files
        objfiles, drop_directories = self.gather_files()

        # Run all s2p through models and save weight dataframes
        output_wraps = self.build_wrap(basepath = os.path.abspath(__file__), filelist = objfiles, droplist = drop_directories)
        jobs_oh, full_commands = self.call_slurm(full_wrap_calls=output_wraps)
        ipdb.set_trace()

        # Determine when jobs are finished
        self.wait_for_jobs_to_finish(job_ids=jobs_oh)

        # Run statistics code
        self.run_final_stats()
        
if __name__=='__main__':
    args = parse_cli_inputs()
    spinobj = spinup(parent_input_directory = args.parent_input_directory, slurm_output_directory = args.slurm_output_directory)
    spinobj()