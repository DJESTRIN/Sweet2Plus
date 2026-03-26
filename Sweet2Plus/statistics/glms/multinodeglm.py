#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name: multinodeglm.py
Description: Submit engelharglm jobs in parallel on SLURM cluster.
Author: David Estrin
Date: 2026-03-26
Version: 1.0
"""
# Load dependencies 
import os
import subprocess
import argparse
import time
from Sweet2Plus.statistics.glms.engelhardglm import currate_data 

# Build custom class for gather all path data and submitting jobs via slurm
class multinodeglm():
    def __init__(self, input_path, conda_environment_name, partition_oh = 'scu-cpu', 
                 email = 'dje4001@med.cornell.edu',  username='dje4001', memory_per_job = 256, tasks_per_job = 3, 
                 cpus_per_task = 8, delete_contents_of_output_folders = False, 
                 delete_pkls=False, number_of_jobs=500, slurm_sleep=60):
        # Set input and output paths
        self.input_path = input_path
        self.zipped_output_path = os.path.join(self.input_path,'temp/')
        self.neuronal_trace_output_path = os.path.join(self.input_path,'neuron_predictions/')
        self.summary_figures_output_path = os.path.join(self.input_path,'neuron_general_metrics/')
        self.slurm_output_path = os.path.join(self.input_path,'slurm_output/')
        self.slurm_error_path = os.path.join(self.input_path,'slurm_errors/')

        # Set other attributes
        self.conda_environment_name = conda_environment_name
        self.partition_oh = partition_oh
        self.username = username 
        self.email = email
        self.memory_per_job = memory_per_job
        self.tasks_per_job = tasks_per_job
        self.cpus_per_task = cpus_per_task
        self.delete_contents_of_output_folders = delete_contents_of_output_folders
        self.delete_pkls = delete_pkls
        self.number_of_jobs = number_of_jobs
        self.slurm_sleep = slurm_sleep # amount of time before checking if jobs are still running on slurm

        # Create output paths
        os.makedirs(self.zipped_output_path , exist_ok=True)
        os.makedirs(self.neuronal_trace_output_path , exist_ok=True)
        os.makedirs(self.summary_figures_output_path , exist_ok=True)
        os.makedirs(self.slurm_output_path , exist_ok=True)
        os.makedirs(self.slurm_error_path , exist_ok=True)
        
    def __call__(self):
        self.grab_neuron_nums()
        self.submit_jobs()
        self.monitor_jobs()
      
    def grab_neuron_nums(self):
        # Grab number of neurons for determining number of jobs. 
        dataset = currate_data([],[],[], self.input_path)
        dataset.load()
        self.number_of_neurons = len(dataset.trans_act)

        # Get groupings of neurons for jobs. 
        self.groupings = [(i*(self.number_of_neurons+1)//self.number_of_jobs, (i+1)*(self.number_of_neurons+1)//self.number_of_jobs - 1) for i in range(self.number_of_jobs)]

    def delete_contents_path(self, path_oh,extensions=['.jpg','.pkl']):
        # Check if the path exists
        if os.path.exists(path_oh):
            # Iterate through all items in the directory
            for item in os.listdir(path_oh):
                item_path = os.path.join(path_oh, item)
                # Check if it's a file or folder
                if os.path.isfile(item_path):
                    for filetype in extensions:
                        if filetype in item_path:
                            try:
                                print(f'File being deleted {item_path}')
                                os.remove(item_path) 
                            except:
                                print(f'Not found {item_path}')

        else:
            print(f"Path '{path_oh}' does not exist.")

    def submit_jobs(self):
        """ Build sbatch command and submit for running """
        # Cancel all previous jobs I am running
        if self.delete_contents_of_output_folders:
            my_first_command = f"scancel -n engelhard_glm"
            first_command_result = subprocess.run([my_first_command], shell=True, capture_output=True, text=True)
            self.delete_contents_path(path_oh = self.slurm_output_path, extensions=['.out'])
            self.delete_contents_path(path_oh = self.slurm_error_path, extensions=['.err'])
            self.delete_contents_path(path_oh = self.zipped_output_path ,extensions=['.gz'])
            self.delete_contents_path(path_oh = self.neuronal_trace_output_path ,extensions=['.jpg'])
            self.delete_contents_path(path_oh = self.summary_figures_output_path,extensions=['.jpg'])


        self.jids = [] # Put jobids into a list for later monitoring
        
        # Loop over groupings
        for (start_neuron,stop_neuron) in self.groupings:
            # Build command line interface command
            my_command = f"sbatch --job-name=engelhard_glm \
                    --mem={self.memory_per_job}G \
                    --ntasks={self.tasks_per_job} \
                    --cpus-per-task={self.cpus_per_task} \
                    --partition={self.partition_oh} \
                    --mail-type=BEGIN,END,FAIL \
                    --mail-user={self.email} \
                    --output={self.slurm_output_path}/%x-%j.out \
                    --error={self.slurm_error_path}/%x-%j.err \
                    --wrap='source ~/.bashrc && conda activate {self.conda_environment_name} && python ./engelhardglm.py \
                    --data_directory {self.input_path} \
                    --drop_directory {self.input_path} \
                    --data_provided \
                    --start_neuron {start_neuron} \
                    --stop_neuron {stop_neuron}'"

            # Run subprocess on command and pull out result. 
            result = subprocess.run([my_command], shell=True, capture_output=True, text=True)
            
            # Append job id to list for monitoring. 
            idoh = result.stdout.strip().split(" ")[-1]
            print(f'Submitted job number: {idoh}')
            self.jids.append(idoh)
        
    def monitor_jobs(self):
        """ Find and monitor my jobs in the slurm queue  """

        def find_my_jobs(original_job_ids,username='dje4001'):
            squeue_result_oh = subprocess.run(f"squeue --noheader -u {username} --format=%A", shell=True, capture_output=True, text=True)
            current_ids = squeue_result_oh.stdout.split()
            running_jobs = [job for job in original_job_ids if job in current_ids]
            if len(running_jobs)>0:
                result = True
            else:
                result = False
            return result, running_jobs
        
        # Continously monitor jobs if running
        running_jobs = self.jids
        result, running_jobs = find_my_jobs(running_jobs,username=self.username)
        while result:
            print(f'jobs {running_jobs} still running')
            time.sleep(self.slurm_sleep) 
            result, running_jobs = find_my_jobs(running_jobs,username=self.username)

def cli_parser():
    parser = argparse.ArgumentParser(description="Get all main directories")

    # Directories
    parser.add_argument('--input_path', type=str, required=True, help='Input path containing processed two-photon data')

    # Options for run
    parser.add_argument('--conda_environment_name', type=str, required=True, help='Conda environment needed for registration')
    parser.add_argument('--partition', type=str, default='scu-cpu', help='slurm partition to use')
    parser.add_argument('--user_email', type=str, default='dje4001@med.cornell.edu', help='Email to use for slurm' )
    parser.add_argument('--memory', type=str, default='128', help='Memory in Gb to be used for each node')
    parser.add_argument('--tasks', type=str, default='8', help='Number of tasks per node')
    parser.add_argument('--cpus_per_task', type=str, default='4', help='Number of cpus per task')

    # Determine whether to monitor jobs
    parser.add_argument('--monitor_only', action='store_true', help='Number of cpus per task')

    # Option for deleting past results
    parser.add_argument('--force_delete_output', action='store_true', help='Number of cpus per task')
    parser.add_argument('--force_delete_pkls', action='store_true', help='Number of cpus per task')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    # Parse command line inputs
    args = cli_parser()
    current_glm_job_object = multinodeglm()
    current_glm_job_object()
    