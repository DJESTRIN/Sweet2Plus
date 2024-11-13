#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: parallel_helper.py
Description: A set of functions used to manage resources when running multi-threading code.
    Ideal example is to be run on the cluster. This code helps optimize the number of processes/threads
    run across cores, without breaking RAM.
Author: David Estrin
Version: 1.0
Date: 10-15-2024
"""
import subprocess
import psutil
import time
import ipdb
# from skopt import gp_minimize
# from skopt.space import Integer
# import numpy as np

def run_and_monitor(cli_command, memory_threshold, time_wait=None):
    if time_wait is not None:
        start_time=time.time() # Set up a stop watch

        # Run process
        process = subprocess.Popen(cli_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            while process.poll() is None: 
                # Get the amount of RAM being used
                memory_info = psutil.virtual_memory()
                available_memory = memory_info.available / memory_info.total
                print(f"Available memory: {available_memory:.2%}")
                
                # Check if available memory is below the threshold
                if available_memory < memory_threshold:
                    print("Memory threshold exceeded. Terminating subprocess.")
                    process.terminate()  # Stop the subprocess
                    return 'OOM'
                
                # Determine change in time
                delta_time=time.time()-start_time
                if delta_time>time_wait:
                    print("Memory usage successful")
                    process.terminate()
                    return 'success'
                
                # Wait for a short time before checking again
                time.sleep(1)
            
            # Get process output if it completes without being terminated
            stdout, stderr = process.communicate()
            
            if stdout:
                print("Subprocess output:", stdout.decode())
            if stderr:
                print("Subprocess error:", stderr.decode())
        
        except Exception as e:
            process.terminate()
            print("An error occurred:", e)
    
    else:
        process = subprocess.Popen(cli_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            while process.poll() is None: 
                # Get the amount of RAM being used
                memory_info = psutil.virtual_memory()
                available_memory = memory_info.available / memory_info.total
                print(f"Available memory: {available_memory:.2%}")
                
                # Check if available memory is below the threshold
                if available_memory < memory_threshold:
                    print("Memory threshold exceeded. Terminating subprocess.")
                    process.terminate()  # Stop the subprocess
                    return 'OOM'
                
                # Wait for a short time before checking again
                time.sleep(1)
            
            # Get process output if it completes without being terminated
            stdout, stderr = process.communicate()
            
            if stdout:
                print("Subprocess output:", stdout.decode())
            if stderr:
                print("Subprocess error:", stderr.decode())
        
        except Exception as e:
            process.terminate()
            print("An error occurred:", e)

def update_command(command,njobs):
    return command+['--njobs',str(njobs)]

def job_optimization(cli_command, memory_threshold, max_n_jobs=99, time_out=180):
    # Ascending optimization to find max n_jobs before OOM error
    for n_jobs_oh in range(4,max_n_jobs):
        print(f'Number of jobs: {n_jobs_oh}')
        cli_command_oh=update_command(cli_command,n_jobs_oh)
        result=run_and_monitor(cli_command=cli_command_oh,memory_threshold=memory_threshold,time_wait=time_out)

        if result=='OOM':
            max_n_jobs_oh=n_jobs_oh-1
            break

        if result=='success':
            continue

    # Descending optimization to find max n_jobs before OOM error
    time_out=None # There will be no max time while descending
    for n_jobs_oh in range(max_n_jobs_oh,1,-1):
        time.sleep(60) # Wait 1 minute for memory usage to go down naturally
        print(f'Number of jobs: {n_jobs_oh}')
        cli_command_oh=update_command(cli_command,n_jobs_oh)
        result=run_and_monitor(cli_command=cli_command_oh,memory_threshold=memory_threshold,time_wait=time_out)
        if result=='OOM':
            continue

def job_objective(n_jobs_oh,cli_command,memory_threshold,time_out):
    cli_command_oh=update_command(cli_command,n_jobs_oh[0])
    result=run_and_monitor(cli_command=cli_command_oh,memory_threshold=memory_threshold,time_wait=time_out)

    if result=='success':
        ans_oh=True
    else:
        ans_oh=False
    return -1 if ans_oh else 0   # Return -1 for True, 0 for False (minimizing -1 = maximizing)

# def bayes_opt():
#     # Define the search space
#     search_space = [Integer(0, 100)]

#     # Perform Bayesian optimization
#     res = gp_minimize(
#         job_objective, 
#         search_space, 
#         n_calls=50,                # Number of guesses (samples)
#         random_state=0,            # For reproducibility
#         acq_func="EI"              # Expected Improvement to balance exploration/exploitation
#     )

# # Get the best result
# best_number = res.x[0]
# best_score = -res.fun  # since we used -1 for True, convert it back

if __name__=='__main__':
    cli_command = ["python", ".\Sweet2Plus\core\CorrelativeActivityAnalysis.py", "--root_data_directory", r"C:\Users\listo\tmt_experiment_2024_working_file"]  # Add your arguments here
    memory_threshold = 0.2  # Set RAM threshold to 80%
    job_optimization(cli_command, memory_threshold, max_n_jobs=101, time_out=180)