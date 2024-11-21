#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module name: CorrelativeActivityAnalysis.py
Description: Primary script for analyzing 
Author: David James Estrin
Version: 1.0
Date: 11-14-2024
"""

""" Written by David James Estrin 
Compare correlation of acitivty:
(1) whole dataset for all animals across time (1,7,14)
    Get average correlation across each neuron in mouse. Then get average +/- sem correlation across mice. 
    GLM or LMM for all neurons average correlation for all mice -> put into dataframe for SAS?
(2) TMT vs non-TMT subgroups 
    Correlate TMT vs TMT neurons, Non-TMT vs Non TMT, TMT vs all and Non TMT vs all. Do this averaged across mice and across time. 
(3) Analyze pre TMT and post TMT Day 1 recordings to see changes in state?
    N dimensions for N neurons where each dimension is average AUC. 
    Does the state change differ with respect to stress (Day 1 vs Day 14)
(4) Replicate findings from CORT study
"""
import ipdb
from Sweet2Plus.core.core import pipeline, corralative_activity 
from behavior import load_serial_output
import numpy as np
import warnings
import tqdm
import pickle
from Sweet2Plus.core.SaveLoadObjs import SaveObj, LoadObj, SaveList, OpenList
import matplotlib.pyplot as plt
import os, glob, re
import pandas as pd
import argparse
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

""" NEED TO PLOT CORRELATIONS..
NEED TO ALSO PLOT AUC values across trial types for each neuron to get averages 

excel sheet needs classification column...
state distances needs to be parsable by day
need auc values across trial types

"""
def run_parallel_correlations(primary_obj):
    print('Generating correlations in parallel right now...')
    data = Parallel(n_jobs=6)(delayed(parallel_correlations)(primary_obj.recordings[subjectnumber]) for subjectnumber in tqdm.tqdm(range(len(primary_obj.recordings))))
    return data

def parallel_correlations(subject_obj_oh):
    """ Runs correlations code but meant for parallel processing 

    Inputs:
    primary_obj -- (obj) a pipeline object containing all data for all subjects in s2p obj formats
    subjectnumber -- (str) the subject on hand that will be analyzed

    Outputs:
    parse_info -- (list) All relevant identifying information for the subject on hand
    correlation_data -- (list) relevant correlation data in a list
    """
    if subject_obj_oh is not None:
        parse_info=[]
        correlation_data=[]
        start_time = subject_obj_oh.all_evts_imagetime[2][0] #Get the first trial time. Baseline activity is everything preceding
        try:
            tmt_start = subject_obj_oh.all_evts_imagetime[3][0] #Get the first trial time. Baseline activity is everything preceding
            tmt_end = subject_obj_oh.all_evts_imagetime[3][4] 

            # Parse traces
            ztracesoh=np.copy(subject_obj_oh.ztraces) #Make a copy of the trace data
            baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time
            rewardztracesoh=ztracesoh[:,int(start_time):int(tmt_start)] 
            tmtztracesoh=ztracesoh[:,int(tmt_start):int(tmt_end)] 
            posttmttracesoh=ztracesoh[:,int(tmt_end):] 

            # Get correlations
            blcorr, correlations = subject_obj_oh.get_activity_correlation(baselineztracesoh,output_filename='baseline_correlation.pdf') #Calculate correlation data
            rewcorr, correlations = subject_obj_oh.get_activity_correlation(rewardztracesoh,output_filename='reward_correlation.pdf') #Calculate correlation data
            tmtcorr, correlations = subject_obj_oh.get_activity_correlation(tmtztracesoh,output_filename='tmt_correlation.pdf') #Calculate correlation data
            posttmtcorr, correlations = subject_obj_oh.get_activity_correlation(posttmttracesoh,output_filename='posttmt_correlation.pdf') #Calculate correlation data
            info = [subject_obj_oh.day,subject_obj_oh.cage,subject_obj_oh.mouse,subject_obj_oh.group] #Get info data
        
            ## Classify whether group one or two is TMT activated
            aucsoh=np.asarray(subject_obj_oh.auc_vals)
            firstzero=aucsoh[np.where(subject_obj_oh.classifications==0)[0][0]]
            firstone=aucsoh[np.where(subject_obj_oh.classifications==1)[0][0]]

            if firstzero[3]>firstone[3]:
                zerolabels='TMT_activated'
                onelabels='NonTMT_activated'
            else:
                zerolabels='NonTMT_activated'
                onelabels='TMT_activated'

            neuron_labels=[]
            for noh in subject_obj_oh.classifications:
                if noh ==1:
                    neuron_labels.append(onelabels)
                else:
                    neuron_labels.append(zerolabels)

        except:
            # Parse traces
            ztracesoh=np.copy(subject_obj_oh.ztraces) #Make a copy of the trace data
            baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time

            # Get correlations
            blcorr, correlations = subject_obj_oh.get_activity_correlation(baselineztracesoh) #Calculate correlation data
            rewcorr,tmtcorr,posttmtcorr=np.nan,np.nan,np.nan
            info = [subject_obj_oh.day,subject_obj_oh.cage,subject_obj_oh.mouse,subject_obj_oh.group] #Get info data
        
        #Append all data to lists
        parse_info.append(info)
        correlation_data.append([blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels])

    else:
        parse_info=[]
        correlation_data=[]

    return [parse_info, correlation_data]

def generate_tall_dataset(parse_info,correlation_data,root_directory,filename='Repeated_Measures_Correlations.csv'):
    """ Converts parse_info and correlation_data into a tall format dataset

    Inputs:
    parse_info -- (list) contains identity information regarding each subject.
    correlation_data -- (list) contains neuronal data for each subject including correlation data.
    root_directory -- (str) location where csv containing data will be saved to. 
     
    Outputs:
    Saves dataset into a csv file """
    # Find the average correlation for each trial type
    av_corrs_data=[]
    for uid in correlation_data:
        try:
            if not uid:
                print('empty list')
            else:
                av_corrs_data.append([[np.nanmean(uid[0][0],axis=0)],[np.nanmean(uid[0][1],axis=0)],[np.nanmean(uid[0][2],axis=0)],[np.nanmean(uid[0][3],axis=0)],uid[0][4]])
        except:
            ipdb.set_trace()

    #Build tall dataset
    counter=0
    for info,data in zip(parse_info,av_corrs_data):
        (bl,rew,tmt,post,neuron_labels)=data
        try:
            for neuron_id,(blv,rewv,tmtv,postv,labelsoh) in enumerate(zip(bl[0],rew[0],tmt[0],post[0],neuron_labels)):
                # list of name, degree, score
                dict={'subject':info[2],'cage':info[1],'session':info[0],'group':info[3],'neuron':neuron_id,'baseline':blv,'reward':rewv,'tmt':tmtv,'posttmt':postv,'classification':labelsoh}
                dfoh=pd.DataFrame(dict,index=[0])
                if counter==0:
                    DF=dfoh
                else:
                    DF=pd.concat([DF,dfoh])
                counter+=1
        except:
            for neuron_id,(blv,labelsoh) in enumerate(bl[0],neuron_labels):
                # list of name, degree, score
                dict={'subject':info[2],'cage':info[1],'session':info[0],'group':info[3],'neuron':neuron_id,'baseline':blv,'reward':np.nan,'tmt':np.nan,'posttmt':np.nan,'classification':labelsoh}
                dfoh=pd.DataFrame(dict,index=[0])
                if counter==0:
                    DF=dfoh
                else:
                    DF=pd.concat([DF,dfoh])
                counter+=1

    ipdb.set_trace()
    # Save tall format dataframe to csv file in root_directory
    DF.to_csv(os.path.join(root_directory,filename), index=False)  

class corralative_activity(corralative_activity):
    def threshold_neurons(self):
        print('MLP file loaded')

class pipeline(pipeline):

    def create_object_from_path(self,paths):
        # check and see if obj file already exists
        imagepath,behpath=paths

        if os.path.isfile(os.path.join(imagepath,'objfile.json')):
            # Load previously made object
            s2p_obj=LoadObj(os.path.join(imagepath,'objfile.json'))
            print('Quick loaded object')

        elif self.skip_new_dirs:
            s2p_obj=None

        else:
            # Create two photon data object and save data
            so_obj = load_serial_output(behpath)
            last_trial = so_obj()

            # Get twophon data object
            s2p_obj = corralative_activity(datapath=imagepath,serialoutput_object=so_obj)
            s2p_obj()
            s2p_obj.get_euclidian_distance()

            SaveObj(FullPath=os.path.join(s2p_obj.datapath,'objfile.json'), s2p_obj_input=s2p_obj)

        return s2p_obj

    def run_parrallel_creation(self):
        # Run process_directory in parallel across directories
        self.recordings = Parallel(n_jobs=self.njobs)(delayed(self.create_object_from_path)(paths) for paths in self.final_list)

            # Collect the results back into the list in the main process
            # self.state_distances.extend(results.state_distances)

    def main(self):
        self.recordings=[]
        self.state_distances=[]
        for i,(imagepath,behpath) in tqdm.tqdm(enumerate(self.final_list), total=len(self.final_list), desc='Current Recording: '):
            try:
                if i==0:
                    continue
                #Get behavior data object
                self.so_obj = load_serial_output(behpath)
                last_trial = self.so_obj()

                # Get twophon data object
                self.s2p_obj = corralative_activity(datapath=imagepath,serialoutput_object=self.so_obj)
                self.s2p_obj()
                self.s2p_obj.get_euclidian_distance()

                SaveObj(FullPath=os.path.join(self.s2p_obj.datapath,'objfile.json'), s2p_obj_input=self.s2p_obj)
                self.state_distances.append(self.s2p_obj.state_distances)

                #Append object as attribute to list
                self.recordings.append(self.s2p_obj)
            except:
                string = f'Error with loop {i}, see {imagepath} or {behpath}'
                print(string)
                
        return self.recordings
    
    def plot_state_distances(self):
        plt.figure(figsize=(10,10),dpi=300)
        for moh in self.state_distances:
            plt.plot(np.asarray(moh))

        plt.savefig('State_distances.jpg')
        aroh = np.asarray(self.state_distances)
        np.save('state_distances.npy',aroh)

class alternative_pipeline(pipeline):
    """ Similar pipeline as above, however, written for reorganized file structure on cluster """
    def __init__(self,base_directory,njobs,skip_new_dirs=False): 
        self.base_directory=base_directory
        self.recordings=[]
        self.state_distances=[]
        self.njobs=njobs
        self.skip_new_dirs=skip_new_dirs

    def find_folders_and_files(self,base_directory):
        # Get all 2P directories
        pattern = re.compile(r'R\d+')
        twop_folders = []

        for root, dirs, files in os.walk(base_directory):
            for dir_name in dirs:
                if pattern.search(dir_name):
                    twop_folders.append(os.path.join(root, dir_name))

        # Get all sync and sens files full paths
        sync_files = glob.glob(os.path.join(base_directory,r'**\*sync*'),recursive=True)
        sens_files = glob.glob(os.path.join(base_directory,r'**\*sens*'),recursive=True)

        # Get all videos and seperate into different behaviors
        all_vids = [glob.glob(os.path.join(base_directory,pat),recursive=True) for pat in (r'**\*.mp4',r'**\*.avi')]
        all_vids = [item for sublist in all_vids for item in sublist]
        head_fixed_front_vids = [vid for vid in all_vids if 'front' in vid]
        head_fixed_side_vids = [vid for vid in all_vids if 'side' in vid]
        tst_vids = [vid for vid in all_vids if 'TST' in vid]
        openfield_vids = [vid for vid in all_vids if 'openfield' in vid]

        return twop_folders, sync_files, sens_files, head_fixed_front_vids, head_fixed_side_vids, tst_vids, openfield_vids

    def match_directories(self,twop_folders):
        # Find and match all 2P image folders with corresponding serial output folders
        self.final_list=[]
        for diroh in twop_folders:
            serialoutput_oh = os.path.dirname(diroh)
            self.final_list.append([diroh,serialoutput_oh])
    
    def __call__(self):
        self.all_dirs = self.find_folders_and_files(self.base_directory)
        self.match_directories(self.all_dirs[0])
        self.run_parrallel_creation() # use to be main

def delete_2p_obj_files(input_directory):
    """Delete a file after single user confirmation."""
    if not os.path.exists(input_directory):
        raise(f"Error: this path  '{input_directory}' does not exist.")
    
    # erify the user wants to delete all s2p objects
    response = input(f"Are you sure you want to delete all Sweet2Plus objects in '{input_directory}'? (yes/no): ").strip().lower()
    if response in {'yes', 'y'}:
        obj_files_in_dir=glob.glob(os.path.join(input_directory,'**\objfile.json*'),recursive=True)
        for obj_file_oh in obj_files_in_dir:
            os.remove(obj_file_oh)
            print(f"'{obj_file_oh}' has been deleted.")
        
    else:
        print(f'You have decided NOT to delete all Sweet2Plus files in {input_directory}')

def all_subjects_pipeline(cli_args,output_file='info_correlation_data.pkl'):
    """ Based on cli arguments, will run all subjects through pipeline. """
    # Determine if final pkl file already created
    if os.path.isfile(os.path.join(cli_args.data_directory,output_file)):
        data = OpenList(FullPath=os.path.join(cli_args.data_directory,output_file))

    else:
        # Create all data object containing correlative activity data + other attributres
        alldata=alternative_pipeline(base_directory=cli_args.data_directory,njobs=cli_args.njobs,skip_new_dirs=cli_args.skip_new_dirs)
        alldata()
        alldata.plot_state_distances()

        # Run and graph statistics for all correlation data. 
        data = run_parallel_correlations(alldata)

        # Save data to pickled file
        SaveList(FullPath=os.path.join(cli_args.data_directory,output_file),complicated_list=data)

    # Seperate data into counter parts and generate tall dataset
    parse_info_oh=[inf for inf, dat in data]
    correlation_data_oh=[dat for inf, dat in data]
    generate_tall_dataset(parse_info=parse_info_oh,correlation_data=correlation_data_oh,root_directory=cli_args.data_directory)

if __name__=='__main__':
    # Set up command line argument parser
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,required=False,help='A parent path containing all of the two-data of interest')
    parser.add_argument('--beh_directory',type=str,required=False,help='A parent path containing all of the two-data of interest')
    parser.add_argument('--njobs',type=int,required=False,help='A parent path containing all of the two-data of interest')
    parser.add_argument('--single_subject_flag',action='store_true',help='run a single folder containing subject data')
    parser.add_argument('--force_redo',action='store_true',help='run a single folder containing subject data')
    parser.add_argument('--skip_new_dirs',action='store_true',help='When added to command line, this will skip over directories that do not have created obj file')
    parser.add_argument('--delete_all_obj_files',action='store_true',help='Deletes all obj files')
    args=parser.parse_args()

    if args.delete_all_obj_files:
        delete_2p_obj_files(input_directory=args.data_directory)

    # Run a single subject's data through pipeline 
    if args.single_subject_flag:
        if args.force_redo: #If True, automatically re-create the objects and save
            # Create a serial output object using provided behavior directory
            so_obj = load_serial_output(args.beh_directory)
            last_trial = so_obj()

            # Get twophon data object
            s2p_obj = corralative_activity(datapath=args.data_directory,serialoutput_object=so_obj)
            s2p_obj()
            s2p_obj.get_euclidian_distance()

            # Save data to json file for quick loading in future. 
            SaveObj(FullPath=os.path.join(s2p_obj.datapath,'objfile.json'), s2p_obj_input=s2p_obj)

        else:  # Otherwise, check and see if the obj file exists. 
            if os.path.isfile(os.path.join(args.data_directory,'objfile.json')):
                # Load previously made object
                s2p_obj=LoadObj(os.path.join(args.data_directory,'objfile.json'))
                print('Quick loaded object')
            
            else:
                # Create a serial output object using provided behavior directory
                so_obj = load_serial_output(args.beh_directory)
                last_trial = so_obj()

                # Get twophon data object
                s2p_obj = corralative_activity(datapath=args.data_directory,serialoutput_object=so_obj)
                s2p_obj()
                s2p_obj.get_euclidian_distance()

                # Save data to json file for quick loading in future. 
                SaveObj(FullPath=os.path.join(s2p_obj.datapath,'objfile.json'), s2p_obj_input=s2p_obj)

    # Run all subjects through pipeline
    else:
        all_subjects_pipeline(cli_args=args)