from SaveLoadObjs import LoadObj
import glob,os
import ipdb
from itertools import combinations
import numpy as np
import pandas as pd

def build_svm_data(obj,max_neurons):
    # For each trial type, get auc of activity for X frames after. 
    all_data=[]
    for trial_data,trial_index in zip(obj.all_evts_imagetime,[0,1,2,3]):
        for trial_oh in trial_data:
            traces_oh=obj.ztraces[:,int(trial_oh):int(trial_oh+5)]
            auc_oh=np.trapz(traces_oh,axis=1)
            auc_mean=np.mean(auc_oh)
            output=np.repeat(auc_mean,max_neurons)
            output[:auc_oh.shape[0]]=auc_oh
            all_data.append([output,trial_index])      

    return all_data


if __name__=='__main__':
    objs = glob.glob(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\**\**\**\*objfile.json*')

    # Calculate the maximumn number of neurons in an animal
    max_neurons=0
    for ln,ooh_str in enumerate(objs):
        ooh=LoadObj(ooh_str)
        if ooh.ztraces.shape[0]>max_neurons:
            max_neurons=ooh.ztraces.shape[0]

    # Generate a dataset for the svm
    y=[]
    count=0
    for ln,ooh_str in enumerate(objs):
        ooh=LoadObj(ooh_str)
        all_data=build_svm_data(ooh,max_neurons)
        for k in all_data:
            auc,trial_name=k
            y.append(trial_name)
            if count==0:
                X=auc
            else:
                X=np.vstack((X,auc))
            count+=1
    y=np.asarray(y)
    ipdb.set_trace()