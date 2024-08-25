from SaveLoadObjs import LoadObj
import glob,os
import ipdb
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    np.save('X.npy',X)
    np.save('y.npy',y)
    ipdb.set_trace()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True,stratifpy=y)
    # clf = svm.SVC()
    # clf.fit(X_train, y_train)
    # predicted = clf.predict(X_train)
    # accuracy_score(y_train,predicted)
    # predicted = clf.predict(X_test)
    # accuracy_score(y_test,predicted)
    ipdb.set_trace()