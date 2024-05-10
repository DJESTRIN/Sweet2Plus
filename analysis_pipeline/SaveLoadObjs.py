import json

def SaveObj(FullPath: str, CurrentObject):
    """ Saves custom objects to json file
    Inputs:
    FullPath -- A string containing the entire directory and filename.json where the object will be saved. 
    CurrentObject -- The object we would like to save

    Outputs:
    None .... File will be saved to specified location
    """
    import json,ipdb
    #Concatenate large list
    big_list=[]

    #Append everything of interest
    big_list.append(CurrentObject.datapath)
    big_list.append(CurrentObject.resultpath)
    big_list.append(CurrentObject.recording_files)
    big_list.append(CurrentObject.probability_files)
    big_list.append(CurrentObject.stat_files)
    big_list.append(str(CurrentObject.neuron_prob.tolist()))
    big_list.append(CurrentObject.traces.tolist())
    big_list.append(CurrentObject.images)
    big_list.append(str(CurrentObject.ztraces_copy.tolist()))
    big_list.append(str(CurrentObject.ztraces.tolist()))
    ipdb.set_trace()
    big_list.append(CurrentObject.resultpath_neur_traces)
    big_list.append(CurrentObject.so.all_evts_imagetime)
    big_list.append(CurrentObject.trial_list)
    big_list.append(CurrentObject.resultpath_neur)
    big_list.append(CurrentObject.first_trial_time)
    big_list.append(CurrentObject.last_trial_time)
    big_list.append(CurrentObject.baselineAUCs.tolist())
    big_list.append(CurrentObject.auc_vals)
    big_list.append(CurrentObject.classifications.tolist())
    
    with open(FullPath, 'w') as file:
        json.dump(big_list, file)

def LoadObj(FullPath: str):
    """ Saves custom objects to json file
    Inputs:
    FullPath -- A string containing the entire directory and filename.json where the object was saved

    Outputs:
    The object you saved in that json file
    """
    with open(FullPath, 'r') as file:
        CurrentObject = json.load(file)
    
    return CurrentObject