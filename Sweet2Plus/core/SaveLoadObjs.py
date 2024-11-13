import json
import numpy as np
from Sweet2Plus.core.core import corralative_activity

def SaveObj(FullPath: str, CurrentObject):
    """ Saves custom objects to json file
    Inputs:
    FullPath -- A string containing the entire directory and filename.json where the object will be saved. 
    CurrentObject -- The object we would like to save

    Outputs:
    None .... File will be saved to specified location
    """
    import json
    import numpy as np

    #Concatenate large list
    big_list=[]

    #Append everything of interest
    big_list.append(CurrentObject.datapath)
    big_list.append(CurrentObject.resultpath)
    big_list.append(CurrentObject.recording_files)
    big_list.append(CurrentObject.probability_files)
    big_list.append(CurrentObject.stat_files)
    big_list.append(CurrentObject.neuron_prob.tolist())
    big_list.append(CurrentObject.traces.tolist())
    big_list.append(CurrentObject.images)
    big_list.append(CurrentObject.ztraces_copy.tolist())
    big_list.append(CurrentObject.ztraces.tolist())
    big_list.append(CurrentObject.resultpath_neur_traces)
    big_list.append(CurrentObject.so.all_evts_imagetime)
    big_list.append(CurrentObject.trial_list)
    big_list.append(CurrentObject.resultpath_neur)
    big_list.append(CurrentObject.first_trial_time)
    big_list.append(CurrentObject.last_trial_time)
    big_list.append(CurrentObject.baselineAUCs.tolist())
    big_list.append(np.asarray(CurrentObject.auc_vals).tolist())
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
    import json
    import numpy as np

    with open(FullPath, 'r') as file:
        big_list = json.load(file)

    
    class dataholder:
        def __init__(self):
            return     
    CurrentObject = dataholder

    #Append everything of interest
    CurrentObject.datapath=big_list[0]
    CurrentObject.resultpath=big_list[1]
    CurrentObject.recording_files=big_list[2]
    CurrentObject.probability_files=big_list[3]
    CurrentObject.stat_files=big_list[4]
    CurrentObject.neuron_prob=big_list[5]
    CurrentObject.traces=np.asarray(big_list[6])
    CurrentObject.images=big_list[7]
    CurrentObject.ztraces_copy=np.asarray(big_list[8])
    CurrentObject.ztraces=np.asarray(big_list[9])
    CurrentObject.resultpath_neur_traces=big_list[10]
    CurrentObject.all_evts_imagetime=big_list[11]
    CurrentObject.trial_list=big_list[12]
    CurrentObject.resultpath_neur=big_list[13]
    CurrentObject.first_trial_time=big_list[14]
    CurrentObject.last_trial_time=big_list[15]
    CurrentObject.baselineAUCs=np.asarray(big_list[16])
    CurrentObject.auc_vals=np.asarray(big_list[17])
    CurrentObject.classifications=np.asarray(big_list[18])

    return CurrentObject

if __name__=='__main__':
    ooh = LoadObj(r'C:\Users\listo\twophoton\summary_data\example_obj.json')