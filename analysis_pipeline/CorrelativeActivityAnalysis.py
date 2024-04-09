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
"""
from twophoton_pipeline_fullstack import pipeline 
import numpy as np

alldata=pipeline(r'C:\Users\listo\tmtassay\TMTAssay\Day1\serialoutput\**\*24*',r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\**\*24*')

def baseline_correlations(primary_obj):
    """ Compare correlation across each neuron in each mouse across times """
    parse_info=[] # Empty list to put animal info data (cage #, mouse # etc)
    correlation_data=[] #Empty list to put correlation data into
    for subjectnumber in range(len(primary_obj.recordings)):    #Loop over subjects
        start_time = primary_obj.recordings[subjectnumber].so.all_evts_imagetime[0] #Get the first trial time. Baseline activity is everything preceding
        ztracesoh=np.copy(primary_obj.recordings[subjectnumber].ztraces) #Make a copy of the trace data
        ztracesoh=ztracesoh[:start_time,:] #Crop trace data 0 --> start time

        corr_ed, correlations = primary_obj.recordings[subjectnumber].get_activity_correlation(ztracesoh) #Calculate correlation data
        info = [primary_obj.recordings[subjectnumber].day,primary_obj.recordings[subjectnumber].cage,primary_obj.recordings[subjectnumber].mouse] #Get info data

        #Append all data to lists
        parse_info.append(info)
        correlation_data.append([corr_ed,correlations])

 # Look at correlation of activity during baseline
        # Look at correlation of activity during US
        # Look at correlation of activity during post-TMT
        # Get PETHS and classify neurons by activity
        # Look at each of above correlations with respect to functional classification of neurons 

