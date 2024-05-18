from SaveLoadObjs import LoadObj
import glob,os
import ipdb
from itertools import combinations
import numpy as np
import pandas as pd

def stripinfo(path):
    _,_,_,_,_,day,_,name=path.split('\\')
    _,cage,mouse,_=name.split('_')
    return day,cage,mouse

def get_activity_correlation(data):
    data=np.asarray(data)
    data=data.T
    data=pd.DataFrame(data)
    correlations=data.corr(method='pearson')
    corr_ed=np.copy(correlations)
    corr_ed[corr_ed==1]=np.nan 
    return corr_ed, correlations

def correlations(objoh):
    """ Compare correlation across each neuron in each mouse across times """
    start_time = objoh.all_evts_imagetime[2][0] #Get the first trial time. Baseline activity is everything preceding
    try:
        tmt_start = objoh.all_evts_imagetime[3][0] #Get the first trial time. Baseline activity is everything preceding
        tmt_end = objoh.all_evts_imagetime[3][4] 

        # Parse traces
        ztracesoh=np.copy(objoh.ztraces) #Make a copy of the trace data
        baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time
        rewardztracesoh=ztracesoh[:,int(start_time):int(tmt_start)] 
        tmtztracesoh=ztracesoh[:,int(tmt_start):int(tmt_end)] 
        posttmttracesoh=ztracesoh[:,int(tmt_end):] 

        # Get correlations
        blcorr, correlations = get_activity_correlation(baselineztracesoh) #Calculate correlation data
        rewcorr, correlations = get_activity_correlation(rewardztracesoh) #Calculate correlation data
        tmtcorr, correlations = get_activity_correlation(tmtztracesoh) #Calculate correlation data
        posttmtcorr, correlations = get_activity_correlation(posttmttracesoh) #Calculate correlation data
        day,cage,mouse=stripinfo(objoh.datapath)
        info = [day,cage,mouse] #Get info data
    
        ## Classify whether group one or two is TMT activated
        aucsoh=np.asarray(objoh.auc_vals)
        firstzero=aucsoh[np.where(objoh.classifications==0)[0][0]]
        firstone=aucsoh[np.where(objoh.classifications==1)[0][0]]

        if firstzero[3]>firstone[3]:
            zerolabels='TMT_activated'
            onelabels='NonTMT_activated'
        else:
            zerolabels='NonTMT_activated'
            onelabels='TMT_activated'

        neuron_labels=[]
        for noh in objoh.classifications:
            if noh ==1:
                neuron_labels.append('TMT_activated')
            else:
                neuron_labels.append('NotTMT_activated')

    except:
        # Parse traces
        ztracesoh=np.copy(objoh.ztraces) #Make a copy of the trace data
        baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time

        # Get correlations
        blcorr, correlations = get_activity_correlation(baselineztracesoh) #Calculate correlation data
        rewcorr,tmtcorr,posttmtcorr,neuron_labels=np.nan,np.nan,np.nan,'NA'
        day,cage,mouse=stripinfo(objoh.datapath)
        info = [day,cage,mouse] #Get info data

    return info,blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels

def sep_correlations(objoh):
    """ Compare correlation across each neuron in each mouse across times """
    start_time = objoh.all_evts_imagetime[2][0] #Get the first trial time. Baseline activity is everything preceding
    try:
        ## Classify whether group one or two is TMT activated
        aucsoh=np.asarray(objoh.auc_vals)
        firstzero=aucsoh[np.where(objoh.classifications==0)[0][0]]
        firstone=aucsoh[np.where(objoh.classifications==1)[0][0]]

        if firstzero[3]>firstone[3]:
            zerolabels=0
            onelabels=1
        else:
            zerolabels=1
            onelabels=0

        neuron_labels=[]
        for noh in objoh.classifications:
            if noh ==1:
                neuron_labels.append(onelabels)
            else:
                neuron_labels.append(zerolabels)

        neuron_labels=np.asarray(neuron_labels)

        ipdb.set_trace()
        tmt_start = objoh.all_evts_imagetime[3][0] #Get the first trial time. Baseline activity is everything preceding
        tmt_end = objoh.all_evts_imagetime[3][4] 

        # Parse traces
        ztracesoh=np.copy(objoh.ztraces) #Make a copy of the trace data



        baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time
        rewardztracesoh=ztracesoh[:,int(start_time):int(tmt_start)] 
        tmtztracesoh=ztracesoh[:,int(tmt_start):int(tmt_end)] 
        posttmttracesoh=ztracesoh[:,int(tmt_end):] 

        # Get correlations
        blcorr, correlations = get_activity_correlation(baselineztracesoh) #Calculate correlation data
        rewcorr, correlations = get_activity_correlation(rewardztracesoh) #Calculate correlation data
        tmtcorr, correlations = get_activity_correlation(tmtztracesoh) #Calculate correlation data
        posttmtcorr, correlations = get_activity_correlation(posttmttracesoh) #Calculate correlation data
        day,cage,mouse=stripinfo(objoh.datapath)
        info = [day,cage,mouse] #Get info data
    
       
    except:
        # Parse traces
        ztracesoh=np.copy(objoh.ztraces) #Make a copy of the trace data
        baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time

        # Get correlations
        blcorr, correlations = get_activity_correlation(baselineztracesoh) #Calculate correlation data
        rewcorr,tmtcorr,posttmtcorr,neuron_labels=np.nan,np.nan,np.nan,'NA'
        day,cage,mouse=stripinfo(objoh.datapath)
        info = [day,cage,mouse] #Get info data

    return info,blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels

def build_tall(big_list):
    for i,uid in enumerate(big_list):
        info,blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels=uid
        rewcorr=np.nanmean(rewcorr,axis=0)
        tmtcorr=np.nanmean(tmtcorr,axis=0)
        posttmtcorr=np.nanmean(posttmtcorr,axis=0)
        blcorr=np.nanmean(blcorr,axis=0)

        count=0
        try:
            for neuron_id,(blv,rewv,tmtv,postv,labelsoh) in enumerate(zip(blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels)):
                dict={'subject':info[2],'cage':info[1],'session':info[0],'neuron':neuron_id,'baseline':blv,'reward':rewv,'tmt':tmtv,'posttmt':postv,'classification':labelsoh}
                dfoh=pd.DataFrame(dict,index=[0])
                
                if count==0:
                    DF=dfoh
                else:
                    DF=pd.concat([DF,dfoh])
                count+=1
        except:
            ipdb.set_trace()

        if i==0:
            DFall=DF
        else:
            DFall=pd.concat([DFall,DF])
    return DFall

def vector_angle(obj):
    data=obj.auc_vals
    trialnames=['Water','Vanilla','PeanutButter','TMT']
    counter=0
    for trial1,trialname1 in zip(data[:-1].T,trialnames[:-1]):
        for trial2,trialname2 in zip(data[1:].T,trialnames[1:]):
            euclid_dist = np.linalg.norm(trial2-trial1)
            vector_angle = np.round(np.degrees(np.arccos(np.dot(trial1,trial2)/(np.linalg.norm(trial1)*np.linalg.norm(trial2)))))
            day,cage,mouse=stripinfo(obj.datapath)

            if trialname1=='TMT' or trialname2=='TMT':
                parser='TMT'
            else:
                parser='NonTMT'

            if counter==0:
                DFinal=pd.DataFrame({'day':day,'cage':cage,'mouse':mouse,'trialname1':trialname1,'trialname2':trialname2,'trialtype':parser,'vectorangle':vector_angle,'vectordis':euclid_dist},index=[counter])
            else:
                dfoh=pd.DataFrame({'day':day,'cage':cage,'mouse':mouse,'trialname1':trialname1,'trialname2':trialname2,'trialtype':parser,'vectorangle':vector_angle,'vectordis':euclid_dist},index=[counter])
                DFinal=pd.concat([DFinal,dfoh])
            counter+=1
    return DFinal

def get_auc_data(objoh):
    day,cage,mouse=stripinfo(objoh.datapath) # Get info data
    info = [day,cage,mouse] #Get info data
    aucsoh=np.asarray(objoh.auc_vals)
    aucsoh=aucsoh[:,:-1]
    firstzero=aucsoh[np.where(objoh.classifications==0)[0][0]]
    firstone=aucsoh[np.where(objoh.classifications==1)[0][0]]

    if firstzero[3]>firstone[3]:
        zerolabels=0
        onelabels=1
    else:
        zerolabels=1
        onelabels=0

    neuron_labels=[]
    for noh in objoh.classifications:
        if noh ==1:
            neuron_labels.append(onelabels)
        else:
            neuron_labels.append(zerolabels)
    
    return info,aucsoh,neuron_labels

def build_auc_tall(listoh):
    # Put AUC data into tall
    for i,(info,aucsoh,neuron_labels) in enumerate(listoh):
        day,cage,mouse=info
        wt,vll,pb,tmt=aucsoh.T
        for neuron_id, (wto,vllo,pbo,tmto,labeloh) in enumerate(zip(wt,vll,pb,tmt,neuron_labels)):
            if i==0:
                DFall=pd.DataFrame({'subject':mouse,'session':day,'cage':cage,'neuron':neuron_id,"label":labeloh,'water':wto,'vanilla':vllo,'peanutbutter':pbo,'tmt':tmto},index=[0])
            else:
                DFoh=pd.DataFrame({'subject':mouse,'session':day,'cage':cage,'neuron':neuron_id,"label":labeloh,'water':wto,'vanilla':vllo,'peanutbutter':pbo,'tmt':tmto},index=[0])
                DFall=pd.concat([DFall,DFoh])
    
    return DFall


if __name__=='__main__':
    objs = glob.glob(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\**\**\**\*objfile.json*')
    auc_list=[]
    for ln,ooh_str in enumerate(objs):
        ooh=LoadObj(ooh_str)
        info,aucsoh,neuron_labels = get_auc_data(ooh)
        auc_list.append([info,aucsoh,neuron_labels])
        # info,blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels = sep_correlations(ooh)
        # Big_list.append([info,blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels])
        # if ln==0:
        #     DFinal=vector_angle(ooh)
        # else:
        #     dfoh=vector_angle(ooh)
        #     DFinal=pd.concat([DFinal,dfoh])
    
    DFaucfinal=build_auc_tall(auc_list)
    DFaucfinal.to_csv('auc_tall.csv',index=False)
    #DFinal.to_csv('vector_statedata.csv',index=False)
    #DFinal2=build_tall(Big_list)
    #DFinal2.to_csv('correlation_by_group.csv',index=False)
    ipdb.set_trace()

