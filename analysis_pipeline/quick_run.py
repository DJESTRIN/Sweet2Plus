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

        tmt_start = objoh.all_evts_imagetime[3][0] #Get the first trial time. Baseline activity is everything preceding
        tmt_end = objoh.all_evts_imagetime[3][4] 

        # Parse traces
        ztracesoh=np.copy(objoh.ztraces) #Make a copy of the trace data

        # Threat activated neurons
        threat_neurons=ztracesoh[np.where(neuron_labels==0)[0],:]
        baselineztracesoh=threat_neurons[:,:int(start_time)] #Crop trace data 0 --> start time
        rewardztracesoh=threat_neurons[:,int(start_time):int(tmt_start)] 
        tmtztracesoh=threat_neurons[:,int(tmt_start):int(tmt_end)] 
        posttmttracesoh=threat_neurons[:,int(tmt_end):] 
        threat_blcorr, correlations = get_activity_correlation(baselineztracesoh) #Calculate correlation data
        threat_rewcorr, correlations = get_activity_correlation(rewardztracesoh) #Calculate correlation data
        threat_tmtcorr, correlations = get_activity_correlation(tmtztracesoh) #Calculate correlation data
        threat_posttmtcorr, correlations = get_activity_correlation(posttmttracesoh) #Calculate correlation data

        # Non-hreat activated neurons
        nonthreat_neurons=ztracesoh[np.where(neuron_labels==1)[0],:]
        baselineztracesoh=nonthreat_neurons[:,:int(start_time)] #Crop trace data 0 --> start time
        rewardztracesoh=nonthreat_neurons[:,int(start_time):int(tmt_start)] 
        tmtztracesoh=nonthreat_neurons[:,int(tmt_start):int(tmt_end)] 
        posttmttracesoh=nonthreat_neurons[:,int(tmt_end):] 
        nonthreat_blcorr, correlations = get_activity_correlation(baselineztracesoh) #Calculate correlation data
        nonthreat_rewcorr, correlations = get_activity_correlation(rewardztracesoh) #Calculate correlation data
        nonthreat_tmtcorr, correlations = get_activity_correlation(tmtztracesoh) #Calculate correlation data
        nonthreat_posttmtcorr, correlations = get_activity_correlation(posttmttracesoh) #Calculate correlation data


        day,cage,mouse=stripinfo(objoh.datapath)
        info = [day,cage,mouse] #Get info data
    except:
        ipdb.set_trace()
    
    return info,threat_blcorr,threat_rewcorr,threat_tmtcorr,threat_posttmtcorr,nonthreat_blcorr,nonthreat_rewcorr,nonthreat_tmtcorr,nonthreat_posttmtcorr,neuron_labels

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

def build_tall_sep(listoh):
    count=0
    for uid in listoh:
        info,threat_blcorr,threat_rewcorr,threat_tmtcorr,threat_posttmtcorr,nonthreat_blcorr,nonthreat_rewcorr,nonthreat_tmtcorr,nonthreat_posttmtcorr,neuron_labels=uid
        day,cage,mouse=info

        # Get average pearson correlation 
        threat_blcorr=np.nanmean(threat_blcorr,axis=1)
        threat_rewcorr=np.nanmean(threat_rewcorr,axis=1)
        threat_tmtcorr=np.nanmean(threat_tmtcorr,axis=1)
        threat_posttmtcorr=np.nanmean(threat_posttmtcorr,axis=1)
        nonthreat_blcorr=np.nanmean(nonthreat_blcorr,axis=1)
        nonthreat_rewcorr=np.nanmean(nonthreat_rewcorr,axis=1)
        nonthreat_tmtcorr=np.nanmean(nonthreat_tmtcorr,axis=1)
        nonthreat_posttmtcorr=np.nanmean(nonthreat_posttmtcorr,axis=1)

        for neuron_id, (bl,rew,tmt,post) in enumerate(zip(threat_blcorr,threat_rewcorr,threat_tmtcorr,threat_posttmtcorr)):
            if count==0:
                DFall=pd.DataFrame({'subject':mouse,'session':day,'cage':cage,'neuron':neuron_id,'label':'ThreatActivated','baseline':bl,'reward':rew,'tmt':tmt,'posttmt':post},index=[0])
            else:
                DFoh=pd.DataFrame({'subject':mouse,'session':day,'cage':cage,'neuron':neuron_id,'label':'ThreatActivated','baseline':bl,'reward':rew,'tmt':tmt,'posttmt':post},index=[0])
                DFall=pd.concat([DFall,DFoh])
            count+=1

        for neuron_id, (bl,rew,tmt,post) in enumerate(zip(nonthreat_blcorr,nonthreat_rewcorr,nonthreat_tmtcorr,nonthreat_posttmtcorr)):
            if count==0:
                DFall=pd.DataFrame({'subject':mouse,'session':day,'cage':cage,'neuron':neuron_id,'label':'NonThreatActivated','baseline':bl,'reward':rew,'tmt':tmt,'posttmt':post},index=[0])
            else:
                DFoh=pd.DataFrame({'subject':mouse,'session':day,'cage':cage,'neuron':neuron_id,'label':'NonThreatActivated','baseline':bl,'reward':rew,'tmt':tmt,'posttmt':post},index=[0])
                DFall=pd.concat([DFall,DFoh])
            count+=1
        
    return DFall

if __name__=='__main__':
    objs = glob.glob(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\**\**\**\*objfile.json*')
    auc_list=[]
    parsed_list=[]
    for ln,ooh_str in enumerate(objs):
        ooh=LoadObj(ooh_str)
        info,aucsoh,neuron_labels = get_auc_data(ooh)
        auc_list.append([info,aucsoh,neuron_labels])
        info,threat_blcorr,threat_rewcorr,threat_tmtcorr,threat_posttmtcorr,nonthreat_blcorr,nonthreat_rewcorr,nonthreat_tmtcorr,nonthreat_posttmtcorr,neuron_labels = sep_correlations(ooh)
        parsed_list.append([info,threat_blcorr,threat_rewcorr,threat_tmtcorr,threat_posttmtcorr,nonthreat_blcorr,nonthreat_rewcorr,nonthreat_tmtcorr,nonthreat_posttmtcorr,neuron_labels])
        
        # Big_list.append([info,blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels])
        # if ln==0:
        #     DFinal=vector_angle(ooh)
        # else:
        #     dfoh=vector_angle(ooh)
        #     DFinal=pd.concat([DFinal,dfoh])
    
    #DFaucfinal=build_auc_tall(auc_list)
    #DFaucfinal.to_csv('auc_tall.csv',index=False)
    #DFinal.to_csv('vector_statedata.csv',index=False)
    #DFinal2=build_tall(Big_list)
    #DFinal2.to_csv('correlation_by_group.csv',index=False)
    DFinal_sep=build_tall_sep(parsed_list)
    ipdb.set_trace()

