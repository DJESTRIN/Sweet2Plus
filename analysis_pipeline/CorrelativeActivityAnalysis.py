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
from twophoton_fullstack import pipeline,corralative_activity 
from behavior import load_serial_output
import numpy as np
import warnings
import tqdm
import pickle
from SaveLoadObjs import SaveObj
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")

""" NEED TO PLOT CORRELATIONS..
NEED TO ALSO PLOT AUC values across trial types for each neuron to get averages """

def correlations(primary_obj):
    """ Compare correlation across each neuron in each mouse across times """
    ipdb.set_trace()
    #Analyze baseline correlations across sessions
    parse_info=[] # Empty list to put animal info data (cage #, mouse # etc)
    correlation_data=[] #Empty list to put correlation data into
    for subjectnumber in range(len(primary_obj.recordings)):    #Loop over subjects
        # Get important times
        start_time = primary_obj.recordings[subjectnumber].so.all_evts_imagetime[2][0] #Get the first trial time. Baseline activity is everything preceding
        try:
            tmt_start = primary_obj.recordings[subjectnumber].so.all_evts_imagetime[3][0] #Get the first trial time. Baseline activity is everything preceding
            tmt_end = primary_obj.recordings[subjectnumber].so.all_evts_imagetime[3][4] 

            # Parse traces
            ztracesoh=np.copy(primary_obj.recordings[subjectnumber].ztraces) #Make a copy of the trace data
            baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time
            rewardztracesoh=ztracesoh[:,int(start_time):int(tmt_start)] 
            tmtztracesoh=ztracesoh[:,int(tmt_start):int(tmt_end)] 
            posttmttracesoh=ztracesoh[:,int(tmt_end):] 

            # Get correlations
            blcorr, correlations = primary_obj.recordings[subjectnumber].get_activity_correlation(baselineztracesoh,output_filename='baseline_correlation.pdf') #Calculate correlation data
            rewcorr, correlations = primary_obj.recordings[subjectnumber].get_activity_correlation(rewardztracesoh,output_filename='reward_correlation.pdf') #Calculate correlation data
            tmtcorr, correlations = primary_obj.recordings[subjectnumber].get_activity_correlation(tmtztracesoh,output_filename='tmt_correlation.pdf') #Calculate correlation data
            posttmtcorr, correlations = primary_obj.recordings[subjectnumber].get_activity_correlation(posttmttracesoh,output_filename='posttmt_correlation.pdf') #Calculate correlation data
            info = [primary_obj.recordings[subjectnumber].day,primary_obj.recordings[subjectnumber].cage,primary_obj.recordings[subjectnumber].mouse] #Get info data
        
            ## Classify whether group one or two is TMT activated
            aucsoh=np.asarray(primary_obj.recordings[subjectnumber].auc_vals)
            firstzero=aucsoh[np.where(primary_obj.recordings[subjectnumber].classifications==0)[0][0]]
            firstone=aucsoh[np.where(primary_obj.recordings[subjectnumber].classifications==1)[0][0]]

            if firstzero[3]>firstone[3]:
                zerolabels='TMT_activated'
                onelabels='NonTMT_activated'
            else:
                zerolabels='NonTMT_activated'
                onelabels='TMT_activated'

            neuron_labels=[]
            for noh in primary_obj.recordings[subjectnumber].classifications:
                if noh ==1:
                    neuron_labels.append(onelabels)
                else:
                    neuron_labels.append(zerolabels)

        except:
            # Parse traces
            ztracesoh=np.copy(primary_obj.recordings[subjectnumber].ztraces) #Make a copy of the trace data
            baselineztracesoh=ztracesoh[:,:int(start_time)] #Crop trace data 0 --> start time

            # Get correlations
            blcorr, correlations = primary_obj.recordings[subjectnumber].get_activity_correlation(baselineztracesoh) #Calculate correlation data
            rewcorr,tmtcorr,posttmtcorr=np.nan,np.nan,np.nan
            info = [primary_obj.recordings[subjectnumber].day,primary_obj.recordings[subjectnumber].cage,primary_obj.recordings[subjectnumber].mouse] #Get info data

        #Append all data to lists
        parse_info.append(info)
        correlation_data.append([blcorr,rewcorr,tmtcorr,posttmtcorr,neuron_labels])

    av_corrs_data=[]
    for uid in correlation_data:
        av_corrs_data.append([[np.nanmean(uid[0],axis=0)],[np.nanmean(uid[1],axis=0)],[np.nanmean(uid[2],axis=0)],[np.nanmean(uid[3],axis=0)],uid[4]])

    #Build tall dataset
    counter=0
    for info,data in zip(parse_info,av_corrs_data):
        (bl,rew,tmt,post,neuron_labels)=data
        try:
            for neuron_id,(blv,rewv,tmtv,postv,labelsoh) in enumerate(zip(bl[0],rew[0],tmt[0],post[0],neuron_labels)):
                # list of name, degree, score
                dict={'subject':info[2],'cage':info[1],'session':info[0],'neuron':neuron_id,'baseline':blv,'reward':rewv,'tmt':tmtv,'posttmt':postv,'classification':labelsoh}
                dfoh=pd.DataFrame(dict,index=[0])
                if counter==0:
                    DF=dfoh
                else:
                    DF=pd.concat([DF,dfoh])
                counter+=1
        except:
            for neuron_id,(blv,labelsoh) in enumerate(bl[0],neuron_labels):
                # list of name, degree, score
                dict={'subject':info[2],'cage':info[1],'session':info[0],'neuron':neuron_id,'baseline':blv,'reward':np.nan,'tmt':np.nan,'posttmt':np.nan,'classification':labelsoh}
                dfoh=pd.DataFrame(dict,index=[0])
                if counter==0:
                    DF=dfoh
                else:
                    DF=pd.concat([DF,dfoh])
                counter+=1

    DF.to_csv('repeatedmeasures_correlations_all.csv', index=False)  
    ipdb.set_trace()
    # Get PETHS and classify neurons by activity
    # Look at each of above correlations with respect to functional classification of neurons 

class corralative_activity(corralative_activity):
    def threshold_neurons(self):
        print('MLP file loaded')

class pipeline(pipeline):
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
                
                ipdb.set_trace()
                SaveObj(FullPath='C:\tmt_assay\object.json', CurrentObject=self.s2p_obj)
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

if __name__=='__main__':
    alldata=pipeline(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\serialoutputdata\Day**\**\*24*' , r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day**\**\*24*')
    alldata()
    alldata.plot_state_distances()
    correlations(alldata)