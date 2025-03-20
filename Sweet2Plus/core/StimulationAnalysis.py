"""
DRN analysis
(1) AUC between each stim. (How long for ITI, get entire ITI. )
    During the stims. --> look into transform 
    compare to baseline period before
    compare to post stim period after
    normalize by time. 
    
(2) Correlation between neurons. 
    pre, during, between and post.
"""
from behavior import load_serial_output
from Sweet2Plus.core.core import pipeline,corralative_activity
import glob,os
import numpy as np
from Sweet2Plus.core.quickgui import quickGUI
import ipdb
import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# Rewrite how behavioral data is loaded
class load_serial_output(load_serial_output):
    def load_data(self):
        file = open(self.path, "r") # Read file
        content = file.read().splitlines() #Separate into correct shape
        alldata=[] #Generate empty list
            
        for line in content: #Go through each line and filter
            try: #If line does not fit specific shape, throws an error. 
                line = line.split(',')
                line = np.asarray(line)
                line = line.astype(float)
                if line.shape[0]!=3: #Hard coded shape in, remove later
                    continue
                alldata.append(line) #Append to empty list
            except:
                continue
            
        alldata=np.asarray(alldata) #Reshape list into numpy array
        self.sync=alldata
 
    def quick_timestamps(self):
        self.starts=[]
        self.finishes=[]
        for k,(i,j) in enumerate(zip(self.sync[:-1,2],self.sync[1:,2])):
            if i==0 and j==1:
                self.starts.append(self.sync[k,0])
            elif i==1 and j==0:
                self.finishes.append(self.sync[k,0])
        self.all_evts_imagetime=[]
        self.all_evts_imagetime.append(self.starts)
        self.all_evts_imagetime.append(self.finishes)

    def __call__(self):
        self.get_file_name()
        self.load_data()
        self.quick_timestamps()


# Rewrite how auc is calculated
class corralative_activity(corralative_activity):
    def __init__(self,datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7):
        super().__init__(datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=cellthreshold)
        self.trial_list = ['LasterON','LasterOFF']
        self.auc_period=10
    
    def get_event_aucs(self,window):
        """ Calculate the AUC for each trial """
        #Set up window 
        window=window*self.ops['fs']
        dataoh = np.copy(self.ztraces_copy) # Copy the numpy array of z traces
        all_aucs=[] #Create list to put results in
        for trial_type in self.so.all_evts_imagetime:
            current_trial=[]
            for trace in dataoh:
                neuron_AUCs=[]

                #Loop over every trial
                for trial in trial_type:
                    auc_oh = np.trapz(trace[int(trial):int(round(trial+window))])
                    neuron_AUCs.append(auc_oh)

                #Average accross trials
                neuron_AUCs=np.asarray(neuron_AUCs)
                neuron_AUCs=neuron_AUCs.mean(axis=0)
                current_trial.append(neuron_AUCs) #Append Average AUC for that trial per neuron
            all_aucs.append(current_trial)
        
        neuron_number = len(all_aucs[0])
        all_values=[]
        for neuron in range(neuron_number):
            # Get AUC data for that neuron
            LaserONAUC=all_aucs[0][neuron]
            LaserOFFAUC=all_aucs[1][neuron]
            values=[LaserONAUC,LaserOFFAUC]
            all_values.append(values)
        
        self.auc_vals = all_values

    def animal_information(self):
            _,_,_,_,self.cage,self.mouse,_,_,_,_,_ = self.datapath.split('_')
            self.day=1

class pipeline(pipeline):
    def match_directories(self):
        # Find and match all 2P image folders with corresponding serial output folders
        behdirs = glob.glob(self.serialoutput_search)
        twoPdirs = glob.glob(self.twophoton_search)
        self.final_list=[]
        for diroh in twoPdirs:
            _,_,_,_,cage,mouse,_,_,_,_,_=diroh.upper().split('_')
            for bdiroh in behdirs:
                _,_,_,cageb,mouseb,_,_ = bdiroh.upper().split('_')
                if cage==cageb and mouse==mouseb:
                    self.final_list.append([diroh,bdiroh])

    def main(self):
        self.recordings=[]
        self.correlation_data=[]
        for i,(imagepath,behpath) in tqdm.tqdm(enumerate(self.final_list), total=len(self.final_list), desc='Current Recording: '):
            #Get behavior data object
            self.so_obj = load_serial_output(behpath)
            self.so_obj()

            # Get twophon data object
            self.s2p_obj = corralative_activity(imagepath,self.so_obj,cellthreshold=0.7)
            self.s2p_obj()

            self.correlation_data.append(self.pull_correlation_data())

            self.recordings.append(self.s2p_obj)
    
    def pull_correlation_data(self):
        # Get first stim,during stim and end stim time points
        stim_start = int(np.asarray(self.s2p_obj.so.all_evts_imagetime[0]).min())
        stim_finish = int(np.asarray(self.s2p_obj.so.all_evts_imagetime[1]).max())
  
        #Parse trace data
        trace_data=np.copy(np.asarray(self.s2p_obj.ztraces))
        prestim_rec=trace_data[:,:stim_start]
        during_rec=trace_data[:,stim_start:stim_finish]
        poststim_rec=trace_data[:,stim_finish:]

        # Get normalized AUC
        prestim_rec_aucs=np.trapz(prestim_rec,axis=1)/len(prestim_rec.T)
        during_rec_aucs=np.trapz(during_rec,axis=1)/len(during_rec.T)
        poststim_rec_aucs=np.trapz(poststim_rec,axis=1)/len(poststim_rec.T)

        # Generate graphs for each correlation
        prestim_corred, _ = self.s2p_obj.get_activity_correlation(prestim_rec,'prestim_correlation.pdf')
        durring_corred, _ = self.s2p_obj.get_activity_correlation(during_rec,'duringstim_correlation.pdf')
        poststim_corred, _ = self.s2p_obj.get_activity_correlation(poststim_rec,'poststim_correlation.pdf')

        # Get average correlation per neuron and output.
        prestim_av_cor=prestim_corred.mean(axis=0)
        durring_av_cor=durring_corred.mean(axis=0)
        poststim_av_cor=poststim_corred.mean(axis=0)

        # Plot average correlation per neuron 
        xticks=['Pre stimulation', 'During stimulation', 'Post stimulation']
        averages = [prestim_av_cor.mean(),durring_av_cor.mean(),poststim_av_cor.mean()]
        errors = [prestim_av_cor.std()/np.sqrt(len(prestim_av_cor)),durring_av_cor.std()/np.sqrt(len(durring_av_cor)),poststim_av_cor.std()/np.sqrt(len(poststim_av_cor))]
        
        plt.figure()
        plt.errorbar(xticks,averages,errors)
        plt.savefig(os.path.join(self.s2p_obj.resultpath_neur,'average_intraneuron_correlation.jpg'))
        plt.close()
        return prestim_av_cor, durring_av_cor, poststim_av_cor, prestim_rec_aucs, during_rec_aucs, poststim_rec_aucs
    
    def build_dataframe(self,output_name):
        try:
            for i,subject in enumerate(self.correlation_data):
                prestim_av_cor, durring_av_cor, poststim_av_cor, prestim_rec_aucs, during_rec_aucs, poststim_rec_aucs=subject
                prestim_av_cor, durring_av_cor, poststim_av_cor, prestim_rec_aucs, during_rec_aucs, poststim_rec_aucs=pd.DataFrame(prestim_av_cor), pd.DataFrame(durring_av_cor), pd.DataFrame(poststim_av_cor), pd.DataFrame(prestim_rec_aucs), pd.DataFrame(during_rec_aucs), pd.DataFrame(poststim_rec_aucs)
                
                #Create labels
                pre,during,post=pd.DataFrame(np.tile(np.asarray('prestim'),len(prestim_av_cor))),pd.DataFrame(np.tile(np.asarray('duringstim'),len(durring_av_cor))),pd.DataFrame(np.tile(np.asarray('poststim'),len(poststim_av_cor)))
                cordf = pd.concat([prestim_av_cor,durring_av_cor,poststim_av_cor])
                labels = pd.concat([pre,during,post])

                # Re-index values
                prestim_rec_aucs=prestim_rec_aucs.reindex(prestim_av_cor.index)
                during_rec_aucs=during_rec_aucs.reindex(durring_av_cor.index)
                poststim_rec_aucs=poststim_rec_aucs.reindex(poststim_av_cor.index)

                # Subject ID, Neuron ID, Period, Correlation value, AUC value
                prestim_rec_aucs['NeuronID'] = prestim_rec_aucs.index
                during_rec_aucs['NeuronID'] = during_rec_aucs.index
                poststim_rec_aucs['NeuronID'] = poststim_rec_aucs.index
                aucs = pd.concat([prestim_rec_aucs,during_rec_aucs,poststim_rec_aucs])
                subjectID = pd.DataFrame(np.tile(np.asarray(f'{self.recordings[i].cage}{self.recordings[i].mouse}'),len(aucs)))

                # Rename column heads
                subjectID.columns=['subjectid']
                aucs.columns=['auc','neuronid']    
                labels.columns=['labels']    
                cordf.columns=['avcorr']        

                # Drop indexes that are wrong
                subjectID = subjectID.reset_index(drop=True)
                labels = labels.reset_index(drop=True)
                aucs = aucs.reset_index(drop=True)
                cordf = cordf.reset_index(drop=True)

                #Create data frame on hand
                dfoh=pd.concat([subjectID,labels,aucs,cordf],axis=1)

                if i==0:
                    dfF=dfoh
                else:
                    dfF=pd.concat([dfF,dfoh],axis=0)
            dfF.to_csv(output_name)
        except:
            ipdb.set_trace()


# Rewrite methods specific to dataset
if __name__=='__main__':
    recordings=pipeline(r'D:\2p_drn_inhibition\serialoutput\*24*\*_*',r'D:\2p_drn_inhibition\twophotonrecordings\24-2-23_deepcad\*24*')
    recordings()
    recordings.build_dataframe(r'D:\2p_drn_inhibition\talldata.csv')


    # python ./StimulationAnalysis.py