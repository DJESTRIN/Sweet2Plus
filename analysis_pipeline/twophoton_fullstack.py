import numpy as np
import suite2p as s2p
import matplotlib.pyplot as plt 
import ipdb
import os,glob
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
import seaborn as sns
from behavior import load_serial_output
from radargraphs import radar_plot
from customs2p import get_s2p,manual_classification
from quickgui import quickGUI
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from SaveLoadObj import SaveObj,LoadObj
import tqdm
sns.set_style('whitegrid')

class parse_s2p(manual_classification):
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=cellthreshold) #Use initialization from previous class 
        # Threshold to determine whether a cell is a cell. 0.7 means only the top 30% of ROIS make it to real dataset as neurons.
        self.trial_list = ['Vanilla','PeanutButter','Water','FoxUrine']

    def __call__(self):
        super().__call__()
        ipdb.set_trace()
        self.parallel_zscore()
        ipdb.set_trace()
        self.plot_all_neurons('Frames','Z-Score + i')
        ipdb.set_trace()
        self.plot_neurons('Frames','Z-Score')
        ipdb.set_trace()
        
    def zscore_trace(self,trace):
        """ Using a sliding window, trace is zscored. 
        The sliding window is offset each iteration of the loop
        This removes any artifacts created by z score. """

        ztrace=[]
        window_width=500
        for rvalue in range(window_width):
            start=rvalue
            stop=rvalue+window_width
            zscored_trace=[]
            for i in range(round((len(trace)/(stop-start))+1)):
                if start>0 and i==0:
                    window=trace[0:start]
                    window=(window-np.mean(window))/np.std(window) #Zscrore winow
                    zscored_trace.append(window)

                if stop>len(trace):
                    window=trace[start:]
                    window=(window-np.mean(window))/np.std(window) #Zscrore winow
                    zscored_trace.append(window)
                    break

                window=trace[start:stop]
                window=(window-np.mean(window))/np.std(window) #Zscrore winow
                start+=window_width
                stop+=window_width
                zscored_trace.append(window)
            
            for i,window in enumerate(zscored_trace):
                if i==0:
                    zscored_trace=window
                else:
                    zscored_trace=np.concatenate((zscored_trace,np.asarray(window)),axis=0)
                
            ztrace.append(zscored_trace)

        ztrace=np.asarray(ztrace)
        ztrace=np.nanmedian(ztrace,axis=0)
        return ztrace
    
    def parallel_zscore(self):
        with Pool() as P:
            self.ztraces = P.map(self.zscore_trace,self.traces)
            self.ztraces = np.asarray(self.ztraces)
            self.ztraces = self.ztraces[~np.isnan(self.ztraces).any(axis=1)]
            self.ztraces_copy=np.copy(self.ztraces)

    def plot_all_neurons(self,x_label,y_label):
        # Plot neuron traces and save them without opening
        dataoh = np.copy(self.ztraces)
        fig,ax=plt.subplots(dpi=1200)
        fig.set_figheight(100)
        fig.set_figwidth(15)
        addit=0
        for i,row in enumerate(dataoh):
            row+=addit
            plt.plot(row)
            addit=np.nanmax(row)

        plt.title('All Neuronal traces')
        ax.set_ylabel(x_label)
        ax.set_ylabel(y_label)

        file_string=os.path.join(self.resultpath_neur,'all_neurons.pdf')
        plt.savefig(file_string)
        plt.close()
        return

    def plot_neurons(self,x_label,y_label):   
        # Set up folder to drop traces
        self.resultpath_neur_traces = os.path.join(self.resultpath_neur,'traces')
        if not os.path.exists(self.resultpath_neur_traces):
            os.mkdir(self.resultpath_neur_traces)

        # Plot neuron traces and save them without opening
        for i,row in enumerate(self.ztraces):
            fig,ax=plt.subplots(dpi=1200)
            plt.plot(row)
            file_string=os.path.join(self.resultpath_neur,f'trace{i}.pdf')
            plt.title(file_string)
            ax.set_ylabel(x_label)
            ax.set_ylabel(y_label)
            plt.savefig(file_string)
            plt.close()

class funcational_classification(parse_s2p):
    def __init__(self,datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=cellthreshold)
        self.so=serialoutput_object #Pass in serial_output_object
        self.auc_period=30

    def __call__(self):
        super().__call__()
        #Create Peths per trial type
        
        self.plot_all_neurons_with_trials('Frames','Z-Score DF',self.trial_list)
        self.get_baseline_AUCs()
        self.get_event_aucs(self.auc_period)
        #self.kmeans_clustering()
        for i,trial_name in enumerate(self.trial_list):
            if not self.so.all_evts_imagetime[i]:
                print(f'There are no {trial_name} trials for this subject: Cage {self.cage} mouse {self.mouse}')
            else:
                self.PETH(self.ztraces_copy,self.so.all_evts_imagetime[i],15,[-10,-5],[0,5],trial_name)

    def plot_all_neurons_with_trials(self,x_label,y_label,trial_list):
        # Create a plot of all neurons with vertical lines per trial type
        color = ['Blue','Green','Black','Red']
        min_trial=self.so.all_evts_imagetime[0][0]
        max_trial=self.so.all_evts_imagetime[0][0]
        for trial_type,col,trial_name in zip(self.so.all_evts_imagetime,color,trial_list):
            try:
                copied_data = np.copy(self.ztraces_copy)
                # Create the figure
                fig,ax=plt.subplots(dpi=1200)
                fig.set_figheight(100)
                fig.set_figwidth(15)
                addit=0

                #Plot each neuron's z-scored trace
                for i,row in enumerate(copied_data):
                    row+=addit
                    plt.plot(row)
                    addit=np.nanmax(row)

                #Plot all vertical lines for the current trial of interest
                for trial in trial_type:
                    if trial<min_trial:
                        min_trial=trial
                    if trial>max_trial:
                        max_trial=trial
                    plt.axvline(x=trial, color=col,ls='--')

                plt.title(f'All Neuronal traces with {trial_name} ticks')
                ax.set_ylabel(x_label)
                ax.set_ylabel(y_label)
                ax = plt.gca()
                ax.set_ylim([-5, addit])
                #ax.set_xlim([min_trial, max_trial])

                file_string=os.path.join(self.resultpath_neur,f'all_neurons_with_{trial_name}trials.pdf')
                plt.savefig(file_string)
                plt.close()
            except:
                continue
                #print('skipped')
        
        #Save the first and last trial times
        self.first_trial_time=min_trial
        self.last_trial_time=max_trial

    def get_baseline_AUCs(self,zeroed=False):
        """ Calculate and graph the Area Under the Curve of periods of interest
        Inputs:
        zeroed -- boolean (default: False) This will zeror the AUC data with respect to the baseline period. 

        Outputs:
        Creates a new attribute containing AUC data across periods
        Plots AUC data and saves into subject's folder
        """
        dataoh = np.copy(self.ztraces)
        self.baselineAUCs=[]
        plt.figure()
        for trace in dataoh:
            #Parse out data and integrate
            pre_period=trace[:int(self.first_trial_time)]
            preauc=np.trapz(pre_period)/len(pre_period)
            during_period=trace[int(self.first_trial_time):int(self.last_trial_time)]
            duringauc=np.trapz(during_period)/len(during_period)
            post_period=trace[int(self.last_trial_time):]
            postauc=np.trapz(post_period)/len(post_period)

            if zeroed:
                #Zero all of the data with respect to baseline
                postauc=postauc-preauc
                duringauc=duringauc-preauc
                preauc=preauc-preauc

            self.baselineAUCs.append([preauc,duringauc,postauc])

            #Add data to graph
            x=['Pre Behavior','During Behavior','Post Behavior baseline']
            plt.plot(x,[preauc,duringauc,postauc],alpha=0.3,color="black")

        #Plot average +/- SEM
        self.baselineAUCs=np.asarray(self.baselineAUCs)
        averages = self.baselineAUCs.mean(axis=0)
        stds = self.baselineAUCs.std(axis=0)
        sems = stds/self.baselineAUCs.shape[0]
        plt.errorbar(x,averages,sems,fmt='-o',color='red',markersize='3')

        file_string=os.path.join(self.resultpath_neur,f'preTMT_postTMT_AUCs.pdf')
        plt.savefig(file_string)
        plt.close()

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
            VanillaAUC=all_aucs[0][neuron]
            PbAUC=all_aucs[1][neuron]
            WaterAUC=all_aucs[2][neuron]
            FoxUrineAUC=all_aucs[3][neuron]
            #labels=['Vanilla','Peanut Butter', 'Water', 'Fox Urine']
            values=[VanillaAUC,PbAUC,WaterAUC,FoxUrineAUC]
            all_values.append(values)
        
        filename = os.path.join(self.resultpath_neur,f'AllNeuronsRadar.pdf')
        radar_plot(self.trial_list,all_values,'All Neurons',filename,single_neuron=False)
        self.auc_vals = all_values
    
    def kmeans_clustering(self):
        def get_silhouette_score(data,kmax):
            """ Citation: https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb """
            sil=[]
            kmax=10
            for k in range(2,kmax+1):
                kmeans = KMeans(n_clusters=k).fit(data)
                labels=kmeans.labels_
                sil.append(silhouette_score(data,labels,metric='euclidean'))
            
            sil = np.asarray(sil)
            final_k = np.where(sil ==sil.max())[0][0]
            return final_k + 2

        auc_array = np.asarray(self.auc_vals)
        auc_array=auc_array[:,:-1]
        auc_array=auc_array[~np.isnan(auc_array).any(axis=1)]
        final_k = get_silhouette_score(auc_array,10)
        print('This is the current K value:')
        print(f'Final K value: {final_k}')
        kmeans = KMeans(n_clusters=final_k).fit(auc_array)

        # Generate Radar plot with Kmeans
        labels=['Vanilla','Peanut Butter', 'Water', 'Fox Urine']
        radar_plot(labels,auc_array,'All Neurons Kmeans','kmeans_radar.pdf',single_neuron=False,Grouping=[kmeans.labels_])
        
    def parse_behavior_df(self,ColumnName):
        #Convert from Pandas dataframe back to numpy
        Event=self.so[ColumnName]
        ImageNumbers=self.so['ImageNumber']
        ImageNumbers=ImageNumbers.to_numpy()
        Event=Event.to_numpy()

        # Find Image number where event occurs
        ImageNumberTS=[]
        for i in range(len(Event)-1):
            if Event[i]==0 and Event[i+1]==1:
                ImageNumberTS.append(ImageNumbers[i])
        
        return ImageNumberTS

    def PETH(self,data,timestamps,window,baseline_period,event_period,event_name):
        """ PETH method will align neuronal trace data to each event of interest. 
        Inputs:
        data: float -- This is a matrix of data where each row contains dF trace data for a single neuron. Each column is a frame/time point
        timestamps: float -- This is the timestamps (taken from load_serial_output class) for trial of interest. 
        window: float default=10 -- The time (seconds) before and after each event that you would like to plot.
        baseline_period: list of two floats default=[-10,-5] -- 
        event_period: list of two floats default=[-10,-5]
        event_name: str -- This string will be used to create a subfolder in figures path. Suggest using the name of the trial type. Example: 'shocktrials'.

        Outputs:
        self.peth_stats -- Class attribute containg a list of important Area Under the Curve statistics for baseline and event. Each element corresponds to stats for a single neuron. 
        PETH graphs -> saved to the provided datapath /figures/neuronal/peths/eventname/peth_neuron{X}.jpg. If given N traces, there will be N peth graphs saved.
        """
        sampling_frequency=self.ops['fs'] # Number of Images taken per second
        window = round(window*sampling_frequency) # Convert the window (s) * the sampling frequency (Frames/s) to get number of frames in window. 

        for i,neuron_trace in enumerate(data):
            heatmap_data=[]
            BL_AUC=[] # Save the baseline AUC stats
            EV_AUC=[] # Save the Event AUC stats
            for time in timestamps:
                trace_event = neuron_trace[int(time-window):int(time+window)]
                heatmap_data.append(trace_event)

                #Calculate AUC for Baseline
                bl_trace=neuron_trace[int(time+round(baseline_period[0]*sampling_frequency)):int(time+round(baseline_period[1]*sampling_frequency))]
                BL_AUC.append(np.trapz(bl_trace))

                #Calculate AUC for Event
                ev_trace=neuron_trace[int(time+round(event_period[0]*sampling_frequency)):int(time+round(event_period[1]*sampling_frequency))]
                EV_AUC.append(np.trapz(ev_trace))

            try:
                mean_trace=np.asarray(heatmap_data).mean(axis=0) # Get Average trace across events for Neuron
            except:
                heatmap_data=heatmap_data[:-1] #The last element is too close to end of session. Best to delete data.
                mean_trace=np.asarray(heatmap_data).mean(axis=0) 
           
            # Plot PETH
            plt.figure(figsize=(15,15),dpi=1200)
            f, axes = plt.subplots(2, 1, sharex='col')
            plt.subplot(2, 1, 1)
            axes[0] = plt.plot(mean_trace)
            ax = plt.subplot(2, 1, 2)
            axes[1].pcolor(heatmap_data)
            #ax.colarbar()
            plt.title(event_name)
            plt.savefig(os.path.join(self.resultpath_neur,f'{event_name}PETH_Neuron{i}.pdf'))
            plt.close()

class corralative_activity(funcational_classification):
    def __init__(self,datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.7):
        super().__init__(datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=cellthreshold)

    def get_activity_heatmap(self,data):
        plt.figure(figsize=(30,30),dpi=300)
        ax = sns.heatmap(data)
        plt.ylabel('Neurons')
        plt.xlabel('Frames')
        plt.savefig(os.path.join(self.resultpath_neur,'general_heatmap.pdf'))

        plt.figure(figsize=(30,30),dpi=300)
        dataav=np.copy(data)
        dataav=np.nanmean(dataav,axis=0)
        plt.plot(dataav,linewidth=3)
        plt.ylabel('Average Z-score(F)', fontsize=15)
        plt.xlabel('Frames', fontsize=15)
        plt.title('Neuronal Population Activity')
        plt.savefig(os.path.join(self.resultpath_neur,'population_activity.pdf'))

    def get_activity_correlation(self,data,output_filename='correlation_analysis.pdf'):
        data=np.asarray(data)
        data=data.T
        data=pd.DataFrame(data)
        correlations=data.corr(method='pearson')

        # Rank order the neurons based on highest to lowest correlations
        cor_rankings=correlations.sum(axis=1)
        order = np.argsort(cor_rankings)
        order = order[::-1] # We want the highest correlations at the top of graph. 
        data_sorted=data[order]
        correlations=data_sorted.corr(method='pearson')

        palleteoh=sns.color_palette("coolwarm", as_cmap=True)
        plt.figure(figsize=(15,15),dpi=300)
        ax = sns.heatmap(correlations,vmin=-0.2,vmax=0.8)
        plt.ylabel('Neuron #')
        plt.xlabel('Neuron #')
        output_filename=os.path.join(self.resultpath_neur,output_filename)
        plt.savefig(output_filename)

        # Get correlation values other than 1. 
        corr_ed=correlations
        corr_ed[corr_ed==1]=np.nan
        return corr_ed, correlations
        
class pipeline():
    """ A general pipeline which pulls data through corralative activity nested class
        The purpose of this class is to allow for user (myself) to quickly pull and analyze all previously calculated data
        in downstream scripts. See Correlative Acitivity Analysis python script. 
    """
    def __init__(self,serialoutput_search,twophoton_search): 
        self.serialoutput_search=serialoutput_search
        self.twophoton_search=twophoton_search

    def match_directories(self):
        # Find and match all 2P image folders with corresponding serial output folders
        behdirs = glob.glob(self.serialoutput_search)
        twoPdirs = glob.glob(self.twophoton_search)
        self.final_list=[]
        for diroh in twoPdirs:
            if 'DAY1' in diroh.upper():
                dday=1
            if 'DAY7' in diroh.upper():
                dday=7
            if 'DAY14' in diroh.upper():
                dday=14
            _,_,_,_,_,_,cage,mouse,_=diroh.upper().split('_')
            for bdiroh in behdirs:
                if 'DAY1' in bdiroh.upper():
                    bday=1
                if 'DAY7' in bdiroh.upper():
                    bday=7
                if 'DAY14' in bdiroh.upper():
                    bday=14
                _,_,_,_,_,_,cageb,mouseb = bdiroh.upper().split('_')
                if cage==cageb and mouse==mouseb and dday==bday:
                    self.final_list.append([diroh,bdiroh])
    
    def main(self):
        self.recordings=[]
        for i,(imagepath,behpath) in tqdm.tqdm(enumerate(self.final_list), total=len(self.final_list), desc='Current Recording: '):
            try:
                #Get behavior data object
                self.so_obj = load_serial_output(behpath)
                self.so_obj()

                # Get twophon data object
                self.s2p_obj = corralative_activity(imagepath,self.so_obj)
                self.s2p_obj()

                #Append object as attribute to list
                self.recordings.append(self.s2p_obj)
            except:
                print('error')
        return self.recordings
    
    def __call__(self):
        self.match_directories()
        self.main()

if __name__=='__main__':
    recordings=pipeline(r'C:\Users\listo\tmtassay\TMTAssay\Day1\serialoutput\**\*24*',r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\**\*24*')
    recordings()