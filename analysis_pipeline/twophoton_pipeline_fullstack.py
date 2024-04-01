import numpy as np
import suite2p as s2p
import matplotlib.pyplot as plt 
import ipdb
import os,glob
import pandas as pd
from multiprocessing import Pool
import seaborn as sns
from behavior import load_serial_output
from radargraphs import radar_plot
sns.set_style('whitegrid')

""" To do list
Add pickling method to load and save objects easily. 
Normalize trace via z-score
High res output for heatmaps =
Have masks imported back into movie to show which neurons are called neurons. 
"""

class get_s2p():
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1):
        #Set input and output directories
        self.datapath=datapath
        self.resultpath=os.path.join(self.datapath,'figures/')
        self.resultpath_so=os.path.join(self.datapath,'figures/serialoutput/')
        self.resultpath_neur=os.path.join(self.datapath,'figures/neuronal/')

        if not os.path.exists(self.resultpath): #Make the figure directory
            os.mkdir(self.resultpath)
        if not os.path.exists(self.resultpath_so): #Make subfolder for serialoutput/behavioral data
            os.mkdir(self.resultpath_so)
        if not os.path.exists(self.resultpath_neur): #Make subfolder for neural data
            os.mkdir(self.resultpath_neur)


        #Set suite2P ops
        self.ops = s2p.default_ops()
        self.ops['batch_size'] = batch_size # we will decrease the batch_size in case low RAM on computer
        self.ops['threshold_scaling'] = threshold_scaling # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
        self.ops['fs'] = fs # sampling rate of recording, determines binning for cell detection
        self.ops['tau'] = tau # timescale of gcamp to use for deconvolution
        self.ops['input_format']="bruker"
        self.ops['blocksize']=blocksize
        self.ops['reg_tif']=reg_tif
        self.ops['reg_tif_chan2']=reg_tif_chan2
        self.ops['denoise']=denoise

        #Set up datapath
        self.db = {'data_path': [self.datapath],}
    
    def __call__(self):
        searchstring=os.path.join(self.datapath,'**/F.npy')
        res = glob.glob(searchstring,recursive=True)
        if not res:
            self.auto_run()
            self.get_reference_image()

    def auto_run(self):
        self.output_all=s2p.run_s2p(ops=self.ops,db=self.db)

    def get_reference_image(self):
        filename=os.path.basename(self.datapath)
        filename_ref=os.path.join(self.resultpath_neur,f'{filename}referenceimage.jpg')
        filename_rigids=os.path.join(self.resultpath_neur,f'{filename}rigid.jpg')
        plt.figure(figsize=(20,20))
        plt.subplot(1, 4, 1)
        plt.imshow(self.output_all['refImg'],cmap='gray')

        plt.subplot(1, 4, 2)
        plt.imshow(self.output_all['max_proj'], cmap='gray')
        plt.title("Registered Image, Max Projection");

        plt.subplot(1, 4, 3)
        plt.imshow(self.output_all['meanImg'], cmap='gray')
        plt.title("Mean registered image")

        plt.subplot(1, 4, 4)
        plt.imshow(self.output_all['meanImgE'], cmap='gray')
        plt.title("High-pass filtered Mean registered image")
        plt.savefig(filename_ref)

class parse_s2p(get_s2p):
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.65):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1) #Use initialization from previous class
        self.cellthreshold=cellthreshold # Threshold to determine whether a cell is a cell. 0.7 means only the top 30% of ROIS make it to real dataset as neurons.

    def get_s2p_outputs(self):
        #Find planes and get recording/probability files
        search_path = os.path.join(self.datapath,'suite2p/plane*/')
        self.recording_files=[]
        self.probability_files=[]
        planes = [result for result in glob.glob(search_path)]
        self.recording_files.append(os.path.join(planes[0],'F.npy'))
        self.probability_files.append(os.path.join(planes[0],'iscell.npy'))
    
        assert (len(self.recording_files)==1 and len(self.probability_files)==1) #Make sure there is only one file.
 
        self.recording_file=self.recording_files[0]
        self.probability_file=self.probability_files[0]
        self.neuron_prob=np.load(self.probability_file)
        self.neuron_prob=self.neuron_prob[:,1]
        self.traces=np.load(self.recording_file)
       
    
    def __call__(self):
        super().__call__()
        self.get_s2p_outputs()
        self.threshold_neurons()
        self.parallel_zscore()
        self.plot_all_neurons('Frames','Z-Score + i')
        self.plot_neurons('Frames','Z-Score')
        
    def threshold_neurons(self):
        self.traces=self.traces[np.where(self.neuron_prob>0.9),:] #Need to add threshold as attirbute
        self.traces=self.traces.squeeze()
        return
        
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
            self.ztraces_copy=np.copy(self.ztraces)

    def plot_all_neurons(self,x_label,y_label):
        # Plot neuron traces and save them without opening
        fig,ax=plt.subplots(dpi=1200)
        fig.set_figheight(100)
        fig.set_figwidth(15)
        addit=0
        for i,row in enumerate(self.ztraces):
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

class corralative_activity(parse_s2p):
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.65):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.65)

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

    def get_activity_correlation(self,data):
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
        plt.savefig(os.path.join(self.resultpath_neur,'correlation_analysis.pdf'))

        # Get correlation values other than 1. 
        corr_ed=correlations
        corr_ed[corr_ed==1]=np.nan

    def general_pipeline(self):
        # Look at correlation of activity during baseline
        # Look at correlation of activity during US
        # Look at correlation of activity during post-TMT
        # Get PETHS and classify neurons by activity
        # Look at each of above correlations with respect to functional classification of neurons 

        a=1
    

class funcational_classification(parse_s2p):
    def __init__(self,datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.65):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,reg_tif_chan2=True,denoise=1,cellthreshold=0.65)
        self.so=serialoutput_object #Pass in serial_output_object

    def __call__(self):
        super().__call__()
        #Create Peths per trial type
        trial_list = ['Vanilla','PeanutButter','Water','FoxUrine']
        self.plot_all_neurons_with_trials('Frames','Z-Score DF',trial_list)
        self.get_baseline_AUCs()
        self.get_event_aucs(30)
        for i,trial_name in enumerate(trial_list):
            self.PETH(self.ztraces_copy,self.so.all_evts_imagetime[i],15,[-10,-5],[0,5],trial_name)

    def plot_all_neurons_with_trials(self,x_label,y_label,trial_list):
        # Create a plot of all neurons with vertical lines per trial type
        color = ['Blue','Green','Black','Red']
        min_trial=self.so.all_evts_imagetime[0][0]
        max_trial=self.so.all_evts_imagetime[0][0]
        for trial_type,col,trial_name in zip(self.so.all_evts_imagetime,color,trial_list):
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
        dataoh = np.copy(self.ztraces_copy)
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
            labels=['Vanilla','Peanut Butter', 'Water', 'Fox Urine']
            values=[VanillaAUC,PbAUC,WaterAUC,FoxUrineAUC]
            all_values.append(values)
        
        radar_plot(labels,all_values,'All Neurons',os.path.join(self.resultpath_neur,f'AllNeuronsRadar.pdf'),single_neuron=False)
        ipdb.set_trace()

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

            mean_trace=np.asarray(heatmap_data).mean(axis=0) # Get Average trace across events for Neuron

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

        # Get Raster-PETH for each neuron's activity across conditions. (10 second before and after)
        # Plot raster-PETHS across trials 

    def classify_neuron(self):
        ipdb.set_trace()
        # Vanilla Event, Peanut Butter Event
        # Trial 1 , 2 , 3 ,4, 5, .. N, 
        # delta AUC1 (Baseline-Event), delta AUC2
    # Classify neurons into sections (Water, TMT, Vanilla, Peanut Butter)
        # Based on change in activity from baseline and fidelity?
    #
        
    def create_labeled_movie(self):
        #Take motion corrected images and overlay mask based on functional classification in python
        a=1

        
def main(serialoutput_search, twophoton_search):
    # Find and match all 2P image folders with corresponding serial output folders
    behdirs = glob.glob(serialoutput_search)
    twoPdirs = glob.glob(twophoton_search)
    final_list=[]
    for diroh in twoPdirs:
        _,cage,mouse,_=diroh.upper().split('_')
        for bdiroh in behdirs:
            _,cageb,mouseb = bdiroh.upper().split('_')
            if cage==cageb and mouse==mouseb:
                final_list.append([diroh,bdiroh])

    recordings=[]
    for i,(imagepath,behpath) in enumerate(final_list):
        if i<4:
            continue

        #Get behavior data object
        so_obj = load_serial_output(behpath)
        so_obj()

        s2p_obj = funcational_classification(imagepath,so_obj)
        s2p_obj()
        #s2p_obj.get_activity_heatmap(s2p_obj.ztraces_copy) #Get the heatmap for whole session
        #s2p_obj.get_activity_correlation(s2p_obj.ztraces_copy) #Get the correlation matrix plot for all neurons
        recordings.append(s2p_obj)

    return recordings

if __name__=='__main__':
    #Set up command line argument parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--headless', action='store_true') #Folder containing two photon's TIFF images
    # parser.add_argument('--subject_two_photon_data',type=str,required=False) #Folder containing two photon's TIFF images
    # parser.add_argument('--serial_output_data',type=str,required=False) #Folder containing the serial outputs from the sync and sens aurduinos
    # parser.add_argument('--deep_lab_cut_data',type=str,required=False) #Folder continaing deeplabcut output data for video.
    # args=parser.parse_args()

    # # Run headless or run main function.
    # if args.headless:
    #     print('headless mode')
    # else:
    recordings=main(r'C:\Users\listo\tmtassay\TMTAssay\Day1\serialoutput\**\*24*',r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\**\*24*')
        # ipdb.set_trace()