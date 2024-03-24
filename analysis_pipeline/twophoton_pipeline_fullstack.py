import os, requests
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import suite2p as s2p
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse 
import ipdb
import os,glob
import pandas as pd
from multiprocessing import Pool
#import seaborn as sns
#sns.set_style('whitegrid')


"""
To do list
Normalize trace via z-score
High res output for heatmaps
Have masks imported back into movie to show which neurons are called neurons. 


"""

class get_s2p():
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1):
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
        self.ops['denoise']=denoise

        #Set up datapath
        self.db = {'data_path': [self.datapath],}
    
    def __call__(self):
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
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1,cellthreshold=0.65):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1) #Use initialization from previous class
        self.cellthreshold=cellthreshold # Threshold to determine whether a cell is a cell. 0.7 means only the top 30% of ROIS make it to real dataset as neurons.

    def get_s2p_outputs(self):
        #Find planes and get recording/probability files
        search_path = os.path.join(self.datapath,'suite2p/plane*/')
        self.recording_files=[]
        self.probability_files=[]
        planes = [result for result in glob.glob(search_path)]
        self.recording_files.append(os.path.join(planes[0],'F.npy'))
        self.probability_files.append(os.path.join(planes[0],'iscell.npy'))

         
        if len(self.recording_files)>1 and len(self.probability_files)>1:
            #Loop over files
            self.__call__
        else:
            self.recording_file=self.recording_files[0]
            self.probability_file=self.probability_files[0]
            self.neuron_prob=np.load(self.probability_file)
            self.neuron_prob=self.neuron_prob[:,1]
            self.traces=np.load(self.recording_file)

        return
    
    def __call__(self):
        super().__call__()
        self.get_s2p_outputs()
        self.threshold_neurons()
        toh=self.traces[10,:]
        self.zscore_trace(toh)
        self.plot_neurons('Frames','F')
        
    def threshold_neurons(self):
        self.traces=self.traces[np.where(self.neuron_prob>0.9),:] #Need to add threshold as attirbute
        self.traces=self.traces.squeeze()
        return
        
    def zscore_trace(self,trace):
        """ Using a sliding window, trace is zscored. 
        The sliding window is offset each iteration of the loop
        This removes any artifacts created by z score. 
        """
        ipdb.set_trace()
        ztrace=[]
        window_width=20
        for rvalue in range(window_width):
            start=rvalue
            stop=rvalue+window_width
            zscored_trace=[]
            ipdb.set_trace()
            for i in range(round(len(trace)/(stop-start))):
                if start>0 and i==0:
                    ipdb.set_trace()
                    window=trace[0:start]
                    window=(window-np.mean(window))/np.std(window) #Zscrore winow
                    zscored_trace.append(window)

                window=trace[start:stop]
                window=(window-np.mean(window))/np.std(window) #Zscrore winow
                start+=window_width
                stop+=window_width
                zscored_trace.append(window)
            
            for i,window in enumerate(zscored_trace):
                if i==0:
                    zscored_trace=
                
            zscored_trace=np.asarray(zscored_trace)
            zscored_trace=zscored_trace.reshape(len(trace),)
            ztrace.append(zscored_trace)
            
        ipdb.set_trace()
        ztrace=np.median(ztrace,axis=0)
        return ztrace
    
    def parallel_zscore(self):
        with Pool() as P:
            self.ztraces = P.map(self.zscore_trace,self.traces)

    def plot_neurons(self,x_label,y_label):   
        # Plot neuron traces and save them without opening
        for i,row in enumerate(self.z_traces):
            fig,ax=plt.subplots()
            row+=i
            plt.plot(row)
            file_string=os.path.join(self.resultpath_neur,f'trace{i}.pdf')
            plt.title(file_string)
            ax.set_ylabel(y_label)
            ax.set_ylabel(x_label)
            plt.savefig(file_string)
            plt.close()
        return

class corralative_activity(parse_s2p):
    def __init__(self,datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1,cellthreshold=0.65):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1,cellthreshold=0.65)

    def get_activity_heatmap(self,data):
        plt.figure(figsize=(15,25))
        plt.imshow(data,cmap='coolwarm')
        plt.ylabel('Neurons')
        plt.xlabel('Frames')
        plt.savefig(os.path.join(self.resultpath_neur,'general_heatmap.jpg'))

    def get_activity_correlation(self,data):
        data=data.T
        data=pd.DataFrame(data)
        correlations=data.corr(method='pearson')
        plt.figure(figsize=(15,15))
        plt.matshow(correlations,cmap='inferno')
        plt.legend()
        plt.ylabel('Neuron #')
        plt.xlabel('Neuron #')
        plt.savefig(os.path.join(self.resultpath_neur,'correlation_analysis.jpg'))


class load_serial_output():
    def __init__(self,path):
        self.path = path 
   
    def __call__(self):
        self.get_file_name()
        self.load_data()
        self.count_trials()
        #self.plot_trials() 
        self.crop_data()
        #self.graph_aurduino_serialoutput_rate()
    
    def get_file_name(self):
        bn = os.path.basename(self.path)
        self.outputfilename=bn+'_plottrials'

    def load_data(self):
        #Search given path for putty files
        search_string = os.path.join(self.path,'*')
        files = glob.glob(search_string)
        #Loop through all files in the path and grab data
        for j,file in enumerate(files):
            file = open(file, "r") # Read file
            content = file.read().splitlines() #Separate into correct shape
            alldata=[] #Generate empty list
            
            if 'sens' in files[j]:
                for line in content: #Go through each line and filter
                    try: #If line does not fit specific shape, throws an error. 
                        line = line.split(',')
                        line = np.asarray(line)
                        line = line.astype(float)
                        if line.shape[0]!=9: #Hard coded shape in, remove later
                            continue
                        alldata.append(line) #Append to empty list
                    except:
                        continue
            
                alldata=np.asarray(alldata) #Reshape list into numpy array
                self.sens=alldata

            # If file name contains sens or sync, put in correct variable name
            if 'sync' in files[j]:
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

    def crop_data(self):
        """ Takes sens and sync data, crops them to only important the experiment
        """
        # Seperate array columns temporarily
        imagecount = self.sync[:,0] #Writing this out for our own sanity
        loopnumber = self.sync[:,1]

        # Roll through image count array to find start and stop
        starts=[]
        stops=[]
        for i,val in enumerate(imagecount[:-1]):
            if imagecount[i]==0 and imagecount[i+1]==1:
                starts.append(i)
            if imagecount[i]>0 and imagecount[i+1]==0:
                stops.append(i)

        if len(stops)<1:
            stops=imagecount[-2]

        if type(stops) is np.float64:
            stops=[int(stops)]

        if len(starts)>1 or len(stops)>1:
            raise Exception("More than 1 start or stop calculated, code must be fixed")

        # Crop data to only the recording
        self.sync = self.sync[starts[0]:stops[0],:]
        loopstart,loopstop=loopnumber[starts[0]],loopnumber[stops[0]]
        loopstart,loopstop=np.where(self.sens[:,0]==loopstart),np.where(self.sens[:,0]==loopstop)
        self.sens=self.sens[int(loopstart[0][0]):int(loopstop[0][0]),:]

        # Convert to pandas dataframe
        self.syncdf = pd.DataFrame({'ImageNumber': self.sync[:, 0], 'LoopNumber': self.sync[:, 1], 'LaserTrigger': self.sync[:, 2]})
        self.sensdf = pd.DataFrame({'LoopNumber': self.sens[:, 0], 'Pressure': self.sens[:, 1], 'Temperature': self.sens[:, 2], \
                                  'Humidity': self.sens[:, 3], 'Time': self.sens[:, 4], 'VanillaBoolean': self.sens[:, 5],\
                                  'PeanutButterBoolean': self.sens[:, 6], 'WaterBoolean': self.sens[:, 7], 'FoxUrineBoolean': self.sens[:, 8],})
        self.behdf = pd.merge(self.syncdf,self.sensdf,on=['LoopNumber'])

    def count_trials(self):
        # Loop through trial types and count total number of trials
        trials=['Vanilla','Peanut Butter', 'Water', 'Fox Urine']
        for i,trial in zip(range(5,9),trials):
            count=0
            for start,stop in zip(self.sens[:-1,i],self.sens[1:,i]):
                if start==0 and stop==1:
                    count+=1
            
            print(f'There were {count} {trial} trials')

    def plot_trials(self):
        # Get start and end
        av=np.sum(self.sens[:,5:],axis=1)
        for i,val in enumerate(av[:-1]):
            val2=av[i+1]
            if val==0 and val2==1:
                start=i
                break

        fav=np.flip(av)
        for i,val in enumerate(fav):
            val2=fav[i+1]
            if val==0 and val2==1:
                print(val2)
                finish=len(fav)-i
                break

        plt.figure()
        for i in range(5,9):
            plt.plot(self.sens[start:finish,i]+(i*1.5-5))
        filename=os.path.join(os.getcwd(),'figures',self.outputfilename+'.jpg')
        pres=self.sens[start:finish,1]
        pres=(pres-pres.min())/(pres.max()-pres.min())+1
        print(pres.min())
        plt.plot(pres)
        filename=os.path.join(os.getcwd(),'figures',self.outputfilename+'pressure.jpg')
        plt.xlabel('Loop Number')
        plt.ylabel('On or Off')
        plt.legend(['Vanilla','Peanut Butter','Water', 'Fox Urine','Normalized Pressure'],loc='upper left')
        plt.savefig(filename)

        #Pressure
        plt.figure()
        plt.plot(self.sens[start:finish,2])
        filename=os.path.join(os.getcwd(),'figures',self.outputfilename+'temp.jpg')
        plt.xlabel('Loop Number')
        plt.ylabel('Temp')
        plt.savefig(filename)
 
        #Pressure
        plt.figure()
        plt.plot(self.sens[start:finish,3])
        filename=os.path.join(os.getcwd(),'figures',self.outputfilename+'humidity.jpg')
        plt.xlabel('Loop Number')
        plt.ylabel('Humidity')
        plt.savefig(filename)  

    def graph_aurduino_serialoutput_rate(self):
        time = self.sens[:,4]
        times=[]
        for i,t in enumerate(time[:-1]):
            times.append(time[i+1]-time[i])

        times=np.asarray(times)
        plt.figure()
        plt.scatter(x=range(len(times)),y=times)
        plt.savefig('timehist.jpg')


class funcational_classification(parse_s2p):
    def __init__(self,datapath,serialoutput_object,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1,cellthreshold=0.65):
        super().__init__(datapath,fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1,cellthreshold=0.65)
        self.so=serialoutput_object.behdf #Pass in serial_output_object

    def PETH(self):
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

        # Get Raster-PETH for each neuron's activity across conditions. (10 second before and after)
        # Plot raster-PETHS across trials 
    # Classify neurons into sections (Water, TMT, Vanilla, Peanut Butter)
        # Based on change in activity from baseline and fidelity?
    #
        
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
    for imagepath,behpath in final_list:
        #Get behavior data object
        so_obj = load_serial_output(behpath)
        so_obj()

        s2p_obj = funcational_classification(imagepath,so_obj)
        s2p_obj()
        s2p_obj.get_activity_heatmap(s2p_obj.traces) #Get the heatmap for whole session
        s2p_obj.get_activity_correlation(s2p_obj.traces) #Get the correlation matrix plot for all neurons
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