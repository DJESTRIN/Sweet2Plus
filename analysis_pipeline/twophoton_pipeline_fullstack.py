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
#import seaborn as sns
#sns.set_style('whitegrid')

class get_s2p():
    def __init__(self,datapath,resultpath=os.getcwd(),fs=1.315235,tau=1,threshold_scaling=2,batch_size=800,blocksize=64,reg_tif=True,denoise=1):
        #Set directories up
        self.datapath=datapath
        self.resultpath=os.path.join(self.datapath,'figures')
        if not os.path.exists(self.resultpath): #Make the figure directory if not already made
            os.mkdir(self.resultpath)

        #Set suite2P ops
        self.ops = s2p.default_ops()
        ipdb.set_trace()
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
        filename_ref=os.path.join(self.resultpath,f'{filename}referenceimage.jpg')
        filename_rigids=os.path.join(self.resultpath,f'{filename}rigid.jpg')
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
    def __init__(self,datapath,resultpath=os.getcwd(),fs=1.315235,tau=1,threshold=2,batchsize=800,blocksize=128,reg_tif=True,denoise=1):
        super().__init__(datapath,resultpath=os.getcwd(),fs=1.315235,tau=1,threshold=2,batchsize=800,blocksize=128,reg_tif=True,denoise=1) #Use initialization from previous class

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
        self.zscore_neurons()
        self.plot_neurons('Frames','F')
        
    def threshold_neurons(self):
        self.traces=self.traces[np.where(self.neuron_prob>0.9),:]
        self.traces=self.traces.squeeze()
        return
        
    def zscore_neurons(self):
        z_scored=[]
        for trace in self.traces:
            trace=trace-np.mean(trace)/np.std(trace)
            z_scored.append(trace)
        self.z_traces=np.asarray(z_scored)
        return
        
    def plot_neurons(self,x_label,y_label):   
        # Plot neuron traces and save them without opening
        for i,row in enumerate(self.traces):
            fig,ax=plt.subplots()
            row+=i
            plt.plot(row)
            file_string=os.path.join(self.resultpath,f'trace{i}.pdf')
            plt.title(file_string)
            ax.set_ylabel(y_label)
            ax.set_ylabel(x_label)
            plt.savefig(file_string)
            plt.close()
        return

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
        ipdb.set_trace()

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

# class funcational_classification(get_s2p,load_serial_output):
#     # Get Raster-PETH for each neuron's activity across conditions. (10 second before and after)?
#         # Plot raster-PETHS across trials 
#     # Classify neurons into sections (Water, TMT, Vanilla, Peanut Butter)
#         # Based on change in activity from baseline and fidelity?
#     #

# class corralative_activity(funcational_classification):
        
def rename_files():
    images = glob.glob(r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\**\*24*\*.tif')
    for image in images:
        newname = image.replace('Ch2','Ch1')
        os.rename(image, newname)

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
        s2p_obj = parse_s2p(imagepath,resultpath='C:\\Users\\listo\\twophoton\\figures\\')
        s2p_obj()
        recordings.append(s2p_obj)

    return recordings

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--subject_two_photon_data',type=str,required=True) #Folder containing two photon's TIFF images
    # parser.add_argument('--serial_output_data',type=str,required=True) #Folder containing the serial outputs from the sync and sens aurduinos
    # parser.add_argument('--deep_lab_cut_data',type=str) #Folder continaing deeplabcut output data for video. 
    # corralative_activity()
    rename_files()
    recordings=main(r'C:\Users\listo\tmtassay\TMTAssay\Day1\serialoutput\**\*24*',r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\**\*24*')
    ipdb.set_trace()


# Presure temp hum plot them 