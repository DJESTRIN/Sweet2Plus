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

class get_s2p():
    def __init__(self,datapath,fs=13,tau=1.25,threshold=2,batchsize=200):
        #Set up ops
        self.ops = s2p.default_ops()
        self.ops['batch_size'] = batchsize # we will decrease the batch_size in case low RAM on computer
        self.ops['threshold_scaling'] = threshold # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
        self.ops['fs'] = fs # sampling rate of recording, determines binning for cell detection
        self.ops['tau'] = tau # timescale of gcamp to use for deconvolution
        self.ops['input_format']="bruker"
        #Set up datapath
        self.db = {'data_path': datapath,}
        ipdb.set_trace()
        
    def register(self):
        self.f_raw = s2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
        self.f_reg = s2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='registered_data.bin', n_frames = f_raw.shape[0]) # Set registered binary file to have same n_frames
        refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest, meanImg_chan2, badframes, yrange, xrange \
            = s2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None, refImg=None, align_by_chan2=False, ops=self.ops)

    def get_ROI(self):
        classfile = suite2p.classification.builtin_classfile
        data = np.load(classfile, allow_pickle=True)[()]
        ops, stat = suite2p.detection_wrapper(f_reg=f_reg, ops=ops, classfile=classfile)

    def get_extraction(self):
        stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(stat, f_reg,
                                                                   f_reg_chan2 = None,ops=ops)
        
    def get_cells(self):
        iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

    def auto_run(self):
        s2p.run_s2p(ops=self.ops,db=self.db)

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

    for imagepath,behpath in final_list:
        images = glob.glob(os.path.join(imagepath,'*.tif*'))
        pathoh = os.path.dirname(images[0])
        ipdb.set_trace()
        s2p_obj = get_s2p(imagepath)
        ipdb.set_trace()
    # for diry in dirsoh:
    #     beh_obj=load_serial_output(diry)
    #     beh_obj()


if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--subject_two_photon_data',type=str,required=True) #Folder containing two photon's TIFF images
    # parser.add_argument('--serial_output_data',type=str,required=True) #Folder containing the serial outputs from the sync and sens aurduinos
    # parser.add_argument('--deep_lab_cut_data',type=str) #Folder continaing deeplabcut output data for video. 
    # corralative_activity()
    rename_files()
    main(r'C:\Users\listo\tmtassay\TMTAssay\Day1\serialoutput\**\*24*',r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\**\*24*')


# Presure temp hum plot them 