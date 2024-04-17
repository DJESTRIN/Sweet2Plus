import matplotlib.pyplot as plt
import numpy as np
import os,glob
import pandas as pd
import ipdb

class load_serial_output():
    def __init__(self,path):
        self.path = path 
   
    def __call__(self):
        self.get_file_name()
        self.load_data()
        self.quick_timestamps()
        last_trial = np.where(self.sync[:,1]==self.all_evts[3][-1]) #Sanity check to validate 2P was recording
        #self.plot_trials() 
        self.crop_data()
        #self.graph_aurduino_serialoutput_rate()
        return last_trial
    
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
            
            if 'sens' in files[j].lower():
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
            if 'sync' in files[j].lower():
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
            stops=imagecount[-1]-1
            stops=np.where(imagecount==stops)
            stops=[int(stops[0][-1])]

        if len(stops)>1:
            stops=[stops[-1]]

        if type(stops) is np.float64 or type(stops) is np.int64:
            stops=[int(stops)]

        if len(starts)>1 or len(stops)>1:
            raise Exception("More than 1 start or stop calculated, code must be fixed")

        # Crop data to only the recording
        self.sync = self.sync[starts[0]:stops[0],:]
        loopstart,loopstop=loopnumber[starts[0]],loopnumber[stops[0]]
        loopstart,loopstop=np.where(self.sens[:,0]==loopstart),np.where(self.sens[:,0]==(loopstop-1))
        self.sens=self.sens[int(loopstart[0][0]):int(loopstop[0][0]),:]

        # Convert to pandas dataframe
        self.syncdf = pd.DataFrame({'ImageNumber': self.sync[:, 0], 'LoopNumber': self.sync[:, 1], 'LaserTrigger': self.sync[:, 2]})
        self.sensdf = pd.DataFrame({'LoopNumber': self.sens[:, 0], 'Pressure': self.sens[:, 1], 'Temperature': self.sens[:, 2], \
                                  'Humidity': self.sens[:, 3], 'Time': self.sens[:, 4], 'VanillaBoolean': self.sens[:, 5],\
                                  'PeanutButterBoolean': self.sens[:, 6], 'WaterBoolean': self.sens[:, 7], 'FoxUrineBoolean': self.sens[:, 8],})
        self.behdf = pd.merge(self.syncdf,self.sensdf,on=['LoopNumber'],how='outer')

    def quick_timestamps(self):
        # Loop through trial types and count total number of trials
        trials=['Vanilla','Peanut Butter', 'Water', 'Fox Urine']
        all_evts=[]
        for i,trial in zip(range(5,9),trials):
            count=0
            all_ts=[]
            for index,(start,stop) in enumerate(zip(self.sens[:-1,i],self.sens[1:,i])):
                if start==0 and stop==1:
                    LoopNumber = self.sens[index,0]
                    count+=1
                    all_ts.append(LoopNumber)
            all_evts.append(all_ts)
            #print(f'There were {count} {trial} trials')
        self.all_evts=all_evts

        # Convert list of loop numbers to 2P Image Numbers
        self.all_evts_imagetime=[]
        for trial_list in self.all_evts:
            all_ts=[]
            for start_time in trial_list:
                try:
                    row_number=np.where(self.sync[:,1]==start_time)[0][0]
                    image_time=self.sync[row_number,0]
                    all_ts.append(image_time)
                except:
                    continue
            self.all_evts_imagetime.append(all_ts)

        # Get Pre Trial Period
        for i,listoh in enumerate(self.all_evts_imagetime):
            if i==0:
                alltimestamps=np.asarray(listoh)
            else:
                alltimestamps=np.concatenate((alltimestamps,np.asarray(listoh)),axis=0)

        self.pretrial_period=[0,alltimestamps[0]]

        # Get Post Trial Period
        if len(alltimestamps)==65:
            self.posttrial_period=[alltimestamps[-1], self.sync[:,0].max()]
        else:
            self.posttrial_period=None
        
        

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
        plt.close()

        #Pressure
        plt.figure()
        plt.plot(self.sens[start:finish,2])
        filename=os.path.join(os.getcwd(),'figures',self.outputfilename+'temp.jpg')
        plt.xlabel('Loop Number')
        plt.ylabel('Temp')
        plt.savefig(filename)
        plt.close()
 
        #Pressure
        plt.figure()
        plt.plot(self.sens[start:finish,3])
        filename=os.path.join(os.getcwd(),'figures',self.outputfilename+'humidity.jpg')
        plt.xlabel('Loop Number')
        plt.ylabel('Humidity')
        plt.savefig(filename)  
        plt.close()

    def graph_aurduino_serialoutput_rate(self):
        time = self.sens[:,4]
        times=[]
        for i,t in enumerate(time[:-1]):
            times.append(time[i+1]-time[i])

        times=np.asarray(times)
        plt.figure()
        plt.scatter(x=range(len(times)),y=times)
        plt.savefig('timehist.jpg')
        plt.close()



if __name__=='__main__':
    behdirs = glob.glob(r'C:\Users\listo\tmtassay\TMTAssay\Day1\serialoutput\**\*24*')
    #behdirs=behdirs[1:]
    working=[]
    for pathoh in behdirs:
        so_obj = load_serial_output(pathoh)
        last_trial = so_obj()
        working.append(last_trial)

