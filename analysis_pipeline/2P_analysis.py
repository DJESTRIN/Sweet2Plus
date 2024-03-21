# -*- coding: utf-8 -*-
"""
Suite 2P post processing analysis 
by Kenneth Johnson & David Estrin 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set_style('whitegrid')

class Mouse():
    def __init__(self,root_dir,recording_file,probability_file):
        self.root=root_dir
        self.neuron_prob=np.load(probability_file)
        self.neuron_prob=self.neuron_prob[:,1]
        self.traces=np.load(recording_file)
        self.threshold_neurons()
        self.zscore_neurons()
        self.plot_neurons('Frames','F')
        return
        
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
        #Create a drop directory for your neuron traces
        drop_directory=self.root+"neuron_traces/"
        isExist = os.path.exists(drop_directory)
        if isExist:
            print('directory already created')
        else:
            os.mkdir(drop_directory)
            
        # Plot neuron traces and save them without opening
        for i,row in enumerate(self.traces):
            fig,ax=plt.subplots()
            row+=i
            plt.plot(row)
            file_string="neuron_number_"+str(i)+".pdf"
            plt.savefig(file_string)
            plt.title(file_string)
            ax.set_ylabel(y_label)
            ax.set_ylabel(x_label)
            plt.close()
        return

mouse_oh=Mouse("D:/21-1-13/01132023-1631-004 _analysis/suite2p/plane0/",
               "D:/21-1-13/01132023-1631-004 _analysis/suite2p/plane0/F.npy",
               "D:/21-1-13/01132023-1631-004 _analysis/suite2p/plane0/iscell.npy")
