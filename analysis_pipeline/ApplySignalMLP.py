import torch
import numpy as np
import glob,os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import ipdb
import tqdm
import matplotlib.pyplot as plt
import argparse
from SignalCNN import downsample_array, normalize_trace

class MLPapply():
    def __init__(self,model_path,data_path,plot_examples=False):
        self.model=torch.load(model_path) #Load model as attribute
        self.open_data(data_path=data_path)
        self.plot_examples=plot_examples

    def downsample_array(self,X):
        return downsample_array(X)

    def open_data(self,data_path):
        """ inputs: data_path -- full path to F.npy file produced by suite2p """
        self.traces=np.load(data_path) #orginal traces
        self.dstraces=self.downsample_array(self.traces) # downsampled traces
        pseudodata=np.ones(shape=(self.dstraces.shape[0],)) #This can be ignored, acts as a place holder
        self.normtraces,_=normalize_trace(traces=self.dstraces,labels=pseudodata) #normalized data
    
    def __call__(self):
        self.run_model()
        self.sep_traces()
        if self.plot_examples:
            self.plot_outputs()

    def run_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        net=self.model.to(device=device)
        self.outputs=[]
        for trace in tqdm.tqdm(self.normtraces):
            output = net(torch.tensor(trace))
            output=torch.argmax(output, dim = 1)
            self.outputs.append(output.cpu().detach().to_numpy())
        self.outputs=np.asarray(self.outputs)
    
    def sep_traces(self):
        self.real_traces=self.traces[self.outputs==1,:]
        self.noise=self.traces[self.outputs==0,:]

    def plot_outputs(self):
        # Generates Figure of Traces classified as real neurons
        pulled = np.random.range(self.real_traces.shape[0],16)
        plt.figure(figsize=(10,10))
        for k in pulled:
            plt.subplot(4,4,k+1)
            plt.plot(self.real_traces[k,:],color='black')
        plt.savefig('RealSignal.jpg')

        # Generate Figure of Traces classified as noise
        pulled = np.random.range(self.noise.shape[0],16)
        plt.figure(figsize=(10,10))
        for k in pulled:
            plt.subplot(4,4,k+1)
            plt.plot(self.noise[k,:],color='black')
        plt.savefig('Noise.jpg')

if __name__=='__main__':
    data=''
    model=''
    mlpoh=MLPapply(model_path=model,data_path=data,plot_examples=True)
    mlpoh()


        

     