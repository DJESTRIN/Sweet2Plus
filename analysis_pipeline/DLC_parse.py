# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:01:13 2024

@author: johns
"""

import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as gb
import os 
import ipdb
from multiprocessing import Pool
from math import dist as dist

class DLCparser():
    def __init__(self,fileoh,threshold=0.60,framenumber=1000):
        self.fileoh=fileoh
        self.framenumber = framenumber
        self.threshold = threshold
        self.data = pd.read_csv(self.fileoh)
        
    def __call__(self):
        self.seperate_dataframe()
        self.get_example_image()
        self.plottrajectory()
        self.get_distance()
        self.plot_distances()
        
        
    def seperate_dataframe(self):
        # Get body parts as column headers
        self.headers = self.data.iloc[:1].to_numpy()
        self.headers = self.headers[0]
        self.headers = self.headers[1::3].astype(str)
        
        #Get data for X and Y coordinates
        self.data=self.data.iloc[1:] #Strip annoying headers
        self.xs=self.data[self.data.columns[1::3]] #Puts specific column of data into new variable  in this example it starts at column 1 and skips by 3 untill the end
        self.xs=self.xs.iloc[1:, :].to_numpy().astype(float)
        self.ys=self.data[self.data.columns[2::3]]
        self.ys=self.ys.iloc[1:, :].to_numpy().astype(float)
        self.ps=self.data[self.data.columns[3::3]]
        self.ps=self.ps.iloc[1:, :].to_numpy().astype(float)
        self.psthreshold = np.where(self.ps >self.threshold, self.ps,np.nan)
        self.xsthreshold = np.where(self.ps>self.threshold,self.xs,np.nan)
        self.ysthreshold = np.where(self.ps>self.threshold,self.ys,np.nan)
        
        
    def get_example_image(self):
        real,_ = self.fileoh.split('DLC')
        real+='.avi'
        cap = cv2.VideoCapture(real)
        breakpoint_oh=0
        while cap.isOpened():
            ret, frame = cap.read()
            if breakpoint_oh==self.framenumber:
                break
            else:
                breakpoint_oh+=1
        self.exampleframe = frame
        
    def plottrajectory(self):
        plt.figure(figsize=(15,15),dpi=300)
        plt.imshow(self.exampleframe,cmap='gray')
        for bodypartx,bodyparty in zip(self.xsthreshold.T,self.ysthreshold.T):
            plt.plot(bodypartx,bodyparty,alpha=0.5)
        
        ax=plt.gca()
        ax.set_xlim([40,800])
        ax.set_ylim([700,220])
        
        ipdb.set_trace()
        filename,_ =self.fileoh.split('DLC') 
        filename += 'trajectory_image.jpg'
        plt.savefig(filename)
        plt.close()
       
    def get_distance(self):
        distance=[]
        for x0,x1,y0,y1 in zip(self.xsthreshold[:-1],self.xsthreshold[1:],self.ysthreshold[:-1],self.ysthreshold[1:]):
            DOH=[]
            for i in range(len(x0)):
                doh = dist([x1[i],y1[i]],[x0[i],y0[i]])
                DOH.append(doh)
            distance.append(np.asarray(DOH))
        self.distance=np.asarray(distance)
            
    def plot_distances(self):
        plt.figure(figsize=(15,15),dpi=300)
        for i,doh in enumerate(self.distance.T):
            plt.subplot(len(self.distance.T),1,i+1)
            plt.tight_layout()
            plt.plot(doh)
            plt.title(self.headers[i])
        filename,_ =self.fileoh.split('DLC') 
        filename += 'distancesperbodypart.pdf'
        plt.savefig(filename)
        plt.close()
        
        #plt.scatter(x, y, kwargs)
def traj_grid(files):
    plt.figure(figsize=(15,15),dpi=400)
    for i, file in enumerate(files):
        image = cv2.imread(file)
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(image)
        if i>7: #DELETE THIS LINE IN FUTURE
            break
    plt.savefig('animaltrajs.pdf')
    plt.close()
        
    
if __name__=='__main__':

    input_video_path = r"C:\\Users\johns\Desktop\Head_Fixed_Side-Kenneth-2024-03-21\videos\\"
    serchstring = os.path.join(input_video_path,"**\\*.csv")
    files = gb.glob(serchstring, recursive=True)
    for fileoh in tqdm.tqdm(files):
        mouseoh = DLCparser(fileoh,0.60,100)
        mouseoh()
    
    serchstring = os.path.join(input_video_path,"**\\*image.jpg")
    files = gb.glob(serchstring, recursive=True)
    traj_grid(files)