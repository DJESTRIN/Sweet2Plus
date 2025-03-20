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

directory = r"C:\Users\johns\Documents\behavior_csv\24-3-18_C4620081_M1_front_CroppedDLC_resnet50_Front_headfixed_behavior_cameraApr28shuffle1_300000.csv"
threshold=0.60   
files = gb.glob(directory, recursive=True)
for fileoh in tqdm.tqdm(files):
    data = pd.read_csv(fileoh)
    data=data.iloc[1:] #Strip annoying headers
    xs=data[data.columns[1::3]] #Puts specific column of data into new variable  in this example it starts at column 1 and skips by 3 untill the end
    xs=xs.iloc[1:, :].to_numpy().astype(float)
    print(xs)
    print (xs.shape)
    ys=data[data.columns[2::3]]
    ys=ys.iloc[1:, :].to_numpy().astype(float)
    ps=data[data.columns[3::3]]
    ps=ps.iloc[1:, :].to_numpy().astype(float)
    psthreshold = np.where(ps >threshold, ps,np.nan)
    xsthreshold = np.where(ps>threshold,xs,np.nan)
    ysthreshold = np.where(ps>threshold,ys,np.nan)
    
    
    for i, (bodypartx,bodyparty) in enumerate(zip(xsthreshold.T,ysthreshold.T)):
        plt.figure(figsize=(15,15),dpi=300)
        plt.xlim(0,600)
        plt.ylim(0,400)
        plt.plot(bodypartx,bodyparty,alpha=0.5)
        filename = f'test_{i}.jpg'
        plt.savefig(os.path.join(r"C:\Users\johns\Documents\behavior_csv\fig",filename))

        