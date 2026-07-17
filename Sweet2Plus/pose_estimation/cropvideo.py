# -*- coding: utf-8 -*-
"""
Crop Behavioral Video
"""
import tqdm
import cv2
import glob as gb
import os 
from multiprocessing import Pool

def appendedname(file):
    directoryname,_ = file.split('.avi', 1)
    directoryname = directoryname + "_Cropped.avi"
    return directoryname

def cropvideo(file,newfilename,fps,x=[625,1300],y=[400,800]):
    cap = cv2.VideoCapture(file)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(newfilename, fourcc, fps, (x[1]-x[0],y[1]-y[0]),isColor=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        framenew = frame[y[0]:y[1],x[0]:x[1]]
        framenew=cv2.resize(framenew,(x[1]-x[0],y[1]-y[0]))
        out.write(framenew)
    cap.release()
    out.release()
 
if __name__=='__main__':

    input_video_path = r"E:\\TMT Experiment\\"
    serchstring = os.path.join(input_video_path,"**\\*.avi")
    files = gb.glob(serchstring, recursive=True)
    for fileoh in tqdm.tqdm(files):
        currentfile = appendedname(fileoh)
        
        if '_Cropped' in fileoh:
            continue
        
        if 'front' in fileoh: 
            cropvideo(fileoh,currentfile,16)
        elif 'side' in fileoh:
            cropvideo(fileoh,currentfile,16,x=[40,800],y=[220,700])
    
#with Pool() as p:
#    p.map(crop,files)
    