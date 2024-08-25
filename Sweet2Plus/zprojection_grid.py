import glob, os
import ipdb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
sessions = glob.glob(r'C:\tmt_assay\tmt_experiment_2024_clean\twophoton_recordings\twophotonimages\Day*')

def sortday(path):
    _,path=path.split("\D")
    value=path[2:]
    return int(value)                  

sessions = sorted(sessions,key=sortday)

all_projs=[]
for j,session in enumerate(sessions):
    print(j)
    vids = glob.glob(session+r'\**\**\motion_corrected*')
    for m,vid in tqdm.tqdm(enumerate(vids),total=len(vids)):
        images = glob.glob(vid+r'\*.tif*')
        for k,image in tqdm.tqdm(enumerate(images),total=len(images)):
            im = Image.open(image)
            if k==0:
                maxproj=np.asarray(im)
                maxproj=maxproj[...,np.newaxis]
            else:
                ci = np.asarray(im)
                ci=ci[...,np.newaxis]
                maxproj = np.concatenate([maxproj, ci],axis=2)
                maxproj=maxproj.max(axis=2)
                maxproj=maxproj[...,np.newaxis]

        all_projs.append([maxproj,m,j])

ipdb.set_trace()
fig=plt.figure(figsize=(15,15),dpi=400)
spot=1
prevday=all_projs[0][2]
for sample in all_projs:
    image,mouse,day=sample
    if day>prevday:
        spot=day*9+1
        prevday=day
    ax1=plt.subplot(9,5,spot)
    plt.imshow(image,cmap='gray')
    plt.axis('off')
    ax1.set_aspect('equal')
    spot+=1
plt.tight_layout()
plt.subplots_adjust(wspace=-0.1, hspace=0)
plt.savefig('GridOfScans.jpg')


import cv2
for sample in all_projs:
    image,mouse,day=sample
    string=r'C:\tmt_assay\paper_figures\zproj'
    string=string+f'\mouse{mouse}day{day}.jpg'
    image=np.squeeze(image, axis=2).astype('uint8')
    cv2.imwrite(string,image)