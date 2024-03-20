import os, requests
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# import suite2p as s2p
import matplotlib as mpl
import argparse 
import ipdb
import os,glob

# class get_s2p():
#     def __init__(self,datapath,fs,tau,threshold,batchsize):
#         #Set up ops
#         self.ops = s2p.default_ops()
#         self.ops['batch_size'] = batchsize # we will decrease the batch_size in case low RAM on computer
#         self.ops['threshold_scaling'] = threshold # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
#         self.ops['fs'] = fs # sampling rate of recording, determines binning for cell detection
#         self.ops['tau'] = tau # timescale of gcamp to use for deconvolution
        
#         #Set up datapath
#         db = {'data_path': datapath,}
        
#     def register(self):
#         self.f_raw = s2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
#         self.f_reg = s2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='registered_data.bin', n_frames = f_raw.shape[0]) # Set registered binary file to have same n_frames
#         refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest, meanImg_chan2, badframes, yrange, xrange \
#             = s2p.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=None, f_raw_chan2=None, refImg=None, align_by_chan2=False, ops=self.ops)

#     def get_ROI(self):
#         classfile = suite2p.classification.builtin_classfile
#         data = np.load(classfile, allow_pickle=True)[()]
#         ops, stat = suite2p.detection_wrapper(f_reg=f_reg, ops=ops, classfile=classfile)

#     def get_extraction(self):
#         stat_after_extraction, F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction_wrapper(stat, f_reg,
#                                                                    f_reg_chan2 = None,ops=ops)
        
#     def get_cells(self):
#         iscell = suite2p.classify(stat=stat_after_extraction, classfile=classfile)

#     def run_automatic(self):
#         s2p.run_s2p(ops=self.ops,db=self.db)

class load_serial_output():
    def __init__(self,path):
        self.path = path 

    def load_data(self):
        search_string = os.path.join(self.path,'*')
        files = glob.glob(search_string)
        for file in files:
            file = open(file, "r")
            content = file.read().splitlines()
            alldata=[]
            for line in content:
                if 'PuTTY' in line:
                    continue
                try:
                    line = line.split(',')
                    line = np.asarray(line)
                    line = line.astype(float)
                    ipdb.set_trace()
                    alldata.append(line)
                except:
                    continue
            
            alldata=np.asarray(alldata)
            ipdb.set_trace()
    # Load behavior data into a nice numpy array
    # Boil down data to timestamps
    # Calculate the number of trials per trial type

# class funcational_classification(get_s2p,load_serial_output):
#     # Get Raster-PETH for each neuron's activity across conditions. (10 second before and after)?
#         # Plot raster-PETHS across trials 
#     # Classify neurons into sections (Water, TMT, Vanilla, Peanut Butter)
#         # Based on change in activity from baseline and fidelity?
#     #

# class corralative_activity(funcational_classification):


if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--subject_two_photon_data',type=str,required=True) #Folder containing two photon's TIFF images
    # parser.add_argument('--serial_output_data',type=str,required=True) #Folder containing the serial outputs from the sync and sens aurduinos
    # parser.add_argument('--deep_lab_cut_data',type=str) #Folder continaing deeplabcut output data for video. 
    # corralative_activity()

    example=load_serial_output(r'C:\Users\listo\tmt_assay\tmt_2P_assay\Day1\serialoutput\24-3-18\24-3-18_C4620081_m1')
    example.load_data()