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
import ipdb

def GetNumpyArray(parent_directory):
    Fnp = np.load(os.path.join(parent_directory, 'F.npy'))
    ICnp = np.load(os.path.join(parent_directory, 'iscell.npy'))
    return Fnp, ICnp

def FilterCells(Fnp, ICnp):
    ipdb.set_trace()
    return Fnp

def GetStats(Fnp, timepoints = [(0,738),(739,1439),(1440,-1)]):
    ipdb.set_trace()
    area = np.trapz(y,x)

    return AUCs, Traces

def GeneratePlots(AUCs, Traces):
    plt.plot(x, y, label="Y = 2X")
    plt.show()

if __name__=='__main__':
    # Get parent directory and create drop directory for dataframe and images 
    parent_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\plane0" 
    drop_directory = r"\\Kenneth-NAS\data\25-3-12\25-3-12\25-3-12_C4856077_M1_SERT_Flp_chrmine_drn_G6M-mpfc_15C_15stim_r1-042\suite2p\analysis" 
    os.makedirs(drop_directory, exist_ok=True)

    # Run current file through analysis. 
    FnpOH, ICnpOH = GetNumpyArray(parent_directory=parent_directory)
    FnpOH = FilterCells(FnpOH, ICnpOH)
    AUCsOH, TracesOH = GetStats(FnpOH, drop_directory)
    GeneratePlots(AUCsOH, TracesOH, drop_directory)

