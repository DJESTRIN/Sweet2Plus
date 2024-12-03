""" Written by David James Estrin 
Compare correlation of acitivty:
(1) What is the average population activity for all neurons across stimuli   ==> get trace and AUCs
(2) What is the population activity across stimuli when subdivided into groups. ==> get trace and AUCs
(3) What percent of neurons are responsive to threat?
"""
import ipdb
from Sweet2Plus.core.SaveLoadObjs import SaveObj, LoadObj, SaveList, OpenList
import matplotlib.pyplot as plt
import os, glob, re
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore")

def get_data(root_dir):
    files = glob.glob(os.path.join(root_dir,'**\objfile.json*'),recursive=True)

    # Loop through files and open them to grab AUC data
    auc_vals=[]
    subject_data=[]
    for file in files:
        objoh = LoadObj(FullPath=file)
        auc_vals.append(objoh.auc_vals)
        subject_data.append([objoh.mouse, objoh.cage, objoh.group, objoh.day])

    return auc_vals, subject_data

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_directory',type=str,required=False,help='A parent path containing all of the two-data of interest')
    args=parser.parse_args()
    aucs, subs = get_data(args.data_directory)
    ipdb.set_trace()

