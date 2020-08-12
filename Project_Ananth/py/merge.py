# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle

############################################### Manual Setting ###############################################

datafile1_folder = "csv/"
datafile1_name = "train_data1"    # name of data file
datafile1_ext = ".csv"
datafile2_folder = "csv/"
datafile2_name = "train_data2"    # name of data file
datafile2_ext = ".csv"
datafile_folder = "csv/"
datafile_name = "train_data"    # name of data file
datafile_ext = ".csv"

############################################################################################################


############################################### Appending Data of file2 to file1 ###############################################

# function for appending data
def merge_data(fl_nm1, fl_nm2, fl_nm):
    ''' Appends data of datafile2 at the end of datafile1 '''
    data_file1 = open(fl_nm1, 'r')
    data_file2 = open(fl_nm2, 'r')
    data_file = open(fl_nm, 'w')
    rows1 = data_file1.readlines()
    rows2 = data_file2.readlines()
    # print(len(rows1), len(rows2))
    j=0
    k=0
    for i in range(len(rows1)+len(rows2)):
        if i%20<10:
            data_file.write(rows1[j])
            j+=1
        else:
            # print(i)
            data_file.write(rows2[k])
            k+=1

merge_data(datafile1_folder+datafile1_name+datafile1_ext, datafile2_folder+datafile2_name+datafile2_ext, datafile_folder+datafile_name+datafile_ext)
############################################################################################################
