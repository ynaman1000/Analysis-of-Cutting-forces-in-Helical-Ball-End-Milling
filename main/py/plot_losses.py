# -*- coding: utf-8 -*-
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt


############################################### Manual Setting ###############################################

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=np.inf, sci_mode=False)
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("CUDA Available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()

datafile_folder = "csv/"
datafile_name = "initLR-0.01_lrUC-2_lrUP-5000_L1-4_N1-270_L2-1_N2-3_2019_8_25_20_27_6"    # name of data file
datafile_ext = ".csv"

dimInput = 2       # input dimension
dimOutput = 3      # output dimension
numIter = 10000    # total number of data points

# For doing post-processing/analysis
plt_max_percent_dev = True   # for plotting max percent deviation in training and validation data vs time step
plt_99per_percent_dev = True    # for plotting 99th percentile percent deviation
y_ax_lim_per_dv = 5          # max value on y-axis for max_dev plots
plt_loss = True              # for plotting loss plots
y_ax_lim_loss = 2        # max value on y-axis for loss plots
fps = 100
num_subplot_rows = 3
num_subplot_cols = 1

############################################################################################################


############################################### Loading Data ###############################################

# Function for loading data
def load_data(fl_nm):
    ''' Returns numpy array of shape (total data points, 7) '''
    data_file = open(fl_nm, 'r')
    data_list = []
    for row in data_file.readlines():
        row_list = row.split(',')
        row_list.pop()
        for i in range(len(row_list)):
            row_list[i] = float(row_list[i])
        data_list.append(row_list)
    data_file.close()
    return np.array(data_list, dtype = np.float32, ndmin = 2)

############################################################################################################


############################################### Defining Animation function ###############################################

def plotter(err_arr, fig_nm, subplot_rows, subplot_cols, y_max, addTitle=None):
    ''' err_arr- numpy array of shape (dimOutput, number of frames)
        fig_nm- string '''

    

    # plotting on each subplot
    
    if subplot_cols == 1 or subplot_rows == 1:
        fig = plt.figure()
        if subplot_cols == 1 and subplot_rows == 1:
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.linspace(0, err_arr.shape[1]-1, err_arr.shape[1]).reshape((err_arr.shape[1]))*fps, (err_arr[cnt, :].reshape((err_arr.shape[1]))), lw=0.3)
        else:
            if subplot_rows == 2:
                # plt.yscale("log", basey=10, subsy=[2,3,4,5,6,7,8,9])
                # major_ticks = np.arange(0, 10**y_max, 10)
                subTitles = ["Training dataset", "Validation dataset"]
                for cnt in range(2):
                    a = fig.add_subplot(2, 1, cnt+1)
                    a.plot(np.linspace(0, err_arr.shape[1]-1, err_arr.shape[1]).reshape((err_arr.shape[1]))*fps, np.log10(err_arr[0, :].reshape((err_arr.shape[1]))), lw=1)
                    a.text(x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=a.transAxes)

                    # axes limit settings
                    # a.set_ylim(-2, 2)
                    # a.set_xlim(0, err_arr.shape[1]*fps+1)
                    # a.set_yticks(major_ticks)
                    a.grid()
            else:
                # plt.yscale("log", basey=10, subsy=[2,3,4,5,6,7,8,9])
                # major_ticks = np.arange(0, 10**y_max, 10)
                subTitles = ["F"+r"$_{tang}$", "F"+r"$_{normal}$", "F"+r"$_{axial}$"]
                for cnt in range(3):
                    a = fig.add_subplot(3, 1, cnt+1)
                    a.plot(np.linspace(0, err_arr.shape[1]-1, err_arr.shape[1]).reshape((err_arr.shape[1]))*fps, err_arr[0, :].reshape((err_arr.shape[1])), lw=1)
                    a.text(x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=a.transAxes)

                    # axes limit settings
                    # a.set_ylim(-2, 2)
                    # a.set_xlim(0, err_arr.shape[1]*fps+1)
                    # a.set_yticks(major_ticks)
                    a.grid()
    else:
        cnt = 0
        # create a figure with subplots
        fig, ax = plt.subplots(subplot_rows, subplot_cols)
        clrs = [['k', 'k'], ['k', 'r'], ['r', 'r']]
        subTitles = ["\u03B2"+r"$_{1}$", "\u03B2"+r"$_{2}$", "\u03B2"+r"$_{3}$", "\u03B4"+r"$_{21}$", "\u03B4"+r"$_{31}$", "\u03B4"+r"$_{32}$"]
        major_ticks = np.arange(0, y_max, 1)
        minor_ticks = np.arange(0, y_max, 0.1)
        for i in range(subplot_rows):
            for j in range(subplot_cols):
                ax[i][j].plot(np.linspace(1, err_arr.shape[1], err_arr.shape[1]).reshape((err_arr.shape[1]))*fps, err_arr[cnt, :].reshape((err_arr.shape[1])), lw=1, color=clrs[i][j])
                ax[i][j].text(x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i][j].transAxes)
                cnt += 1

                # axes limit settings
                ax[i][j].set_ylim(0, y_max)
                ax[i][j].set_xlim(0, err_arr.shape[1]*fps+1)

                # y-axis ticks setting
                ax[i][j].set_yticks(major_ticks)
                ax[i][j].set_yticks(minor_ticks, minor=True)
                ax[i][j].grid(which='both', axis='y')
                ax[i][j].grid(which='minor', axis='y', alpha=0.2)
                ax[i][j].grid(which='major', axis='y', alpha=0.5) 
                ax[i][j].tick_params(labelsize="x-small")           

    if addTitle=="loss":
        fig.suptitle("Mean squared percent error vs number of iterations on log"+r"$_{10}$"+" scale", fontsize=10)
    elif addTitle=="train_max_per_dev":
        fig.suptitle("Training set- Max. percentile percent deviation vs number of iterations", fontsize=10)
    elif addTitle=="val_max_per_dev":
        fig.suptitle("Validation set- Max. percent deviation vs number of iterations", fontsize=10)
    elif addTitle=="train_99per_per_dev":
        fig.suptitle("Training set- 99"+r"$^{th}$"+" percentile percent deviation vs number of iterations", fontsize=10)
    elif addTitle=="val_99per_per_dev":
        fig.suptitle("Validation set- 99"+r"$^{th}$"+" percentile percent deviation vs number of iterations", fontsize=10)

    plt.savefig(fig_nm + ".png")
    # plt.savefig(fig_nm + ".png", dpi=1200)

    # closing all figures
    plt.close('all')

############################################################################################################

fig_name = "plots/" + datafile_name
if plt_loss:
    file_name = datafile_name + "_train_loss"
    train_loss_arr  = load_data(datafile_folder+file_name+datafile_ext)
    assert train_loss_arr.shape == (numIter/fps, 1), train_loss_arr.shape
    train_loss_arr = train_loss_arr.transpose()
    file_name = datafile_name + "_val_loss"
    val_loss_arr  = load_data(datafile_folder+file_name+datafile_ext)
    assert val_loss_arr.shape == (numIter/fps, 1), val_loss_arr.shape
    val_loss_arr = val_loss_arr.transpose()
    print("Plotting loss.....")
    loss_arr = np.append(train_loss_arr, val_loss_arr, axis=0)
    plotter(loss_arr, fig_name+"_loss", 2, 1, y_ax_lim_loss, "loss")

if plt_max_percent_dev:
    file_name = datafile_name + "_train_max_per_dev"
    train_max_per_dev_arr  = load_data(datafile_folder+file_name+datafile_ext)
    assert train_max_per_dev_arr.shape == (numIter/fps, dimOutput), train_max_per_dev_arr.shape
    train_max_per_dev_arr = train_max_per_dev_arr.transpose()
    file_name = datafile_name + "_val_max_per_dev"
    val_max_per_dev_arr  = load_data(datafile_folder+file_name+datafile_ext)
    assert val_max_per_dev_arr.shape == (numIter/fps, dimOutput), val_max_per_dev_arr.shape
    val_max_per_dev_arr = val_max_per_dev_arr.transpose()
    print("Plotting max percent deviation.....")
    plotter(train_max_per_dev_arr, fig_name+"_train_max_per_dev", num_subplot_rows, num_subplot_cols, y_ax_lim_per_dv, "train_max_per_dev")
    plotter(val_max_per_dev_arr, fig_name+"_val_max_per_dev", num_subplot_rows, num_subplot_cols, y_ax_lim_per_dv, "val_max_per_dev")

if plt_99per_percent_dev:
    file_name = datafile_name + "_train_99per_per_dev"
    train_99per_per_dev_arr  = load_data(datafile_folder+file_name+datafile_ext)
    assert train_99per_per_dev_arr.shape == (numIter/fps, dimOutput), train_99per_per_dev_arr.shape
    train_99per_per_dev_arr = train_99per_per_dev_arr.transpose()
    file_name = datafile_name + "_val_99per_per_dev"
    val_99per_per_dev_arr  = load_data(datafile_folder+file_name+datafile_ext)
    assert val_99per_per_dev_arr.shape == (numIter/fps, dimOutput), val_99per_per_dev_arr.shape
    val_99per_per_dev_arr = val_99per_per_dev_arr.transpose()
    print("Plotting 99th percentile percent deviation.....")
    plotter(train_99per_per_dev_arr, fig_name+"_train_99per_per_dev", num_subplot_rows, num_subplot_cols, y_ax_lim_per_dv, "train_99per_per_dev")
    plotter(val_99per_per_dev_arr, fig_name+"_val_99per_per_dev", num_subplot_rows, num_subplot_cols, y_ax_lim_per_dv, "val_99per_per_dev")
