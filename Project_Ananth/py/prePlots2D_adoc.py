import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib

datafile_folder = "csv/"
datafile_name = "pre_adoc"
datafile_ext = ".csv"

dimInput = 2       # input dimension
dimOutput = 3      # output dimension

numN = 3
numE = 10
numDataPts = numN*numE     # total number of data points

plt_folder = "plots/"
plt_name = "pre_adoc"
num_subplot_rows = 3    # number of rows of subplot array in plot
num_subplot_cols = 1    # number of columns of subplot array in plot	

############################################### Loading Data ###############################################

# Function for loading data
def load_data(fl_nm):
    ''' Returns numpy array of shape (total data points, dimInput+dimOutput) '''
    data_file = open(fl_nm, 'r')
    data_list = []
    for row in data_file.readlines():
        row_list = row.split(',')
        for i in range(len(row_list)):
            row_list[i] = float(row_list[i])
        data_list.append(row_list)
    data_file.close()   
    return np.array(data_list, dtype = np.float32, ndmin = 2)


data_arr  = load_data(datafile_folder+datafile_name+datafile_ext)
assert data_arr.shape == (numDataPts, dimInput+dimOutput), data_arr.shape

data_input, data_output = np.split(data_arr, [dimInput], axis=1)
x_data1 = data_input[:numE, 1]
x_data2 = data_input[-numE:, 1]
y_data = np.empty([numN, numE, dimOutput])
for i in range(numN):
    for j in range(numE):
        y_data[i, j, :] = data_output[numE*i+j, :]

print(y_data.shape)
print(x_data1, x_data2)

############################################################################################################

############################################### Defining Plotting function ###############################################

def plotter(x_data1, x_data2, y_data, fig_nm, subplot_rows, subplot_cols, y_max):
    ''' x_data- numpy array of shape (1, numE)
        y_data- numpy array of shape (numN, numE, dimOutput)
        fig_nm- string '''

    # create a figure with subplots
    fig, ax = plt.subplots(subplot_rows, subplot_cols)

    # plotting on each subplot
    cnt = 0
    l=[0]*numN
    clrs = ['r', 'r', 'r']
    lnStyls = [":", "--", "-",]
    subTitles = ["F"+r"$_{tang}$", "F"+r"$_{normal}$", "F"+r"$_{axial}$"]
    for i in range(subplot_rows):
        if subplot_cols>1:
            for j in range(subplot_cols):
                for k in range(numN):
                    if k<3:
                        l[k], = ax[i][j].plot(x_data1.reshape((numE)), y_data[k, :, cnt].reshape((numE)), ls=lnStyls[k], lw=0.5, color=clrs[k])
                    else:
                        l[k], = ax[i][j].plot(x_data2.reshape((numE)), y_data[k, :, cnt].reshape((numE)), ls=lnStyls[k], lw=0.5, color=clrs[k])

                matplotlib.axes.Axes.text(ax[i][j], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i][j].transAxes)

                # axes limit settings
                if cnt<3:
                    y_max = 5*(cnt+2)
                    major_ticks = np.arange(0, y_max, 5)
                    minor_ticks = np.arange(0, y_max, 0.5)
                else:
                    y_max=1
                    major_ticks = np.arange(0, y_max, 0.2)
                    minor_ticks = np.arange(0, y_max, 0.02)
                ax[i][j].set_ylim(0, y_max)
                ax[i][j].set_xlim(0, 0.6)

                # y-axis ticks setting    
                ax[i][j].set_yticks(major_ticks)
                ax[i][j].set_yticks(minor_ticks, minor=True)
                ax[i][j].grid(which='both', axis='y')
                ax[i][j].grid(which='minor', axis='y', alpha=0.2)
                ax[i][j].grid(which='major', axis='y', alpha=0.5)
                ax[i][j].tick_params(labelsize="x-small")

                cnt += 1
        else:
            for k in range(numN):
                if k<3:
                    l[k], = ax[i].plot(x_data1.reshape((numE)), y_data[k, :, cnt].reshape((numE)), ls=lnStyls[k], lw=0.8, color=clrs[k])
                else:
                    l[k], = ax[i].plot(x_data2.reshape((numE)), y_data[k, :, cnt].reshape((numE)), ls=lnStyls[k], lw=0.8, color=clrs[k])

            matplotlib.axes.Axes.text(ax[i], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i].transAxes)

            # axes limit settings
            # if cnt<3:
            #     y_max = 5*(cnt+2)
            #     major_ticks = np.arange(0, y_max, 5)
            #     minor_ticks = np.arange(0, y_max, 0.5)
            # else:
            #     y_max=1
            #     major_ticks = np.arange(0, y_max, 0.2)
            #     minor_ticks = np.arange(0, y_max, 0.02)
            # ax[i].set_ylim(0, y_max)
            # ax[i].set_xlim(0, 0.6)

            # y-axis ticks setting    
            # ax[i].set_yticks(major_ticks)
            # ax[i].set_yticks(minor_ticks, minor=True)
            # ax[i].grid(which='both', axis='y')
            # ax[i].grid(which='minor', axis='y', alpha=0.2)
            # ax[i].grid(which='major', axis='y', alpha=0.5)
            # ax[i].tick_params(labelsize="x-small")

            cnt += 1


    fig.suptitle("Cutting forces vs Feedrate for different values of ADOC", fontsize=16)
    fig.legend(tuple(l) , ("ADOC=1", "ADOC=1.4", "ADOC=1.8"), loc = 'lower center', ncol=6, labelspacing=0. )
    plt.savefig(fig_nm + ".svg")
    # plt.savefig(fig_nm + ".png", dpi=1200)

    # closing all figures
    plt.close('all')

############################################################################################################




plotter(x_data1, x_data2, y_data, plt_folder+plt_name, num_subplot_rows, num_subplot_cols, 1)
