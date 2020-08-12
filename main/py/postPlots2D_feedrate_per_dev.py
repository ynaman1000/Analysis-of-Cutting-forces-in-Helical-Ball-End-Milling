import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib

datafile_folder = "csv/"
datafile_name = "per_dev_feedrate"
datafile_ext = ".csv"

dimInput = 2       # input dimension
dimOutput = 3      # output dimension

numN = 5
numE = 3
numDataPts = numN*numE     # total number of data points

plt_folder = "plots/"
plt_name = "per_dev_feedrate"
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
x_data = data_input[:, 0]
y_data = np.empty([numE, numN, dimOutput])
for i in range(numN):
    for j in range(numE):
        y_data[j, i, :] = data_output[numE*i+j, :]

print(y_data.shape)
print(x_data)

############################################################################################################

############################################### Defining Animation function ###############################################

def plotter(x_data, y_data, fig_nm, subplot_rows, subplot_cols, y_max):
    ''' x_data- numpy array of shape (1, numE)
        y_data- numpy array of shape (numN, numE, dimOutput)
        fig_nm- string '''

    # create a figure with subplots
    fig, ax = plt.subplots(subplot_rows, subplot_cols)

    # plotting on each subplot
    cnt = 0
    l=[0]*numE
    clrs = ['k', 'r', 'b']
    subTitles = ["F"+r"$_{feed}$", "F"+r"$_{normal}$", "F"+r"$_{axial}$"]
    for i in range(subplot_rows):
        if subplot_cols>1:
            for j in range(subplot_cols):
                for k in range(numE):
                    l[k], = ax[i][j].plot(x_data[5*k:5*(k+1)].reshape((numN)), y_data[k, :, cnt].reshape((numN)), ls='-', lw=0.3, color=clrs[k])

                matplotlib.axes.Axes.text(ax[i][j], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i][j].transAxes)

                # axes limit settings
                major_ticks = np.arange(0, y_max, 0.04)
                minor_ticks = np.arange(0, y_max, 0.004)
                ax[i][j].set_ylim(0, y_max)
                ax[i][j].set_xlim(0, 10)

                # y-axis ticks setting    
                ax[i][j].set_yticks(major_ticks)
                ax[i][j].set_yticks(minor_ticks, minor=True)
                ax[i][j].grid(which='both', axis='y')
                ax[i][j].grid(which='minor', axis='y', alpha=0.2)
                ax[i][j].grid(which='major', axis='y', alpha=0.5)
                ax[i][j].tick_params(labelsize="x-small")

                cnt += 1
        else:
            for k in range(numE):
                l[k], = ax[i].plot(x_data[5*k:5*(k+1)].reshape((numN)), y_data[k, :, cnt].reshape((numN)), ls='-', lw=0.7, color=clrs[k])

            matplotlib.axes.Axes.text(ax[i], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i].transAxes)

            # axes limit settings
            # major_ticks = np.arange(0, y_max, 0.04)
            # minor_ticks = np.arange(0, y_max, 0.004)
            # ax[i][j].set_ylim(0, y_max)
            # ax[i][j].set_xlim(0, 10)

            # # y-axis ticks setting    
            # ax[i][j].set_yticks(major_ticks)
            # ax[i][j].set_yticks(minor_ticks, minor=True)
            # ax[i][j].grid(which='both', axis='y')
            # ax[i][j].grid(which='minor', axis='y', alpha=0.2)
            # ax[i][j].grid(which='major', axis='y', alpha=0.5)
            # ax[i][j].tick_params(labelsize="x-small")

            cnt += 1
    ax[2].set_xlabel("ADOC(mm)")

    # fig.suptitle("Percent deviations vs ADOC for different values of Feedrate", fontsize=14)
    fig.legend(tuple(l) , ("Feedrate = 0.2 mm", "Feedrate = 0.24 mm", "Feedrate = 0.28 mm"), loc = 'lower center', bbox_to_anchor=(0.5, 0.89), ncol=6, labelspacing=0. )
    plt.savefig(fig_nm + ".svg")
    # plt.savefig(fig_nm + ".png", dpi=1200)

    # closing all figures
    plt.close('all')

############################################################################################################




plotter(x_data, y_data, plt_folder+plt_name, num_subplot_rows, num_subplot_cols, 0.2)
