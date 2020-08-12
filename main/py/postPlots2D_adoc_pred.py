import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib

datafile_folder = "csv/"
datafile_name_act = "pred_act_adoc"
datafile_name_pred = "pred_adoc"
datafile_ext = ".csv"

dimInput = 2       # input dimension
dimOutput = 3      # output dimension

numN_act = 3
numE_act = 5
numDataPts_act = numN_act*numE_act     # total number of data points
numN_pred = 3
numE_pred = 5
numDataPts_pred = numN_act*numE_pred     # total number of data points

plt_folder = "plots/"
plt_name = "pred_adoc"
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


data_arr_act  = load_data(datafile_folder+datafile_name_act+datafile_ext)
assert data_arr_act.shape == (numDataPts_act, dimInput+dimOutput), data_arr_act.shape
print(data_arr_act.shape)

data_input_act, data_output_act = np.split(data_arr_act, [dimInput], axis=1)
x_data_act = np.empty([numN_act, numE_act])
for i in range(numN_act):
    x_data_act[i, :] = data_input_act[numE_act*i:numE_act*(i+1), 1]
y_data_act = np.empty([numN_act, numE_act, dimOutput])
for i in range(numN_act):
    for j in range(numE_act):
        y_data_act[i, j, :] = data_output_act[numE_act*i+j, :]

data_arr_pred  = load_data(datafile_folder+datafile_name_pred+datafile_ext)
assert data_arr_pred.shape == (numDataPts_pred, dimInput+dimOutput), data_arr_pred.shape
print(data_arr_pred.shape)

data_input_pred, data_output_pred = np.split(data_arr_pred, [dimInput], axis=1)
x_data_pred = np.empty([numN_pred, numE_pred])
for i in range(numN_pred):
    x_data_pred[i, :] = data_input_pred[numE_pred*i:numE_pred*(i+1), 1]
y_data_pred = np.empty([numN_pred, numE_pred, dimOutput])
for i in range(numN_pred):
    for j in range(numE_pred):
        y_data_pred[i, j, :] = data_output_pred[numE_pred*i+j, :]

print(y_data_act.shape)
print(y_data_pred.shape)

############################################################################################################

############################################### Defining Animation function ###############################################

def plotter(x_data_act, y_data_act, x_data_pred, y_data_pred, fig_nm, subplot_rows, subplot_cols, y_max):
    ''' x_data- numpy array of shape (1, numE_act)
        y_data- numpy array of shape (numN_act, numE_act, dimOutput)
        fig_nm- string '''

    # create a figure with subplots
    fig, ax = plt.subplots(subplot_rows, subplot_cols)

    # plotting on each subplot
    cnt = 0
    l_act=[0]*numN_act
    l_pred=[0]*numN_pred
    clrs = ['m', 'b', 'k', 'r']
    ylbls = ["Tang. force(N)", "Normal force(N)", "Axial force(N)"]
    mrkrs = [".", "+", "x"]
    subTitles = ["F"+r"$_{feed}$", "F"+r"$_{normal}$", "F"+r"$_{axial}$"]
    for i in range(subplot_rows):
        for j in range(numN_act):
            l_act[j], = ax[i].plot(x_data_act[j].reshape((numE_act)), y_data_act[j, :, i].reshape((numE_act)), lw=1, color=clrs[1+j])
            l_pred[j], = ax[i].plot(x_data_pred[j].reshape((numE_pred)), y_data_pred[j, :, i].reshape((numE_pred)), marker=mrkrs[j], ms=3, lw=0, color=clrs[0])
        matplotlib.axes.Axes.text(ax[i], x=0.5, y=0.9, s=subTitles[i], horizontalalignment='center',verticalalignment='center', transform=ax[i].transAxes)
        ax[i].set_ylabel(ylbls[i])
    ax[2].set_xlabel("Feedrate(mm)")

        # if subplot_cols>1:
        #     for j in range(subplot_cols):
        #         for k in range(numN_act):
        #             if k<3:
        #                 l_act[k], = ax[i][j].plot(x_data_act1.reshape((numE_act)), y_data_act[k, :, cnt].reshape((numE_act)), ls=lnStyls[k], lw=0.3, color=clrs[k])
        #                 l_pred[k], = ax[i][j].plot(x_data_pred1.reshape((numE_pred)), y_data_pred[k, :, cnt].reshape((numE_pred)), marker="x", ms=3, lw=0, color=clrs[k])
        #             else:
        #                 l_act[k], = ax[i][j].plot(x_data_act2.reshape((numE_act)), y_data_act[k, :, cnt].reshape((numE_act)), ls=lnStyls[k], lw=0.3, color=clrs[k])
        #                 l_pred[k], = ax[i][j].plot(x_data_pred2.reshape((numE_pred)), y_data_pred[k, :, cnt].reshape((numE_pred)), marker="x", ms=3, lw=0, color=clrs[k])

        #         matplotlib.axes.Axes.text(ax[i][j], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i][j].transAxes)

        #         # axes limit settings
        #         if cnt<3:
        #             y_max = 5*(cnt+2)
        #             major_ticks = np.arange(0, y_max, 5)
        #             minor_ticks = np.arange(0, y_max, 0.5)
        #         else:
        #             y_max=1
        #             major_ticks = np.arange(0, y_max, 0.2)
        #             minor_ticks = np.arange(0, y_max, 0.02)
        #         ax[i][j].set_ylim(0, y_max)
        #         ax[i][j].set_xlim(0, 0.6)

        #         # y-axis ticks setting    
        #         ax[i][j].set_yticks(major_ticks)
        #         ax[i][j].set_yticks(minor_ticks, minor=True)
        #         ax[i][j].grid(which='both', axis='y')
        #         ax[i][j].grid(which='minor', axis='y', alpha=0.2)
        #         ax[i][j].grid(which='major', axis='y', alpha=0.5)
        #         ax[i][j].tick_params(labelsize="x-small")

        #         cnt += 1
        # else:
        #     for k in range(numN_act):
        #         if k<3:
        #             l_act[k], = ax[i].plot(x_data_act1.reshape((numE_act)), y_data_act[k, :, cnt].reshape((numE_act)), ls=lnStyls[k], lw=0.7, color=clrs[k])
        #             l_pred[k], = ax[i].plot(x_data_pred1.reshape((numE_pred)), y_data_pred[k, :, cnt].reshape((numE_pred)), marker="x", ms=3, lw=0, color=clrs[k])
        #         else:
        #             l_act[k], = ax[i].plot(x_data_act2.reshape((numE_act)), y_data_act[k, :, cnt].reshape((numE_act)), ls=lnStyls[k], lw=0.7, color=clrs[k])
        #             l_pred[k], = ax[i].plot(x_data_pred2.reshape((numE_pred)), y_data_pred[k, :, cnt].reshape((numE_pred)), marker="x", ms=3, lw=0, color=clrs[k])

        #     matplotlib.axes.Axes.text(ax[i], x=0.5, y=0.9, s=subTitles[cnt], horizontalalignment='center',verticalalignment='center', transform=ax[i].transAxes)

        #     # axes limit settings
        #     # if cnt<3:
        #     #     y_max = 5*(cnt+2)
        #     #     major_ticks = np.arange(0, y_max, 5)
        #     #     minor_ticks = np.arange(0, y_max, 0.5)
        #     # else:
        #     #     y_max=1
        #     #     major_ticks = np.arange(0, y_max, 0.2)
        #     #     minor_ticks = np.arange(0, y_max, 0.02)
        #     # ax[i][j].set_ylim(0, y_max)
        #     # ax[i][j].set_xlim(0, 0.6)

        #     # # y-axis ticks setting    
        #     # ax[i][j].set_yticks(major_ticks)
        #     # ax[i][j].set_yticks(minor_ticks, minor=True)
        #     # ax[i][j].grid(which='both', axis='y')
        #     # ax[i][j].grid(which='minor', axis='y', alpha=0.2)
        #     # ax[i][j].grid(which='major', axis='y', alpha=0.5)
        #     # ax[i][j].tick_params(labelsize="x-small")

        #     cnt += 1

    # fig.suptitle("Cutting forces vs Feedrate for different values of ADOC", fontsize=16)
    # fig.legend(tuple(l_act+l_pred) , ("\u03B7=1", "\u03B7=2", "\u03B7=3", "\u03B7=4", "\u03B7=6", "\u03B7=8", "\u03B7=1", "\u03B7=2", "\u03B7=3", "\u03B7=4", "\u03B7=6", "\u03B7=8"), loc = 'lower center', ncol=6, prop={'size': 7}, labelspacing=0. )
    # fig.legend(tuple(l_act) , ("ADOC=1", "ADOC=1.4", "ADOC=1.8", "Predicted Values"), loc = 'lower center', ncol=6, labelspacing=0. )
    fig.legend(tuple(l_act)+tuple(l_pred) , ("ADOC = 1 mm", "ADOC = 1.4 mm", "ADOC = 1.8 mm"), loc = 'lower center', bbox_to_anchor=(0.5, 0.89), ncol=6, labelspacing=0. )
    # fig.legend(tuple(l_act+[l_pred[0]]) , ("\u03B7=1", "\u03B7=2", "\u03B7=3", "\u03B7=4", "\u03B7=6", "\u03B7=8", "Predicted Values"), loc = 'lower center', ncol=7, prop={'size': 7}, labelspacing=0. )
    plt.savefig(fig_nm + ".svg")
    # plt.savefig(fig_nm + ".png", dpi=1200)

    # closing all figures
    plt.close('all')

############################################################################################################


plotter(x_data_act, y_data_act, x_data_pred, y_data_pred, plt_folder+plt_name, num_subplot_rows, num_subplot_cols, 1)
