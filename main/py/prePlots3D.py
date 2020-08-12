# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

datafile_folder = "csv/"
# datafile_name = "train_data_0-3"
datafile_name = "train_data"
datafile_ext = ".csv"

dimInput = 2
dimOutput = 3
numN = 91
numE = 10
numDataPts = numN*numE     # total number of data points

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

############################################################################################################

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
n = np.linspace(1, 1.9, numN)
e = np.linspace(0.2, 0.29, numE)
Xn, Ye = np.meshgrid(n, e, indexing="ij")
Z = np.empty((dimOutput, numN, numE))

for i in range(numN):
	for j in range(numE):
		Z[:, i, j] = data_output[numE*i+j, :]

for i in range(dimOutput):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(Xn, Ye, Z[i], cmap=cm.seismic, linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0, 6)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.savefig("plots/output_0-3_" + repr(i+1) + ".png")
    plt.savefig("plots/output_3-10_" + repr(i+1) + ".png")
    plt.show()
