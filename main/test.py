# -*- coding: utf-8 -*-
import torch
import numpy as np
from random import shuffle

############################################### Manual Setting ###############################################

# np.set_printoptions(threshold=np.inf)
# torch.set_printoptions(threshold=np.inf, sci_mode=False)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
print("CUDA Available:", torch.cuda.is_available())

# datafile_folder = "/home/naman/OneDrive/Summer_Projects/DL_in_FEM_Project/Vibration_Analysis/Project/Proj4_n_varied_bet_3_and_10/Test_Data/"
datafile_folder = "Test_Data/"
datafile_name = "test_data"    # name of data file
datafile_ext = ".csv"

# modelfile_folder = "Analysis_files/new/"
modelfile_folder = "Analysis_files/final_tuning/Case1_L1_N1/"
modelfile_folder = "Analysis_files/final_tuning/Case2_L1_N1/"
# modelfile_names = ["initLR-0.049569852132014834_lrUC-5_lrUP-3528_L1-3_N1-200_L2-1_N2-1_2019_7_6_14_14_25_1"]
modelfile_names = [
"initLR-0.01_lrUC-5_lrUP-5000_L1-3_N1-173_L2-1_N2-3_2019_7_7_2_22_25_1"
                    ]
modelfile_ext = ".pth"

dimInput = 2
dimOutput = 6
numDataPts = 50*100

doLayerNorm = True       # for performing layer normalization on each layer
aF = torch.nn.PReLU        # activation functions

make_report = True         # for making report
# file_folder = "Report_files/final_tuning/cross/Case1_L1_N1/"
file_folder = "Report_files/final_tuning/same/Case2_L1_N1/"

############################################################################################################


############################################### Loading Data ###############################################

# function for loading data
def load_data(fl_nm):
    ''' Returns numpy array of shape (total data points, 7) '''
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
data_arr = data_arr.transpose()

# defining data variables as numpy arrays
test_data_input_arr = data_arr[0:dimInput].transpose()
test_data_output_arr = data_arr[dimInput:].transpose()

# converting data variables to torch tensors
test_data_input = torch.from_numpy(test_data_input_arr).to(device)
assert test_data_input.shape[1] == dimInput
test_data_output = torch.from_numpy(test_data_output_arr).to(device)
assert test_data_output.shape[1] == dimOutput

############################################################################################################


############################################### Defining Model Class ###############################################

class Model(torch.nn.Module):
    ''' Using activationFun activation function in all hidden layes '''
    def __init__(self, inputSize, outputSize, hlwaf, n1, hlwoaf, n2, ln, activationFun):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential()
        # Adding hidden layers without activation function
        if hlwoaf > 0:
            self.addLayer(inputSize, outputSize*(n2**1))
            for i in range(hlwoaf-1):
                self.addLayer(outputSize*(n2**(i+1)), outputSize*(n2**(i+2)))
        # Adding hidden layers with activation function
            self.addLayer(outputSize*(n2**hlwoaf), n1, ln, activationFun)
        else:
            self.addLayer(inputSize, n1, ln, activationFun)
        for _i in range(hlwaf-1):
            self.addLayer(n1, n1, ln, activationFun)
        # Adding hidden layers without activation function
        if hlwoaf > 0:
            self.addLayer(n1, outputSize*(n2**hlwoaf))
            for i in range(hlwoaf):
                self.addLayer(outputSize*(n2**(hlwoaf-i)), outputSize*(n2**(hlwoaf-i-1)))
        else:
            self.addLayer(n1, outputSize)

    def addLayer(self, inDim, outDim, LN=False, actFun=None):
        if LN:
            if actFun != None:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim), torch.nn.LayerNorm(outDim), actFun())
            else:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim), torch.nn.LayerNorm(outDim))
        else:
            if actFun != None:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim), actFun())
            else:
                self.layers = torch.nn.Sequential(self.layers, torch.nn.Linear(inDim, outDim))

    def forward(self, inputData):
        pred = self.layers(inputData)
        return pred

#####################################################################################################################


for modelfile_name in modelfile_names:
    L1i = modelfile_name.find("_L1-")
    N1i = modelfile_name.find("_N1-")
    L2i = modelfile_name.find("_L2-")
    N2i = modelfile_name.find("_N2-")
    L1 = int(modelfile_name[L1i+4])                   # number of hidden layers with activation function
    N1 = int(modelfile_name[N1i+4:N1i+4+3])                # number of units per hidden layer with activation function
    L2 = int(modelfile_name[L2i+4])                   # number of hidden layers without activation function
    N2 = int(modelfile_name[N2i+4])                # hidden layers without activation function will have units = dimOutput*(N2**?)

    model = Model(dimInput, dimOutput, L1, N1, L2, N2, doLayerNorm, aF)
    model = Model(dimInput, dimOutput, L1, N1, L2, N2, doLayerNorm, aF)
    model.load_state_dict(torch.load(modelfile_folder+modelfile_name+modelfile_ext))
    model.eval()
    model.to(device)
    test_pred = model(test_data_input)
    test_per_dev = torch.div(torch.abs(test_pred - test_data_output), test_data_output)*100

    test_max_per_dev = torch.max(test_per_dev, dim=0)   # not a tensor, see return type of torch.max
    test_max_per_dev = test_max_per_dev.values.detach().cpu().numpy().reshape(dimOutput)
    test_mean_per_dev = torch.mean(test_per_dev, dim=0)
    test_per_dev_sorted, __ = torch.sort(test_per_dev, dim=0)
    test_99per_per_dev = test_per_dev_sorted[int(test_per_dev.shape[0]*0.99), :]
    test_95per_per_dev = test_per_dev_sorted[int(test_per_dev.shape[0]*0.95), :]
    test_95per_per_dev = test_95per_per_dev.detach().cpu().numpy().reshape(dimOutput)
    test_99per_per_dev = test_99per_per_dev.detach().cpu().numpy().reshape(dimOutput)

    test_pred = test_pred.detach().cpu().numpy().reshape((numDataPts, dimOutput))
    test_per_dev = test_per_dev.detach().cpu().numpy().reshape((numDataPts, dimOutput))
    # print(test_per_dev)


    if make_report:
        file_name = file_folder + modelfile_name
        print("Working on ", file_name, ".............................................")
        report = open(file_name + ".csv", 'w+')
        report.write(','.join(["", "n", "eta", "beta1", "beta2", "beta3", "x21", "x31", "x32", "\n"]))
        for i in range(numDataPts):
            report.write(','.join(["Actual"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_data_output[i].tolist()] + ["\n"]))
            report.write(','.join(["Prediction"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_pred[i].tolist()] + ["\n"]))
            report.write(','.join(["Per Dev"] + [repr(e) for e in test_data_input[i].tolist()] + [repr(e) for e in test_per_dev[i].tolist()] + ["\n", "\n"]))

        report.write(','.join(["\nMean", "", ""] + [repr(e) for e in test_mean_per_dev.tolist()]))
        report.write(','.join(["\n95th percentile", "", ""] + [repr(e) for e in test_95per_per_dev.tolist()]))
        report.write(','.join(["\n99th percentile", "", ""] + [repr(e) for e in test_99per_per_dev.tolist()]))
        report.write(','.join(["\nWorst", "", ""] + [repr(e) for e in test_max_per_dev.tolist()] + ["\n"]))

        print("Closing report file.")
        report.close()
    else:
        for i in range(numDataPts):
            print(test_data_output[i])
            print(test_pred[i])
            print(test_per_dev[i])
            print("\n")
