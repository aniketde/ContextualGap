from __future__ import division
import numpy as np
from scipy import io
import DataCreate.DataCreate as DC
# import CrossValidation.CrossValidation as CV
import Algorithms.UCBTimeT as UCBT
import os
import sys
import warnings
import bandwidthselection.bandwidths as bwest
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import time


if __name__ == '__main__':
    clArgs = sys.argv
    RunNumber = int(clArgs[1])
    # start_time = time.time()

    ########################################################################################################################
    # This is a main file.
    # 
    #
    ########################################################################################################################


    ### Initialize the algorithm
    ########################################################################################################################
    N_valid = 1 #No. of data points per arm in validation set
    N = 1 #Algorithm starts with one random example assigned to each arm. This is a cold start problem.
    data_flag_multiclass = 'spacecraft_labexperimental' # 'spacecraft_labexperimental' 'synthetic'
    data_flag_multiclass_filename = 'spacecraft_labexperimental' # 'synthetic'
    reward_scenario =  'spacecraft_labbased' #   'spacecraft_labbased' 'synthetic'

    crossval_flag = 0 # 0 if you don't need cross validation, 2 for batch learning way of doing cross valid, 3 for silverman's rule
    Main_Program_flag = 1  # This variable is 0 when doing cross validation and 1 when running main block. Don't change this.
    # In future release we will get rid of this.

    #Random seeds so that we use same sequence of data points while evaluating algorithms
    randomSeedsTrain = np.array([15485867, 15486277, 15486727, 15487039,
                                 15485917, 15486281, 15486739, 15487049,
                                 15485927, 15486283, 15486749, 15487061,
                                 15485933, 15486287, 15486769, 15487067,
                                 15485941, 15486347, 15486773, 15487097,
                                 15485959, 15486421, 15486781, 15487103,
                                 15485989, 15486433, 15486791, 15487139,
                                 15485993, 15486437, 15486803, 15487151,
                                 15486013, 15486451, 15486827, 15487177,
                                 15486041, 15486469, 15486833, 15487237,
                                 15486047, 15486481, 15486857, 15487243,
                                 15486059, 15486487, 15486869, 15487249,
                                 15486071, 15486491, 15486871, 15487253])

    randomSeedsTest = np.array([15486101, 15486511, 15486883, 15487271,
                                15486139, 15486517, 15486893, 15487291,
                                15486157, 15486533, 15486907, 15487309,
                                15486173, 15486557, 15486917, 15487313,
                                15486181, 15486571, 15486929, 15487319,
                                15486193, 15486589, 15486931, 15487331,
                                15486209, 15486649, 15486953, 15487361,
                                15486221, 15486671, 15486967, 15487399,
                                15486227, 15486673, 15486997, 15487403,
                                15486241, 15486703, 15487001, 15487429,
                                15486257, 15486707, 15487007, 15487457,
                                15486259, 15486719, 15487019, 15487469])

    ########################################################################################################################


    ### List of algorithms
    ########################################################################################################################
    # These are different algorithm you could run and compare. You can add your own aglorithm in the list by modifying DataCreate
    # and GaussianKernels files in the respective libraries.
    #algorithm_list = ['KTL-UCB-TaskSimEst', 'Lin-UCB-Ind', 'KTL-UCB-TaskSim', 'Lin-UCB-Pool']
    algorithm_list = ['Lin-UCB-Ind']
    ucb_flag = 'KernelTS'#'EpsilonGreedy' # 'UniformSampling'  'BayesGap', 'KernelUCB', 'KernelUCBMod' 'KernelTS'
    ########################################################################################################################


    ### cross validation Block
    ########################################################################################################################

    Parameter_Dict = dict()


    ########################################################################################################################


    ### Main Algorithm Block
    ########################################################################################################################
    if data_flag_multiclass == 'spacecraft_labexperimental':
        TCV_train = 1000
        TCV_validate = 1000
        TEV_train = 3000
        TEV_test = 950
    elif data_flag_multiclass == 'synthetic':
        TCV_train = 2000
        TCV_validate = 1000
        TEV_train = 2000
        TEV_test = 1000



    Main_Program_flag = 1 # This variable is 0 when doing cross validation and 1 when running main block. Don't change this.
    # In future release we will get rid of this.
    # Runs = 10 #Number of times we repeat the experiments.
    # Results_dict = di000
    # n_grid = 5
    # nStartbw = -3
    # nEndbw = 0
    # bw_x_grid = np.logspace(nStartbw, nEndbw, n_grid)
    # nStartgamma = -6
    # nEndgamma = -2
    # gamma_grid = np.logspace(nStartgamma, nEndgamma, n_grid)

    #load values from cross validation
    ipfilestr = './ExperimentResults/' + data_flag_multiclass + '/' + ucb_flag + 'CV.mat'
    cvData = io.loadmat(ipfilestr)
    bw_x_grid = cvData['bw_x_grid'][0]
    gamma_grid = cvData['gamma_grid'][0]
    minGrid = cvData['MinGrid']
    cvData = None   #delete data please

    bw_x = bw_x_grid[minGrid[0][0]]
    gamma = gamma_grid[minGrid[1][0]]
    alpha = 1

    Nsr = 50  # number of points to test simple regret for

    #Initialization
    AverageRegretGrid = 0
    AverageAccuracyGrid = 0
    RegretUCBRuns= np.zeros([TEV_train,])
    CumRegretUCBRuns= np.zeros([TEV_train,])
    SimpleRegretUCBRuns= np.zeros([int(np.floor(TEV_train/Nsr)),])
    # SimpleRegretUCBRuns = np.zeros((Runs,))

    # for RunNumber in range(0,Runs):
    #    run numbering
    start_time = time.time()
    algo = 0
    algorithm_flag = algorithm_list[0]
    # for algorithm_flag in algorithm_list:

    # random seed
    rngTest = np.random.RandomState(randomSeedsTest[RunNumber])

    # Get the train data. This is just one example assigned to each arm randomly when N = 1 (cold start)
    DataXY = DC.TrainDataCollect(data_flag_multiclass, N_valid, N, randomSeedsTest[RunNumber],
                                 Main_Program_flag, bw_x, gamma, TCV_train, TCV_validate, TEV_train, TEV_test)
    print("Algorithm " + "Run number" + str(RunNumber))
    # print(algorithm_flag, RunNumber)
    # Run the bandit algorithm and get regret/reward with selected arm
    AverageRegret, AverageAccuracy, CumregretUCB, CumSimpleregretUCB, Selected_Arm_T, Exact_Arm_T, Best_Arm_T, Task_sim_dict, Best_Arm_test, Selected_Arm_test = UCBT.ContextBanditUCBRunForTSteps(DataXY, TEV_train, bw_x, gamma, alpha, algorithm_flag, reward_scenario, Nsr, TEV_test, Main_Program_flag, ucb_flag)

    # Store the result
    AverageRegretGrid = AverageRegret
    AverageAccuracyGrid = AverageAccuracy
    RegretUCBRuns = CumregretUCB
    CumRegretUCBRuns = np.cumsum(CumregretUCB)
    SimpleRegretUCBRuns = CumSimpleregretUCB

    CodeRunTime= time.time()-start_time

    print "code runtime="+str(CodeRunTime)

    # save data
    datadict = dict( gamma = gamma, bw_x = bw_x, AverageRegretGrid=AverageRegretGrid, AverageAccuracyGrid=AverageAccuracyGrid,
            RegretUCBRuns=RegretUCBRuns, CumRegretUCBRuns=CumRegretUCBRuns, SimpleRegretUCBRuns=SimpleRegretUCBRuns,
            data_flag_multiclass=data_flag_multiclass, reward_scenario=reward_scenario, DataXY=Task_sim_dict,
            TCV_train=TCV_train, TCV_validate=TCV_validate, TEV_train=TEV_train, TEV_test=TEV_test, Nsr=Nsr,
            alpha=alpha, Best_Arm_test=Best_Arm_test, Selected_Arm_test=Selected_Arm_test )
    filestr = './ExperimentResults/' + data_flag_multiclass + '/' + ucb_flag + 'Run_'+str(RunNumber)+'.mat'
    io.savemat(filestr, datadict)
