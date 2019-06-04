from __future__ import division
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import sklearn.datasets as datasets
import scipy.io as spio
import RewardProcess
from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise as Kern
import KernelCalculation.GaussianKernels as GK
from numpy import genfromtxt

def TrainDataCollect(data_flag_multiclass,N_valid,N,RandomSeedNumber,Main_Program_flag, bw_x, gamma, NCV_train, NCV_validate, NEV_train, NEV_test):
    rng = np.random.RandomState(RandomSeedNumber)
    DataXY = dict()

    if data_flag_multiclass == 'synthetic':
        DataXY['TDependentLoss'] = 0
        T = 6100
        A = 20
        np.random.seed(seed=0)
        Features = 2.0*np.pi*np.random.rand(T,1)

        '''
        #Used this during dependent syntehtic data experiments
        Features = np.linspace(0,2*np.pi,T)
        Features = Features[:,np.newaxis]
        '''
        Labels_full = np.zeros((T,))
        for i in range(0,T):
            Labels_full[i] = np.argmax([np.sin(j*Features[i]) for j in range(0,A)])




    elif data_flag_multiclass == 'spacecraft_synthetic':
        DataXY['IPFileName']='./Datasets/synthetic_sc_lbeqn_5d.mat'
        sc_synth = spio.loadmat(DataXY['IPFileName'])
        context_full = sc_synth['context']    # contexts not related to loss function are handled outside
        context_full = context_full # + 0.00001*(np.random.rand(context_full.shape[0], context_full.shape[1]) - 0.5)
        bs = sc_synth['B_s']*1e-4   # convert to 1e-5 Tesla
        br = sc_synth['B_ref']*1e-4
        RP = RewardProcess.RewardProcess('spacecraft')
        dataset_type = '3-axis-norot'
        br_uhc, bs_uhc = RP.InitializeDataset(dataset_type, bs, br)

        T = br.shape[1]
        A = bs.shape[1]
        # context = np.zeros((T,9*A))
        # context[:,0::3] = np.absolute(context_full[0].T)
        # context[:,1::3] = np.absolute(context_full[1].T)
        # context[:,2::3] = np.absolute(context_full[2].T)
        # context = np.zeros((T,A))
        context = context_full.T
        # context = np.concatenate((np.absolute(context_full.T)/3000, np.sign(context_full.T)), axis=1)

        # modifications to context
        Cstd = np.std(context, axis=0)
        Cmax = context.max(axis=0)

        # contextmin = context.min(axis=0)
        # context = (context - context.min(axis=0))
        # contextcube = context.max(axis=0)
        context = context/np.amin(context.max(axis=0))
        # context = context/4998.5418
        # context = context / np.concatenate((Cmax[0:A * A], Cstd[A * A:]))
        # context = context / np.concatenate(((np.ones((A*A,)), Cstd[A*A:])))
        Tvec = np.arange(T)
        Features = np.concatenate((context, Tvec[:, np.newaxis]), axis=1)
        # Features = context

        Labels_full = np.zeros((T,))
        for tt in range(0,T):
            B_S = bs_uhc[:,:,tt]
            B_R = br_uhc[:,tt]
            rewarr = -10*(np.linalg.norm(B_R[:,np.newaxis] - B_S, axis=0))**2
            # Labels_full[tt] = np.argmax(rewarr)
            Labels_full[tt] = np.amax(rewarr)

        # Labels = 1.0*(Labels_full == 0)
        # Labels = Labels_full

        DataXY['TDependentLoss'] = 1

    elif data_flag_multiclass == 'spacecraft_labexperimental':
        DataXY['IPFileName'] = './Datasets/largecoil_postproc.mat'
        sc_synth = spio.loadmat(DataXY['IPFileName'])
        context_full = sc_synth['telMat']  # contexts not related to loss function are handled outside
        bNoise = sc_synth['bNorm'].T
        context_full = context_full[:,:bNoise.shape[0]]
        # context_full = context_full  # + 0.00001*(np.random.rand(context_full.shape[0], context_full.shape[1]) - 0.5)
        RP = RewardProcess.RewardProcess('spacecraft_labbased')
        dataset_type = '3-axis-labbased'
        br = np.zeros((1, bNoise.shape[0]))
        br_uhc, bs_uhc = RP.InitializeDataset(dataset_type, bNoise, br)

        T = br.shape[1]
        A = bNoise.shape[1]
        # context = np.zeros((T,9*A))
        # context[:,0::3] = np.absolute(context_full[0].T)
        # context[:,1::3] = np.absolute(context_full[1].T)
        # context[:,2::3] = np.absolute(context_full[2].T)
        # context = np.zeros((T,A))
        context = context_full.T
        # context = np.concatenate((np.absolute(context_full.T)/3000, np.sign(context_full.T)), axis=1)

        # modifications to context
        Cstd = np.std(context, axis=0)
        Cmax = context.max(axis=0)

        # contextmin = context.min(axis=0)
        # context = (context - context.min(axis=0))
        # contextcube = context.max(axis=0)
        context = context / np.amin(context.max(axis=0))
        # context = context/4998.5418
        # context = context / np.concatenate((Cmax[0:A * A], Cstd[A * A:]))
        # context = context / np.concatenate(((np.ones((A*A,)), Cstd[A*A:])))
        Tvec = np.arange(T)
        Features = np.concatenate((context, Tvec[:, np.newaxis]), axis=1)
        # Features = context

        Labels_full = np.zeros((T,))
        for tt in range(0, T):
            rewarr = -bNoise[tt,:]
            # Labels_full[tt] = np.argmax(rewarr)
            Labels_full[tt] = np.amax(rewarr)

        # Labels = 1.0*(Labels_full == 0)
        # Labels = Labels_full

        DataXY['TDependentLoss'] = 1

    elif data_flag_multiclass =='spacecraft_experimental':
        sc_exp = spio.loadmat('./Datasets/MAB_DATASET_GRIFEX3_2p.mat')
        context = sc_exp['context']    # contexts not related to loss function are handled outside
        bs = sc_exp['B_s']*1e-4   # convert to 1e-5 Tesla
        br = sc_exp['B_ref']*1e-4
        RP = RewardProcess.RewardProcess('spacecraft')
        dataset_type = '1-axis-norot'
        br_uhc, bs_uhc = RP.InitializeDataset(dataset_type, bs, br)

        T = br.shape[0]
        A = bs.shape[1]

        contextmin = context.min(axis=0)
        context = (context - context.min(axis=0))
        contextcube = context.max(axis=0)
        context = context / context.max(axis=0)

        Tvec = np.arange(T)
        Features = np.concatenate((context[:, (0,1,2,3,4,5,6,7,8,9)], Tvec[:,np.newaxis] ), axis=1)
        TDataset = T

        Labels = np.zeros((T,))
        for tt in range(0,T):
            B_S = bs_uhc[tt,:]
            B_R = br_uhc[tt,0]
            rewarr = -np.absolute((B_R**2 - B_S**2))/1e-10 + 5
            Labels[tt] = np.argmax(rewarr)
            # Labels[tt] = np.max(rewarr)

        DataXY['TDependentLoss'] = 1

    if np.min(Labels_full) == 1:
        Labels_full = Labels_full - 1
    elif np.min(Labels_full) == -1:
        Labels_full = Labels_full + 1
    # A = np.unique(Labels).shape[0]  # number of classes
    # Features_valid, Features_train_test, Labels_valid, Labels_train_test = train_test_split(Features, Labels,
    #                                                                                         train_size=int(A * N_valid),
    #                                                                                         shuffle=False,
    #                                                                                         stratify=None) # Labels
    # splitting manually as sklearn 0.18 doesn't have shuffle=false option


    if Main_Program_flag == 0:
        Features = Features[:NCV_train + NCV_validate+A*N,:]
        Labels = Labels_full[:NCV_train + NCV_validate+A*N]
    else:
        Features = Features[(NCV_train+NCV_validate+A*N):(NCV_train+NCV_validate+NEV_train + NEV_test+2*A*N),:]
        Labels = Labels_full[(NCV_train + NCV_validate+A*N) : (NCV_train + NCV_validate + NEV_train + NEV_test+2*A*N)]

    if Main_Program_flag == 0:
        Features_valid = Features[0:int(A * N_valid), :]
        Labels_valid = Labels[0:int(A * N_valid)]
        Features_train_test = Features[int(A * N_valid):, :]
        Labels_train_test = Labels[int(A * N_valid):]
        Features_train = np.copy(Features_valid)
        Features_test = np.copy(Features_train_test)
        Labels_train = np.copy(Labels_valid)
        Labels_test = np.copy(Labels_train_test)
    else:

        Features_train_test = np.copy(Features)
        Labels_train_test = np.copy(Labels)

        Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features_train_test,
                                                                                    Labels_train_test,
                                                                                    train_size=int(
                                                                                        A * N),
                                                                                    random_state=RandomSeedNumber + 17,
                                                                                    stratify=None) # Labels_train_test
        '''
        # Used this during dependent synthetic data experiments
        Features_train = Features_train_test[0:int(A*N),:]
        Labels_train = Labels_train_test[0:int(A*N)]
        Features_test = Features_train_test[int(A * N):, :]
        Labels_test = Labels_train_test[int(A * N):]
        '''

        # Features_train.shape, Features_test.shape, A*N, Labels_train
    if Main_Program_flag == 0:
        M = N_valid
    else:
        M = N

    idx = rng.permutation(A * M)
    Features_train = Features_train[idx, :]
    Labels_train = Labels_train[idx]
    #
    # Labels_test_hist = np.histogram(Labels_test, A)
    # minimum_number_label = np.min(Labels_test_hist[0])
    # Features_test_dummy = np.zeros([A * minimum_number_label, Features_test.shape[1]])
    # Labels_test_dummy = np.zeros([A * minimum_number_label])
    #
    # for aa in range(0, A):
    #     idx = np.where(Labels_test == aa)[0]
    #     # idx = np.asarray(idx)
    #     # idx = idx[0,:]
    #     # idx.astype(int)
    #     Features_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label, :] = Features_test[
    #                                                                                         idx[:minimum_number_label],
    #                                                                                         :]
    #     Labels_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label] = Labels_test[
    #         idx[:minimum_number_label]]
    # idx = rng.permutation(A * minimum_number_label)
    # Features_test = Features_test_dummy[idx, :]
    # Labels_test = Labels_test_dummy[idx]

    for aa in range(0, A):
        XTrain = Features_train[aa * M:(aa + 1) * M, :]
        LabelsTrain = Labels_train[aa * M:(aa + 1) * M]
        # TTrain = Time_train[aa * M:(aa + 1) * M]
        YTrain = np.zeros([M])
        YTrain[LabelsTrain == aa] = 1
        Total_Features = np.copy(XTrain)
        Arm_rewards = np.copy(YTrain)
        KMatrix = np.zeros((M,M))
        if(DataXY['TDependentLoss']==0):
            KMatrix[:,:] = Kern.rbf_kernel(Total_Features, Total_Features, bw_x)
        elif(DataXY['TDependentLoss']==1):
            KMatrix[:, :] = Kern.rbf_kernel(Total_Features[:,:-1], Total_Features[:,:-1], bw_x)

        KInvMatrix = np.linalg.inv(KMatrix + gamma * np.identity(KMatrix.shape[0]))
        AInvMatrix = (1/gamma)*np.eye(Features_train.shape[1])
        # Time_stamps = np.copy(TTrain)

        # # Save training data for KTL UCB
        # train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
        # DataXY[train_datasetKTLUCB] = np.copy(Total_Features)
        #
        # train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
        # DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)
        #
        # # Save training data for KTLEst UCB
        # train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
        # DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)
        #
        # train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
        # DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)

        # Save training data for Lin UCB
        train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
        DataXY[train_datasetLinUCB] = np.copy(Total_Features)

        train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
        DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

        # train_timeLinUCB = 'Train_Labels'

        # Save training data for Pool UCB
        train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
        DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

        train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
        DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)

        # Save balanced training data for Lin UCB Balanced
        train_datasetBalUCB = 'Train_Datasets_BalUCB' + str(int(aa))
        DataXY[train_datasetBalUCB] = np.copy(Total_Features)

        train_labelsBalUCB = 'Train_Labels_BalUCB' + str(int(aa))
        DataXY[train_labelsBalUCB] = np.copy(Arm_rewards)

        # Kernel Matrix
        # train_KMatrix = 'Train_KMatrix' + str(int(aa))
        # DataXY[train_KMatrix] = KMatrix

        # Inverse Kernel Matrix
        train_KInvMatrix = 'Train_KInvMatrix' + str(int(aa))
        DataXY[train_KInvMatrix] = KInvMatrix

        # Thompson sampling parameters
        train_AInvMatrix = 'Train_AInvMatrix' + str(int(aa))
        DataXY[train_AInvMatrix] = AInvMatrix

        train_bVector = 'Train_bVector' + str(int(aa))
        DataXY[train_bVector] = np.zeros((1,Features_train.shape[1]))


    DataXY['Testfeatures'] = np.copy(Features_test)
    DataXY['armTest'] = np.copy(Labels_test)
    DataXY['NoOfArms'] = A
    DataXY['DatasetName'] = data_flag_multiclass
    return DataXY


def AllDataCollect(DataXY,algorithm_flag):
    # Get total samples and samples in each dataset
    A = DataXY['NoOfArms']
    total_samples = 0
    samples_per_task = np.zeros([A, 1])
    for i in range(0, A):
        if algorithm_flag == 'Lin-UCB-Ind':
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
            train_dataset_bal = 'Train_Datasets_BalUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)
        elif algorithm_flag == 'Thompson-Sampling':
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
        #print DataXY.keys()
        if(DataXY['TDependentLoss']==0):
            X = np.copy(DataXY[train_dataset])
        elif(DataXY['TDependentLoss']==1):
            XT = np.copy(DataXY[train_dataset])
            X=XT[:,:-1]

        total_samples = total_samples + X.shape[0]
        samples_per_task[i] = X.shape[0]

    # Collect all labels and all features
    y = np.zeros(total_samples)
    X_total = np.zeros([total_samples, X.shape[1]])
    # KMatrices = []
    KInvMatrices = []
    rr = 0
    for i in range(0, A):
        if algorithm_flag == 'Lin-UCB-Ind':
            train_labels = 'Train_Labels_LinUCB' + str(i)
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
            train_labels_bal = 'Train_Labels_BalUCB' + str(i)
            train_dataset_bal = 'Train_Datasets_BalUCB' + str(i)
            # train_KMatrix = 'Train_KMatrix' + str(i)
            train_KInvMatrix = 'Train_KInvMatrix' + str(i)
            KInvMatrices.append(DataXY[train_KInvMatrix])
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_labels = 'Train_Labels_PoolUCB' + str(i)
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)
            train_KInvMatrix = 'Train_KInvMatrix' + str(i)
            KInvMatrices.append(DataXY[train_KInvMatrix])
        elif algorithm_flag == 'Thompson-Sampling':
            train_labels = 'Train_Labels_LinUCB' + str(i)
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
            train_AInvMatrix = 'Train_AInvMatrix' + str(i)
            train_bVector = 'Train_bVector' + str(i)
            KInvMatrices.append([DataXY[train_AInvMatrix], DataXY[train_bVector]])

        labels = np.copy(DataXY[train_labels])
        y[rr:rr + labels.shape[0]] = np.copy(DataXY[train_labels])
        if(DataXY['TDependentLoss']==0):
            X_total[rr:rr + labels.shape[0], :] = np.copy(DataXY[train_dataset])
        elif(DataXY['TDependentLoss']==1):
            XT_total = np.copy(DataXY[train_dataset])
            X_total[rr:rr + labels.shape[0], :] = XT_total[:,:-1]
        rr = rr + labels.shape[0]

    return total_samples, samples_per_task, y, X_total, KInvMatrices


def AddData(DataXY,arm_tt,algorithm_flag,X_test,reward_test,tt, bw_x, gamma):
    if algorithm_flag == 'Lin-UCB-Ind':
        train_labels = 'Train_Labels_LinUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_LinUCB' + str(arm_tt)
        test_label = 'Test_Labels_LinUCB'
        last_roundXTest = 'Test_Datasets_LinUCB'
        train_labels_bal = 'Train_Labels_BalUCB' + str(arm_tt)
        train_dataset_bal = 'Train_Datasets_BalUCB' + str(arm_tt)
        # train_KMatrix = 'Train_KMatrix' + str(arm_tt)
        train_KInvMatrix = 'Train_KInvMatrix' + str(arm_tt)

        Total_Features = np.copy(DataXY[train_dataset])
        Arm_rewards = np.copy(DataXY[train_labels])

        if (DataXY['TDependentLoss'] == 0):
            KInvMatrix = GK.UpdateKernelMatrix(DataXY[train_KInvMatrix], Total_Features, X_test, bw_x, gamma)
        elif (DataXY['TDependentLoss'] == 1):
            KInvMatrix = GK.UpdateKernelMatrix(DataXY[train_KInvMatrix], Total_Features[:, :-1], X_test[:, :-1], bw_x,
                                               gamma)

        DataXY[train_KInvMatrix] = np.copy(KInvMatrix)

    elif algorithm_flag == 'Lin-UCB-Pool':
        train_labels = 'Train_Labels_PoolUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_PoolUCB' + str(arm_tt)
        test_label = 'Test_Labels_PoolUCB'
        last_roundXTest = 'Test_Datasets_PoolUCB'
        Total_Features = np.copy(DataXY[train_dataset])
        Arm_rewards = np.copy(DataXY[train_labels])

    elif algorithm_flag == 'Thompson-Sampling':
        train_labels = 'Train_Labels_LinUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_LinUCB' + str(arm_tt)
        test_label = 'Test_Labels_LinUCB'
        last_roundXTest = 'Test_Datasets_LinUCB'
        train_AInvMatrix = 'Train_AInvMatrix' + str(arm_tt)
        train_bVector = 'Train_bVector' + str(arm_tt)
        Total_Features = np.copy(DataXY[train_dataset])
        Arm_rewards = np.copy(DataXY[train_labels])

        if (DataXY['TDependentLoss'] == 0):
            Xtemp = X_test
        elif (DataXY['TDependentLoss'] == 1):
            Xtemp = X_test[:,:-1]

        # rank one update to inverse
        AIX = (DataXY[train_AInvMatrix]).dot(Xtemp.T)
        AXscale = 1/(1 + Xtemp.dot(AIX))
        AInvMatrix = DataXY[train_AInvMatrix] - AXscale*np.outer(AIX,AIX)
        bVector = DataXY[train_bVector] + reward_test * X_test

        DataXY[train_AInvMatrix] = np.copy(AInvMatrix)
        DataXY[train_bVector] = np.copy(bVector)



    Total_Features = np.append(Total_Features, X_test, axis=0)
    reward_test = np.ones([1])*reward_test
    Arm_rewards = np.append(Arm_rewards, reward_test, axis=0)

    DataXY[train_dataset] = np.copy(Total_Features)
    DataXY[train_labels] = np.copy(Arm_rewards)
    DataXY[last_roundXTest] = np.copy(X_test)

    if tt == 0:
        armSelectedTT =  np.ones([1])*arm_tt #np.empty([0])
    else:
        armSelectedTT = np.copy(DataXY[test_label])
        armSelectedTT = np.append(armSelectedTT, np.ones([1])*arm_tt, axis=0)
    armSelectedTT = armSelectedTT.astype(int)

    DataXY[test_label] = np.copy(armSelectedTT)

    # # Balanced dataset accounting: works only for binary 0-1 rewards
    # Arm_rewards_bal = np.copy(DataXY[train_labels_bal])
    # Total_Features_bal = np.copy(DataXY[train_dataset_bal])
    # Nones = np.sum(Arm_rewards_bal)
    # Nzeros = np.sum(1.0 - Arm_rewards_bal)
    # RewBal = Nones - Nzeros
    # if (RewBal >= 0 and reward_test == 0 ):
    #     # KInvMatrix = GK.UpdateKernelMatrix(DataXY[train_KInvMatrix], Total_Features_bal, X_test, bw_x, gamma)
    #     Arm_rewards_bal = np.append(Arm_rewards_bal, reward_test, axis=0)
    #     Total_Features_bal = np.append(Total_Features_bal, X_test, axis=0)
    #     DataXY[train_dataset_bal] = np.copy(Total_Features_bal)
    #     DataXY[train_labels_bal] = np.copy(Arm_rewards_bal)
    #     # DataXY[train_KMatrix] = np.copy(KMatrix)
    #     # DataXY[train_KInvMatrix] = np.copy(KInvMatrix)
    # elif (RewBal <= 0 and reward_test == 1):
    #     # KInvMatrix = GK.UpdateKernelMatrix(DataXY[train_KInvMatrix], Total_Features_bal, X_test, bw_x, gamma)
    #     Arm_rewards_bal = np.append(Arm_rewards_bal, reward_test, axis=0)
    #     Total_Features_bal = np.append(Total_Features_bal, X_test, axis=0)
    #     DataXY[train_dataset_bal] = np.copy(Total_Features_bal)
    #     DataXY[train_labels_bal] = np.copy(Arm_rewards_bal)
    #     # DataXY[train_KMatrix] = np.copy(KMatrix)
    #     # DataXY[train_KInvMatrix] = np.copy(KInvMatrix)

    return DataXY
