
from __future__ import division
import numpy as np
import sklearn.metrics.pairwise as Kern
import DataCreate.DataCreate as DC
import KernelCalculation.GaussianKernels as GK


#This function runs the bandit algorithm at time t
def ContextBanditUCB(DataXY, tt, bw_x, gamma, alpha,algorithm_flag):
    #KTLUCB
    total_samples, samples_per_task, y, X_total, KInvMatrices = DC.AllDataCollect(DataXY,algorithm_flag)

    #print algorithm_flag, X_total.shape
    samples_per_task = samples_per_task.astype(int)
    A = DataXY['NoOfArms']

    #K_sim,Task_sim,DataXY = GK.GetKernelMatrix(DataXY,X_total,A,total_samples,samples_per_task,bw_x,algorithm_flag)
    Task_sim = np.identity(A)
    # Run the UCB estimate
    # KRR Estimate using training data and direct inverse

    # InvTerm = np.linalg.inv(K_sim + gamma * np.identity(K_sim.shape[0]))

    reward = np.zeros([A, 1])
    reward_est = np.zeros([A])
    reward_conf= np.zeros([A])
    UCB = np.zeros([A])
    LCB = np.zeros([A])
    armTest = DataXY['armTest']
    aa = armTest[tt]
    Testfeatures = DataXY['Testfeatures']
    if (DataXY['TDependentLoss']==0):
        X_test = Testfeatures[tt, :]
    elif (DataXY['TDependentLoss']==1):
        X_test = Testfeatures[tt, :-1]

    X_test = X_test[np.newaxis, :]
    rr1 = 0
    if algorithm_flag == 'Lin-UCB-Ind':         #for loop inside if for optimization
        for aa in range(0, A):
            K_x = np.zeros((X_total.shape[0], X_test.shape[0]))
            rr = 0
            for i in range(0, A):
                Xi = X_total[rr:rr + samples_per_task[i][0], :]
                K_x[rr:rr + samples_per_task[i][0], :] = Task_sim[i, aa] * Kern.rbf_kernel(Xi,X_test,bw_x)
                rr = rr + samples_per_task[i][0]


            k_x_a = Kern.rbf_kernel(X_test, X_test, bw_x)

            K_x_short = K_x[rr1:rr1 + samples_per_task[aa][0]]
            reward_est[aa] = np.transpose(K_x_short).dot(KInvMatrices[aa]).dot(y[rr1:rr1 + samples_per_task[aa][0]])
            reward_conf[aa] = k_x_a - np.transpose(K_x_short).dot(KInvMatrices[aa]).dot(K_x_short)
            rr1 = rr1 + samples_per_task[aa][0]

            '''
            if  k_x_a - np.transpose(K_x).dot(InvTerm).dot(eta).dot(K_x) < 0:
                print tt, aa, reward_conf
                reward_conf = 0.0
            '''

            UCB[aa] = reward_est[aa] + (alpha) * np.sqrt(reward_conf[aa])
            LCB[aa] = reward_est[aa] - (alpha) * np.sqrt(reward_conf[aa])
    elif algorithm_flag == 'Thompson-Sampling':
        for aa in range(0,A):
            AInv = KInvMatrices[aa][0]
            b = KInvMatrices[aa][1]
            muhata = AInv.dot(b.T)
            mutilde = np.random.multivariate_normal(muhata[:,0], alpha*AInv)
            reward_est[aa] = X_test.dot(mutilde)


    X_test = Testfeatures[tt, :]
    X_test = X_test[np.newaxis, :]
    
    return UCB,LCB,np.sqrt(reward_conf),reward_est,X_test,DataXY