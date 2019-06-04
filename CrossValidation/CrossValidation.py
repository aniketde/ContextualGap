
from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import KernelCalculation.GaussianKernels as GK
from sklearn.cross_validation import KFold
import Algorithms.UCBTimeT as UCBT

#alpha estimation for Kernel Ridge Regression
def alphaEst(lambda_reg,Y_train,K_train):
    inv_term_est = np.linalg.inv(K_train + lambda_reg * np.identity(K_train.shape[0]))

    alpha_est = np.dot(inv_term_est,Y_train)

    alpha_est = alpha_est[np.newaxis, :]

    return alpha_est

#Main cross-validation block
def CrossValidRegression(bw_x_grid,lambda_reg_grid,fold_cv,algorithm_flag, data_flag_multiclass, N_valid,N,randomSeeds,Main_Program_flag):
    DataXY = DC.TrainDataCollect(data_flag_multiclass, N_valid,N, randomSeeds[0],Main_Program_flag)
    A = DataXY['NoOfArms']
    total_samples, samples_per_task, y, X_total = DC.AllDataCollect(DataXY,algorithm_flag)
    samples_per_task = samples_per_task.astype(int)
    kf = KFold(samples_per_task[0], n_folds=fold_cv)
    err = np.zeros([bw_x_grid.shape[0],lambda_reg_grid.shape[0]])
    for bb1 in range(0, bw_x_grid.shape[0]):
        bw_x = bw_x_grid[bb1]
        for ll in range(0, lambda_reg_grid.shape[0]):
            lambda_reg = lambda_reg_grid[ll]
            err_cv = np.zeros([fold_cv, 1])
            cv = 0
            print("parameters")
            print(bb1, ll)
            for train_index, test_index in kf:
                ind_all = np.linspace(0, total_samples - 1, total_samples)
                ind_all = ind_all.astype(int)

                ind_test = np.zeros([A * test_index.shape[0]]).astype(int)
                samples_per_task_test = np.copy(samples_per_task)
                samples_per_task_train = np.copy(samples_per_task)
                for ii in range(0, A):
                    # print  ii,ind_test[ii*test_index.shape[0]:(ii+1)*test_index.shape[0]].shape,ind_all[ii*samples_per_task[ii]:(test_index.shape[0]+ii*samples_per_task[ii])].shape
                    ind_test[ii * test_index.shape[0]:(ii + 1) * test_index.shape[0]] = ind_all[
                                                                                        ii * samples_per_task[ii]:(
                                                                                        test_index.shape[0] + ii *
                                                                                        samples_per_task[ii])]
                    samples_per_task_test[ii] = int(test_index.shape[0])
                    samples_per_task_train[ii] = int(train_index.shape[0])

                ind_train = np.delete(ind_all, ind_test)
                X_train = X_total[ind_train, :]
                X_test = X_total[ind_test, :]
                Y_train = y[ind_train]
                Y_test = y[ind_test]

                K_train,Task_sim =  GK.GetKernelMatrixCV(X_train,A,X_train.shape[0],samples_per_task_train,bw_x,algorithm_flag)

                alpha_est = alphaEst(lambda_reg, Y_train, K_train)

                K_test =  GK.GetTestKernelMatrixCV(X_train,X_test,A,Task_sim,samples_per_task_train,samples_per_task_test,bw_x)
                # print alpha_est.shape, K_test.shape
                Y_est = np.dot(alpha_est, K_test)
                Y_est = Y_est[0, :]

                err_cv[cv] = np.linalg.norm(Y_test - Y_est)
                cv = cv + 1
            err[bb1, ll] = np.mean(err_cv)


    bb1_min,ll_min = np.where(err == err.min())
    bb1_min = np.ones([1])* bb1_min
    ll_min = np.ones([1]) * ll_min
    bw_x_est = bw_x_grid[bb1_min[0]]
    lambda_reg_est = lambda_reg_grid[ll_min[0]]
    print("minimium Error is: " + str(err.min()/Y_test.shape[0]))
    return bw_x_est,lambda_reg_est




def CrossValidRegret(param_grid,algorithm_flag,numb_exp,data_flag,data_flag_multiclass,N_valid, N,randomSeeds,Main_Program_flag):

    FinalRegret = np.zeros([param_grid.shape[1]])
    for ii in range(0,param_grid.shape[1]):
        print(ii, param_grid[:,ii])
        bw_x = param_grid[0,ii]
        gamma = param_grid[1,ii]
        alpha = param_grid[2,ii]
        Regret = np.zeros([numb_exp])
        Accuracy = np.zeros([numb_exp])
        for nn in range(0, numb_exp):
            DataXY = DC.TrainDataCollect(data_flag_multiclass, N_valid, N, randomSeeds[0], Main_Program_flag)
            rng = np.random.RandomState(randomSeeds[nn])
            T = DataXY['Testfeatures'].shape[0]
            ShuffleIndex = rng.permutation(T)
            # print DataXY['Testfeatures'].shape, ShuffleIndex.shape
            Testfeatures = np.copy(DataXY['Testfeatures'])
            armTest = np.copy(DataXY['armTest'])
            Testfeatures = Testfeatures[ShuffleIndex, :]
            armTest = armTest[ShuffleIndex]
            DataXY['Testfeatures'] = np.copy(Testfeatures)


            Regret[nn], Accuracy[nn], regretUCB, Selected_Arm_T, Exact_Arm_T, Task_sim_dict = UCBT.ContextBanditUCBRunForTSteps(
                DataXY, T, data_flag, bw_x, gamma, alpha, algorithm_flag)



        FinalRegret[ii] = np.mean(Regret)
    ind_min = np.where(FinalRegret == FinalRegret.min())

    bw_x_est = param_grid[0,ind_min]
    bw_prob_est = param_grid[1,ind_min]
    bw_prod_est = param_grid[2,ind_min]
    gamma_est = param_grid[3,ind_min]
    alpha_est = param_grid[4,ind_min]
    print("minimium Average Regret is: " + str(FinalRegret.min()))
    return bw_x_est, bw_prob_est, bw_prod_est, gamma_est,alpha_est

