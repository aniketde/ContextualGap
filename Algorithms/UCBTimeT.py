from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import BanditModels.Unified_Bandit as Bandits

# Thic function runs the bandit algorithm for T steps
def ContextBanditUCBRunForTSteps(DataXY,T,bw_x,gamma,alpha,algorithm_flag, reward_scenario, Nsr, T_test, Main_Program_Flag, ucb_flag):
    accuracy_UCB = 0.0
    #rewardAccu = np.zeros([T])
    CumregretUCB = np.zeros([T])
    SimpleRegret = np.zeros((int(np.floor(T/Nsr))))
    Bandit_Algo = Bandits.Unified_Bandit(bw_x, gamma, alpha,
                                 algorithm_flag, reward_scenario)


    Selected_Arm_T = np.zeros([T])
    Exact_Arm_T =  np.zeros([T])
    Best_Arm_T = np.zeros([T])
    # Task_sim_dict = dict()
    Best_Arm_test = np.zeros((T_test,))
    Selected_Arm_test = np.zeros((T_test,))

    for tt in range(0, T):
        arm_tt, best_arm_tt,X_test, DataXY = Bandit_Algo.get_Arm_And_X_test_And_Data(DataXY, tt, ucb_flag)
        armTest = DataXY['armTest']
        ind_arm = armTest[tt]
        # true_reward = 1.0
        if (DataXY['TDependentLoss']==0):
            Bandit_Algo.update_Collected_Rewards(tt, ind_arm, X_test)
        elif(DataXY['TDependentLoss']==1):
            tstp = X_test[0,-1]
            Bandit_Algo.update_Collected_Rewards(tstp, ind_arm, X_test)

        (rewardAccu, rewardMax) = Bandit_Algo.get_Collected_Rewards()


        Selected_Arm_T[tt] = arm_tt
        Best_Arm_T[tt] = best_arm_tt
        Exact_Arm_T[tt] = ind_arm

        # if int(ind_arm) == int(arm_tt):
        #     accuracy_UCB += 1

        accuracy_UCB += np.absolute((rewardMax[tt]-rewardAccu[tt]))
        CumregretUCB[tt] = rewardMax[tt] - rewardAccu[tt]

        #Add Data
        DataXY = DC.AddData(DataXY, arm_tt, algorithm_flag, X_test, rewardAccu[tt], tt, bw_x, gamma)

        if tt % Nsr == 0 and tt != 0:
            print("iteration number, true class, UCB class,true reward, UCB reward, Algorithm ")
            print tt, int(ind_arm), int(arm_tt), int(best_arm_tt), rewardMax[tt], rewardAccu[tt], algorithm_flag
            print str(tt) + " Error of "+ algorithm_flag + " :"+ str( accuracy_UCB / tt)

            if Main_Program_Flag:
                idxsr = int(np.floor(tt / Nsr))
                for tt1 in range(T, T + T_test):
                    arm_tt1, best_arm_tt1, X_test1, DataXY1 = Bandit_Algo.get_Arm_And_X_test_And_Data(DataXY, tt1, ucb_flag)
                    armTest1 = DataXY['armTest']
                    ind_arm1 = armTest1[tt1]
                    if DataXY['DatasetName'] == 'spacecraft_labexperimental':
                        (rewardAccu1, rewardMax1) = Bandit_Algo.RP.RewardFromAlpha(best_arm_tt1, ind_arm1, X_test1[0, -1])
                    elif DataXY['DatasetName'] == 'multiclass' or DataXY['DatasetName'] == 'ordinal_regression':
                        (rewardAccu1, rewardMax1) = Bandit_Algo.RP.RewardFromLabelsMulticlass(ind_arm1, best_arm_tt1)
                    elif DataXY['DatasetName'] == 'synthetic':
                        (rewardAccu1, rewardMax1) = Bandit_Algo.RP.RewardFromSynthetic(best_arm_tt1, ind_arm1, X_test1)

                    SimpleRegret[idxsr] += rewardMax1 - rewardAccu1

                SimpleRegret[idxsr] = SimpleRegret[idxsr] / float(T_test)

        if tt == T-1:
            print "iteration number, true class, UCB class,true reward, UCB reward, Algorithm "
            print tt, int(ind_arm), int(arm_tt), int(best_arm_tt),rewardMax[tt], rewardAccu[tt], algorithm_flag
            print str(tt) + " Error of "+ algorithm_flag + " :"+ str( accuracy_UCB / tt)

            idxsr = int(np.floor(tt / Nsr))
            for tt1 in range(T, T + T_test):
                arm_tt1, best_arm_tt1, X_test1, DataXY1 = Bandit_Algo.get_Arm_And_X_test_And_Data(DataXY, tt1, ucb_flag)
                armTest1 = DataXY['armTest']
                ind_arm1 = armTest1[tt1]
                if DataXY['DatasetName'] == 'spacecraft_labexperimental':
                    (rewardAccu1, rewardMax1) = Bandit_Algo.RP.RewardFromAlpha(best_arm_tt1, ind_arm1, X_test1[0, -1])
                elif DataXY['DatasetName'] == 'multiclass' or DataXY['DatasetName'] == 'ordinal_regression':
                    (rewardAccu1, rewardMax1) = Bandit_Algo.RP.RewardFromLabelsMulticlass(ind_arm1, best_arm_tt1)
                elif DataXY['DatasetName'] == 'synthetic':
                    (rewardAccu1, rewardMax1) = Bandit_Algo.RP.RewardFromSynthetic(best_arm_tt1, ind_arm1, X_test1)

                Selected_Arm_test[tt1-T] = arm_tt1
                Best_Arm_test[tt1-T] = best_arm_tt1

                SimpleRegret[idxsr] += rewardMax1 - rewardAccu1

            SimpleRegret[idxsr] = SimpleRegret[idxsr] / float(T_test)
        
        # if tt %(T/4) == 0:
        #     Task_sim_dict[algorithm_flag + '_Task_Sim_' + str(tt)] = DataXY[algorithm_flag + '_TaskSim']
    AverageRegret = np.sum(CumregretUCB) / float(T)
    AverageAccuracy = accuracy_UCB / float(T)

    Simpleregret = np.ones([Best_Arm_T.shape[0]])
    a = Exact_Arm_T == Best_Arm_T
    print(a.shape)
    Simpleregret[a] = 0

    #print Simpleregret
    # CumSimpleregretUCB = np.zeros(Best_Arm_T.shape[0])
    # jj = Best_Arm_T.shape[0]
    # for ii in range(0,Simpleregret.shape[0]):
    #     CumSimpleregretUCB[jj-ii-1] = np.sum(Simpleregret[jj-ii-1:])/float(ii+1.0)

    Task_sim_dict = DataXY
    #print CumSimpleregretUCB
    return AverageRegret,AverageAccuracy,CumregretUCB,SimpleRegret,Selected_Arm_T,Exact_Arm_T,Best_Arm_T,Task_sim_dict, Best_Arm_test, Selected_Arm_test
