from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import DataCreate.RewardProcess as RewardProcess
import Algorithms.UCB as UCB
import warnings
warnings.filterwarnings("ignore")

#f_bw <- bw_x_estKTLUCB
#s_bw <- bw_prod_estKTLUCB
#gamma <- gammaKTLUCB
#t_bw <- bw_prob_estKTLUCB
class Unified_Bandit:
    def __init__(self,f_bw, gamma, alpha,type, reward_scenario):
        self.type = type
        self.f_bw = f_bw
        self.gamma = gamma
        self.alpha = alpha
        self.collected_rewards = []
        self.maximum_rewards = []
        self.RP = RewardProcess.RewardProcess(reward_scenario)

    #data <- DataXY
    #time_Step <- tt
    def get_Arm_And_X_test_And_Data(self, data, time_Step, ucb_flag):
        UCBound, LCBound, reward_conf, reward_est,X_test, data = UCB.ContextBanditUCB(data, time_Step, self.f_bw, self.gamma,self.alpha, self.type)
        if ucb_flag == 'BayesGap':
            diff_UCB_LCB = np.zeros(LCBound.shape[0])
            B = np.zeros(LCBound.shape[0])
            for aa in range(0,UCBound.shape[0]):
                for bb in range(0,UCBound.shape[0]):
                    if aa == bb:
                        diff_UCB_LCB[bb] = -100000000 #Some negative large number
                    else:
                        # diff_UCB_LCB[bb] = UCBound[aa] - LCBound[bb]
                        diff_UCB_LCB[bb] = UCBound[bb] - LCBound[aa]
                B[aa] = np.max(diff_UCB_LCB)
            J = np.argmin(B)
            UCB_J = np.copy(UCBound)
            UCB_J[J] = -100000000 #Some negative large number
            j = np.argmax(UCB_J)
            if reward_conf[J] < reward_conf[j]:
                self.selected_arm = j
            else:
                self.selected_arm = J
            self.best_arm = J
        elif ucb_flag == 'KernelUCB':
            self.selected_arm = np.argmax(UCBound)
            # self.best_arm = np.argmax(reward_est)
            self.best_arm = np.argmax(UCBound)
        elif ucb_flag == 'KernelUCBMod':
            self.selected_arm = np.argmax(UCBound)
            self.best_arm = np.argmax(reward_est)
            #self.best_arm = np.argmax(UCBound)
        elif ucb_flag == 'UniformSampling':
            self.selected_arm = np.random.randint(0, UCBound.shape[0])
            self.best_arm = np.argmax(reward_est)
        elif ucb_flag == 'EpsilonGreedy':
            # epsilon sampling
            epsilonVal = 0.99**(time_Step)
            rsamp = np.random.uniform()
            if (rsamp < epsilonVal):
                self.selected_arm = np.random.randint(0, UCBound.shape[0])
            else:
                self.selected_arm = np.argmax(reward_est)

            self.best_arm = np.argmax(reward_est)
        elif ucb_flag == 'ThompsonSampling':
            self.selected_arm = np.argmax(reward_est)
            self.best_arm = np.argmax(reward_est)
        elif ucb_flag == 'KernelTS':
            sigma = (UCBound - LCBound) / 2.0
            rewsamp = np.random.multivariate_normal(reward_est, np.diag(sigma))
            self.selected_arm = np.argmax(rewsamp)
            self.best_arm = np.argmax(rewsamp)

        return self.selected_arm, self.best_arm, X_test, data
    

        
    def update_Collected_Rewards(self, time_Step,ind_arm, X_test):
        # print ind_arm,  self.selected_arm
        if self.RP.dataset == 'multiclass':
            (rew, rewmax) = self.RP.RewardFromLabelsMulticlass(ind_arm, self.selected_arm)
            self.collected_rewards.append(rew)
            self.maximum_rewards.append(rewmax)
        elif self.RP.dataset == 'ordinal_regression':
            (rew, rewmax) = self.RP.RewardFromLabelsMulticlass(ind_arm, self.selected_arm)
            self.collected_rewards.append(rew)
            self.maximum_rewards.append(rewmax)
        elif self.RP.dataset == 'spacecraft':
            # Working with reward process
            # alpha = np.zeros((self.RP.bs_uhc.shape[0], self.RP.bs_uhc.shape[1]))
            # alpha[:,self.selected_arm] = 1
            rew, rewmax = self.RP.RewardFromAlpha(self.selected_arm, ind_arm, time_Step)
            self.collected_rewards.append(rew)
            self.maximum_rewards.append(rewmax)
        elif self.RP.dataset == 'spacecraft_labbased':
            rew, rewmax = self.RP.RewardFromAlpha(self.selected_arm, ind_arm, time_Step)
            self.collected_rewards.append(rew)
            self.maximum_rewards.append(rewmax)
        elif self.RP.dataset == 'synthetic':
            rew, rewmax = self.RP.RewardFromSynthetic(self.selected_arm, ind_arm, X_test)
            self.collected_rewards.append(rew)
            self.maximum_rewards.append(rewmax)


    # def get_Simple_Rewards(self, time_Step):



    #All these functions are needed for data_flag = 3
    def set_All_User_Context(self,context):
        self.User_Context = context
        
    def set_All_Arm_Context(self,context):
        self.Arm_Context = context
    
    def set_Reward_Distribution(self,rew_Dist):
        self.Reward_Distribution = rew_Dist
    
    def get_User_Context(self,time_Step):
        #Check the size of user context if it is 0, then raise an exception
        x = self.User_Context[time_Step,:]
        return x[np.newaxis, :]
        
    def get_Arm_Context(self):
        #Check the size of arm context if it is 0, then raise an exception
        return  self.Arm_Context
        
    def get_Reward_Distribuiton(self):
        return self.Reward_Distribution
        
    def get_Rotatated_Context(self, user_context, index):
        mat = np.array([[np.cos(self.Arm_Context[index, 0]), -np.sin(self.Arm_Context[index, 0])],[np.sin(self.Arm_Context[index, 0]), np.cos(self.Arm_Context[index, 0])]])
        return mat.dot(user_context.T).T
       

    def get_Reward_For_All_Arms(self,data,time_Step):
        A = data['NoOfArms']
        RewardforAllArms = np.zeros([A])
        user_Context = self.get_User_Context(time_Step)
        for aa in range(0, A):
            RewardforAllArms[aa] =  self.get_User_Context_Based_Reward(user_Context, aa)
        return RewardforAllArms
    
    def get_Collected_Rewards(self):
        return self.collected_rewards, self.maximum_rewards


    #All the following functions are needed for data_flag = 4,5

    def set_Reward_Function_Flag(self,reward_funct):
        self.Reward_Funct = reward_funct



