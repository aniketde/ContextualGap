# Reward Process for running more complex rewards for UCB setup

from __future__ import division
import numpy as np
from scipy import io as spio

class RewardProcess:
    def __init__(self, dataset):
        if(dataset == 'multiclass'):     # Multiclass classficiation dataset
            self.dataset = 'multiclass'
        elif(dataset == 'ordinal_regression'):
            self.dataset = 'ordinal_regression'
            self.dataset_type = 'negative-mag'  #This is actually reward type, misnomer due to s/c dataset where reward type depended on dataset type
        elif(dataset == 'spacecraft'):
            self.dataset = 'spacecraft'
            input_dataset = './Datasets/synthetic_sc_lbeqn_5d.mat'
            data = spio.loadmat(input_dataset)
            # context_full = data['context']    # contexts not related to loss function are handled outside
            bs = data['B_s']*1e-4   # convert to 1e-5 Tesla
            br = data['B_ref']*1e-4
            self.dataset_type = '3-axis-norot'
            self.br_uhc, self.bs_uhc = self.InitializeDataset(self.dataset_type, bs, br)
        elif(dataset == 'spacecraft_labbased'):
            self.dataset = 'spacecraft_labbased'
            input_dataset = './Datasets/largecoil_postproc.mat'
            data = spio.loadmat(input_dataset)
            bs = data['bNorm']
            br = np.zeros((1, bs.shape[0]))
            self.dataset_type = '3-axis-labbased'
            self.br_uhc, self.bs_uhc = self.InitializeDataset(self.dataset_type, bs, br)
        elif(dataset == 'synthetic'):
            self.dataset = 'synthetic'
            self.scaling = 1.0

    def InitializeDataset(self, dataset_type, bs, br):
        if self.dataset_type == '3-axis-norot':
            #scaling to unit hypercube
            bs_uhc = np.copy(bs)
            bs_uhc_unsc = np.copy(bs)

            for ii in range(0,3):
                bs_uhc[ii] = (bs[ii].T - np.mean(bs[ii],axis=1) + np.mean(br[ii,:])).T
                bs_uhc_unsc[ii] = np.copy(bs_uhc[ii])
                # bs_uhc[ii] = (bs_uhc[ii].T/np.amax(bs_uhc[ii],axis=1)).T      #for scenario 2

            br_uhc = np.copy(br)
        elif self.dataset_type == '1-axis-norot':
            bs_uhc = np.copy(bs)
            ms = bs.shape[1]
            bs_uhc = bs - np.mean(bs, axis=0) + np.mean(br)*np.ones((1,ms))

            br_uhc = np.copy(br)
        elif self.dataset_type == '3-axis-labbased':
            br_uhc = np.copy(br)
            bs_uhc = np.copy(bs)

        return br_uhc, bs_uhc

    def RewardFromLabelsMulticlass(self,true_label, est_label):
        if self.dataset == 'multiclass':
            if true_label == est_label:
                rew = 1
            else:
                rew = 0
            rew = np.ones([1]) * rew
            # print true_label, est_label, rew
            rewmax = 1.0
        if self.dataset == 'ordinal_regression':
            if self.dataset_type == 'negative-mag':
                rew = -np.abs(true_label - est_label)
            elif self.dataset_type == 'negative-ls':
                rew = -np.power(true_label - est_label, 2)
            rewmax = 0.0
        return rew, rewmax

    def RewardFromAlpha(self, alpha, ind_arm, time_step):
        if self.dataset_type == '3-axis-norot':
            B_S = self.bs_uhc[:,:,int(time_step)]
            B_R = self.br_uhc[:,int(time_step)]
            # B_c = np.array([alpha[0].dot(B_S[0]),
            #                         alpha[1].dot(B_S[1]),
            #                         alpha[2].dot(B_S[2])])
            #
            # rew = -(np.linalg.norm(B_R - B_c))**2

            # CAUTION: REWMAX works with m discrete magnetometer arms only
            # rewarr = np.zeros((self.bs_uhc.shape[1],1))
            rewarr = -10*(np.linalg.norm(B_R[:,np.newaxis] - B_S, axis=0))**2
            rew = rewarr[alpha]
            # rewmaxarm = np.argmax(rewarr)
            # rew = 1.0*(alpha == rewmaxarm)
            rewmax = ind_arm
        if self.dataset_type == '1-axis-norot':
            B_S = self.bs_uhc[time_step,:]
            B_R = self.br_uhc[time_step,0]

            rewarr = -np.absolute(B_R**2 - B_S**2)/1e-10 + 5
            # rewmaxarm = np.argmax(rewarr)
            # rew = 1.0*(alpha == rewmaxarm)
            # rewmax = 1.0
            rew = rewarr[alpha]
            rewmax = np.ndarray.max(rewarr)
        if self.dataset_type == '3-axis-labbased':
            B_S = self.bs_uhc[:, int(time_step)]
            rewarr = -B_S   #no change needed as all processing is done and this is only noise
            rew = rewarr[alpha]
            rewmax = ind_arm

        return rew, rewmax

    def RewardFromSynthetic(self, alpha, true_value, context):
        rew = self.scaling*np.sin(alpha*context)
        rewmax = self.scaling*np.sin(true_value*context)#true_value
        return rew[0], rewmax

