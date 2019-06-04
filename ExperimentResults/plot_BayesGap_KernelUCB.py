# Python code to plot all data

import numpy as np
from scipy import io
print('here')
import matplotlib.pyplot as plt

# plt.style.use('seaborn-whitegrid')

# load all the datasets
nRuns = 10
dBG = []
dKUCB = []
dKUCBMod = []
dUS = []
dEG = []
dKTS = []

ipdataset = 'synthetic' # 'spacecraft_labexperimental'
# vals = [1,2,3,4,7,8,9,10]
vals = range(1, nRuns+1)
# vals = [1]

for ii in vals:
    Dtemp = io.loadmat('./'+ipdataset+'/run_alpha_1/BayesGapRun_' + str(ii))
    dBG.append(Dtemp['SimpleRegretUCBRuns'])
    Dtemp = io.loadmat('./'+ipdataset+'/run_alpha_1/KernelUCBRun_' + str(ii))
    dKUCB.append(Dtemp['SimpleRegretUCBRuns'])
    Dtemp = io.loadmat('./' + ipdataset + '/run_alpha_1/KernelUCBModRun_' + str(ii))
    dKUCBMod.append(Dtemp['SimpleRegretUCBRuns'])
    Dtemp = io.loadmat('./' + ipdataset + '/run_alpha_1/UniformSamplingRun_' + str(ii))
    dUS.append(Dtemp['SimpleRegretUCBRuns'])
    Dtemp = io.loadmat('./' + ipdataset + '/run_alpha_1/EpsilonGreedyRun_' + str(ii))
    dEG.append(Dtemp['SimpleRegretUCBRuns'])
    Dtemp = io.loadmat('./' + ipdataset + '/run_alpha_1/KernelTSRun_' + str(ii))
    dKTS.append(Dtemp['SimpleRegretUCBRuns'])


dBGarray = np.concatenate(dBG)
dBGvar = np.var(dBGarray, axis=0)
dBGmean = np.mean(dBGarray, axis=0)

dKUCBarray = np.concatenate(dKUCB)
dKUCBvar = np.var(dKUCBarray, axis=0)
dKUCBmean = np.mean(dKUCBarray, axis=0)

dKUCBModarray = np.concatenate(dKUCBMod)
dKUCBModvar = np.var(dKUCBModarray, axis=0)
dKUCBModmean = np.mean(dKUCBModarray, axis=0)

dUSarray = np.concatenate(dUS)
dUSvar = np.var(dUSarray, axis=0)
dUSmean = np.mean(dUSarray, axis=0)

dEGarray = np.concatenate(dEG)
dEGvar = np.var(dEGarray, axis=0)
dEGmean = np.mean(dEGarray, axis=0)

dKTSarray = np.concatenate(dKTS)
dKTSvar = np.var(dKTSarray, axis=0)
dKTSmean = np.mean(dKTSarray, axis=0)

S = dBGmean.shape
y1BG = dBGmean[1:]-np.sqrt(dBGvar[1:])/2
y2BG = dBGmean[1:] + np.sqrt(dBGvar[1:])/2

y1KUCB = dKUCBmean[1:] - np.sqrt(dKUCBvar[1:])/2
y2KUCB = dKUCBmean[1:] + np.sqrt(dKUCBvar[1:])/2

y1KUCBMod = dKUCBModmean[1:] - np.sqrt(dKUCBModvar[1:])/2
y2KUCBMod = dKUCBModmean[1:] + np.sqrt(dKUCBModvar[1:])/2

y1US = dUSmean[1:] - np.sqrt(dUSvar[1:])/2
y2US = dUSmean[1:] + np.sqrt(dUSvar[1:])/2

y1EG = dEGmean[1:] - np.sqrt(dEGvar[1:])/2
y2EG = dEGmean[1:] + np.sqrt(dEGvar[1:])/2

y1KTS = dKTSmean[1:] - np.sqrt(dKTSvar[1:])/2
y2KTS = dKTSmean[1:] + np.sqrt(dKTSvar[1:])/2

S2 = dKUCBmean.shape

plt.figure(num=None, figsize=(10.5,9.5), facecolor='w')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=20)


plt.plot(range(50,50*S[0],50),dBGmean[1:], 'bo-',label='Contextual Gap')
plt.plot(range(50,50*S2[0],50), dKUCBmean[1:], 'g+-', label='Kernel-UCB')
plt.plot(range(50,50*S2[0],50), dKUCBModmean[1:], 'k+-', label='Kernel-UCB-Mod')
plt.plot(range(50,50*S[0],50), dUSmean[1:], 'rx-', label='Uniform Sampling')
plt.plot(range(50,50*S[0],50), dEGmean[1:], 'yv-', label='Epsilon Greedy')
plt.plot(range(50,50*S[0],50), dKTSmean[1:], 'm^-', label='Kernel TS')


plt.fill_between(range(50,50*S[0],50), y1BG, y2BG, where=y2BG>=y1BG, color='blue', alpha=0.2)
plt.fill_between(range(50,50*S2[0],50), y1KUCB, y2KUCB, where=y2KUCB>=y1KUCB, color='green', alpha=0.2)
plt.fill_between(range(50,50*S2[0],50), y1KUCBMod, y2KUCBMod, where=y2KUCBMod>=y1KUCBMod, color='black', alpha=0.2)
plt.fill_between(range(50,50*S[0],50), y1US, y2US, where=y2US>=y1US, color='red', alpha=0.2)
plt.fill_between(range(50,50*S[0],50), y1EG, y2EG, where=y2US>=y1US, color='yellow', alpha=0.2)
plt.fill_between(range(50,50*S[0],50), y1KTS, y2KTS, where=y2US>=y1US, color='magenta', alpha=0.2)

plt.yscale('log')
plt.legend()
plt.xlabel('Length of Exploration Phase')
plt.ylabel('Simple Regret During Exploitation Phase')
plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth=0.25, color='black')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth=0.25, color='black')
plt.tight_layout()
plt.savefig(ipdataset+'_run_alpha_1.png')
plt.show()


print("here")
