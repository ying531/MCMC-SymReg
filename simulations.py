#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:40:29 2020

@author: ying
"""

import os
base_path = "/Users/ying/Dropbox/Guo_Jin_Kang_Shared/SymReg/codes/MCMC-SymReg"
os.chdir(base_path)
from funcs import Operator, Node
from funcs import grow, genList, shrink, upgOd, allcal, display, getHeight, getNum, numLT, upDepth, Express, fStruc
from funcs import ylogLike, newProp, Prop, auxProp
from bsr import BSR

import numpy as np
import pandas as pd
from scipy.stats import invgamma
from scipy.stats import norm
import sklearn
from sklearn.datasets import load_iris
import copy
import matplotlib.pyplot as plt
#import pymc3 as pm
import seaborn
import random
#import data_generate_funcs as dg
import time

# =============================================================================
# # data processing
# =============================================================================
data_path = "/Users/ying/Dropbox/Guo_Jin_Kang_Shared/SymReg/codes/real_data/Concrete_Data.xls"
data = pd.read_excel(data_path)
data.columns=['cement','blast','flyash','water','superp','coarse','fine','age','ccs']
train_data = data.iloc[:,0:8]
train_data.iloc[:,0] = train_data.iloc[:,0]/100
train_data.iloc[:,1] = train_data.iloc[:,1]/100
train_data.iloc[:,2] = train_data.iloc[:,2]/100
train_data.iloc[:,3] = train_data.iloc[:,3]/100
train_data.iloc[:,4] = train_data.iloc[:,4]/10
train_data.iloc[:,5] = train_data.iloc[:,5]/100
train_data.iloc[:,6] = train_data.iloc[:,6]/100
train_data.iloc[:,7] = train_data.iloc[:,7]/100

train_y = data.iloc[:,8]

train_inds = random.sample(range(1030),721)
train_inds = np.sort(train_inds).tolist()
test_inds = [x for x in range(1030) if x not in train_inds]

test_data = train_data.iloc[test_inds,:]
test_y = train_y.iloc[train_inds]
train_data = train_data.iloc[train_inds,:]
train_data.index = range(721)
train_y = train_y.iloc[train_inds]
train_y.index = range(721)

# =============================================================================
# # y=exp(x0) + 5 * cos(x1)
# =============================================================================

random.seed(1)
n = 100
x1 = np.random.uniform(-3, 3, n)
x2 = np.random.uniform(-3, 3, n)
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
train_data = pd.concat([x1, x2], axis=1)
train_y = 1.35* train_data.iloc[:,0]*train_data.iloc[:,1] + 5.5*np.sin((train_data.iloc[:,0]-1)*(train_data.iloc[:,1]-1))

xx1 = np.random.uniform(-3, 3, 30)
xx2 = np.random.uniform(-3,3,30)
xx1 = pd.DataFrame(xx1)
xx2 = pd.DataFrame(xx2)
test_data = pd.concat([xx1, xx2], axis=1)
test_y = 1.35* test_data.iloc[:,0]*test_data.iloc[:,1] + 5.5*np.sin((test_data.iloc[:,0]-1)*(test_data.iloc[:,1]-1))

xx1 = np.random.uniform(-6, 6, 30)
xx2 = np.random.uniform(-6,6,30)
xx1 = pd.DataFrame(xx1)
xx2 = pd.DataFrame(xx2)
test2_data = pd.concat([xx1, xx2], axis=1)
test2_y = 1.35* test2_data.iloc[:,0]*test2_data.iloc[:,1] + 5.5*np.sin((test2_data.iloc[:,0]-1)*(test2_data.iloc[:,1]-1))


package = 'dg.'
my_func = 'f2'
# number of simple signals
K = 3
MM = 50

hyper_params = [{'treeNum': 3, 'itrNum':50, 'alpha1':0.4, 'alpha2':0.4, 'beta':-1}]
my_bsr = BSR(K,MM)
my_bsr.train(train_data, train_y, MM, K)
fitted_y = my_bsr.fit(test_data)

my_bsr.display_trees()


plt.figure(figsize=(10,5))
plt.hist(train_data.iloc[:,0])
plt.title("distribution of x0")
plt.show()


#print("========finish========")
#print("the beta of linear regrgession:")
#print(Beta)
#print("This time function: {}".format(my_func))
# plt.yticks(np.arange(0, 50, 5))
# plt.grid(color='gray', linestyle='--', linewidth=1, alpha=0.3)
# cols = ['r', 'b', 'g', 'k', 'm']
# # plt.plot(np.arange(len(errList)),errList,color=cols[0],linewidth=1)
# # plt.plot(np.arange(len(testList)),testList,color=cols[1],linewidth=1)
#
# for i in np.arange(K):
#     plt.plot(np.arange(len(ErrLists[i][-10:-1])), ErrLists[i][-10:-1], color=cols[i], linewidth=1)
# # plt.savefig('errplot.png',dpi=400)
# plt.show()

# =============================================================================
# # diagnose
# =============================================================================

# print('---------------------')
# print("Gelman-Rubin Diagnosis:", pm.diagnostics.gelman_rubin(LLHS))
#
# props = []
# for i in np.arange(K):
#     props.append(sum(TotalLists[i]))
# print('---------------------')
# print("Number of all proposals:", props)

# =============================================================================
# # Genetic Programming
# =============================================================================

import gplearn
from gplearn.genetic import SymbolicRegressor


def _protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)

expo = gplearn.functions.make_function(_protected_exponent, name='exp', arity=1)

est_gp = SymbolicRegressor(population_size=200,
                           generations=100, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                           function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos'],
                           metric='rmse')

est_gp.fit(train_data, train_y)

gpNodes = [31,48,34,30,49,61,61,74,86,113,118,111,109,108,107,106,105,105,104,104,104]
gpErrs = [75.65,28.55,18.67,15.21,9.92,8.37,7.24,6.36,4.58,2.73,0.1,0.06,0.058,0.057,0.058,0.058,0.057,0.059,0.059,0.057,0.058]



plt.figure(figsize=(10,8),dpi=200)
plt.title("train and test error during training",size=20)
plt.plot(errList,label='BSR training error',color='b')
plt.scatter(range(0,len(errList)),errList,marker='^',facecolors='none',edgecolors='b')
plt.plot(testList,label='BSR test error',color='g')
plt.scatter(range(0,len(testList)),testList,marker='v',facecolors='none',edgecolors='g')
plt.plot(gpErrs,label='GP training error',color='r')
plt.scatter(range(0,len(gpErrs)),gpErrs,marker='o',facecolors='none',edgecolors='r')
plt.legend(fontsize=20)
plt.xlabel('iterations of training')
plt.ylabel('RMSEs')
plt.show()

plt.figure(figsize=(10,8),dpi=200)
plt.title("complexity (node counts) during training",size=20)
plt.plot(nodeCounts,label='BSR nodes',color='b')
plt.scatter(range(0,len(nodeCounts)),nodeCounts,marker='v',facecolors='none',edgecolors='b')
plt.plot(gpNodes,label='GP nodes',color='g')
plt.scatter(range(0,len(gpNodes)),gpNodes,facecolors='none',edgecolors='g')
plt.legend(fontsize=20)
plt.xlabel('iterations of training')
plt.ylabel('number of nodes')
plt.show()









print('---------------------')
print("Estimated model by Genetic Programming:")
print(est_gp._program)
estd = est_gp.predict(test_data)
gperror = np.sqrt(sum((estd - test_y) ** 2) / len(estd))
print("-----")
print("RMSE of test data:", round(gperror, 5))
# gpderror = np.sqrt(sum((estd-test_yy)**2)/len(estd))
# print("RMSE of denoised test data:",round(gpderror,5))

print("This time function: {}".format(my_func))




#==============================================================================
# ## ffx
#==============================================================================

import ffx 


random.seed(1)
n = 100
x1 = np.random.uniform(0.1, 5.9, n)
x2 = np.random.uniform(0.1, 5.9, n)
x1 = pd.DataFrame(x1)
x2 = pd.DataFrame(x2)
train_data = pd.concat([x1, x2], axis=1)
train_y = 2.5 * np.power(train_data.iloc[:, 0],4) - 1.3 * np.power(train_data.iloc[:, 0],3)+0.5*np.power(train_data.iloc[:, 1],2)-1.7* train_data.iloc[:, 0]
xx1 = []
xx2 = []
for i in np.arange(31):
    for j in np.arange(31):
        xx1.append(-0.15 + 0.2 * i)
        xx2.append(-0.15 + 0.2 * j)
xx1 = pd.DataFrame(xx1)
xx2 = pd.DataFrame(xx2)
test_data = pd.concat([xx1, xx2], axis=1)
test_y = 2.5 * np.power(test_data.iloc[:, 0],4) - 1.3 * np.power(test_data.iloc[:, 0],3)+0.5*np.power(test_data.iloc[:, 1],2)-1.7* test_data.iloc[:, 0]

train_X = train_data
test_X = test_data

models = ffx.run(train_X, train_y, test_X, test_y, ["predictor_a", "predictor_b"])
for model in models:
    yhat = model.simulate(test_X)
    print(model)

FFX = ffx.FFXRegressor()
FFX.fit(train_X, train_y)
print("Prediction:", FFX.predict(test_X))
print("Score:", FFX.score(test_X, test_y))