#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:40:29 2020

@author: ying
"""

import os
base_path = "" # the path of project
os.chdir(base_path)
from funcs import Operator, Node
from funcs import grow, genList, shrink, upgOd, allcal, display, getHeight, getNum, numLT, upDepth, Express, fStruc
from funcs import ylogLike, newProp, Prop, auxProp
from bsr_class import BSR

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


# number of simple signals
K = 3
MM = 50

hyper_params = [{'treeNum': 3, 'itrNum':50, 'alpha1':0.4, 'alpha2':0.4, 'beta':-1}]
# initialize
my_bsr = BSR(K,MM)
# train (need to fill in parameters)
my_bsr.train()
# fit new values
fitted_y = my_bsr.fit()
# display fitted trees
my_bsr.display_trees()
