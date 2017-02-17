# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:00:57 2017

@author: Hajime
"""

import pandas as pd
import numpy as np
import random as rnd
import scipy.stats as stats
import scipy.optimize as opt
import json as json
import matplotlib as mpl
from math import exp
from matplotlib import pyplot as plt
from statsmodels import tsa
from sklearn import linear_model
import sys

from RustPurchase_simulate import SimulateDynShare 

sim = SimulateDynShare()
sim.Simulate()
data = sim.data
share = data['share']
char = data['char']

dpara_guess = np.array([.4,3.,.7])
nu_var=1.
rho=8.
beta=.75
phi_guess = .9
v_var_guess = .5

def Gen_states_def(phi, v_var,S_gridN,S_gridmin,S_gridmax,S_grid,\
    S_grid_bounds,S_gridsize):
    v_dist = stats.norm(scale=np.sqrt(v_var))
    S_trans_mat_cons = np.zeros([S_gridN,S_gridN])
    for i in range(S_gridN):
        for j in range(S_gridN):
            s_current = S_grid[i]
            s_next = S_grid[j]
            s_diff = s_next-phi*s_current
            if s_next==S_gridmin:
                S_trans_mat_cons[i,j] = v_dist.cdf(s_diff+S_gridsize/2)-0.
            elif s_next==S_gridmax:
                S_trans_mat_cons[i,j] = 1.-v_dist.cdf(s_diff-S_gridsize/2)
            else:
                S_trans_mat_cons[i,j] = v_dist.cdf(s_diff+S_gridsize/2)-v_dist.cdf(s_diff-S_gridsize/2)
                
    states_def ={ 'S_gridN':S_gridN,'S_gridmin':S_gridmin,'S_gridmax':S_gridmax,'S_grid':S_grid,\
    'S_grid_bounds':S_grid_bounds,'S_gridsize':S_gridsize,'S_trans_mat_cons':S_trans_mat_cons }
    return states_def



def contraction_mapping(beta,\
                        S_gridN,S_gridmin,S_gridmax,S_grid,\
                        S_grid_bounds,S_gridsize,S_trans_mat_cons,\
                        threshold=1e-6, maxiter=10e3):
    achieved = True
    k = 0
    U_myopic = np.c_[np.zeros(S_gridN),S_grid].T
    EV_new = np.c_[S_grid,np.zeros(S_gridN)].T
    norm = threshold + 100.
    while norm>threshold:
        EV = EV_new
        val = U_myopic + beta*EV
        expval = np.exp(val)
        v0 = np.log( np.sum( expval, axis=0 ) )
        v1 = np.zeros(S_gridN)
        val_f = np.c_[v0,v1].T #EV in each future state
        EV_new =  np.dot( S_trans_mat_cons, val_f.T).T #assuming the transition is same across action.
        k=k+1
        if k>maxiter:
            achieved=False
            break
        norm = np.max(np.abs(EV_new - EV))
    return EV_new, U_myopic

def Calc_share_hat(zeta_seq, zeta_grid, EV_grid, beta):
    nobs = len(zeta_seq)
    val_grid = np.c_[np.zeros(nobs), zeta_grid] + beta*EV_grid
    pchoice_grid = choice_prob(val_grid)
    pchoice_seq = pchoice_grid[:,zeta_seq]
    
    share_hat = np.zeros(nobs)
    for i in range(nobs):
        if loc_firstobs[i]:
            remain = 1.
        share_hat = remain*pchoice_seq[1,i]
        remain = remain-share_obs
    return share_hat
    
def choice_prob(val): #input 2 by S_gridN matrix
    n,s = val.shape
    expval = np.exp(val)
    expval_sum = np.tile(  np.sum(expval, axis=0), n).reshape([n,s])
    p_choice = expval/expval_sum
    return p_choice


def FP_zeta(share_obs, phi_guess, v_var_guess,\
            EV_grid,
            S_gridN,S_gridmin,S_gridmax,S_grid,S_grid_bounds,S_gridsize,\
            beta,\
            threshold=1e-6, maxiter=10e3):
    nobs = len(share_obs)
    zeta_hat = 5.*np.ones(nobs)
    zeta_hat_new = 5.*np.ones(nobs)
    norm = 100.
    while norm>threshold:    
        zeta_hat = zeta_hat_new
        share_hat = Calc_share_hat(zeta_hat,S_grid, EV_grid, beta)
        update = np.log(share_obs)-np.log(share_hat)
        zeta_hat_new = zeta_hat+update
        norm = np.max( np.abs(zeta_hat_new-zeta_hat) )
    return zeta_hat

def Unpdate_phi():
    pass

def Calc_eta():
    pass

def Calc_nu():
    pass
    
    
    
    
    
    
