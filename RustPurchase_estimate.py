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
data_unobs = sim.data_unobs

S_gridN=data['states_def']['S_gridN']
S_gridmin=data['states_def']['S_gridmin']
S_gridmax=data['states_def']['S_gridmax']
S_grid=data['states_def']['S_grid']
S_grid_bounds=data['states_def']['S_grid_bounds']
S_gridsize=data['states_def']['S_gridsize']


dpara_guess = np.array([.4,3.,.7])
nu_var=1.
rho=8.
beta=.75
phi_guess = .9
v_var_guess = .5
class EstimateDynShare:
    def __init__(self, data, S_gridN = 3,S_gridmin = -1.,S_gridmax = 1.,
                 beta=.75, data_unobs=None):
        
        #state space
        self.S_grid,self.S_gridsize = np.linspace(S_gridmin, S_gridmax, num=S_gridN, retstep=True)
        self.S_grid_bounds = np.linspace(S_gridmin+self.S_gridsize/2, S_gridmax-self.S_gridsize/2,num=S_gridN-1)
        self.S_setting ={ 'S_gridN':S_gridN,'S_gridmin':S_gridmin,'S_gridmax':S_gridmax,'S_grid':self.S_grid,\
        'S_grid_bounds':self.S_grid_bounds,'S_gridsize':self.S_gridsize}
        
        #read data
        self.share_obs = data['share']
        self.char = data['char']
        self.loc_firstobs = data['loc_firstobs']
        self.loc_lastobs = data['loc_lastobs']
        self.mktid = data['mktid']
        self.age = data['age']
        self.prodid = data['prodid']
        self.nobs = len(self.share_obs)
        try:
            self.iv_nu_other = data['iv_nu_other']
        except KeyError:
            self.iv_nu_other = None


    def Gen_states_def(self,phi, v_var,S_gridN,S_gridmin,S_gridmax,S_grid,\
        S_grid_bounds,S_gridsize):
        v_dist = stats.norm(scale=np.sqrt(v_var))
        S_trans_mat_cons = np.zeros([S_gridN,S_gridN])
        
        S_grid_lbound = np.append(-np.inf, S_grid_bounds)
        S_grid_ubound = np.append(S_grid_bounds,np.inf)
        S_trans_mat_cons=v_dist.cdf( S_grid_ubound-phi*S_grid.reshape([-1,1]) )- v_dist.cdf(S_grid_lbound-phi*S_grid.reshape([-1,1]))
        '''
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
        '''         
        states_def ={ 'S_gridN':S_gridN,'S_gridmin':S_gridmin,'S_gridmax':S_gridmax,'S_grid':S_grid,\
        'S_grid_bounds':S_grid_bounds,'S_gridsize':S_gridsize,'S_trans_mat_cons':S_trans_mat_cons }
        return states_def
    
    def contraction_mapping(self,beta,\
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

    def choice_prob(self,val): #input 2 by S_gridN matrix
        n,s = val.shape
        expval = np.exp(val)
        expval_sum = np.tile(  np.sum(expval, axis=0), n).reshape([n,s])
        p_choice = expval/expval_sum
        return p_choice

    def Calc_share_hat(self,zeta_seq, S_grid, S_grid_bounds , EV_grid, beta):
        nobs = len(zeta_seq)
        '''
        val_grid = np.c_[np.zeros(len(S_grid)),S_grid].T + beta*EV_grid
        pchoice_grid = choice_prob(val_grid)
        S_seq = np.digitize( zeta_seq, self.S_grid_bounds ).astype(int)
        pchoice_seq = pchoice_grid[:,S_seq]
        '''
        S_seq = np.digitize( zeta_seq, S_grid_bounds ).astype(int)
        EV_seq = EV_grid[:,S_seq]
        val_seq = np.c_[np.zeros(len(zeta_seq)), zeta_seq].T + beta* EV_seq
        pchoice_seq = self.choice_prob(val_seq)
        share_hat = np.zeros(nobs)
        '''
        for i in range(nobs):
            if loc_firstobs[i]:
                remain = 1.
            share_hat[i] = remain*pchoice_seq[1,i]
            remain = remain-share_hat[i]
        '''
        for i in range(self.nprod):
            remain = np.cumprod( np.append(1., pchoice_seq[0,self.prodid==i]) )[:-1]
            share_hat[self.prodid==i] = remain*pchoice_seq[1,self.prodid==i]
        return share_hat

    def FP_zeta(self,beta,share_obs, phi_guess, v_var_guess,\
                EV_grid,
                S_gridN,S_gridmin,S_gridmax,S_grid,S_grid_bounds,S_gridsize,\
                threshold=1e-6, maxiter=10e3,step=.05):
        nobs = len(share_obs)
        zeta_hat = 1.*np.ones(nobs)
        zeta_hat_new = 1.*np.ones(nobs)
        norm = 100.
        k=0
        while norm>threshold:    
            zeta_hat = zeta_hat_new
            share_hat = self.Calc_share_hat(zeta_hat,S_grid, S_grid_bounds, EV_grid, beta)
            if np.any(share_hat==0):
                zeta_hat[np.where(share_hat==0)]=zeta_hat[np.where(share_hat==0)]+.0001
                share_hat = self.Calc_share_hat(zeta_hat,S_grid, S_grid_bounds, EV_grid, beta)            
            update = step*( np.log(share_obs)-np.log(share_hat) )
            zeta_hat_new = zeta_hat+update
            norm = np.max( np.abs(zeta_hat_new-zeta_hat) )
            k=k+1
        return zeta_hat

    def Unpdate_phi(self,zeta_seq,x,iv_phi):
        nobs=len(zeta_seq)
        phi, f = self.tsls_1d(x=x,y=zeta_seq,iv=iv_phi)
        eta = zeta_seq-phi*x
        return phi, eta


    def Update_rho(self,eta_seq):
        nobs=len(eta_seq)
        x = eta_seq[~self.loc_lastobs]
        y = eta_seq[~self.loc_firstobs]
        '''
        if iv_rho.shape[0]==nobs:
            iv_rho = iv_rho[~self.loc_lastobs,:]
        rho, f = self.tsls_1d(x,y,iv_rho)
        '''
        lr1=linear_model.LinearRegression(fit_intercept=False)
        lr1.fit(x,y)
        rho = lr1.coef_
        nu = y-rho*x
        return rho, nu
        
    def tsls_1d(self,x,y,iv):    
        nobs = len(y)
        x=x.reshape([nobs,-1])
        y=y.reshape([nobs,-1])
        z = np.c_[x, iv]
        N_inst = z.shape[1]
        invA = np.linalg.solve( np.dot(z.T,z), np.identity(N_inst) )
        temp1 = np.dot(x.T,z)
        temp2 = np.dot(y.T,z)
        temp3 = np.dot(np.dot(temp1,invA),temp1.T) #x'z(z'z)^{-1}z'x
        temp4 = np.dot(np.dot(temp1,invA),temp2.T) #x'z(z'z)^{-1}z'y
        bhat = temp4/temp3
        gmmresid = y - np.dot(x, bhat)
        temp5=np.dot(gmmresid.T, z)
        f=np.dot(np.dot(temp5/nobs, invA),(temp5.T)/nobs)
        return bhat, f
    
    def Calc_gmm(self, nu_short, iv_nu, W=None):
        nobs_nu = len(nu_short)
        nobs_iv_nu = iv_nu.shape[0]               
        nu=nu_short.reshape([nobs_nu,-1])
        nu_fd = nu_short[1:]-nu_short[:-1]
        iv_nu_lagged = iv_nu[1:,:]
        nu_comb = np.vstack((nu_short, nu_fd) )
        iv_comb = np.vstack( (iv_nu, iv_nu_lagged) )
        n_iv = iv_comb.shape[1]
        if W is None:
            W = np.linalg.solve( np.dot(iv_comb.T,iv_comb), np.identity(n_iv) )
        temp1 = np.dot(nu_comb.T, iv_comb)
        gmm = np.dot( np.dot( temp1, W ), temp1.T)
        return gmm
    
    def Create_iv_nu(zeta_seq):
        zeta_lag1=zeta_seq[~self.loc_lastobs]
        zeta_lag2=zeta_seq[~self.loc_lastobs2]
        iv_nu = np.c_[zeta_lag1[~self.loc_firstobs],zeta_lag2]
        if self.iv_nu_other is not None:
            iv_nu = np.c_[iv_nu,iv_nu_other]
        return iv_nu
        
    def make_gmmobj(self):
        def gmmobj(para):
            phi_guess = para[0]
            v_var_guess = para[1]
            states_def = self.Gen_states_def(phi=phi_guess, v_var=v_var_guess,**self.S_setting)
            EV_grid,U_myopic = self.contraction_mapping(beta=self.beta,**states_def)
            zeta_hat = self.FP_zeta(share_obs=self.share_obs, phi_guess=phi_guess, v_var_guess=v_var_guess,\
                                    EV_grid=EV_grid, beta=self.beta, **self.S_setting)
            phi_guess,eta_seq = self.Update_phi(zeta_seq=zeta_hat,x=self.char,iv_phi=self.iv_phi)
            iv_rho = self.Create_iv_rho(zeta_seq=zeta_hat)
            rho_guess, nu_seq = self.Update_rho(eta_seq=eta_seq)
            nu_short = nu[((1-self.firstobs)*(1-self.firstobs2)).astype(bool)]
            f = self.Calc_gmm(nu,iv_nu=iv_nu)
            return f
        return gmmobj
    
    def fit(self):
        self.iv_phi = 
        self.loc_firstobs2 = 
        self.loc_lastobs2 = 
        if self.iv_nu_other is not None and self.iv_nu_other.shape[0]==self.nobs:
            self.iv_nu_other = self.iv_nu_other[((1-self.firstobs)*(1-self.firstobs2)).astype(bool)]
        
