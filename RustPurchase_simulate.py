# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:40:43 2017

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

class SimulateDynShare:
    def __init__(self, S_gridN = 3,S_gridmin = -1.,S_gridmax = 1.,\
                 T=10,N_cons=10000,nprod=5,nchar=3,\
                 dpara=np.ones(3) , beta=.75,mean_x=0.,var_x=1.,cov_x=0.,nu_var = 1.,rho = .8,\
                 flag_char_dyn=1):
        
        #state space
        self.S_grid,self.S_gridsize = np.linspace(S_gridmin, S_gridmax, num=S_gridN, retstep=True)
        self.S_grid_bounds = np.linspace(S_gridmin+self.S_gridsize/2, S_gridmax-self.S_gridsize/2,num=S_gridN-1)
        self.S_setting ={ 'S_gridN':S_gridN,'S_gridmin':S_gridmin,'S_gridmax':S_gridmax,'S_grid':self.S_grid,\
        'S_grid_bounds':self.S_grid_bounds,'S_gridsize':self.S_gridsize}
        
        #data num
        self.T=T
        self.nprod=nprod
        self.nobs=T*nprod
        self.prodid = np.repeat(np.arange(nprod),T)
        self.mktid = np.tile( np.arange(T),nprod )
        self.loc_firstobs = np.append( True, self.prodid[1:]!=self.prodid[:-1] )
        self.loc_lastobs = np.append( self.prodid[1:]!=self.prodid[:-1], True )
        self.N_cons=N_cons

        #Mean utility simulation
        self.char_setting = {'mean_x':mean_x,'var_x':var_x,'cov_x':cov_x,'nchar':nchar,'nobs':self.nobs,'flag_char_dyn':flag_char_dyn}
        
        #parameters
        self.nu_var=nu_var
        self.rho=rho
        self.beta=beta
        self.dpara=dpara
        

    def CreateChar(self,mean_x,var_x,cov_x,nchar,nobs,flag_char_dyn):
        m = mean_x * np.ones(nchar-1)
        v = cov_x * np.ones([nchar-1,nchar-1])
        np.fill_diagonal(v, var_x )
        if flag_char_dyn==0:
            chars = np.random.multivariate_normal(m,v,nprod) #Assuming static char
            chars = np.c_[np.ones([nprod,1]), chars]
            char = chars[prodid]
        if flag_char_dyn==1:
            chars = np.random.multivariate_normal(m,v,nobs) #Assuming dynamic char
            chars = np.c_[np.ones([nobs,1]), chars]
            char = chars
        return char

    def Gen_eta(self):
        nu = np.random.normal(size=[self.T,self.nprod])*np.sqrt(self.nu_var)
        eta_seq = np.zeros([self.T,self.nprod])
        eta_seq[0,:] = nu[0,:]
        for t in range(1,self.T):
            eta_seq[t,:] = self.rho*eta_seq[t-1,:]+nu[t-1,:]
        return eta_seq
        
    def Gen_zeta(self):
        self.char = self.CreateChar(**self.char_setting)
        eta = self.Gen_eta().T.flatten()
        U_char = np.dot(self.char, self.dpara)
        zeta = U_char + eta
        return zeta
    
    def Gen_S_seq(self):
        zeta_seq = self.Gen_zeta()
        S_seq = np.digitize( zeta_seq, self.S_grid_bounds ).astype(int)
        return S_seq,zeta_seq
    '''
    def Calc_phi(self,zeta_mat): #input T by nprod matrix of zeta. common phi across products.
        lr = linear_model.LinearRegression(fit_intercept=False)
        x = zeta_mat[:-1,:].T.flatten().reshape([-1,1])
        y = zeta_mat[1:,:].T.flatten()
        lr.fit(x,y)
        phi = lr.coef_
        v = y - np.dot(x,phi)
        v_var = np.var(v)
        return phi, v_var
    ''' 
    def Calc_phi(self,zeta_vec): #input T by nprod matrix of zeta. common phi across products.
        lr = linear_model.LinearRegression(fit_intercept=False)
        x = zeta_vec[~self.loc_lastobs].reshape([-1,1])
        y = zeta_vec[~self.loc_firstobs]
        lr.fit(x,y)
        phi = lr.coef_
        v = y - np.dot(x,phi)
        v_var = np.var(v)
        return phi, v_var
    
    #Generate Transition Matrix for consumer
    def Gen_states_def(self,phi, v_var,S_gridN,S_gridmin,S_gridmax,S_grid,\
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


    def Simulate(self):
        #Generate the real transition
        S_real,zeta_real = self.Gen_S_seq()
        
        #gen states transition from consumer perspective
        phi, v_var = self.Calc_phi(zeta_real)
        
        self.states_def=self.Gen_states_def(phi,v_var,**self.S_setting)
        #Calculate choice probability    
        EV_grid, U_myopic = self.contraction_mapping(beta=self.beta,**self.states_def)
        val_grid = U_myopic + self.beta*EV_grid
        pchoice_grid = self.choice_prob(val_grid)
        
        self.pchoice_grid = pchoice_grid
        #Simulate share        
        pchoice_seq = pchoice_grid[:,S_real]
        self.S_real=S_real
        self.pchoice_seq=pchoice_seq
        '''
        remain = np.zeros(self.nobs)
        for i in range(0,self.nprod):
            remain[self.prodid==i]=np.cumprod( np.append(1.,pchoice_seq[0,self.prodid==i]))[:-1]
        self.remain=remain
        share_seq = remain*pchoice_seq[1,:]
        '''
        share_obs = np.zeros(self.nobs)
        for i in range(self.nobs):
            if self.loc_firstobs[i]:
                remain=self.N_cons
            buy_obs = np.random.binomial(remain, p=pchoice_seq[1,i])
            remain = remain-buy_obs
            share_obs[i] = buy_obs/self.N_cons
        self.share_obs = share_obs
        
        self.data = {'char':self.char,'share':share_obs}

        
       
if __name__=='__main__':
    sim = SimulateDynShare()
    sim.Simulate()


