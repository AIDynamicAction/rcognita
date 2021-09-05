#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains auxiliary functions

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
from numpy.random import rand
from numpy.matlib import repmat
import scipy.stats as st
from scipy import signal
import matplotlib.pyplot as plt

def rej_sampling_rvs(dim, pdf, M):
    """
    Random variable (pseudo)-realizations via rejection sampling
    
    Parameters
    ----------
    dim : : integer
        dimension of the random variable
    pdf : : function
        desired probability density function
    M : : number greater than 1
        it must hold that :math:`\\text{pdf}_{\\text{desired}} \le M \\text{pdf}_{\\text{proposal}}`.
        This function uses a normal pdf with zero mean and identity covariance matrix as a proposal distribution.
        The smaller `M` is, the fewer iterations to produce a sample are expected.

    Returns
    -------
    A single realization (in general, as a vector) of the random variable with the desired probability density

    """
    
    # Use normal pdf with zero mean and identity covariance matrix as a proposal distribution
    normal_RV = st.multivariate_normal(cov=np.eye(dim))
    
    # Bound the number of iterations to avoid too long loops
    max_iters = 1e3
    
    curr_iter = 0
    
    while curr_iter <= max_iters:
        proposal_sample = normal_RV.rvs()
    
        unif_sample = rand()
        
        if unif_sample < pdf(proposal_sample) / M / normal_RV.pdf(proposal_sample):
            return proposal_sample
        
def to_col_vec(argin):
    """
    Convert input to a column vector

    """
    if argin.ndim < 2:
        return np.reshape(argin, (argin.size, 1))
    elif argin.ndim ==2:
        if argin.shape[0] < argin.shape[1]:
            return argin.T
        else:
            return argin

def rep_mat(argin, n, m):
    """
    Ensures 1D result
    
    """
    return np.squeeze(repmat(argin, n, m))

def push_vec(matrix, vec):
    return np.vstack([matrix[1:,:], vec])

def uptria2vec(mat):
    """
    Convert upper triangular square sub-matrix to column vector
    
    """    
    n = mat.shape[0]
    
    vec = np.zeros( (int(n*(n+1)/2)) )
    
    k = 0
    for i in range(n):
        for j in range(n):
            vec[j] = mat[i, j]
            k += 1
            
    return vec

class ZOH:
    """
    Zero-order hold
    
    """    
    def __init__(self, init_time=0, init_val=0, sample_time=1):
        self.time_step = init_time
        self.sample_time = sample_time
        self.currVal = init_val
        
    def hold(self, signal_val, t):
        timeInSample = t - self.time_step
        if timeInSample >= self.sample_time: # New sample
            self.time_step = t
            self.currVal = signal_val

        return self.currVal
    
class DFilter:
    """
    Real-time digital filter
    
    """
    def __init__(self, filter_num, filter_den, buffer_size=16, init_time=0, init_val=0, sample_time=1):
        self.Num = filter_num
        self.Den = filter_den
        self.zi = rep_mat( signal.lfilter_zi(filter_num, filter_den), 1, init_val.size)
        
        self.time_step = init_time
        self.sample_time = sample_time
        self.buffer = rep_mat(init_val, 1, buffer_size)
        
    def filt(self, signal_val, t=None):
        # Sample only if time is specified
        if t is not None:
            timeInSample = t - self.time_step
            if timeInSample >= self.sample_time: # New sample
                self.time_step = t
                self.buffer = push_vec(self.buffer, signal_val)
        else:
            self.buffer = push_vec(self.buffer, signal_val)
        
        bufferFiltered = np.zeros(self.buffer.shape)
        
        for k in range(0, signal_val.size):
                bufferFiltered[k,:], self.zi[k] = signal.lfilter(self.Num, self.Den, self.buffer[k,:], zi=self.zi[k, :])
        return bufferFiltered[-1,:]
    
def dss_sim(A, B, C, D, uSqn, x0, y0):
    """
    Simulate output response of a discrete-time state-space model
    """
    if uSqn.ndim == 1:
        return y0, x0
    else:
        ySqn = np.zeros( [ uSqn.shape[0], C.shape[0] ] )
        xSqn = np.zeros( [ uSqn.shape[0], A.shape[0] ] )
        x = x0
        ySqn[0, :] = y0
        xSqn[0, :] = x0 
        for k in range( 1, uSqn.shape[0] ):
            x = A @ x + B @ uSqn[k-1, :]
            xSqn[k, :] = x
            ySqn[k, :] = C @ x + D @ uSqn[k-1, :]
            
        return ySqn, xSqn     
    
def upd_line(line, newX, newY):
    line.set_xdata( np.append( line.get_xdata(), newX) )
    line.set_ydata( np.append( line.get_ydata(), newY) )  
    
def reset_line(line):
    line.set_data([], [])     
 
def upd_scatter(scatter, newX, newY):
    scatter.set_offsets( np.vstack( [ scatter.get_offsets().data, np.c_[newX, newY] ] ) )
    
def upd_text(textHandle, newText):
    textHandle.set_text(newText)    
    
def on_key_press(event, anm):  
    """
    Key press event handler for a ``FuncAnimation`` animation object

    """
    if event.key==' ':
        if anm.running:
            anm.event_source.stop()
            
        else:
            anm.event_source.start()
        anm.running ^= True
    elif event.key=='q':
        plt.close('all')
        raise Exception('exit')    