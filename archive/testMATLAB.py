#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:25:50 2020

@author: pavel
"""

import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'FOLDER_WITH_MATLAB_SCRIPTS',nargout=0)

from numpy.random import randn

us = randn(2, 100)
ys = randn(3, 100)

us_ml = matlab.double(us.tolist())
ys_ml = matlab.double(ys.tolist())

A, B, C, D = eng.mySSest_simple(ys_ml, us_ml, 0.01, 1, nargout=4)

print(A)
