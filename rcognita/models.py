#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for models
"""

# class models

class model_SS:
     """
         Class of estimated models
         
         So far, uses just the state-space structure:
             
     .. math::
         \\begin{array}{ll}
 			\\hat x^+ & = A \\hat x + B u \\newline
 			y^+  & = C \\hat x + D u,
         \\end{array}                 
         
     Attributes
     ---------- 
     A, B, C, D : : arrays of proper shape
         State-space model parameters
     x0set : : array
         Initial state estimate
         
     **When introducing your custom model estimator, adjust this class**    
         
     """
     
     def __init__(self, A, B, C, D, x0est):
         self.A = A
         self.B = B
         self.C = C
         self.D = D
         self.x0est = x0est
         
     def upd_pars(self, Anew, Bnew, Cnew, Dnew):
         self.A = Anew
         self.B = Bnew
         self.C = Cnew
         self.D = Dnew
         
     def updateIC(self, x0setNew):
         self.x0set = x0setNew
         
class model_NN:
    ...
    # if self.dt != dt -> error