#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains classes to be used in fitting system models.

Updates to come.

"""
import numpy as np
import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

from utilities import uptria2vec, nc
import numpy as np


class ModelSS:
    """
    State-space model
            
    .. math::
        \\begin{array}{ll}
			\\hat x^+ & = A \\hat x + B u, \\newline
			y^+  & = C \\hat x + D u.
        \\end{array}                 
        
    Attributes
    ---------- 
    A, B, C, D : : arrays of proper shape
        State-space model parameters.
    x0set : : array
        Initial state estimate.
            
    """

    def __init__(self, A, B, C, D, x0est):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x0est = x0est
        self.model_name = "state-space"

    def upd_pars(self, Anew, Bnew, Cnew, Dnew):
        self.A = Anew
        self.B = Bnew
        self.C = Cnew
        self.D = Dnew

    def updateIC(self, x0setNew):
        self.x0set = x0setNew


class ModelNN:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(f"Class {self.__class__} is not yet implemented.")


class ModelPolynomial:
    def __init__(self, model_name="quad-lin"):
        self.model_name = model_name

    def __call__(self, v1, v2, weights):
        return self.compute(v1, v2, weights)

    def compute(self, v1, v2, weights):

        v1 = nc.to_col(v1)
        v2 = nc.to_col(v2)
        chi = nc.concatenate([v1, v2])
        if self.model_name == "quad-lin":
            polynom = nc.to_col(uptria2vec(nc.outer(chi, chi)))
            polynom = nc.concatenate([polynom, chi]) ** 2
        elif self.model_name == "quadratic":
            polynom = nc.to_col(uptria2vec(nc.outer(chi, chi))) ** 2
        elif self.model_name == "quad-nomix":
            polynom = chi * chi
        elif self.model_name == "quad-mix":
            polynom = nc.concatenate([v1 ** 2, nc.kron(v1, v2), v2 ** 2])

        result = nc.dot(weights, polynom)

        return result

    def gradient(self, vector, weights):

        gradient = nc.autograd(self.compute, vector, [], weights)

        return gradient


class ModelQuadForm:
    def __init__(self, R1=None, R2=None, model_name="quadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, v1, v2):
        return self.compute(v1, v2)

    def compute(self, v1, v2):
        result = nc.matmul(nc.matmul(v1.T, self.R1), v2)

        result = nc.squeeze(result)

        return result


class ModelBiquadForm:
    def __init__(self, R1=None, R2=None, model_name="biquadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, v1, v2):
        return self.compute(v1, v2)

    def compute(self, v1, v2):
        result = nc.matmul(nc.matmul(v1.T ** 2, self.R2), v2 ** 2) + nc.matmul(
            nc.matmul(v1.T, self.R1), v2
        )
        result = nc.squeeze(result)

        return result
