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

from npcasadi_api import SymbolicHandler
from utilities import uptria2vec
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

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def compute(self, v1, v2, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)

        chi = npcsd.concatenate([v1, npcsd.array(v2, array_type="SX")])
        if self.model_name == "quad-lin":
            polynom = npcsd.concatenate([uptria2vec(np.outer(chi, chi)), chi])
        elif self.model_name == "quadratic":
            polynom = npcsd.concatenate([uptria2vec(np.outer(chi, chi))])
        elif self.model_name == "quad-nomix":
            polynom = chi * chi
        elif self.model_name == "quad-mix":
            polynom = npcsd.concatenate([v1 ** 2, npcsd.kron(v1, v2), v2 ** 2])

        return polynom


class ModelQuadForm:
    def __init__(self, R1=None, R2=None, model_name="quadratic"):
        self.model_name = model_name
        self.R1 = R1
        self.R2 = R2

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

    def compute(self, v1, v2, is_symbolic=False):
        npcsd = SymbolicHandler(is_symbolic)
        if self.model_name == "quadratic":
            result = npcsd.matmul(npcsd.matmul(v1.T, self.R1), v2)
        elif self.model_name == "biquadratic":
            result = npcsd.matmul(
                npcsd.matmul(v1.T ** 2, self.R2), v2 ** 2
            ) + npcsd.matmul(npcsd.matmul(v1.T, self.R1), v2)

        return result
