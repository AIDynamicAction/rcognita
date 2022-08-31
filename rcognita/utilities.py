#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains auxiliary functions.

Remarks: 

- All vectors are treated as of type [n,]
- All buffers are treated as of type [L, n] where each row is a vector
- Buffers are updated from bottom to top

"""

import numpy as np
import os, sys
from enum import IntEnum

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
from numpy.random import rand
from numpy.matlib import repmat
import scipy.stats as st
from scipy import signal
import matplotlib.pyplot as plt
import casadi

import inspect
import warnings

try:
    import casadi

    CASADI_TYPES = tuple(
        x[1] for x in inspect.getmembers(casadi.casadi, inspect.isclass)
    )
except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting casadi failed. You may still use rcognita, but"
        + " without symbolic optimization capability. ",
        UserWarning,
        __file__,
        42,
    )
    CASADI_TYPES = []
import types

try:
    import torch

    TORCH_TYPES = tuple(x[1] for x in inspect.getmembers(torch, inspect.isclass))
except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting pytorch failed. You may still use rcognita, but"
        + " without pytorch optimization capability. ",
        UserWarning,
        __file__,
        42,
    )
    TORCH_TYPES = []


class RCType(IntEnum):
    TORCH = 3
    CASADI = 2
    NUMPY = 1


TORCH = RCType.TORCH
CASADI = RCType.CASADI
NUMPY = RCType.NUMPY


def is_CasADi_typecheck(*args):
    return CASADI if any([isinstance(arg, CASADI_TYPES) for arg in args]) else False


def is_Torch_typecheck(*args):
    return TORCH if any([isinstance(arg, TORCH_TYPES) for arg in args]) else False


def type_inference(*args, **kwargs):
    is_CasADi = is_CasADi_typecheck(*args, *kwargs.values())
    is_Torch = is_Torch_typecheck(*args, *kwargs.values())
    if is_CasADi + is_Torch > 4:
        raise TypeError(
            "There is no support for simultaneous usage of both NumPy and CasADi"
        )
    else:
        result_type = max(is_CasADi, is_Torch, NUMPY)
        return result_type


def safe_unpack(argin):
    if isinstance(argin, (list, tuple)):
        return argin
    else:
        return (argin,)


def decorateAll(decorator):
    class MetaClassDecorator(type):
        def __new__(meta, classname, supers, classdict):
            for name, elem in classdict.items():
                if (
                    type(elem) is types.FunctionType
                    and (name != "__init__")
                    and not isinstance(elem, staticmethod)
                ):
                    classdict[name] = decorator(classdict[name])
            return type.__new__(meta, classname, supers, classdict)

    return MetaClassDecorator


@decorateAll
def metaclassTypeInferenceDecorator(func):
    def wrapper(*args, **kwargs):
        rc_type = kwargs.get("rc_type")
        if rc_type is not None:
            del kwargs["rc_type"]
            return func(rc_type=rc_type, *args, **kwargs)
        else:

            return func(rc_type=type_inference(*args, **kwargs), *args, **kwargs)

    return wrapper


class RCTypeHandler(metaclass=metaclassTypeInferenceDecorator):
    def cos(self, x, rc_type=NUMPY):
        if rc_type == NUMPY:
            return np.cos(x)
        elif rc_type == TORCH:
            return torch.cos(x)
        elif rc_type == CASADI:
            return casadi.cos(x)

    def sin(self, x, rc_type=NUMPY):
        if rc_type == NUMPY:
            return np.sin(x)
        elif rc_type == TORCH:
            return torch.sin(x)
        elif rc_type == CASADI:
            return casadi.sin(x)

    def hstack(self, tup, rc_type=NUMPY):
        rc_type = type_inference(*tup)

        if rc_type == NUMPY:
            return np.hstack(tup)
        elif rc_type == TORCH:
            return torch.hstack(tup)
        elif rc_type == CASADI:
            return casadi.horzcat(*tup)

    def vstack(self, tup, rc_type=NUMPY):
        rc_type = type_inference(*tup)

        if rc_type == NUMPY:
            return np.vstack(tup)
        elif rc_type == TORCH:
            return torch.vstack(tup)
        elif rc_type == CASADI:
            return casadi.vertcat(*tup)

    def push_vec(self, matrix, vec, rc_type=NUMPY):
        return self.vstack([matrix[1:, :], vec.T], rc_type=rc_type)

    def reshape_CasADi_as_np(self, array, dim_params, rc_type=NUMPY):
        result = casadi.SX(*dim_params)
        n_rows = dim_params[0]
        n_cols = dim_params[1]
        for i in range(n_rows):
            result[i, :] = array[i * n_cols : (i + 1) * n_cols]

        return result

    def reshape(self, array, dim_params, rc_type=NUMPY):

        if rc_type == CASADI:
            if isinstance(dim_params, (list, tuple)):
                if len(dim_params) > 1:
                    return self.reshape_CasADi_as_np(array, dim_params)
                else:
                    return casadi.reshape(array, dim_params[0], 1)
            elif isinstance(dim_params, int):
                return casadi.reshape(array, dim_params, 1)
            else:
                raise TypeError(
                    "Wrong type of dimension parameter was passed.\
                         Possible cases are: int, [int], [int, int, ...]"
                )

        elif rc_type == NUMPY:
            return np.reshape(array, dim_params)

        elif rc_type == TORCH:
            return torch.reshape(array, dim_params)

    def array(self, array, prototype=None, rc_type=NUMPY):
        rc_type = type_inference(prototype)

        if rc_type == NUMPY:
            return np.array(array)
        elif rc_type == TORCH:
            return torch.tensor(array)
        elif rc_type == CASADI:
            casadi_constructor = type(prototype)
            return casadi_constructor(array)

    def ones(self, argin, prototype=None, rc_type=NUMPY):
        if rc_type == NUMPY:
            return np.ones(argin)
        elif rc_type == TORCH:
            return torch.ones(argin)
        elif rc_type == CASADI:
            casadi_constructor = type(prototype) if prototype else casadi.DM

            if isinstance(argin, int):
                return casadi_constructor.ones(argin)
            elif isinstance(argin, (tuple, list)):
                return casadi_constructor.ones(*argin)

    def zeros(self, argin, prototype=None, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.zeros(argin)
        elif rc_type == TORCH:
            return torch.zeros(argin)
        elif rc_type == CASADI:
            casadi_constructor = type(prototype) if prototype is not None else casadi.DM

            if isinstance(argin, int):
                return casadi_constructor.zeros(argin)
            elif isinstance(argin, (tuple, list)):
                return casadi_constructor.zeros(*argin)

    def concatenate(self, argin, rc_type=NUMPY):
        rc_type = type_inference(*safe_unpack(argin))

        if rc_type == NUMPY:
            return np.concatenate(argin)
        elif rc_type == TORCH:
            return torch.cat(argin)
        elif rc_type == CASADI:
            if isinstance(argin, (list, tuple)):
                if len(argin) > 1:
                    argin = [rc.to_col(x) for x in argin]
                    return casadi.vertcat(*argin)
                else:
                    raise NotImplementedError(
                        f"Concatenation is not implemented for argument of type {type(argin)}."
                        + "Possible types are: list, tuple"
                    )

    def rep_mat(self, array, n, m, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.squeeze(np.tile(array, (n, m)))
        elif rc_type == TORCH:
            return torch.tile(array, (n, m))
        elif rc_type == CASADI:
            return casadi.repmat(array, n, m)

    def matmul(self, A, B, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.matmul(A, B)
        elif rc_type == TORCH:
            A = torch.tensor(A).double()
            B = torch.tensor(B).double()
            return torch.matmul(A, B)
        elif rc_type == CASADI:
            return casadi.mtimes(A, B)

    def casadi_outer(self, v1, v2, rc_type=NUMPY):

        if not is_CasADi_typecheck(v1):
            v1 = self.array_symb(v1)

        return casadi.horzcat(*[v1 * v2_i for v2_i in v2.nz])

    def outer(self, v1, v2, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.outer(v1, v2)
        elif rc_type == TORCH:
            return torch.outer(v1, v2)
        elif rc_type == CASADI:
            return self.casadi_outer(v1, v2)

    def sign(self, x, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.sign(x)
        elif rc_type == TORCH:
            return torch.sign(x)
        elif rc_type == CASADI:
            return casadi.sign(x)

    def abs(self, x, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.abs(x)
        elif rc_type == TORCH:
            return torch.abs(x)
        elif rc_type == CASADI:
            return casadi.fabs(x)

    def min(self, array, rc_type=NUMPY):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(array)

        if rc_type == NUMPY:
            return np.min(array)
        elif rc_type == TORCH:
            return torch.min(array)
        elif rc_type == CASADI:
            return casadi.fmin(*array)

    def max(self, array, rc_type=NUMPY):
        if isinstance(array, (list, tuple)):
            rc_type = type_inference(array)

        if rc_type == NUMPY:
            return np.max(array)
        elif rc_type == TORCH:
            return torch.max(array)
        elif rc_type == CASADI:
            return casadi.max(*safe_unpack(array))

    def to_col(self, argin, rc_type=NUMPY):
        arin_shape = self.shape(argin)

        if len(arin_shape) > 1:
            if self.shape(argin)[0] < self.shape(argin)[1]:
                return argin.T
            else:
                return argin
        else:
            if rc_type == NUMPY:
                return np.reshape(argin, (argin.size, 1))
            elif rc_type == TORCH:
                return torch.reshape(argin, (argin.size()[0], 1))

    def dot(self, A, B, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.dot(A, B)
        elif rc_type == TORCH:
            return torch.dot(A, B)
        elif rc_type == CASADI:
            return casadi.dot(A, B)

    def sqrt(self, x, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.sqrt(x)
        elif rc_type == TORCH:
            return torch.sqrt(x)
        elif rc_type == CASADI:
            return casadi.sqrt(x)

    def shape(self, array, rc_type=NUMPY):

        if rc_type == CASADI:
            return array.size()
        elif rc_type == NUMPY:
            return np.shape(array)
        elif rc_type == TORCH:
            return array.size()

    def func_to_lambda_with_params(
        self, func, *params, var_prototype=None, rc_type=NUMPY
    ):

        if rc_type == NUMPY or rc_type == TORCH:
            if params:
                return lambda x: func(x, *params)
            else:
                return lambda x: func(x)
        else:
            try:
                x_symb = casadi.SX.sym("x", self.shape(var_prototype))
            except NotImplementedError:
                x_symb = casadi.SX.sym("x", *safe_unpack(self.shape(var_prototype)), 1)

            if params:
                return func(x_symb, *safe_unpack(params)), x_symb
            else:
                return func(x_symb), x_symb

    def lambda2symb(self, lambda_func, x_symb, rc_type=NUMPY):
        return lambda_func(x_symb)

    def if_else(self, c, x, y, rc_type=NUMPY):

        if rc_type == CASADI:
            res = casadi.if_else(c, x, y)
            return res
        elif rc_type == TORCH or rc_type == NUMPY:
            return x if c else y

    def kron(self, A, B, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.kron(A, B)
        elif rc_type == TORCH:
            return torch.kron(A, B)
        elif rc_type == CASADI:
            return casadi.kron(A, B)

    def array_symb(self, tup=None, literal="x", rc_type=NUMPY, prototype=None):
        if prototype is not None:
            shape = self.shape(prototype)
        else:
            shape = tup

        if isinstance(shape, tuple):
            if len(tup) > 2:
                raise ValueError(
                    f"Not implemented for number of dimensions grreater than 2. Passed: {len(tup)}"
                )
            else:
                return casadi.SX.sym(literal, *tup)

        elif isinstance(tup, int):
            return casadi.SX.sym(literal, tup)

        else:
            raise TypeError(
                f"Passed an invalide argument of type {type(tup)}. Takes either int or tuple data types"
            )

    def norm_1(self, v, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.linalg.norm(v, 1)
        elif rc_type == TORCH:
            return torch.linalg.norm(v, 1)
        elif rc_type == CASADI:
            return casadi.norm_1(v)

    def norm_2(self, v, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.linalg.norm(v, 2)
        elif rc_type == TORCH:
            return torch.linalg.norm(v, 2)
        elif rc_type == CASADI:
            return casadi.norm_2(v)

    def logic_and(self, a, b, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.logical_and(a, b)
        elif rc_type == TORCH:
            return torch.logical_and(a, b)
        elif rc_type == CASADI:
            return casadi.logic_and(a, b)

    def squeeze(self, v, rc_type=NUMPY):

        if rc_type == NUMPY:
            return np.squeeze(v)
        elif rc_type == TORCH:
            return torch.squeeze(v)
        elif rc_type == CASADI:
            return v

    def uptria2vec(self, mat, rc_type=NUMPY):
        if rc_type == NUMPY:
            result = mat[np.triu_indices(self.shape(mat)[0])]
            return result
        elif rc_type == TORCH:
            result = mat[torch.triu_indices(self.shape(mat)[0])]
            return result
        elif rc_type == CASADI:
            n = self.shape(mat)[0]

            vec = rc.zeros((int(n * (n + 1) / 2)), prototype=mat)

            k = 0
            for i in range(n):
                for j in range(i, n):
                    vec[k] = mat[i, j]
                    k += 1

            return vec

    @staticmethod
    def DM(mat):
        return casadi.DM(mat)

    @staticmethod
    def SX(mat):
        return casadi.SX(mat)

    @staticmethod
    def autograd(func, x, *args):
        return casadi.Function("f", [x, *args], [casadi.gradient(func(x, *args), x)])


rc = RCTypeHandler()


def rej_sampling_rvs(dim, pdf, M):
    """
    Random variable (pseudo)-realizations via rejection sampling.
    
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
    A single realization (in general, as a vector) of the random variable with the desired probability density.

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


def push_vec(matrix, vec):
    return rc.vstack([matrix[1:, :], vec.T])


def uptria2vec(mat):
    """
    Convert upper triangular square sub-matrix to column vector.
    
    """


class ZOH:
    """
    Zero-order hold.
    
    """

    def __init__(self, init_time=0, init_val=0, sample_time=1):
        self.time_step = init_time
        self.sample_time = sample_time
        self.currVal = init_val

    def hold(self, signal_val, t):
        timeInSample = t - self.time_step
        if timeInSample >= self.sample_time:  # New sample
            self.time_step = t
            self.currVal = signal_val

        return self.currVal


class DFilter:
    """
    Real-time digital filter.
    
    """

    def __init__(
        self,
        filter_num,
        filter_den,
        buffer_size=16,
        init_time=0,
        init_val=0,
        sample_time=1,
    ):
        self.Num = filter_num
        self.Den = filter_den
        self.zi = rc.rep_mat(
            signal.lfilter_zi(filter_num, filter_den), 1, init_val.size
        )

        self.time_step = init_time
        self.sample_time = sample_time
        self.buffer = rc.rep_mat(init_val, 1, buffer_size)

    def filt(self, signal_val, t=None):
        # Sample only if time is specified
        if t is not None:
            timeInSample = t - self.time_step
            if timeInSample >= self.sample_time:  # New sample
                self.time_step = t
                self.buffer = push_vec(self.buffer, signal_val)
        else:
            self.buffer = push_vec(self.buffer, signal_val)

        bufferFiltered = np.zeros(self.buffer.shape)

        for k in range(0, signal_val.size):
            bufferFiltered[k, :], self.zi[k] = signal.lfilter(
                self.Num, self.Den, self.buffer[k, :], zi=self.zi[k, :]
            )
        return bufferFiltered[-1, :]


def dss_sim(A, B, C, D, uSqn, x0, y0):
    """
    Simulate output response of a discrete-time state-space model.
    """
    if uSqn.ndim == 1:
        return y0, x0
    else:
        ySqn = np.zeros([uSqn.shape[0], C.shape[0]])
        xSqn = np.zeros([uSqn.shape[0], A.shape[0]])
        x = x0
        ySqn[0, :] = y0
        xSqn[0, :] = x0
        for k in range(1, uSqn.shape[0]):
            x = A @ x + B @ uSqn[k - 1, :]
            xSqn[k, :] = x
            ySqn[k, :] = C @ x + D @ uSqn[k - 1, :]

        return ySqn, xSqn


def upd_line(line, newX, newY):
    line.set_xdata(np.append(line.get_xdata(), newX))
    line.set_ydata(np.append(line.get_ydata(), newY))


def reset_line(line):
    line.set_data([], [])


def upd_scatter(scatter, newX, newY):
    scatter.set_offsets(np.vstack([scatter.get_offsets().data, np.c_[newX, newY]]))


def upd_text(textHandle, newText):
    textHandle.set_text(newText)


def on_key_press(event, anm):
    """
    Key press event handler for a ``FuncAnimation`` animation object.

    """
    if event.key == " ":
        if anm.running:
            anm.event_source.stop()

        else:
            anm.event_source.start()
        anm.running ^= True
    elif event.key == "q":
        plt.close("all")
        raise Exception("exit")
