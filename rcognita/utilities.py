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


def is_CasADi_typecheck(*args):
    return any([isinstance(arg, CASADI_TYPES) for arg in args])


def decorateAll(decorator):
    class MetaClassDecorator(type):
        def __new__(meta, classname, supers, classdict):
            for name, elem in classdict.items():
                if type(elem) is types.FunctionType and (name != "__init__"):
                    classdict[name] = decorator(classdict[name])
            return type.__new__(meta, classname, supers, classdict)

    return MetaClassDecorator


@decorateAll
def typeInferenceDecorator(func):
    def wrapper(*args, **kwargs):
        is_symbolic = kwargs.get("is_symbolic")
        if not is_symbolic is None:
            del kwargs["is_symbolic"]
        return func(
            is_symbolic=(is_CasADi_typecheck(*args, *kwargs.values()) or is_symbolic),
            *args,
            **kwargs,
        )

    return wrapper


class SymbolicHandler(metaclass=typeInferenceDecorator):
    def cos(self, x, is_symbolic=False):
        return casadi.cos(x) if is_symbolic else np.cos(x)

    def sin(self, x, is_symbolic=False):
        return casadi.sin(x) if is_symbolic else np.sin(x)

    def hstack(self, tup, is_symbolic=False):
        is_symbolic = is_CasADi_typecheck(*tup)
        return casadi.horzcat(*tup) if is_symbolic else np.hstack(tup)

    def push_vec(self, matrix, vec, is_symbolic=False):
        return self.vstack([matrix[1:, :], vec.T])

    def vstack(self, tup, is_symbolic=False):
        is_symbolic = is_CasADi_typecheck(*tup)
        return casadi.vertcat(*tup) if is_symbolic else np.vstack(tup)

    def reshape_CasADi_as_np(self, array, dim_params, is_symbolic=False):
        result = casadi.MX(*dim_params)
        n_rows = dim_params[0]
        n_cols = dim_params[1]
        for i in range(n_rows):
            result[i, :] = array[i * n_cols : (i + 1) * n_cols]

        return result

    def reshape(self, array, dim_params, is_symbolic=False):
        if is_symbolic:
            if isinstance(dim_params, list) or isinstance(dim_params, tuple):
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
        else:
            return np.reshape(array, dim_params)

    def array(self, array, prototype=None, is_symbolic=False):
        array_type = type(prototype)
        return array_type(array) if is_CasADi_typecheck(array_type) else np.array(array)

    def ones(self, tup, is_symbolic=False):
        if isinstance(tup, int):
            return casadi.DM.ones(tup) if is_symbolic else np.ones(tup)
        else:
            return casadi.DM.ones(*tup) if is_symbolic else np.ones(tup)

    def zeros(self, tup, prototype=None, is_symbolic=False):
        if isinstance(tup, int):
            if is_symbolic:
                array_type = type(prototype)
                try:
                    return array_type.zeros(tup)
                except:
                    ValueError(f"Invalid array type:{array_type}")
            else:
                return np.zeros(tup)
        else:
            if is_symbolic:
                array_type = type(prototype)
                try:
                    return array_type.zeros(*tup)
                except:
                    ValueError(f"Invalid array type:{array_type}")
            else:
                return np.zeros(tup)

    def concatenate(self, tup, is_symbolic=False):
        if len(tup) > 1:
            is_symbolic = is_CasADi_typecheck(*tup)
            if is_symbolic:
                return casadi.vertcat(*tup)
            else:
                return np.concatenate(tup)

    def rep_mat(self, array, n, m, is_symbolic=False):
        return casadi.repmat(array, n, m) if is_symbolic else rep_mat(array, n, m)

    def matmul(self, A, B, is_symbolic=False):
        return casadi.mtimes(A, B) if is_symbolic else np.matmul(A, B)

    def inner_product(self, A, B, is_symbolic=False):
        return casadi.dot(A, B) if is_symbolic else np.inner(A, B)

    def casadi_outer(self, v1, v2, is_symbolic=False):
        if not is_CasADi_typecheck(v1):
            v1 = self.array_symb(v1)

        return casadi.horzcat(*[v1 * v2_i for v2_i in v2.nz])

    def outer(self, v1, v2, is_symbolic=False):
        return self.casadi_outer(v1, v2) if is_symbolic else np.outer(v1, v2)

    def sign(self, x, is_symbolic=False):
        return casadi.sign(x) if is_symbolic else np.sign(x)

    def abs(self, x, is_symbolic=False):
        return casadi.fabs(x) if is_symbolic else np.abs(x)

    def min(self, array, is_symbolic=False):
        return casadi.fmin(*array) if is_symbolic else np.min(array)

    def max(self, array, is_symbolic=False):
        is_symbolic = is_CasADi_typecheck(*array)
        if is_symbolic:
            if isinstance(array, list):
                array = self.vstack(tuple(array), is_symbolic=True)
            elif isinstance(array, tuple):
                array = self.vstack(array, is_symbolic=True)
            return casadi.mmax(array)
        else:
            return np.max(array)

    def to_col(self, argin, is_symbolic=False):
        if is_symbolic:
            if self.shape(argin)[0] < self.shape(argin)[1]:
                return argin.T
            else:
                return argin
        else:
            return to_col_vec(argin)

    def dot(self, A, B, is_symbolic=False):
        return casadi.dot(A, B) if is_symbolic else A @ B

    def sqrt(self, x, is_symbolic=False):
        return casadi.sqrt(x) if is_symbolic else np.sqrt(x)

    def shape(self, array, is_symbolic=False):
        return (
            array.size()
            if isinstance(array, (casadi.MX, casadi.DM, casadi.MX))
            else np.shape(array)
        )

    def func_to_lambda_with_params(self, func, *params, x0=None, is_symbolic=False):

        if not is_symbolic:
            return lambda x: func(x, *params)
        else:
            try:
                x_symb = casadi.MX.sym("x", self.shape(x0))
            except NotImplementedError as e:  #####
                x_symb = casadi.MX.sym("x", *self.shape(x0), 1)

            if params:
                return func(x_symb, *params), x_symb
            else:
                return func(x_symb), x_symb

    def lambda2symb(self, lambda_func, x_symb, is_symbolic=False):
        return lambda_func(x_symb)

    def if_else(self, c, x, y, is_symbolic=False):
        if is_symbolic:
            res = casadi.if_else(c, x, y)
            return res
        else:
            return x if c else y

    def kron(self, A, B, is_symbolic=False):
        return casadi.kron(A, B) if is_symbolic else np.kron(A, B)

    @staticmethod
    def autograd(func, x, *args, is_symbolic=False):
        return casadi.Function("f", [x, *args], [casadi.gradient(func(x, *args), x)])

    def array_symb(self, tup=None, literal="x", is_symbolic=False, prototype=None):
        if not prototype is None:
            tup = self.shape(prototype)
        if isinstance(tup, tuple):
            if len(tup) > 2:
                raise ValueError(
                    f"Not implemented for number of dimensions grreater than 2. Passed: {len(tup)}"
                )
            else:
                return casadi.MX.sym(literal, *tup)

        elif isinstance(tup, int):
            return casadi.MX.sym(literal, tup)

        else:
            raise TypeError(
                f"Passed an invalide argument of type {type(tup)}. Takes either int or tuple data types"
            )

    def norm_1(self, v, is_symbolic=False):
        return casadi.norm_1(v) if is_symbolic else np.linalg.norm(v, 1)

    def norm_2(self, v, is_symbolic=False):
        return casadi.norm_2(v) if is_symbolic else np.linalg.norm(v, 2)

    def logic_and(self, a, b, is_symbolic=False):
        return casadi.logic_and(a, b) if is_symbolic else (a and b)

    def squeeze(self, v, is_symbolic=False):
        return v if is_symbolic else np.squeeze(v)


nc = SymbolicHandler()


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


def to_col_vec(argin):
    """
    Convert input to a column vector.

    """
    if argin.ndim < 2:
        return np.reshape(argin, (argin.size, 1))
    elif argin.ndim == 2:
        if argin.shape[0] < argin.shape[1]:
            return argin.T
        else:
            return argin


def rep_mat(argin, n, m):
    """
    Ensures 1D result.
    
    """
    return np.squeeze(repmat(argin, n, m))


def push_vec(matrix, vec):
    return nc.vstack([matrix[1:, :], vec.T])


def uptria2vec(mat):
    """
    Convert upper triangular square sub-matrix to column vector.
    
    """
    n = mat.shape[0]

    vec = nc.zeros((int(n * (n + 1) / 2)), prototype=mat)

    k = 0
    for i in range(n):
        for j in range(i, n):
            vec[k] = mat[i, j]
            k += 1

    return vec


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
        self.zi = rep_mat(signal.lfilter_zi(filter_num, filter_den), 1, init_val.size)

        self.time_step = init_time
        self.sample_time = sample_time
        self.buffer = rep_mat(init_val, 1, buffer_size)

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
