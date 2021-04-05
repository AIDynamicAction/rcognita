# class model

# 	def get_next_state()


# class model_SS(model)
# 	class model:

import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod

class model(ABC):
    """
        Class of estimated models
    """
    @abstractmethod
    def upd_pars(self):
        pass

    @abstractmethod
    def updateIC(self): #update initial condition
        pass

class model_SS(model):
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


class model_NN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, logger=None):
        super(model_NN, self).__init__()
        pass


    def forward(self, x):
        pass


    def prediction(self, data):
        pass


    def train_model(self, data, labels, criterion, optimizer, epochs):
        pass


    def save_state(self, filename):
        print("***************Weights saved!**********************")
        torch.save(self.state_dict(), f'model_weights/{filename}.pth')


    def upload_weights(self, filename):
        print("***************Weights uploaded!**********************")
        self.load_state_dict(torch.load(f'model_weights/{filename}'))
        self.eval()
