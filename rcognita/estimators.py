import numpy as np
from .utilities import dss_sim, push_vec, rc
import warnings

try:
    import sippy
except ModuleNotFoundError:
    warnings.warn_explicit(
        "\nImporting sippy failed. You may still use rcognita, but"
        + " without model identification capability. \nRead on how"
        + " to install sippy at https://github.com/AIDynamicAction/rcognita\n",
        UserWarning,
        __file__,
        33,
    )

import torch
import torch.optim as optim
import torch.nn as nn


import rcognita.models as models


class EstimatorSS:
    def __init__(self, model, model_est_checks, model_order=3):
        self.model_order = model_order
        self.model = model
        self.model_est_checks = model_est_checks
        self.model_stack = []
        for _ in range(self.model_est_checks):
            self.model_stack.append(self.model)

    def _estimate_model(self, observation, t):
        """
        Estimate model parameters by accumulating data buffers ``action_buffer`` and ``observation_buffer``.
        
        """

        # Update buffers when using RL or requiring estimated model

        time_in_est_period = t - self.est_clock

        # Estimate model if required
        if time_in_est_period >= self.model_est_period:
            # Update model estimator's internal clock
            self.est_clock = t

            try:
                # Using Github:CPCLAB-UNIPI/SIPPY
                # method: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S, PARSIM-K
                SSest = sippy.system_identification(
                    self.observation_buffer,
                    self.action_buffer,
                    id_method="N4SID",
                    SS_fixed_order=self.model_order,
                    SS_D_required=False,
                    SS_A_stability=False,
                    # SS_f=int(self.buffer_size/12),
                    # SS_p=int(self.buffer_size/10),
                    SS_PK_B_reval=False,
                    tsample=self.sampling_time,
                )

                self.my_model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)

                # ToDo: train an NN via Torch
                # NN_wgts = NN_train(...)

            except:
                print("Model estimation problem")
                self.my_model.upd_pars(
                    rc.zeros([self.model_order, self.model_order]),
                    rc.zeros([self.model_order, self.dim_input]),
                    rc.zeros([self.dim_output, self.model_order]),
                    rc.zeros([self.dim_output, self.dim_input]),
                )

            # Model checks
            if self.model_est_checks > 0:
                # Update estimated model parameter stacks
                self.model_stack.pop(0)
                self.model_stack.append(self.model)

                # Perform check of stack of models and pick the best
                tot_abs_err_curr = 1e8
                for k in range(self.model_est_checks):
                    A, B, C, D = (
                        self.model_stack[k].A,
                        self.model_stack[k].B,
                        self.model_stack[k].C,
                        self.model_stack[k].D,
                    )
                    x0est, _, _, _ = np.linalg.lstsq(C, observation)
                    Yest, _ = dss_sim(
                        A, B, C, D, self.action_buffer, x0est, observation
                    )
                    mean_err = np.mean(Yest - self.observation_buffer, axis=0)

                    # DEBUG ===================================================================
                    # ================================Interm output of model prediction quality
                    # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                    # dataRow = []
                    # for k in range(dim_output):
                    #     dataRow.append( mean_err[k] )
                    # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                    # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                    # print( table )
                    # /DEBUG ===================================================================

                    tot_abs_err = np.sum(np.abs(mean_err))
                    if tot_abs_err <= tot_abs_err_curr:
                        tot_abs_err_curr = tot_abs_err
                        self.model.upd_pars(SSest.A, SSest.B, SSest.C, SSest.D)

                # DEBUG ===================================================================
                # ==========================================Print quality of the best model
                # R  = '\033[31m'
                # Bl  = '\033[30m'
                # x0est,_,_,_ = np.linalg.lstsq(ctrlStat.C, observation)
                # Yest,_ = dssSim(ctrlStat.A, ctrlStat.B, ctrlStat.C, ctrlStat.D, ctrlStat.action_buffer, x0est, observation)
                # mean_err = np.mean(Yest - ctrlStat.observation_buffer, axis=0)
                # headerRow = ['diff y1', 'diff y2', 'diff y3', 'diff y4', 'diff y5']
                # dataRow = []
                # for k in range(dim_output):
                #     dataRow.append( mean_err[k] )
                # rowFormat = ('8.5f', '8.5f', '8.5f', '8.5f', '8.5f')
                # table = tabulate([headerRow, dataRow], floatfmt=rowFormat, headers='firstrow', tablefmt='grid')
                # print(R+table+Bl)
                # /DEBUG ===================================================================

        # Update initial state estimate
        x0est, _, _, _ = np.linalg.lstsq(self.model.C, observation)
        self.model.updateIC(x0est)

        if t >= self.model_est_stage:
            # Drop probing noise
            self.is_prob_noise = 0


class Estimator_RNN:
    """
    Class of model estimators based on recurrent neural networks
    """

    def __init__(
        self,
        dim_observation,
        dim_action,
        dim_hidden,
        buffer_size,
        model=None,
        Nbackprops=1,
        t0=0,
        sampling_time=0.1,
    ):
        self.buffer_size = buffer_size

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        if model is None:
            self.model = models.ModelRNN(
                None, self.dim_observation, self.dim_action, self.dim_hidden
            )

        else:
            self.model = model

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.observation_buffer = np.zeros((self.buffer_size, self.dim_observation))
        self.action_buffer = np.zeros((self.buffer_size, self.dim_action))

        self.Nbackprops = Nbackprops

    def receive_sys_IO(self, t, observation, action):
        # push observation, action to buffers -- see functionality in controllers.py, line 1463

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time:  # New sample
            self.ctrl_clock = t

            self.observation_buffer = push_vec(self.observation_buffer, observation)
            self.action_buffer = push_vec(self.action_buffer, action)

    def update_params(self):
        """
        Update neural network weights
        """

        # Torch backprop (Nbackprops times, say) on loss = model accuracy over buffers

        self.loss.backward()

    def output_loss(self):
        """
        Return current loss
        """

        self.loss = 0

        for i in range(self.buffer_size - 1):
            # y_pred = self.model.model_out(np.concatenate((self.observation_buffer[i, :], self.action_buffer[i, :])))
            y_pred = self.model.model_out(
                self.observation_buffer[i, :], self.action_buffer[i, :]
            )

            # loss += np.linalg.norm((y_pred.detach().numpy() - self.observation_buffer[i + 1, :]))
            self.loss += self.criterion(
                y_pred, torch.tensor(self.observation_buffer[i + 1, :])
            )

        return self.loss.detach().numpy()
