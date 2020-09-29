import warnings

# scipy
import scipy as sp, signal
from scipy.optimize import minimize, basinhopping

# numpy
import numpy as np
from numpy.random import rand, randn
from numpy.matlib import repmat
import sippy  # Github:CPCLAB-UNIPI/SIPPY

# rcognita
from . import utilities
from .base_classes_controllers import EndiControllerBase

class ActorCritic(EndiControllerBase, utilities.Generic):
    """
    Optimal controller (a.k.a. agent) class. Actor-Critic model.

    ----------
    Parameters
    ----------

    system : object of type `System` class
        object of type System (class)

    t0 : int
        * Initial value of the controller's internal clock

    t1 : int
        * End value of controller's internal clock

    actor_control_horizon : int
        Number of prediction steps. actor_control_horizon=1 means the controller is purely data-driven and doesn't use prediction.

    buffer_size : int
        The size of the buffer to store data for model estimation. The bigger the buffer, the more accurate the estimation may be achieved. Using a larger buffer results in better model estimation at the expense of computational cost.

    ctrl_mode : int
        Modes with online model estimation are experimental
        * 0 : manual constant control (only for basic testing)
        * -1 : nominal parking controller (for benchmarking optimal controllers)
        * 1 : model-predictive control (MPC). Prediction via discretized true model
        * 2 : adaptive MPC. Prediction via estimated model
        * 3 : RL: Q-learning with buffer_size roll-outs of running cost. Prediction via discretized true model
        * 4 : RL: Q-learning with buffer_size roll-outs of running cost. Prediction via estimated model
        * 5 : RL: stacked Q-learning. Prediction via discretized true model
        * 6 : RL: stacked Q-learning. Prediction via estimated model

        * Modes 1, 3, 5 use model for prediction, passed into class exogenously. This could be, for instance, a true system model
        * Modes 2, 4, 6 use an estimated online

    critic_mode : int
        Choice of the structure of the critic's feature vector
        * 1 - Quadratic-linear
        * 2 - Quadratic
        * 3 - Quadratic, no mixed terms
        * 4 - Quadratic, no mixed terms in input and output

    critic_update_time : float
        * Time between critic updates

    r_cost_struct : int
        * Choice of the running cost structure. A typical choice is quadratic of the form [y, u].T * R1 [y, u], where R1 is the (usually diagonal) parameter matrix. For different structures, R2 is also used.
        * 1 - quadratic chi.T @ R1 @ chi
        * 2 - 4th order chi**2.T @ R2 @ chi**2 + chi.T @ R2 @ chi

    sample_time : int or float
        Controller's sampling time (in seconds). The system itself is continuous as a physical process while the controller is digital.
        * the higher the sampling time, the more chattering in the control might occur. It even may lead to instability and failure to park the robot
        * smaller sampling times lead to higher computation times
        * especially controllers that use the estimated model are sensitive to sampling time, because inaccuracies in estimation lead to problems when propagated over longer periods of time. Experiment with sample_time and try achieve a trade-off between stability and computational performance

    step_size : float
        * Prediction step size in `J` (in seconds). Is the time between the computation of control inputs and outputs J. Should be a multiple of `sample_time`.

    estimator_buffer_fill : int
        * Initial phase to fill the estimator's buffer before applying optimal control (in seconds)

    estimator_buffer_power : int
        * Power of probing noise during an initial phase to fill the estimator's buffer before applying optimal control

    estimator_update_time : float
        * In seconds, the time between model estimate updates. This constant determines how often the estimated parameters are updated. The more often the model is updated, the higher the computational burden is. On the other hand, more frequent updates help keep the model actual.

    stacked_model_params : int
        * Estimated model parameters can be stored in stacks and the best among the `stacked_model_params` last ones is picked.
        * May improve the prediction quality somewhat

    model_order : int
        * The order of the state-space estimation model. We are interested in adequate predictions of y under given u's. The higher the model order, the better estimation results may be achieved, but be aware of overfitting.

    gamma : float
        * Discounting factor
        * number in (0, 1]
        * Characterizes fading of running costs along horizon

    ----------
    Attributes
    ----------

    A, B, C, D : float vectors
        * state-space model parameters (vectors)

    my_model : object of type `_model` class

    R1, R2 : float vectors
        * running cost parameters

    u_min, u_max : float vectors
        * denoting the min and max control action values

    u_buffer : float vector
        * buffer of previous controls

    y_buffer : float vector
        * buffer of previous outputs

    ----------
    References
    ----------
    .. [1] Osinenko, Pavel, et al. "Stacked adaptive dynamic programming with unknown system model." IFAC-PapersOnLine 50.1 (2017): 4150-4155

    """

    def __init__(self,
                 system,
                 t0=0,
                 t1=15,
                 buffer_size=10,
                 r_cost_struct=1,
                 sample_time=0.2,
                 gamma=1,
                 actor_control_horizon=10,
                 ctrl_mode=3,
                 critic_mode=3,
                 critic_update_time=0.1,
                 step_size=0.3,
                 estimator_update_time=0.1,
                 estimator_buffer_fill=6,
                 estimator_buffer_power=2,
                 stacked_model_params=0,
                 model_order=3):

        super(ActorCritic, self).__init__(system, t0, t1, buffer_size, r_cost_struct, sample_time, step_size, gamma)

        """

        CONTROLLER-RELATED ATTRIBUTES

        """
        self.is_prob_noise = 0
        self.estimator_buffer_power = estimator_buffer_power
        self.estimator_buffer_fill = estimator_buffer_fill
        self.estimator_update_time = estimator_update_time
        self.stacked_model_params = stacked_model_params
        self.model_order = model_order

        # model params
        A = np.zeros([self.model_order, self.model_order])
        B = np.zeros([self.model_order, self.dim_input])
        C = np.zeros([self.dim_output, self.model_order])
        D = np.zeros([self.dim_output, self.dim_input])
        x0_est = np.zeros(self.model_order)

        self.my_model = utilities._model(A, B, C, D, x0_est)
        self.model_stack = []

        for k in range(self.stacked_model_params):
            self.model_stack.append(self.my_model)

        # number of prediction steps
        self.actor_control_horizon = actor_control_horizon

        # time between critic updates
        self.critic_update_time = critic_update_time

        # integrated cost
        self.critic_mode = critic_mode
        self.critic_clock = self.t0

        # control mode
        self.ctrl_mode = ctrl_mode

        self.u_min = np.tile(self.min_bounds, actor_control_horizon)
        self.u_max = np.tile(self.max_bounds, actor_control_horizon)


        # critic weights conditional logic
        if self.critic_mode == 1:
            self.num_critic_weights = int(((self.dim_output + self.dim_input) + 1) * (
                self.dim_output + self.dim_input) / 2 + (self.dim_output + self.dim_input))

            self.w_min = -1e3 * np.ones(self.num_critic_weights)
            self.w_max = 1e3 * np.ones(self.num_critic_weights)

        elif self.critic_mode == 2:
            self.num_critic_weights = int(((self.dim_output + self.dim_input) + 1) * (self.dim_output + self.dim_input) / 2)

            self.w_min = np.zeros(self.num_critic_weights)
            self.w_max = 1e3 * np.ones(self.num_critic_weights)

        elif self.critic_mode == 3:
            self.num_critic_weights = int(self.dim_output + self.dim_input)
            self.w_min = np.zeros(self.num_critic_weights)
            self.w_max = 1e3 * np.ones(self.num_critic_weights)

        elif self.critic_mode == 4:
            self.num_critic_weights = int((self.dim_output * 2) + (self.dim_input * 2))

            self.w_min = -1e3 * np.ones(self.num_critic_weights)
            self.w_max = 1e3 * np.ones(self.num_critic_weights)

        self.W_prev = np.ones(int(self.num_critic_weights))

    def _estimate_model(self, t, y):
        """
        Estimate model parameters by accumulating data buffers `u_buffer` and `y_buffer`

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample
            # Update buffers when using RL or requiring estimated model
            if self.ctrl_mode in (2, 3, 4, 5, 6):
                time_in_est_period = t - self.est_clock

                # Estimate model if required by ctrlStatMode
                if (time_in_est_period >= estimator_update_time) and (self.ctrl_mode in (2, 4, 6)):
                    # Update model estimator's internal clock
                    self.est_clock = t

                    try:
                        SSest = sippy.system_identification(self.y_buffer,
                                                            self.u_buffer,
                                                            id_method='N4SID',
                                                            SS_fixed_order=self.model_order,
                                                            SS_D_required=False,
                                                            SS_A_stability=False,
                                                            SS_PK_B_reval=False,
                                                            tsample=self.sample_time)

                        self.my_model.updatePars(SSest.A, SSest.B, SSest.C, SSest.D)

                    except:
                        print('Model estimation problem')
                        self.my_model.updatePars(np.zeros([self.model_order, self.model_order]),
                                                 np.zeros(
                                                     [self.model_order, self.dim_input]),
                                                 np.zeros(
                                                     [self.dim_output, self.model_order]),
                                                 np.zeros([self.dim_output, self.dim_input]))

                    # Model checks
                    if self.stacked_model_params > 0:
                        # Update estimated model parameter stacks
                        self.model_stack.pop(0)
                        self.model_stack.append(self.model)

                        # Perform check of stack of models and pick the best
                        totAbsErrCurr = 1e8
                        for k in range(self.stacked_model_params):
                            A, B, C, D = self.model_stack[k].A, self.model_stack[k].B, self.model_stack[k].C, self.model_stack[k].D
                            x0_est, _, _, _ = np.linalg.lstsq(C, y)
                            y_est, _ = self._dss_sim(
                                A, B, C, D, self.u_buffer, x0_est, y)
                            meanErr = np.mean(y_est - self.y_buffer, axis=0)

                            totAbsErr = np.sum(np.abs(meanErr))
                            if totAbsErr <= totAbsErrCurr:
                                totAbsErrCurr = totAbsErr
                                self.my_model.updatePars(
                                    SSest.A, SSest.B, SSest.C, SSest.D)

            # Update initial state estimate
            x0_est, _, _, _ = np.linalg.lstsq(self.my_model.C, y)
            self.my_model.updateIC(x0_est)

            if t >= self.estimator_buffer_fill:
                    # Drop probing noise
                self.is_prob_noise = 0

    def _actor(self, U, y_obs, N, W, step_size, ctrl_mode):
        """
        This method normally should not be altered. The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of actor
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        actor_opt_method = 'SLSQP'
        
        if actor_opt_method == 'trust-constr':
            actor_opt_options = {'maxiter': 300, 'disp': False}
        
        else:
            actor_opt_options = {'maxiter': 300, 'disp': False, 'ftol': 1e-7}

        isGlobOpt = 0

        bnds = sp.optimize.Bounds(self.u_min, self.u_max, keep_feasible=True)

        u_horizon = np.tile(U, self.actor_control_horizon)

        try:
            if isGlobOpt:
                minimizer_kwargs = {
                    'method': actor_opt_method,
                    'bounds': bnds,
                    'tol': 1e-7,
                    'options': actor_opt_options
                }

                U = basinhopping(lambda U: self._actor_cost(U, y_obs, N, W, step_size, ctrl_mode),
                                 u_horizon,
                                 minimizer_kwargs=minimizer_kwargs,
                                 niter=10).x

            else:
                warnings.filterwarnings('ignore')
                
                U = minimize(lambda U: self._actor_cost(U, y_obs, N, W, step_size, ctrl_mode),
                             u_horizon,
                             method=actor_opt_method,
                             tol=1e-7,
                             bounds=bnds,
                             options=actor_opt_options).x


        except ValueError:
            print("Actor's optimizer failed. Returning default action")
            U = u_horizon

        return U[:self.dim_input] # Return first action

    def _actor_cost(self, U, y_obs, N, W, step_size, ctrl_mode):
        U_2d = np.reshape(U, (N, self.dim_input))
        Y = np.zeros([N, self.dim_output])

        # populate predictions over actor horizon
        if ctrl_mode in (1,3,5):
            Y[0, :] = y_obs
            x = self.system_state

            for k in range(1, self.actor_control_horizon):
                x = x + step_size * self.sys_dynamics([], x, U_2d[k - 1, :], [], self.m, self.I, self.dim_state, self.is_disturb)

                Y[k, :] = self.sys_output(x)

        elif ctrl_mode in (2,4,6):
            myU_upsampled = U_2d.repeat(int(step_size / self.sample_time), axis=0)
            
            Yupsampled, _ = self._dss_sim(self.my_model.A, self.my_model.B, self.my_model.C, self.my_model.D, myU_upsampled, self.my_model.x0_est, y_obs)
            
            Y = Yupsampled[::int(step_size / self.sample_time)]

        # compute cost-to-go for k steps in Y

        J = 0

        if ctrl_mode in (1, 2):
            for k in range(N):
                J += self.gamma**k * self.running_cost(Y[k, :], U_2d[k, :])

        elif ctrl_mode in (3, 4, 5, 6):
            for k in range(N - 1):
                J += self.gamma**k * self.running_cost(Y[k, :], U_2d[k, :])
            
            J += W @ self._phi(Y[-1, :], U_2d[-1, :])

            if ctrl_mode in (4,5):
                    J /= N

        return J

    def _critic(self, W, u_buffer, y_buffer):
        """ Critic

        Customization
        -------------

        This method normally should not be altered, adjust `controller._critic_cost` instead.
        The only customization you might want here is regarding the optimization algorithm

        """

        # Optimization method of critic
        # Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        critic_opt_method = 'SLSQP'
        
        if critic_opt_method == 'trust-constr':
            critic_opt_options = {'maxiter': 200, 'disp': False}
        
        else:
            warnings.filterwarnings('ignore')
            critic_opt_options = {'maxiter': 200, 'disp': False, 'ftol': 1e-7}

        bnds = sp.optimize.Bounds(self.w_min, self.w_max, keep_feasible=True)

        W_new = minimize(lambda W: self._critic_cost(W, u_buffer, y_buffer), W,
                     method=critic_opt_method,
                     tol=1e-7,
                     bounds=bnds,
                     options=critic_opt_options).x

        return W_new

    def _critic_cost(self, W, u_buffer, y_buffer):
        """ Cost function of the critic

        Currently uses value-iteration

        """
        Jc = 0

        for k in range(1, self.buffer_size-1):
            y_prev = y_buffer[k - 1, :]
            y_next = y_buffer[k, :]
            u_prev = u_buffer[k - 1, :]
            u_next = u_buffer[k, :]

            # Temporal difference
            e = self.running_cost(y_prev, u_prev) + (self.gamma * W @ self._phi(y_next, u_next)) - (W @ self._phi(y_prev, u_prev))

            Jc += (1 / 2) * e**2

        return Jc

    def compute_action(self, t, y_obs):
        """ Main method. """
        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample
            # Update controller's internal clock
            self.ctrl_clock = t

            if self.ctrl_mode in (1, 2):

                # Apply control when model estimation phase is over
                if self.is_prob_noise and (self.ctrl_mode == 2):
                    return self.estimator_buffer_power * (rand(self.dim_input) - 0.5)

                elif not self.is_prob_noise and (self.ctrl_mode == 2):
                    u = self._actor(self.u_curr, y_obs, self.actor_control_horizon, [], self.step_size, self.ctrl_mode)

                elif (self.ctrl_mode == 1):
                    u = self._actor(self.u_curr, y_obs, self.actor_control_horizon, [], self.step_size, self.ctrl_mode)

            elif self.ctrl_mode in (3, 4, 5, 6):
                time_in_critic_update_time = t - self.critic_clock

                # Update data buffers
                self.u_buffer = np.vstack([self.u_buffer, self.u_curr])[-self.buffer_size:, :]
                self.y_buffer = np.vstack([self.y_buffer, y_obs])[-self.buffer_size:, :]

                # Critic: minimize critic cost and return new weights
                if time_in_critic_update_time >= self.critic_update_time:
                    self.critic_clock = t

                    W_new = self._critic(self.W_prev, self.u_buffer, self.y_buffer)

                else:
                    W_new = self.W_prev

                # Actor: Apply control by minimizing actor cost
                if self.is_prob_noise and (self.ctrl_mode in (4, 6)):
                    u = self.estimator_buffer_power * (rand(self.dim_input) - 0.5)
                
                elif not self.is_prob_noise and (self.ctrl_mode in (4, 6)):
                    u = self._actor(self.u_curr, y_obs, self.actor_control_horizon, W_new, self.step_size, self.ctrl_mode)

                elif self.ctrl_mode in (3, 5):
                    u = self._actor(self.u_curr, y_obs, self.actor_control_horizon, W_new, self.step_size, self.ctrl_mode)

            self.u_curr = u
            self.W_prev = W_new

            return u

        else:
            return self.u_curr

    def _dss_sim(self, A, B, C, D, uSqn, x0, y0):
        """
        Simulate output response of a discrete-time state-space model
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

    def _phi(self, y, u=None):
        """
        Feature vector of critic

        In Q-learning mode, it uses both `y` and `u`. In value function approximation mode, it should use just `y`

        Customization
        -------------

        If you decide to switch to a non-linearly parametrized approximator, you need to alter the terms like `W @ self._phi( y, u )`
        within `controller._critic_cost`

        """

        if u is not None:
            chi = np.concatenate([y, u])
        else:
            chi = y

        if self.critic_mode == 1:
            return np.concatenate([_uptria2vec(np.kron(chi, chi)), chi])

        elif self.critic_mode == 2:
            return np.concatenate([_uptria2vec(np.kron(chi, chi))])

        elif self.critic_mode == 3:
            return chi * chi

        elif self.critic_mode == 4:
            if u is not None:
                return np.concatenate([y**2, np.kron(y, u), u**2])
            else:
                return np.concatenate([chi, np.kron(chi, chi), chi**2])
