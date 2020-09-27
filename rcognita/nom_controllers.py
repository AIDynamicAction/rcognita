# scipy
from scipy.optimize import minimize

# numpy
import numpy as np

# rcognita
from . import utilities

class NominalController(utilities.Generic):
    """
    This is a class of nominal controllers used for benchmarking of optimal controllers.
    Specification should be provided for each individual case (system)

    The controller is sampled.

    For a three-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_])

    Parameters
    ----------
    m : int
        * mass of robot

    I : int
        * Inertia around vertical axis of the robot

    ctrl_gain : int
        * Controller gain

    t0 : int
        * Initial value of the controller's internal clock

    sample_time : int or float
        * Controller's sampling time (in seconds)

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(self,
                 t0=0,
                 m=10,
                 I=1,
                 ctrl_gain=10,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 sample_time=0.1):

        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = np.array([[f_min, f_max], [m_min, m_max]])
        self.ctrl_clock = t0
        self.sample_time = sample_time
        self.u_curr = np.zeros(2)
        self.m = m
        self.I = I

    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation

        """
        self.ctrl_clock = t0
        self.u_curr = np.zeros(2)

    def _zeta(self, x_ni, theta):
        """
        utilities.Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)

        """

        sigma_tilde = x_ni[0] * np.cos(theta) + x_ni[1] * \
            np.sin(theta) + np.sqrt(np.abs(x_ni[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4 * x_ni[0]**3 - 2 * \
            np.abs(x_ni[2])**3 * np.cos(theta) / sigma_tilde**3

        nablaF[1] = 4 * x_ni[1]**3 - 2 * \
            np.abs(x_ni[2])**3 * np.sin(theta) / sigma_tilde**3

        nablaF[2] = (3 * x_ni[0] * np.cos(theta) + 3 * x_ni[1] * np.sin(theta) + 2 *
                     np.sqrt(np.abs(x_ni[2]))) * x_ni[2]**2 * np.sign(x_ni[2]) / sigma_tilde**3

        return nablaF

    def _kappa(self, x_ni, theta):
        """
        Stabilizing controller for NI-part

        """
        kappa_val = np.zeros(2)

        G = np.zeros([3, 2])
        G[:, 0] = np.array([1, 0, x_ni[1]])
        G[:, 1] = np.array([0, 1, -x_ni[0]])

        zeta_val = self._zeta(x_ni, theta)

        kappa_val[0] = - np.abs(np.dot(zeta_val, G[:, 0])
                                )**(1 / 3) * np.sign(np.dot(zeta_val, G[:, 0]))
        kappa_val[1] = - np.abs(np.dot(zeta_val, G[:, 1])
                                )**(1 / 3) * np.sign(np.dot(zeta_val, G[:, 1]))

        return kappa_val

    def _Fc(self, x_ni, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation

        """

        sigma_tilde = x_ni[0] * np.cos(theta) + x_ni[1] * \
            np.sin(theta) + np.sqrt(np.abs(x_ni[2]))

        F = x_ni[0]**4 + x_ni[1]**4 + np.abs(x_ni[2])**3 / sigma_tilde

        z = eta - self._kappa(x_ni, theta)

        return F + 1 / 2 * np.dot(z, z)

    def _theta_minimizer(self, x_ni, eta):
        theta_init = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {'maxiter': 50, 'disp': False}

        theta_val = minimize(lambda theta: self._Fc(x_ni, eta, theta), theta_init,
                             method='trust-constr', tol=1e-6, bounds=bnds, options=options).x

        return theta_val

    def _cart_to_nh(self, cart_coords):
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: `\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        x_ni = np.zeros(3)
        eta = np.zeros(2)

        xc = cart_coords[0]
        yc = cart_coords[1]
        alpha = cart_coords[2]
        v = cart_coords[3]
        omega = cart_coords[4]

        x_ni[0] = alpha
        x_ni[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        x_ni[2] = - 2 * (yc * np.cos(alpha) - xc * np.sin(alpha)) - \
            alpha * (xc * np.cos(alpha) + yc * np.sin(alpha))

        eta[0] = omega
        eta[1] = (yc * np.cos(alpha) - xc * np.sin(alpha)) * omega + v

        return [x_ni, eta]

    def _nh_to_cartctrl(self, x_ni, eta, u_ni):
        """
        Get control for Cartesian NI from NH coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: `\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)


        """

        uCart = np.zeros(2)

        uCart[0] = self.m * (u_ni[1] + x_ni[1] * eta[0]**2 +
                             1 / 2 * (x_ni[0] * x_ni[1] * u_ni[0] + u_ni[0] * x_ni[2]))
        uCart[1] = self.I * u_ni[0]

        return uCart

    def compute_action(self, t, y):
        """
        See algorithm description in [[1]_], [[2]_]

        **This algorithm needs full-state measurement of the robot**

        References
        ----------
        .. [1] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
               via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

        .. [2] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sample_time:  # New sample

            # This controller needs full-state measurement
            x_ni, eta = self._cart_to_nh(y)
            theta_star = self._theta_minimizer(x_ni, eta)
            kappa_val = self._kappa(x_ni, theta_star)
            z = eta - kappa_val
            u_ni = - self.ctrl_gain * z
            u = self._nh_to_cartctrl(x_ni, eta, u_ni)

            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0],
                                   self.ctrl_bnds[k, 1])

            self.u_curr = u

            return u

        else:
            return self.u_curr