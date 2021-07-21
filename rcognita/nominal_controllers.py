import numpy as np

class ctrl_ni_nominal_3wrobot:
    """
    Nominal parking controller for NI using disassembled subgradients

    """

    def __init__(self, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.uCurr = np.zeros(2)

    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation

        """
        self.ctrl_clock = t0
        self.uCurr = np.zeros(2)

    def _zeta(self, xNI):
        """
        Analytic disassembled subgradient, without finding minimizer theta

        """

        #                                 3
        #                             |x |
        #         4     4             | 3|
        # L(x) = x  +  x  +  ----------------------------------=   min F(x)
        #         1     2                                        theta
        #                     /     / 2   2 \             \ 2
        #                     | sqrt| x + x   | + sqrt|x | |
        #                     \     \ 1   2 /        | 3| /
        #                        \_________  __________/
        #                                 \/
        #                               sigma
        #                                                3
        #                                            |x |
        #                4     4                     | 3|
        # F(x; theta) = x  +  x  +  ----------------------------------------
        #                1     2
        #                           /                                      \ 2
        #                           | x cos theta + x sin theta + sqrt|x | |
        #                           \ 1             2                | 3|  /
        #                              \_______________  ______________/
        #                                              \/
        #                                             sigma~


        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt(abs(xNI[2]));

        nablaL = np.zeros(3)

        nablaL[0] = 4*xNI[0]**3 + np.abs(xNI[2])**3/sigma**3 * 1/np.sqrt( xNI[0]**2 + xNI[1]**2 )**3 * 2 * xNI[0];
        nablaL[1] = 4*xNI[1]**3 + np.abs(xNI[2])**3/sigma**3 * 1/np.sqrt( xNI[0]**2 + xNI[1]**2 )**3 * 2 * xNI[1];
        nablaL[2] = 3 * np.abs(xNI[2])**2 * np.sign(xNI[2]) + np.abs(xNI[2])**3 / sigma**3 * 1/np.sqrt(np.abs(xNI[2])) * np.sign(xNI[2]);

        theta = 0

        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3
        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3
        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3

        if xNI[0] == 0 and xNI[1] == 0:
            return nablaF
        else:
            return nablaL

    def _kappa(self, xNI):
        """ Stabilizing controller for NI-part """
        kappa_val = np.zeros(2)

        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])

        zeta_val = self._zeta(xNI)

        kappa_val[0] = - np.abs( np.dot( zeta_val, G[:,0] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,0] ) )
        kappa_val[1] = - np.abs( np.dot( zeta_val, G[:,1] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,1] ) )

        return kappa_val

    def _F(self, xNI, eta, theta):
        """ Marginal function for NI """

        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))
        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde**2
        z = eta - self._kappa(xNI, theta)

        return F + 1/2 * np.dot(z, z)

    def _Cart2NH(self, coords_Cart):
        """ Transformation from Cartesian coordinates to non-holonomic (NH) coordinates   """

        xNI = np.zeros(3)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]

        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )

        return xNI

    def _NH2ctrl_Cart(self, xNI, uNI):
        """ Get control for Cartesian NI from NH coordinates """

        uCart = np.zeros(2)
        uCart[0] = uNI[1] + 1/2 * uNI[0] * ( xNI[2] + xNI[0] * xNI[1] )
        uCart[1] = uNI[0]

        return uCart

    def compute_action(self, t, y):
        """

        """

        time_in_sample = t - self.ctrl_clock

        if time_in_sample >= self.sampling_time: # New sample
            # Update internal clock
            self.ctrl_clock = t

            xNI = self._Cart2NH( y )
            kappa_val = self._kappa(xNI)
            uNI = self.ctrl_gain * kappa_val
            u = self._NH2ctrl_Cart(xNI, uNI)

            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])

            self.uCurr = u

            return u

        else:
            return self.uCurr

    def compute_action_vanila(self, y):
        """ Same as :func:`~ctrl_nominal_3wrobot_NI.compute_action`, but without invoking the internal clock """

        xNI = self._Cart2NH( y )
        kappa_val = self._kappa(xNI)
        uNI = self.ctrl_gain * kappa_val
        u = self._NH2ctrl_Cart(xNI, uNI)

        self.uCurr = u

        return u

    def compute_LF(self, y):

        xNI = self._Cart2NH( y )
        sigma = np.sqrt( xNI[0]**2 + xNI[1]**2 ) + np.sqrt( np.abs(xNI[2]) )

        return xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma**2


class ctrl_endi_nominal_3wrobot:
    """
    This is a class of nominal controllers for 3-wheel robots used for benchmarking of other controllers.

    The controller is sampled.

    For a 3-wheel robot with dynamical pushing force and steering torque (a.k.a. ENDI - extended non-holonomic double integrator) [[1]_], we use here
    a controller designed by non-smooth backstepping (read more in [[2]_], [[3]_])

    Attributes
    ----------
    m, I : : numbers
        Mass and moment of inertia around vertical axis of the robot
    ctrl_gain : : number
        Controller gain
    t0 : : number
        Initial value of the controller's internal clock
    sampling_time : : number
        Controller's sampling time (in seconds)

    References
    ----------
    .. [1] W. Abbasi, F. urRehman, and I. Shah. “Backstepping based nonlinear adaptive control for the extended
           nonholonomic double integrator”. In: Kybernetika 53.4 (2017), pp. 578–594

    ..   [2] Matsumoto, R., Nakamura, H., Satoh, Y., and Kimura, S. (2015). Position control of two-wheeled mobile robot
             via semiconcave function backstepping. In 2015 IEEE Conference on Control Applications (CCA), 882–887

    ..   [3] Osinenko, Pavel, Patrick Schmidt, and Stefan Streif. "Nonsmooth stabilization and its computational aspects." arXiv preprint arXiv:2006.14013 (2020)

    """

    def __init__(self, m, I, ctrl_gain=10, ctrl_bnds=[], t0=0, sampling_time=0.1):
        self.m = m
        self.I = I
        self.ctrl_gain = ctrl_gain
        self.ctrl_bnds = ctrl_bnds
        self.ctrl_clock = t0
        self.sampling_time = sampling_time

        self.uCurr = np.zeros(2)

    def reset(self, t0):
        """
        Resets controller for use in multi-episode simulation

        """
        self.ctrl_clock = t0
        self.uCurr = np.zeros(2)

    def _zeta(self, xNI, theta):
        """
        Generic, i.e., theta-dependent, subgradient (disassembled) of a CLF for NI (a.k.a. nonholonomic integrator, a 3wheel robot with static actuators)

        """

        #                                 3
        #                             |x |
        #         4     4             | 3|
        # V(x) = x  +  x  +  ----------------------------------=   min F(x)
        #         1     2                                        theta
        #                     /     / 2   2 \             \ 2
        #                    | sqrt| x + x   | + sqrt|x |  |
        #                     \     \ 1   2 /        | 3| /
        #                        \_________  __________/
        #                                 \/
        #                               sigma
        #                                         3
        #                                     |x |
        #            4     4                     | 3|
        # F(x; theta) = x  +  x  +  ----------------------------------------
        #            1     2
        #                        /                                     \ 2
        #                        | x cos theta + x sin theta + sqrt|x | |
        #                        \ 1             2                | 3| /
        #                           \_______________  ______________/
        #                                            \/
        #                                            sigma~

        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        nablaF = np.zeros(3)

        nablaF[0] = 4*xNI[0]**3 - 2 * np.abs(xNI[2])**3 * np.cos(theta)/sigma_tilde**3

        nablaF[1] = 4*xNI[1]**3 - 2 * np.abs(xNI[2])**3 * np.sin(theta)/sigma_tilde**3

        nablaF[2] = ( 3*xNI[0]*np.cos(theta) + 3*xNI[1]*np.sin(theta) + 2*np.sqrt(np.abs(xNI[2])) ) * xNI[2]**2 * np.sign(xNI[2]) / sigma_tilde**3

        return nablaF

    def _kappa(self, xNI, theta):
        """
        Stabilizing controller for NI-part

        """
        kappa_val = np.zeros(2)

        G = np.zeros([3, 2])
        G[:,0] = np.array([1, 0, xNI[1]])
        G[:,1] = np.array([0, 1, -xNI[0]])

        zeta_val = self._zeta(xNI, theta)

        kappa_val[0] = - np.abs( np.dot( zeta_val, G[:,0] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,0] ) )
        kappa_val[1] = - np.abs( np.dot( zeta_val, G[:,1] ) )**(1/3) * np.sign( np.dot( zeta_val, G[:,1] ) )

        return kappa_val

    def _Fc(self, xNI, eta, theta):
        """
        Marginal function for ENDI constructed by nonsmooth backstepping. See details in the literature mentioned in the class documentation

        """

        sigma_tilde = xNI[0]*np.cos(theta) + xNI[1]*np.sin(theta) + np.sqrt(np.abs(xNI[2]))

        F = xNI[0]**4 + xNI[1]**4 + np.abs( xNI[2] )**3 / sigma_tilde

        z = eta - self._kappa(xNI, theta)

        return F + 1/2 * np.dot(z, z)

    def _minimizer_theta(self, xNI, eta):
        thetaInit = 0

        bnds = sp.optimize.Bounds(-np.pi, np.pi, keep_feasible=False)

        options = {'maxiter': 50, 'disp': False}

        theta_val = minimize(lambda theta: self._Fc(xNI, eta, theta), thetaInit, method='trust-constr', tol=1e-6, bounds=bnds, options=options).x

        return theta_val

    def _Cart2NH(self, coords_Cart):
        """
        Transformation from Cartesian coordinates to non-holonomic (NH) coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)

        """

        xNI = np.zeros(3)
        eta = np.zeros(2)

        xc = coords_Cart[0]
        yc = coords_Cart[1]
        alpha = coords_Cart[2]
        v = coords_Cart[3]
        omega = coords_Cart[4]

        xNI[0] = alpha
        xNI[1] = xc * np.cos(alpha) + yc * np.sin(alpha)
        xNI[2] = - 2 * ( yc * np.cos(alpha) - xc * np.sin(alpha) ) - alpha * ( xc * np.cos(alpha) + yc * np.sin(alpha) )

        eta[0] = omega
        eta[1] = ( yc * np.cos(alpha) - xc * np.sin(alpha) ) * omega + v

        return [xNI, eta]

    def _NH2ctrl_Cart(self, xNI, eta, uNI):
        """
        Get control for Cartesian NI from NH coordinates
        See Section VIII.A in [[1]_]

        The transformation is a bit different since the 3rd NI eqn reads for our case as: :math:`\\dot x_3 = x_2 u_1 - x_1 u_2`

        References
        ----------
        .. [1] Watanabe, K., Yamamoto, T., Izumi, K., & Maeyama, S. (2010, October). Underactuated control for nonholonomic mobile robots by using double
               integrator model and invariant manifold theory. In 2010 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 2862-2867)


        """

        uCart = np.zeros(2)

        uCart[0] = self.m * ( uNI[1] + xNI[1] * eta[0]**2 + 1/2 * ( xNI[0] * xNI[1] * uNI[0] + uNI[0] * xNI[2] ) )
        uCart[1] = self.I * uNI[0]

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

        if time_in_sample >= self.sampling_time: # New sample
            self.ctrl_clock = t
            # This controller needs full-state measurement
            xNI, eta = self._Cart2NH( y )
            theta_star = self._minimizer_theta(xNI, eta)
            kappa_val = self._kappa(xNI, theta_star)
            z = eta - kappa_val
            uNI = - self.ctrl_gain * z
            u = self._NH2ctrl_Cart(xNI, eta, uNI)

            if self.ctrl_bnds.any():
                for k in range(2):
                    u[k] = np.clip(u[k], self.ctrl_bnds[k, 0], self.ctrl_bnds[k, 1])

            self.uCurr = u

            return u

        else:
            return self.uCurr

    def compute_action_vanila(self, y):
        """
        Same as :func:`~ctrl_endi_nominal_3wrobot.compute_action`, but without invoking the internal clock

        """

        xNI, eta = self._Cart2NH( y )
        theta_star = self._minimizer_theta(xNI, eta)
        kappa_val = self._kappa(xNI, theta_star)
        z = eta - kappa_val
        uNI = - self.ctrl_gain * z
        for k in range(len(uNI)):
            uNI[k] = np.clip(uNI[k], -1, 1)

        u = self._NH2ctrl_Cart(xNI, eta, uNI)

        self.uCurr = u

        return u
