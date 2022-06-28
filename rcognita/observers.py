import numpy as np


class KalmanFilter:
    def __init__(
        self, t0, my_sys, sys_noise_cov, observ_noise_cov, prior_est_cov, state_init,
    ):

        self.my_sys = my_sys

        self.posterior_state_est = state_init
        self.prior_state_est = None
        self.dim_state = self.posterior_state_est.shape[0]
        self.estimation_buffer = []

        self.sys_noise_cov = sys_noise_cov
        self.observ_noise_cov = observ_noise_cov

        self.posterior_est_cov = np.eye(self.dim_state)
        self.prior_est_cov = prior_est_cov

        self.est_clock = t0

    def predict_state(self, action, dt):
        Q = self.sys_noise_cov
        J = self.my_sys.Jacobi_system_matrix(self.posterior_state_est)
        P_posterior_prev = self.posterior_est_cov

        self.prior_state_est = self.posterior_state_est + dt * self.my_sys._state_dyn(
            [], self.posterior_state_est, action
        )

        self.prior_est_cov = J @ P_posterior_prev @ J.T + Q

    def correct_state(self, observation):
        z = np.array(observation)
        J_h = self.my_sys.Jacobi_observation_matrix()
        P_pred = self.prior_est_cov
        R = self.observ_noise_cov

        K = np.array(P_pred @ J_h.T @ np.linalg.inv(J_h @ P_pred @ J_h.T + R))
        self.posterior_state_est = self.prior_state_est + K @ (
            z - self.my_sys.out(self.prior_state_est)
        )
        self.posterior_est_cov = (np.eye(self.dim_state) - K @ J_h) @ P_pred
        print(f"Kalman gain:\n{K}, \n Post est cov: \n{self.posterior_est_cov}")

    def compute_estimate(self, t, observation, action):
        dt = t - self.est_clock
        self.est_clock = t
        print(f"TRACE OF COV EST MATRIX:{self.prior_est_cov.trace()}\n")
        self.predict_state(action, dt)
        self.correct_state(observation)
        self.estimation_buffer.append(self.posterior_state_est)
        return self.posterior_state_est
