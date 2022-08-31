from rcognita.utilities import rc


class TabularScenario:
    def __init__(self, actor, critic, n_iters):
        self.actor = actor
        self.critic = critic
        self.n_iters = n_iters

    def run(self):
        for _ in range(self.n_iters):
            self.actor.update()
            self.critic.update()


class OnlineScenario:
    def __init__(self, simulator, controller, actor, critic, logger, datafiles, t1):
        self.simulator = simulator
        self.controller = controller
        self.actor = actor
        self.critic = critic
        self.logger = logger
        self.t1 = t1
        self.accum_obj_val = 0
        self.datafile = datafiles[0]
        t = t_prev = 0

        while True:

            self.my_simulator.sim_step()

            t_prev = t

            (t, _, observation, state_full,) = self.simulator.get_sim_step_data()

            delta_t = t - t_prev

            if self.save_trajectory:
                self.trajectory.append(rc.concatenate((state_full, t), axis=None))
            if self.control_mode == "nominal":
                action = self.controller.compute_action_sampled(t, observation)
            else:
                action = self.controller.compute_action_sampled(t, observation)

            self.my_sys.receive_action(action)

            running_obj = self.running_objective(observation, action)
            self.upd_accum_obj(observation, action, delta_t)
            accum_obj = self.accum_obj_val

            if not self.no_print:
                self.logger.print_sim_step(
                    t, state_full, action, running_obj, accum_obj
                )

            if self.is_log:
                self.logger.log_data_row(
                    self.datafile, t, state_full, action, running_obj, accum_obj,
                )
