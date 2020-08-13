from LearnRLSK.RLframe import *
from LearnRLSK.utilities import *
import numpy as np
import warnings
import sys
import argparse


class Simulation:
    """class to create simulation and run simulation."""

    def __init__(self,
                 dim_state=5,
                 dim_input=2,
                 dim_output=5,
                 dimDisturb=2,
                 m=10,
                 I=1,
                 t0=0,
                 t1=100,
                 n_runs=1,
                 a_tol=1e-5,
                 r_tol=1e-3,
                 x_min=-10,
                 x_max=10,
                 y_min=-10,
                 y_max=10,
                 dt=0.05,
                 mod_est_phase=2,
                 model_order=5,
                 prob_noise_pow=8,
                 mod_est_checks=0,
                 f_man=-3,
                 n_man=-1,
                 f_min=-5,
                 f_max=5,
                 m_min=-1,
                 m_max=1,
                 nactor=6,
                 buffer_size=200,
                 r_cost_struct=1,
                 n_critic=50,
                 gamma=1,
                 critic_struct=3,
                 is_log_data=0,
                 is_visualization=1,
                 is_print_sim_step=1,
                 is_disturb=0,
                 is_dyn_ctrl=0,
                 ctrl_mode=5):
        """init."""
        self.dim_state = dim_state
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dimDisturb = dimDisturb
        self.m = m
        self.I = I
        self.sigma_q = 1e-3 * np.ones(dimDisturb)
        self.mu_q = np.zeros(dimDisturb)
        self.tau_q = np.ones(dimDisturb)
        self.t0 = t0
        self.t1 = t1
        self.n_runs = n_runs
        self.x0 = np.zeros(dim_state)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi / 2
        self.u0 = 0 * np.ones(dim_input)
        self.q0 = 0 * np.ones(dimDisturb)
        self.a_tol = a_tol
        self.r_tol = r_tol
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dt = dt
        self.mod_est_phase = mod_est_phase
        self.modEstPeriod = 1 * dt
        self.model_order = model_order
        self.prob_noise_pow = prob_noise_pow
        self.mod_est_checks = mod_est_checks
        self.f_man = f_man
        self.n_man = n_man
        self.uMan = np.array([f_man, n_man])
        self.f_min = f_min
        self.f_max = f_max
        self.m_min = m_min
        self.m_max = m_max
        self.nactor = nactor
        self.predStepSize = 5 * dt
        self.buffer_size = buffer_size
        self.r_cost_struct = r_cost_struct
        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        self.n_critic = n_critic
        self.gamma = gamma
        self.critic_period = 5 * dt
        self.critic_struct = critic_struct
        self.is_log_data = is_log_data
        self.is_visualization = is_visualization
        self.is_print_sim_step = is_print_sim_step
        self.is_disturb = is_disturb
        self.is_dyn_ctrl = is_dyn_ctrl
        self.ctrl_mode = ctrl_mode

    def run_sim(self):
        """run sim."""
        ctrlBnds = np.array([[self.f_min, self.f_max], [self.m_min, self.m_max]])

        # environment
        sys = system(self.dim_state,
                     self.dim_input,
                     self.dim_output,
                     self.dimDisturb,
                     pars=[self.m, self.I],
                     ctrlBnds=ctrlBnds)

        alpha0 = self.x0[2]

        # agent
        myNominalCtrl = nominalController(self.m,
                                          self.I,
                                          ctrlGain=0.5,
                                          ctrlBnds=ctrlBnds,
                                          t0=self.t0,
                                          samplTime=self.dt)

        agent = controller(self.dim_input,
                           self.dim_output,
                           self.ctrl_mode,
                           ctrlBnds=ctrlBnds,
                           t0=self.t0,
                           samplTime=self.dt,
                           Nactor=self.nactor,
                           predStepSize=self.predStepSize,
                           sysRHS=sys._stateDyn,
                           sysOut=sys.out,
                           xSys=self.x0,
                           probNoisePow=self.prob_noise_pow,
                           modEstPhase=self.mod_est_phase,
                           modEstPeriod=self.modEstPeriod,
                           bufferSize=self.buffer_size,
                           modelOrder=self.model_order,
                           modEstChecks=self.mod_est_checks,
                           gamma=self.gamma,
                           Ncritic=self.n_critic,
                           criticPeriod=self.critic_period,
                           criticStruct=self.critic_struct,
                           rcostStruct=self.r_cost_struct,
                           rcostPars=[self.R1, self.R2])

        # simulator
        if self.is_dyn_ctrl:
            ksi0 = np.concatenate([self.x0, self.q0, self.u0])
        else:
            ksi0 = np.concatenate([self.x0, self.q0])

        simulator = sp.integrate.RK45(sys.closedLoop,
                                      self.t0, ksi0, self.t1,
                                      max_step=self.dt / 2,
                                      first_step=1e-6,
                                      atol=self.a_tol,
                                      rtol=self.r_tol)

        # extras
        dataFiles = logdata(self.n_runs, save=self.is_log_data)

        if self.is_print_sim_step:
            warnings.filterwarnings('ignore')

        # main loop

        if self.is_visualization:
            myAnimator = animator(objects=(simulator,
                                           sys,
                                           myNominalCtrl,
                                           agent,
                                           dataFiles,
                                           ctrlSelector,
                                           printSimStep,
                                           logDataRow),
                                  pars=(self.dim_state,
                                        self.x0,
                                        self.u0,
                                        self.t0,
                                        self.t1,
                                        ksi0,
                                        self.x_min,
                                        self.x_max,
                                        self.y_min,
                                        self.y_max,
                                        self.f_min,
                                        self.f_max,
                                        self.m_min,
                                        self.m_max,
                                        self.ctrl_mode,
                                        self.uMan,
                                        self.n_runs,
                                        self.is_print_sim_step,
                                        self.is_log_data))

            anm = animation.FuncAnimation(myAnimator.simFig,
                                          myAnimator.animate,
                                          init_func=myAnimator.initAnim,
                                          blit=False,
                                          interval=self.dt / 1e6,
                                          repeat=True)

            anm.running = True
            myAnimator.simFig.canvas.mpl_connect('key_press_event',
                                                 lambda event: onKeyPress(event, anm))
            myAnimator.simFig.tight_layout()
            plt.show()

        else:
            currRun = 1
            dataFile = dataFiles[0]

            while True:
                simulator.step()
                t = simulator.t
                ksi = simulator.y
                x = ksi[0:dim_state]
                y = sys.out(x)
                u = ctrlSelector(
                    t, y, self.uMan, myNominalCtrl, agent, ctrl_mode)
                sys.receiveAction(u)
                agent.receiveSysState(sys._x)
                agent.update_icost(y, u)
                xCoord = ksi[0]
                yCoord = ksi[1]
                alpha = ksi[2]
                v = ksi[3]
                omega = ksi[4]
                icost = agent.icostVal

                if self.is_print_sim_step:
                    printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u)

                if self.is_log_data:
                    logDataRow(dataFile, t, xCoord, yCoord,
                               alpha, v, omega, icost, u)

                if t >= self.t1:
                    if is_print_sim_step:
                        print('.....................................Run {run:2d} done.....................................'.format(
                            run=currRun))
                    currRun += 1
                    if currRun > n_runs:
                        break

                    if self.is_log_data:
                        dataFile = dataFiles[currRun - 1]

                    # Reset simulator
                    simulator.status = 'running'
                    simulator.t = t0
                    simulator.y = ksi0

                    if ctrl_mode > 0:
                        agent.reset(self.t0)
                    else:
                        myNominalCtrl.reset(self.t0)
                    icost = 0


def main(args=None):
    """main."""
    if args is not None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-dim_state', type=int, default=5, help="description")
        parser.add_argument('-dim_input', type=int, default=2, help="description")
        parser.add_argument('-dim_output', type=int, default=5, help="description")
        parser.add_argument('-dim_disturb', type=int, default=2, help="description")
        parser.add_argument('-m', type=int, default=10, help="description")
        parser.add_argument('-I', dest="I", type=int, default=1, help="description")
        parser.add_argument('-t0', type=int, default=0, help="description")
        parser.add_argument('-t1', type=int, default=100, help="description")
        parser.add_argument('-n_runs', type=int, default=1, help="description")
        parser.add_argument('-a_tol', type=float, default=1e-5, help="description")
        parser.add_argument('-r_tol', type=float, default=1e-3, help="description")
        parser.add_argument('-x_min', type=int, default=-10, help="description")
        parser.add_argument('-x_max', type=int, default=10, help="description")
        parser.add_argument('-y_min', type=int, default=-10, help="description")
        parser.add_argument('-y_max', type=int, default=10, help="description")
        parser.add_argument('-dt', type=float, default=0.05, help="description")
        parser.add_argument('-mod_est_phase', type=int, default=2, help="description")
        parser.add_argument('-model_order', type=int, default=5, help="description")
        parser.add_argument('-prob_noise_pow', type=int, default=8, help="description")
        parser.add_argument('-mod_est_checks', type=int, default=0, help="description")
        parser.add_argument('-f_man', type=int, default=-3, help="description")
        parser.add_argument('-n_man', type=int, default=-1, help="description")
        parser.add_argument('-f_min', type=int, default=5, help="description")
        parser.add_argument('-f_max', type=int, default=5, help="description")
        parser.add_argument('-m_min', type=int, default=-1, help="description")
        parser.add_argument('-m_max', type=int, default=1, help="description")
        parser.add_argument('-nactor', type=int, default=6, help="description")
        parser.add_argument('-buffer_size', type=int, default=200, help="description")
        parser.add_argument('-r_cost_struct', type=int, default=1, help="description")
        parser.add_argument('-n_critic', type=int, default=50, help="description")
        parser.add_argument('-gamma', type=int, default=1, help="description")
        parser.add_argument('-critic_struct', type=int, default=3, help="description")
        parser.add_argument('-is_log_data', type=int, default=0, help="description")
        parser.add_argument('-is_visualization', type=int, default=1, help="description")
        parser.add_argument('-is_print_sim_step', type=int, default=1, help="description")
        parser.add_argument('-is_disturb', type=int, default=0, help="description")
        parser.add_argument('-is_dyn_ctrl', type=int, default=0, help="description")
        parser.add_argument('-ctrl_mode', type=int, default=5, help="description")

        args = parser.parse_args()

        dim_state=args.dim_state,
        dim_input=args.dim_input,
        dim_output=args.dim_output,
        dim_disturb=args.dim_disturb,
        m=args.m,
        I=args.I,
        t0=args.t0,
        t1=args.t1,
        n_runs=args.n_runs,
        a_tol=args.a_tol,
        r_tol=args.r_tol,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        dt=args.dt,
        mod_est_phase=args.mod_est_phase,
        model_order=args.model_order,
        prob_noise_pow=args.prob_noise_pow,
        mod_est_checks=args.mod_est_checks,
        f_man=args.f_man,
        n_man=args.n_man,
        f_min=args.f_min,
        f_max=args.f_max,
        m_min=args.m_min,
        m_max=args.m_max,
        nactor=args.nactor,
        buffer_size=args.buffer_size,
        r_cost_struct=args.r_cost_struct,
        n_critic=args.n_critic,
        gamma=args.gamma,
        critic_struct=args.critic_struct,
        is_log_data=args.is_log_data,
        is_visualization=args.is_visualization,
        is_print_sim_step=args.is_print_sim_step,
        is_disturb=args.is_disturb,
        is_dyn_ctrl=args.is_dyn_ctrl,
        ctrl_mode=args.ctrl_mode

        sim = Simulation(dim_state, dim_input, dim_output, dim_disturb, m, I, t0, t1, n_runs, a_tol, r_tol, x_min, x_max, y_min, y_max, dt, mod_est_phase, model_order, prob_noise_pow, mod_est_checks, f_man, n_man, f_min, f_max, m_min, m_max, nactor, buffer_size, r_cost_struct, n_critic, gamma, critic_struct, is_log_data, is_visualization, is_print_sim_step, is_disturb, is_dyn_ctrl, ctrl_mode)

    else:
        sim = Simulation()

    sim.run_sim()

if __name__ == "__main__":
    command_line_args = sys.argv[1:]
    main(command_line_args)
