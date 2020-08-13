from LearnRLSK.RLframe import *
from LearnRLSK.utilities import *
import numpy as np
import warnings
import sys


class Simulation:
    """class to create simulation and run simulation."""

    def __init__(self,
                 dimState=5,
                 dimInput=2,
                 dimOutput=5,
                 dimDisturb=2,
                 m=10,
                 I=1,
                 t0=0,
                 t1=100,
                 Nruns=1,
                 atol=1e-5,
                 rtol=1e-3,
                 xMin=-10,
                 xMax=10,
                 yMin=-10,
                 yMax=10,
                 dt=0.05,
                 modEstPhase=2,
                 modelOrder=5,
                 probNoisePow=8,
                 modEstChecks=0,
                 Fman=-3,
                 Nman=-1,
                 Fmin=-5,
                 Fmax=5,
                 Mmin=-1,
                 Mmax=1,
                 Nactor=6,
                 bufferSize=200,
                 rcostStruct=1,
                 Ncritic=50,
                 gamma=1,
                 criticStruct=3,
                 isLogData=0,
                 isVisualization=1,
                 isPrintSimStep=1,
                 isDisturb=0,
                 isDynCtrl=0,
                 ctrlMode=5):
        """init."""
        self.dimState = dimState
        self.dimInput = dimInput
        self.dimOutput = dimOutput
        self.dimDisturb = dimDisturb
        self.m = m
        self.I = I
        self.sigma_q = 1e-3 * np.ones(dimDisturb)
        self.mu_q = np.zeros(dimDisturb)
        self.tau_q = np.ones(dimDisturb)
        self.t0 = t0
        self.t1 = t1
        self.Nruns = Nruns
        self.x0 = np.zeros(dimState)
        self.x0[0] = 5
        self.x0[1] = 5
        self.x0[2] = np.pi / 2
        self.u0 = 0 * np.ones(dimInput)
        self.q0 = 0 * np.ones(dimDisturb)
        self.atol = atol
        self.rtol = rtol
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.dt = dt
        self.modEstPhase = modEstPhase
        self.modEstPeriod = 1 * dt
        self.modelOrder = modelOrder
        self.probNoisePow = probNoisePow
        self.modEstChecks = modEstChecks
        self.Fman = Fman
        self.Nman = Nman
        self.uMan = np.array([Fman, Nman])
        self.Fmin = Fmin
        self.Fmax = Fmax
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.Nactor = Nactor
        self.predStepSize = 5 * dt
        self.bufferSize = bufferSize
        self.rcostStruct = rcostStruct
        self.R1 = np.diag([10, 10, 1, 0, 0, 0, 0])
        self.R2 = np.array([[10, 2, 1, 0, 0],
                            [0, 10, 2, 0, 0],
                            [0, 0, 10, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
        self.Ncritic = Ncritic
        self.gamma = gamma
        self.criticPeriod = 5 * dt
        self.criticStruct = criticStruct
        self.isLogData = isLogData
        self.isVisualization = isVisualization
        self.isPrintSimStep = isPrintSimStep
        self.isDisturb = isDisturb
        self.isDynCtrl = isDynCtrl
        self.ctrlMode = ctrlMode


    def run_sim(self):
        """run sim."""
        ctrlBnds = np.array([[self.Fmin, self.Fmax], [self.Mmin, self.Mmax]])

        # environment
        sys = system(self.dimState,
                     self.dimInput,
                     self.dimOutput,
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

        agent = controller(self.dimInput,
                           self.dimOutput,
                           self.ctrlMode,
                           ctrlBnds=ctrlBnds,
                           t0=self.t0,
                           samplTime=self.dt,
                           Nactor=self.Nactor,
                           predStepSize=self.predStepSize,
                           sysRHS=sys._stateDyn,
                           sysOut=sys.out,
                           xSys=self.x0,
                           probNoisePow=self.probNoisePow,
                           modEstPhase=self.modEstPhase,
                           modEstPeriod=self.modEstPeriod,
                           bufferSize=self.bufferSize,
                           modelOrder=self.modelOrder,
                           modEstChecks=self.modEstChecks,
                           gamma=self.gamma,
                           Ncritic=self.Ncritic,
                           criticPeriod=self.criticPeriod,
                           criticStruct=self.criticStruct,
                           rcostStruct=self.rcostStruct,
                           rcostPars=[self.R1, self.R2])

        # simulator
        if self.isDynCtrl:
            ksi0 = np.concatenate([self.x0, self.q0, self.u0])
        else:
            ksi0 = np.concatenate([self.x0, self.q0])

        simulator = sp.integrate.RK45(sys.closedLoop,
                                      self.t0, ksi0, self.t1,
                                      max_step=self.dt / 2,
                                      first_step=1e-6,
                                      atol=self.atol,
                                      rtol=self.rtol)

        # extras
        dataFiles = logdata(self.Nruns, save=self.isLogData)

        if self.isPrintSimStep:
            warnings.filterwarnings('ignore')


        # main loop

        if self.isVisualization:
            myAnimator = animator(objects=(simulator,
                                  sys,
                                  myNominalCtrl,
                                  agent,
                                  dataFiles,
                                  ctrlSelector,
                                  printSimStep,
                                  logDataRow),
                                  pars=(self.dimState,
                                        self.x0,
                                        self.u0,
                                        self.t0,
                                        self.t1,
                                        ksi0,
                                        self.xMin,
                                        self.xMax,
                                        self.yMin,
                                        self.yMax,
                                        self.Fmin,
                                        self.Fmax,
                                        self.Mmin,
                                        self.Mmax,
                                        self.ctrlMode,
                                        self.uMan,
                                        self.Nruns,
                                        self.isPrintSimStep,
                                        self.isLogData))

            anm = animation.FuncAnimation(myAnimator.simFig, 
                                          myAnimator.animate, 
                                          init_func = myAnimator.initAnim, 
                                          blit=False, 
                                          interval=self.dt/1e6, 
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
                x = ksi[0:dimState]
                y = sys.out(x)
                u = ctrlSelector(t, y, self.uMan, myNominalCtrl, agent, ctrlMode)
                sys.receiveAction(u)
                agent.receiveSysState(sys._x)
                agent.update_icost(y, u)
                xCoord = ksi[0]
                yCoord = ksi[1]
                alpha = ksi[2]
                v = ksi[3]
                omega = ksi[4]
                icost = agent.icostVal

                if self.isPrintSimStep:
                    printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u)

                if self.isLogData:
                    logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u)

                if t >= self.t1:  
                    if isPrintSimStep:
                        print('.....................................Run {run:2d} done.....................................'.format(run = currRun))
                    currRun += 1
                    if currRun > Nruns:
                        break

                    if self.isLogData:
                        dataFile = dataFiles[currRun-1]

                    # Reset simulator
                    simulator.status = 'running'
                    simulator.t = t0
                    simulator.y = ksi0

                    if ctrlMode > 0:
                        agent.reset(self.t0)
                    else:
                        myNominalCtrl.reset(self.t0)
                    icost = 0

def main(args=None):
    sim = Simulation()
    sim.run_sim()

if __name__ == "__main__":
    command_line_args = sys.argv[1:]

    dimState = 5,
    dimInput = 2,
    dimOutput = 5,
    dimDisturb = 2,
    m = 10,
    t0 = 0,
    t1 = 100,
    Nruns = 1,
    atol = 1e-5,
    rtol = 1e-3,
    xMin = -10,
    xMax = 10,
    yMin = -10,
    yMax = 10,
    dt = 0.05,
    modEstPhase = 2,
    modelOrder = 5,
    probNoisePow = 8,
    modEstChecks = 0,
    Fman = -3,
    Nman = -1,
    Fmin = -5,
    Fmax = 5,
    Mmin = -1,
    Mmax = 1,
    Nactor = 6,
    bufferSize = 200,
    rcostStruct = 1,
    Ncritic = 50,
    gamma = 1,
    criticStruct = 3,
    isLogData = 0,
    isVisualization = 1,
    isPrintSimStep = 1,
    isDisturb = 0,
    isDynCtrl = 0,
    ctrlMode = 5

    main(command_line_args)
