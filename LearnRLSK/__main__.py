from Rlframe import *
from utilities import *
import numpy

#%% User settings: parameters

#------------------------------------system
# System
dimState = 5
dimInput = 2
dimOutput = 5
dimDisturb = 2

# System parameters
m = 10 # [kg]
I = 1 # [kg m^2]

# Disturbance
sigma_q = 1e-3 * np.ones(dimDisturb)
mu_q = np.zeros(dimDisturb)
tau_q = np.ones(dimDisturb)

#------------------------------------simulation
t0 = 0
t1 = 100
Nruns = 1

x0 = np.zeros(dimState)
x0[0] = 5
x0[1] = 5
x0[2] = np.pi/2

u0 = 0 * np.ones(dimInput)

q0 = 0 * np.ones(dimDisturb)

# Solver
atol = 1e-5
rtol = 1e-3

# xy-plane
xMin = -10
xMax = 10
yMin = -10
yMax = 10

#------------------------------------digital elements
# Digital elements sampling time
dt = 0.05 # [s], controller sampling time
# sampleFreq = 1/dt # [Hz]

# Parameters
# cutoff = 1 # [Hz]

# Digital differentiator filter order
# diffFiltOrd = 4

#------------------------------------model estimator
modEstPhase = 2 # [s]
modEstPeriod = 1*dt # [s]

modelOrder = 5

probNoisePow = 8

# Model estimator stores models in a stack and recall the best of modEstChecks
modEstChecks = 0

#------------------------------------controller
# u[0]: Pushing force F [N]
# u[1]: Steering torque M [N m]

# Manual control
Fman = -3
Nman = -1
uMan = np.array([Fman, Nman])

# Control constraints
Fmin = -5
Fmax = 5
Mmin = -1
Mmax = 1

# Control horizon length
Nactor = 6

# Should be a multiple of dt
predStepSize = 5*dt # [s]

# Size of data buffers (used, e.g., in model estimation and critic)
bufferSize = 200

#------------------------------------RL elements
# Running cost structure and parameters
# Notation: chi = [y, u]
# 1     - quadratic chi.T R1 chi 
# 2     - 4th order chi**2.T R2 chi**2 + chi.T R2 chi
# R1, R2 must be positive-definite
rcostStruct = 1

R1 = np.diag([10, 10, 1, 0, 0, 0, 0])  # No mixed terms, full-state measurement
# R1 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
# R1 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
# R1 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# R2 = np.diag([10, 10, 1, 0, 0])  # No mixed terms
R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in y
# R2 = np.array([[10, 2, 1, 1, 1], [0, 10, 2, 1, 1], [0, 0, 10, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])  # mixed terms in chi

# Critic stack size, not greater than bufferSize
Ncritic = 50

# Discounting factor
gamma = 1

# Critic is updated every criticPeriod seconds
criticPeriod = 5*dt # [s]

# Critic structure choice
# 1 - quadratic-linear
# 2 - quadratic
# 3 - quadratic, no mixed terms
# 4 - W[0] y[0]^2 + ... W[p-1] y[p-1]^2 + W[p] y[0] u[0] + ... W[...] u[0]^2 + ... 
criticStruct = 3

#%% User settings: main switches

isLogData = 0
isVisualization = 1
isPrintSimStep = 1

isDisturb = 0

# Static or dynamic controller
isDynCtrl = 0

# Control mode
#
#   Modes with online model estimation are experimental
#
# 0     - manual constant control (only for basic testing)
# -1    - nominal parking controller (for benchmarking optimal controllers)
# 1     - model-predictive control (MPC). Prediction via discretized true model
# 2     - adaptive MPC. Prediction via estimated model
# 3     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via discretized true model
# 4     - RL: Q-learning with Ncritic roll-outs of running cost. Prediction via estimated model
# 5     - RL: stacked Q-learning. Prediction via discretized true model
# 6     - RL: stacked Q-learning. Prediction via estimated model
ctrlMode = 5




#%% Initialization

#------------------------------------environment
sys = system(dimState, dimInput, dimOutput, dimDisturb,
             pars=[m, I],
             ctrlBnds=np.array([[Fmin, Fmax], [Mmin, Mmax]]))

y0 = sys.out(x0)

xCoord0 = x0[0]
yCoord0 = x0[1]
alpha0 = x0[2]
alphaDeg0 = alpha0/2/np.pi

#------------------------------------agent
ctrlBnds = np.array([[Fmin, Fmax], [Mmin, Mmax]])

myNominalCtrl = nominalController(m, I, ctrlGain=0.5, ctrlBnds=ctrlBnds, t0=t0, samplTime=dt)

agent = controller(dimInput, dimOutput, ctrlMode, ctrlBnds=ctrlBnds, t0=t0, samplTime=dt, Nactor=Nactor, predStepSize=predStepSize,
                 sysRHS=sys._stateDyn, sysOut=sys.out, xSys=x0,
                 probNoisePow = probNoisePow, modEstPhase=modEstPhase, modEstPeriod=modEstPeriod, bufferSize=bufferSize,
                 modelOrder=modelOrder, modEstChecks=modEstChecks,
                 gamma=gamma, Ncritic=Ncritic, criticPeriod=criticPeriod, criticStruct=criticStruct, rcostStruct=rcostStruct, rcostPars=[R1, R2])

#------------------------------------simulator
if isDynCtrl:
    ksi0 = np.concatenate([x0, q0, u0])
else:
    ksi0 = np.concatenate([x0, q0])

simulator = sp.integrate.RK45(sys.closedLoop, 
                              t0, ksi0, t1, max_step = dt/2, first_step=1e-6, atol=atol, rtol=rtol)

#------------------------------------extras

if isLogData:
	logdata()

# Do not display annoying warnings when print is on
if isPrintSimStep:
    warnings.filterwarnings('ignore')


#%% Main loop

if isVisualization:
    myAnimator = animator(objects=(simulator, sys, myNominalCtrl, agent, dataFiles, ctrlSelector, printSimStep, logDataRow),
                          pars=(x0, u0, t0, t1, ksi0, xMin, xMax, yMin, yMax, ctrlMode, uMan, Nruns, isPrintSimStep, isLogData))
    
    cId = myAnimator.simFig.canvas.mpl_connect('key_press_event', onKeyPress)
    
    anm = animation.FuncAnimation(myAnimator.simFig, myAnimator.animate,
                                  init_func= myAnimator.initAnim,
                                  blit=False, interval=dt/1e6, repeat=False)
    
    anm.running = True
    
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
        
        u = ctrlSelector(t, y, uMan, myNominalCtrl, agent, ctrlMode)
        
        sys.receiveAction(u)
        agent.receiveSysState(sys._x)
        agent.update_icost(y, u)
        
        xCoord = ksi[0]
        yCoord = ksi[1]
        alpha = ksi[2]
        v = ksi[3]
        omega = ksi[4]
        
        r = agent.rcost(y, u)
        icost = agent.icostVal
        
        if isPrintSimStep:
            printSimStep(t, xCoord, yCoord, alpha, v, omega, icost, u)
            
        if isLogData:
            logDataRow(dataFile, t, xCoord, yCoord, alpha, v, omega, icost, u)
        
        if t >= t1:  
            if isPrintSimStep:
                print('.....................................Run {run:2d} done.....................................'.format(run = currRun))
                
            currRun += 1
            
            if currRun > Nruns:
                break
                
            if isLogData:
                dataFile = dataFiles[currRun-1]
            
            # Reset simulator
            simulator.status = 'running'
            simulator.t = t0
            simulator.y = ksi0
            
            if ctrlMode > 0:
                agent.reset(t0)
            else:
                myNominalCtrl.reset(t0)
            
            icost = 0      