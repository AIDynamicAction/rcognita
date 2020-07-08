# learnRL-py
Learning reinforcement learning (in Python) 

Stacked Q-learning, model is known, one single episode demo:

<img src="https://github.com/OsinenkoP/learnRL-py/blob/master/data-standard.gif" width="500" />

N-rollout Q-learning, model is learned on-the-fly in one single episode:

<img src="https://github.com/OsinenkoP/learnRL-py/blob/master/data-experimental.gif" width="500" />

## Abstract

This is a framework of **reinforcement learning** with a rich variety of settings, aimed specifically at studying and research.
The playground is based upon model of a mobile robot, referred to as the so called "extended non-holonomic double integrator" (ENDI).
See [these notes](ENDI-notes.pdf) for its description.
A custom model can be easily implemented -- read below.
A flowchart of the overall code can be found in [here](Python-RL-flowchart.pdf).
Basically, an *agent* (referred to also as the *controller*) is attached to the *environment* (the system) and generates *actions* so as to minimize running costs (also called rewards, utility, stage costs etc.) over an infinite horizon in future.
This software package aims at the objective of parking the robot, as an example.
The controller is multi-modal and allows comparison with various baselines (nominal parking controller, model-predictive controller with or without on-the-fly model estimation).

Please see [paper 1](https://www.sciencedirect.com/science/article/pii/S2405896317312570) and [paper 2](https://arxiv.org/abs/2006.14034) for some underlying theory.

Also, see [an analogous package](https://github.com/OsinenkoP/learnRL) in MATLAB®.

**The code was developed in Python 3.7, tested in Python 3.6, 3.7.**

## Main content of the package

* [RLframe.py](RLframe.py) - main code, configured for quick demonstration
* [RLframe-experimental.py](RLframe-experimental.py) - the same, but configured for demonstration of control under estimated model
* [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py) - the same with model estimation and optimization called from MATLAB®
* [data-experimental.csv](data-experimental.csv) - data generated from running [RLframe-experimental.py](RLframe-experimental.py)
* [simulateData.py](simulateData.py) - program to *play back* a simulation episode
* [mySSest_simple.m](mySSest_simple.m) - a simple wrapper for the MATLAB's system identification routine of state-space model parameter estimation (used in [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py))
* [optCtrl.m](optCtrl.m) - actor routine from MATLAB® (used in [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py))

## General description

The flowchart in [Python-RL-flowchart.pdf](Python-RL-flowchart.pdf) pretty much explains how the different parts of this software package interact with each other.
The main ingredients of it are:

* the system
* the nominal controller
* the optimal controller consisting of a model estimator, the critic and the actor

Specifically, the environment (or rather the *system* interacting with the environment) possesses an internal state **x** and generates output **y**, being controlled by input **u**.
The running cost **r** is an abstract instantenous characteristic of performance as a function of the current **y**, **u**.
The *agent*, a.k.a. the *controller*, acquires the output **y**, the running cost **r** and generates a *policy*, a.k.a. a control law, which is a function that assignes a *control action* to each **y**.
The common goal is to generate such a policy that yields the best long-run performance in terms of an infinite horizon of running costs.
This is a general, abstract description.
Specific details depend on the concrete nature of the system, the running cost, the reinforcement learning (RL) method.
For instance, the system may be deterministic or stochastic.
The controller as well.
In the deterministic case, the controller assigns a fixed action to each output.
In the stochastic case, the policy is described by a probability distribution and actions are sampled from it.

## Usage

### Preliminaries

First, take care of the dependencies.
For the basic scenario, you will require:
* [`tabulate`](https://pypi.org/project/tabulate/)
* [`mpldatacursor`](https://pypi.org/project/mpldatacursor/)
* [`svgpath2mpl`](https://pypi.org/project/svgpath2mpl/)
* [`sippy`](https://github.com/CPCLAB-UNIPI/SIPPY)

These can be installed by

`pip install <package>`

or put directly into the code as

`!pip install <package>`

if you want to use it as an IPython notebook.

Regarding `sippy`, follow the instructions from the respective repository.
To get their `setup.py` working, you might need to do

    pip install nbconvert
    pip install cmake
    pip install scikit-build
    sudo apt-get install gfortran

Scroll down to [here](#running) for the standard scenario, without MATLAB.

### Usage with MATLAB

To run [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py), additional work is required.
First, download a suitable [MATLAB Runtime](https://www.mathworks.com/products/compiler/matlab-runtime.html).
Make sure that the Python and MATLAB Runtime verions are compatible.
If necessary, create a virtual environment with a suitable Python version, install needed packages there and proceed.
You will need to setup `matlab.engine` for Python which, in case you have MATLAB installed (which you should if you want to use this variant), can be found in

`<MATLABroot>/extern/engines/python`

Check usage with [testMATLAB.py](testMATLAB.py) in which specify `FOLDER_WITH_MATLAB_SCRIPTS` properly.
The same applies to [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py), section `Initialization`.

Read this [nice guide(]https://mscipio.github.io/post/matlab-from-python/) on how to use MATLAB in Python.

### Running

If everything was installed properly, simply run [RLframe.py](RLframe.py) in whatever environment you prefer.
[Spyder](https://www.spyder-ide.org/) is a nice one designed specifically for scientific computing.
To get the visualization working properly, you might need to set `backend` option in the tab `IPython console` to automatic.
You can always switch it off in case of problems.
See the customization description below.
If you choose to log data by setting `isLogData` to `1`, data files for each episode (or run) will be created in `data` folder.
To play back a data file, run `simulateData.py` whereby specify the file name in `dataFile` of section `Initialization`.
You can choose to print each time step, visualize stactically or dynamically, as if the simulator were running.

### Preconfigurations

As mentioined above, the following configurations come with this software, namely:

* [RLframe.py](RLframe.py) - configured for quick demonstration
* [RLframe-experimental.py](RLframe-experimental.py) - the same, but configured for demonstration of control under estimated model
* [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py) - the same with model estimation and optimization called from MATLAB®

In general , it is not well understood how good purely data-driven controllers can work.
In contrast to a pure data-driven RL, classical controllers use at least some knowledge of the physical process model.
The last two configurations above are achieved by experimentation with settings listed in [basic customization](#basic-customization).
They do not use the true model and estimate one on-the-fly.
All in all, it is an extremely challenging task to combine control with real-time model learning.
The rationale behind the described customization is as follows: to park the robot, or as we formally call it to *stabilize*, the controller needs to look several steps ahead.
We should notice here that we are talking about a rather brief preliminary learning phase before control, and not about a multi-trial RL where many episodes are to be accomplished before the controller learns proper behavior.
That is, the controller is to learn how to park the robot *in one try*, i.e., one episode.
Classical controllers are usually meant to function this way.
They may require a rather elaborate design, but come with formal guarantees of proper functioning.
In contrast, RL is a plug-and-play approach, that *per se* needs no insight into the system's nature and model.
What is achieved in the experimental configurations is a suitable balance between the controller sampling (although it was set to plausible 10 ms) and prediction horizon so as to stabilize the system (in an approximate sense, frankly) and at the same time to keep the prediction error reasonably small.
The longer the prediction horizon, the more estimation error is accumulated which might lead to improper controller behavior.
So far, the computation time is rather large in [RLframe-experimental.py](RLframe-experimental.py), [RLframe-experimental-MATLAB.py](RLframe-experimental-MATLAB.py) (although depending on your gear) and further tuning is required to reduce it.
Therefore, [data-experimental.csv](data-experimental.csv) is provided for use with [simulateData.py](simulateData.py) to produce a quick playback of an experimental run.

### Basic customization

All the customization settings are pretty much in section `Initialization` of [RLframe.py](RLframe.py).
The essential ones are listed below.

* Subsection `system`: contains system dimensions and parameters, as well as the parameters of the disturbance if it is required. 
* Subsection `simulation`:
  * `t0`, `t1`: start time and stop time of one episode (usually, `t0=0` and `t1` is the episode duration)
  * `Nruns`: number of episodes. After an episode, the system is reset to the initial state, whereas all the learned parameters continue to get updated. This emulates multi-trial RL
  * `x0, u0, q0`: initial values of the state, control and disturbance
  * `atol, rtol`: sensitivity of the solver. The lower the values, the more accurate the simulation results are
  * `xMin, xMax, yMin, yMax`: used so far rather for visualization only, but may be integrated into the actor as constraints
* Subsection `digital elements`
  * `dt`: controller sampling time. The system itself is continuous as a physical process while the controller is digital.
Things to note:
    1. the higher the sampling time, the more chattering in the control might occur. It even may lead to instability and failure to park the robot
    1. smaller sampling times lead to higher computation times
    1. especially controllers that use the estimated model are sensitive to sampling time, because inaccuracies in estimation lead to problems when propagated over longer periods of time. Experiment with `dt` and try achieve a trade-off between stability and computational performance
* Subsection `model estimator`:
  * `modEstPhase` [in seconds]: an initial phase to fill the estimator's buffer before applying optimal control
  * `modEstPeriod` [in seconds]: time between model estimate updates. This constant determines how often the estimated parameters are updated. The more often the model is updated, the higher the computational burden is. On the other hand, more frequent updates help keep the model actual.
  * `modEstBufferSize`: The size of the buffer to store data for model estimation. The bigger the buffer, the more accurate the estimation may be achieved. For successful model estimation, the system must be sufficiently excited. Using bigger buffers is a way to achieve this.
`modEstBufferSize` is measured in numbers of periods of length `dt`
  * `modelOrder`
  The order of the state-space estimation model `x_next = A x + B u, y = C x + D u`. We are interested in adequate predictions of `y` under given `u`'s. The higher the model order, the better estimation results may be achieved, but be aware of overfitting
  * (experimental) `modEstchecks`: estimated model parameters can be stored in stacks and the best among the `modEstchecks` last ones is picked. May improve the prediction quality somewhat
* Subsection `controller`:
  * `Nactor`: number of prediction steps. `Nactor=1` means the controller is purely **data-driven** and doesn't use prediction.
  * `predStepSize` [in seconds]: please, refer to [Python-RL-flowchart.pdf](Python-RL-flowchart.pdf)
* Subsection `RL elements`
  * `rcostStruct`: choice of the running cost structure. The explanation is given in the code. A typical choice is quadratic of the form `[y, u].T * R1 [y, u]`, where `R1` is the (usually diagonal) parameter matrix. For different structures, `R2` is also used.
  * `Ncritic`: critic stack size. The critic optimizes the *temporal error* which is a measure of critic's ability to capture the optimal infinite-horizon cost (a.k.a. the *value function*). The temporal errors are stacked up using the said buffer. The principle here is pretty much the same as with the model estimation: accuracy against performance
  * `criticPeriod` [in seconds]: the same meaning as `modEstPeriod`
  * `criticStruct`: choice of the structure of critic's feature vector. See the code for detailed description
* Subsection `main switches`: all the settings are self-explained. The controller modes are as follows:
  1. model-predictive control (MPC)
  1. MPC with estimated model, a.k.a. adaptive MPC, or AMPC
  1. RL/ADP (as `Nactor`-step roll-out Q-learning) using true model for prediction
  1. RL/ADP (as `Nactor`-step roll-out Q-learning) using estimated model for prediction 
  1. RL/ADP (as normalized [stacked Q-learning](https://doi.org/10.1016/j.ifacol.2017.08.803) with horizon `Nactor`) using true model for prediction
  1. RL/ADP (as normalized stacked Q-learning with horizon `Nactor`) using estimated model for prediction
  These methods are also described in detail in [Python-RL-flowchart.pdf](Python-RL-flowchart.pdf). Special modes: `0` -- manual control, `10` -- nominal parking controller, designed by a special technique called *nonsmooth backstepping* (read more [here](https://arxiv.org/abs/2006.14013) and references therein).

To check the model prediction quality, you can ucomment `DEBUG` lines in `actor` function.

## Advanced customization

* **Custom system**: setup dimensions and provide your system description in `sysStateDyn` and `sysOut`. Either provide your nominal controller or remove it altogether
* **Custom running cost**: adjust `rcost`
* **Custom critic**: either adjust the feature vector `Phi` while defining `dimCrit` accordingly, or change `W @ Phi(y, u)` in `criticCost` to what you require. It may be, say, a non-linearly parametrized structure like a deep neural net with multiple hidden layers. Here, we talk about Q-function approximation which depends on both **y** and **u**. If you choose an RL method that approximates the value function, the critic will depend only on **y**
* **Custom RL method**: (related to the previous one) adjust `actor`, `critic` or even change the general structure of the method in subsection `control: RL` of `ctrlStat` if further customization is required
* **Custom model estimator**: adjust subsection `model update` in `ctrlStat`. Be aware that this will also require adjusting `actorCost`, specifically, the part with prediction

## Features for future extension

* **ctrlDyn**: this is a dynamical controller which can be considered as an extension of the system state vector. RL is usually understood as a static controller, i.e., a one which assigns actions directly to outputs. A dynamical controller does this indirectly, via an internal state as intermediate link. Dynamical controllers can overcome some limitations of static controllers, but it goes beyond this description

## Closing remarks

Please contact [me](mailto:p.osinenko@skoltech.ru) for any inquiries and don't forget to give me credit for usage of this code.
If you are interested in stacked Q-learning, kindly read the [paper](https://www.sciencedirect.com/science/article/pii/S2405896317312570).
