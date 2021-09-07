<p align="center">
<img src="https://raw.githubusercontent.com/AIDynamicAction/rcognita/master/gfx/logo/rcognita-logo.png" width=40% height=40% />
</p>

`rcognita` is a framework for hybrid agent-enviroment simultion.
The hybrid setting here means the actions are updated at discrete moments in time, whereas the environment dynamics are modelled time-continuous.
A detailed documentation is available [here](https://aidynamicaction.github.io/rcognita/).

## Example run with a mobile robot simulation

<p align="center">
	<img src="https://raw.githubusercontent.com/AIDynamicAction/rcognita/master/gfx/demo/3wheel_robot_exm_run.gif" width=40% />
</p>

# Table of content

- [Installation](#Installation)
- [General description](#General-description)
- [Usage](#Usage)
  * [Settings](#Settings)
  * [Advanced customization](#Advanced-customization)
  * [Experimental things](#Experimental-things)
- [Closing remarks](#Closing-remarks)

# Installation

This instruction is for Unix-based systems, assuming a terminal and Python3 interpreter.

```
git clone https://github.com/AIDynamicAction/rcognita
cd rcognita
python3 setup.py install
```

Notice that your Python 3 interpreter might be called something else, say, just `python`

# General description

[To table of content](#Table-of-content)

`rcognita` Python package is designed for hybrid simulation of agents and environments (generally speaking, not necessarily reinforcement learning agents).
Its main idea is to have an explicit implementation of sampled controls with user-defined sampling time specification.
The package consists of several modules, namely, `controllers`, `loggers`, `models`, `simulator`, `systems`, `utilities`, `visuals` and a collection of main modules (presets) for each agent-environment configuration.

[This flowchart](./flowcharts/rcognita-flowchart-RLstab.pdf) shows interaction of the core `rcognita` classes contained in the said modules (the latter are not shown on the diagram).

The main module is a preset, on the flowchart a 3-wheel robot.
It initializes the system (the environment), the controllers (the agents, e. g., a safe agent, a benchmarking agent, a reinforcement learning agent etc.), the visualization engine called animator, the logger and the simulator.
The latter is a multi-purpose device for simulating agent-environment loops of different types (specified by sys type).

Depending on `sys_type`, the environment can either be described by a differential equation (including stochastic ones), a difference equation (for discrete-time systems), or by a probability distribution (for, e. g., Markov decision processes).

The parameter `dt` determines the maximal step size for the numerical solver in case of differential equations.
The main method of this class is `sim_step` which performs one solver step, whereas reset re-initializes the simulator after an episode.

The `Logger` class is an interface defining stubs of a print-to-console method print sim step, and print-to-file method log data row, respectively.
Concrete loggers realize these methods.

A similar class inheritance scheme is used in animator, and system.
The core data of animator’s subclasses are `objects`, which include entities to be updated on the screen, and their parameters stored in `pars`.

A concrete realization of a system interface must realize `sys_dyn`, which is the “right-handside” of the environment description, optionally disturbance dynamics via `disturb_dyn`, optionally controller dynamics (if the latter is, e. g., time-varying), and the output function `out`.
The method `receive_action` gets a control action and stores it.
Everything is packed together in the `closed_loop_rhs` for the use in `Simulator`.

Finally, the `controllers` module contains various agent types.
One of them is `CtrlRLStab` – the class of stabilizing reinforcement learning agents as shown in [this flowchart](./flowcharts/rcognita-flowchart-RLstab.pdf).
Notice it contains an explicit specification of the sampling time.
The data `SafeCtrl` is required to specify the stabilizing constraints and also to initialize the optimizer inside the `actor_critic` method, which in turns fetches the cost function from the `actor_critic_cost` method.
The method `compute_action` essentially watches the internal clock and performs an action updates when a time sample has elapsed.

Auxiliary modules of the package are `models` and `utilities` which provide auxiliary functions and data structures, such as neural networks.

# Usage

[To table of content](#Table-of-content)

After the package is installed, you may just `python` run one of the presets found [here](./presets).
The naming concention is `main_ACRONYM`, where `ACRONYM` is actually related to the system (environment). 
You may create your own by analogy.

## Settings

These are made in a preset file.
Some are more or less self-evident, like `is_log_data`.
The crucial ones are:

* `dt`: [in seconds] controller sampling time. Relevant if the system itself is continuous as a physical process while the controller is digital
* `Nactor`: number of prediction steps. `Nactor=1` means the controller is purely **data-driven** and doesn't use prediction. Say, stacked QL will turn into the usual SARSA (in VI form)
* `pred_step_size`: [in seconds] determines how fine the resolution of the prediction horizon is. Larger `pred_step_size` will result in a larger effective horizon length
* 

Miscellaneous settings:

* `t0`, `t1`: start time and stop time of one episode (usually, `t0=0` and `t1` is the episode duration)
* `atol, rtol`: sensitivity of the solver. The lower the values, the more accurate the simulation results are
* `Nruns`: number of episodes. After an episode, the system is reset to the initial state, whereas all the learned parameters continue to get updated. This emulates multi-episode RL

## Advanced customization

[To table of content](#Table-of-content)

* **Custom environments**: realize `system` interface in the `systems` module. You might need nominal controllers for that, as well as an animator, a logger etc.
* **Custom running cost**: adjust `rcost` in controllers
* **Custom AC method**: simplest way -- by adding a new mode and updating `_actor_cost`, `_critic_cost` and, possibly, `_actor`, `_critic`. For deep net AC structures, use, say, [PyTorch](https://pytorch.org/)
* **Custom model estimator**: so far, the framework offers a state-space model structure. You may use any other one. In case of neural nets, use, e.g., [PyTorch](https://pytorch.org/)

## Experimental things

[To table of content](#Table-of-content)

An interface for dynamical controllers, which can be considered as extensions of the system state vector, is provided in `_ctrl_dyn` of the `systems` module.
RL is usually understood as a static controller, i.e., a one which assigns actions directly to outputs.
A dynamical controller does this indirectly, via an internal state as intermediate link. 
ynamical controllers can overcome some limitations of static controllers.

The package was tested with online model estimation using [SIPPY](https://github.com/CPCLAB-UNIPI/SIPPY). So far, there is no straightforward way to install it (under Windows at least).
The functionality is implemented and enabled via `is_est_model`.
Related parameters can be found in the documentation of the `ctrl_opt_pred` class.
Updates to come.

# Closing remarks

Please contact [me](mailto:p.osinenko@gmail.com) for any inquiries and don't forget to give me credit for usage of this code.
If you are interested in stacked Q-learning, kindly read the [paper](https://arxiv.org/abs/2007.03999).

[To table of content](#Table-of-content)

```
@misc{rcognita00,
author =   {Pavel Osinenko},
title =    {Rcognita: a framework for hybrid agent-enviroment simultion},
howpublished = {\url{https://github.com/AIDynamicAction/rcognita}},
year = {2020}
}
```

Original author: P. Osinenko, 2020
