<p align="center">
	<img src="./docs/rcognita-logo.png" width=40% height=40% />
</p>

<p align="center">
<br />
<br />
Rcognita is a framework for DP and RL algorithm development, testing, and simulation.
</p>

Installation:
* clone repo
* run in terminal ```python3 setup.py install```

Example of terminal comand:
```{r, engine='bash', count_lines}
python3 main_3wrobot.py --mode 5 --ndots 20 --radius 5 --dt 0.05
```
Arguments for comand line for single run:
- mode - *default 3*
- dt - *default 0.05*
- Nactor - *default 6*
- pred_step_size - *default 5*
- init_x - used only if one point required and only with init_y and init_alpha *default None*
- init_y - used only if one point required and only with init_x and init_alpha *default None*
- init_alpha - could be used separetely for one point (without init_x and init_y) *default None*
- ndots - number of dots for simulation *default 25*
- radius - *default 5*
- folder - *default None will be created folder with name of current hour*
- is_log_data - saving date in csv file in folder data/date/hour(or_name) *default True*
- is_print_sim_step - printing info about each step in terminal *default False*
- is_visualization - will visualize interface of rcognita and all process *default False*

