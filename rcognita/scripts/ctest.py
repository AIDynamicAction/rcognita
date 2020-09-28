from rcognita import *
import argparse

def main(args=None):
    """main"""
    
    # environment
    sys = EndiSystem() # already creates bot #1
    sys.add_bots(-5,-5) # creates bot #2
    sys.add_bots(-7,7) # creates bot #3

    agent1 = ActorCritic(sys,
                        sample_time=0.3,
                        step_size=0.3,
                        ctrl_mode=3,
                        critic_mode=3,
                        buffer_size=15,
                        actor_control_horizon=15,
                        t1=20,
                        estimator_update_time=0.3)

    agent2 = ActorCritic(sys,
                        sample_time=0.45,
                        step_size=0.3,
                        ctrl_mode=3,
                        critic_mode=3,
                        buffer_size=10,
                        actor_control_horizon=10,
                        t1=20,
                        estimator_update_time=0.3)

    agent3 = ActorCritic(sys,
                        sample_time=0.6,
                        step_size=0.3,
                        ctrl_mode=3,
                        critic_mode=3,
                        buffer_size=10,
                        actor_control_horizon=10,
                        t1=20,
                        estimator_update_time=0.3)

    nominal_ctrl = NominalController()
    nominal_ctrl2 = NominalController()
    nominal_ctrl3 = NominalController()

    # sim = Simulation(sys, agent1, nominal_ctrl)
    sim = Simulation(sys, [agent1, agent2, agent3], [nominal_ctrl, nominal_ctrl2, nominal_ctrl3])
    
    sim.run_simulation(n_runs=2, 
                    is_visualization=False, 
                    close_plt_on_finish=False, 
                    show_annotations=True, 
                    print_summary_stats=True, 
                    print_statistics_at_step=True)

if __name__ == "__main__":
    main()
