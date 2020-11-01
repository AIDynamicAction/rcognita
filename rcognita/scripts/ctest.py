from rcognita import *
import argparse

def main(args=None):
    """main"""
    
    # environment
    sys = EndiSystem(initial_x=7, initial_y=7)
    # sys.add_bots(-5,-5) # creates bot #2
    # sys.add_bots(-7,7) # creates bot #3
    # sys.add_obstacle()
    # sys.add_obstacle(obs_type='Circle', coords=(2,3), radius=1.5)
    # sys.add_obstacle(obs_type='Circle', coords=(7,3), radius=0.5)

    agent1 = ActorCritic(sys,
                        sample_time=0.3,
                        step_size=0.6,
                        ctrl_mode=3,
                        critic_mode=3,
                        buffer_size=10,
                        actor_control_horizon=7,
                        t1=20,
                        obs_penalty=10**8)

    # agent2 = ActorCritic(sys,
    #                     sample_time=0.6,
    #                     step_size=0.3,
    #                     ctrl_mode=3,
    #                     critic_mode=3,
    #                     buffer_size=10,
    #                     actor_control_horizon=10,
    #                     t1=20)

    # agent3 = ActorCritic(sys,
    #                     sample_time=0.75,
    #                     step_size=0.3,
    #                     ctrl_mode=3,
    #                     critic_mode=3,
    #                     buffer_size=10,
    #                     actor_control_horizon=10,
    #                     t1=20)

    sim = Simulation(sys, agent1)
    # sim = Simulation(sys, [agent1, agent2, agent3])
    
    sim.run_simulation(n_runs=2,
                    is_visualization=True, 
                    exit_py_on_finish=False, 
                    show_annotations=True, 
                    show_summary_stats=True, 
                    print_statistics_at_step=False)


if __name__ == "__main__":
    main()
