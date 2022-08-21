import os, sys
import numpy as np

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)

import rcognita

if os.path.abspath(rcognita.__file__ + "/../..") == PARENT_DIR:
    info = (
        f"this script is being run using "
        f"rcognita ({rcognita.__version__}) "
        f"located in cloned repository at '{PARENT_DIR}'. "
        f"If you are willing to use your locally installed rcognita, "
        f"run this script ('{os.path.basename(__file__)}') outside "
        f"'rcognita/presets'."
    )
else:
    info = (
        f"this script is being run using "
        f"locally installed rcognita ({rcognita.__version__}). "
        f"Make sure the versions match."
    )
print("INFO:", info)

from rcognita import optimizers
from pipeline_3wrobot import Pipeline3WRobot
import matplotlib.pyplot as plt


class Pipeline3WRobotCasadi(Pipeline3WRobot):
    def initialize_optimizers(self):

        opt_options = {
            "print_time": 0,
            "ipopt.max_iter": 200,
            "ipopt.print_level": 0,
            "ipopt.acceptable_tol": 1e-7,
            "ipopt.acceptable_obj_change_tol": 1e-4,
        }

        self.actor_optimizer = optimizers.CasADiOptimizer(
            opt_method="ipopt", opt_options=opt_options
        )
        self.critic_optimizer = optimizers.CasADiOptimizer(
            opt_method="ipopt", opt_options=opt_options,
        )


def main():
    pipeline = Pipeline3WRobotCasadi()
    pipeline.execute_pipeline()
    # DEBUG ===================================================================
    # if pipeline.control_mode == "RLSTAB":
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(2, 2, 1)
    #     plt.plot(
    #         np.array(pipeline.my_ctrl_benchm.critic_values, dtype="float")[:, 2],
    #         np.array(pipeline.my_ctrl_benchm.critic_values, dtype="float")[:, 0],
    #         "g",
    #         label="critic values",
    #     )
    #     plt.plot(
    #         np.array(pipeline.my_ctrl_benchm.critic_values, dtype="float")[:, 2],
    #         np.array(pipeline.my_ctrl_benchm.critic_values, dtype="float")[:, 1],
    #         "r",
    #         label="critic_diff_factual",
    #     )
    #     plt.xlabel("t [s]")
    #     plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.plot(
    #     np.array(pipeline.my_ctrl_benchm.actor.g_actor_values, dtype="float")[:, 1],
    #     np.array(pipeline.my_ctrl_benchm.actor.g_actor_values, dtype="float")[:, 0],
    #     label="g_actor",
    # )
    # plt.plot(
    #     np.array(pipeline.my_ctrl_benchm.critic.g_critic_values, dtype="float")[
    #         :, 1
    #     ],
    #     np.array(pipeline.my_ctrl_benchm.critic.g_critic_values, dtype="float")[
    #         :, 0
    #     ],
    #     label="g_critic",
    # )
    # plt.xlabel("t [s]")
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.plot(
    #     np.array(pipeline.my_ctrl_benchm.g_actor_values, dtype="float")[:, 1],
    #     np.array(pipeline.my_ctrl_benchm.g_actor_values, dtype="float")[:, 0],
    #     label="g_actor",
    # )
    # plt.plot(
    #     np.array(pipeline.my_ctrl_benchm.g_critic_values, dtype="float")[:, 1],
    #     np.array(pipeline.my_ctrl_benchm.g_critic_values, dtype="float")[:, 0],
    #     label="g_critic",
    # )
    # plt.xlabel("t [s]")
    # plt.legend()

    # plt.subplot(2, 2, 4)
    # plt.plot(
    #     np.array(pipeline.my_ctrl_benchm.g_emergency_critic_deriv, dtype="float")[
    #         :, 1
    #     ],
    #     np.array(pipeline.my_ctrl_benchm.g_emergency_critic_deriv, dtype="float")[
    #         :, 0
    #     ],
    #     label="g_critic_deriv",
    # )
    # plt.plot(
    #     np.array(
    #         pipeline.my_ctrl_benchm.g_emerency_critic_diff_weights, dtype="float"
    #     )[:, 1],
    #     np.array(
    #         pipeline.my_ctrl_benchm.g_emerency_critic_diff_weights, dtype="float"
    #     )[:, 0],
    #     label="g_critic_diff_weights",
    # )
    # plt.xlabel("t [s]")

    # plt.legend()
    # /DEBUG ===================================================================

    # if pipeline.save_trajectory:
    #     trajectory = np.linalg.norm(np.array(pipeline.trajectory)[:, :5], 2, axis=1)
    #     ts = np.array(pipeline.trajectory)[:, 5]
    #     plt.subplot(2, 2, 2)
    #     plt.plot(ts, trajectory, label="||x||")
    #     plt.xlabel("t [s]")
    #     plt.legend()
    #     plt.show()


if __name__ == "__main__":
    main()
