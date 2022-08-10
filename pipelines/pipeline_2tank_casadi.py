import os, sys

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
from pipeline_2tank import Pipeline2Tank


class Pipeline2TankCasADi(Pipeline2Tank):
    def optimizers_initialization(self):

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


if __name__ == "__main__":

    Pipeline2TankCasADi().pipeline_execution()
