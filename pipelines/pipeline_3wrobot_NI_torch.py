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
from pipeline_3wrobot_NI import Pipeline3WRobotNI


class Pipeline3WRobotNITorch(Pipeline3WRobotNI):
    def initialize_optimizers(self):

        opt_options_torch = {"lr": 0.000005, "momentum": 0.9}
        opt_options_scipy = {
            "maxiter": 500,
            "maxfev": 5000,
            "disp": False,
            "adaptive": True,
            "xatol": 1e-7,
            "fatol": 1e-7,
        }
        self.actor_optimizer = optimizers.SciPyOptimizer(
            opt_method="SLSQP", opt_options=opt_options_scipy
        )

        self.critic_optimizer = optimizers.TorchOptimizer(opt_options_torch)


if __name__ == "__main__":

    Pipeline3WRobotNITorch().execute_pipeline()
