import os, sys
import pickle5 as pickle

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)
from presets import (
    pipeline_3wrobot,
    pipeline_3wrobot_NI,
    pipeline_2tank,
    configs,
)


class Pipeline3WRobotTest(pipeline_3wrobot.Pipeline3WRobot):
    def generate_trajectory(self):
        args = {"no_visual": True, "save_trajectory": True}
        self.pipeline_execution(args)
        with open("./refs/trajectory_3wrobot.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


class Pipeline3WRobotNITest(pipeline_3wrobot_NI.Pipeline3WRobotNI):
    def generate_trajectory(self):
        args = {"no_visual": True, "save_trajectory": True}
        self.pipeline_execution(args)
        with open("./refs/trajectory_3wrobot_NI.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


class Pipeline2TankTest(pipeline_2tank.Pipeline2Tank):
    def generate_trajectory(self):
        args = {"no_visual": True, "save_trajectory": True}
        self.pipeline_execution(args)
        with open("./refs/trajectory_2tank.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


if __name__ == "__main__":
    os.chdir(os.path.abspath(__file__ + "/.."))

    pipeline3Wrobot = Pipeline3WRobotTest()
    pipeline3Wrobot.generate_trajectory()
    pipeline3Wrobot.config_to_pickle()

    pipeline3WrobotNI = Pipeline3WRobotNITest()
    pipeline3WrobotNI.generate_trajectory()
    pipeline3WrobotNI.config_to_pickle()

    pipeline2Tank = Pipeline2TankTest()
    pipeline2Tank.generate_trajectory()
    pipeline2Tank.config_to_pickle()
