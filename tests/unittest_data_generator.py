import os, sys
import pickle5 as pickle

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)

from presets import (
    pipeline_3wrobot,
    pipeline_3wrobot_NI,
    pipeline_2tank,
)


class PresetPipeline3WRobotTest(pipeline_3wrobot.Pipeline3WRobot):
    def generate_trajectory(self):
        self.save_trajectory = True
        self.pipeline_execution()
        with open("./refs/trajectory_3wrobot.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


class PresetPipeline3WRobotNITest(pipeline_3wrobot_NI.Pipeline3WRobot):
    def generate_trajectory(self):
        self.save_trajectory = True
        self.pipeline_execution()
        with open("./refs/trajectory_3wrobot.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


class PresetPipeline2TankTest(pipeline_2tank.Pipeline3WRobotNI):
    def generate_trajectory(self):
        self.save_trajectory = True
        self.pipeline_execution()
        with open("./refs/trajectory_2tank.pickle", "wb") as trajectory:
            pickle.dump(self.trajectory, trajectory)


if __name__ == "__main__":
    PresetPipeline3WRobotTest().config_to_pickle()
    PresetPipeline3WRobotTest().generate_trajectory()
    PresetPipeline3WRobotNITest().config_to_pickle()
    PresetPipeline3WRobotNITest().generate_trajectory()
    PresetPipeline2TankTest().config_to_pickle()
    PresetPipeline2TankTest().generate_trajectory()
