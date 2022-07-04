import os, sys
import pickle5 as pickle
from unittest_data_generator import (
    Pipeline3WRobotTest,
    Pipeline3WRobotNITest,
    Pipeline2TankTest,
)
import pytest
from numpy.testing import assert_allclose

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)
from cli_test_helpers import ArgvContext


class TestPipeline3WRobot(Pipeline3WRobotTest):
    @pytest.fixture(autouse=True)
    def load_trajectory(self):
        with open("./refs/trajectory_3wrobot.pickle", "rb") as f:
            self.reference_trajectory = pickle.load(f)

    def test_run_animate(self):
        with ArgvContext("run_animate"):
            self.pipeline_execution()

    def test_run_raw(self):
        with ArgvContext("run_raw", "--no_visual"):
            self.pipeline_execution()

    def test_run_raw_sql(self):
        with ArgvContext("run_raw_sql", "--no_visual", "--ctrl_mode", "SQL"):
            self.pipeline_execution()

    def test_trajectory(self, load_trajectory):
        with ArgvContext("trajectory", "--no_visual", "--save_trajectory"):
            self.pipeline_execution()
            assert_allclose(self.trajectory, self.reference_trajectory)


class TestPipeline3WRobotNI(Pipeline3WRobotNITest):
    @pytest.fixture(autouse=True)
    def load_trajectory(self):
        with open("./refs/trajectory_3wrobot_NI.pickle", "rb") as f:
            self.reference_trajectory = pickle.load(f)

    def test_run_animate(self):
        with ArgvContext("run_animate"):
            self.pipeline_execution()

    def test_run_raw(self):
        with ArgvContext("run_raw", "--no_visual"):
            self.pipeline_execution()

    def test_run_raw_sql(self):
        with ArgvContext("run_raw_sql", "--no_visual", "--ctrl_mode", "SQL"):
            self.pipeline_execution()

    def test_trajectory(self, load_trajectory):
        with ArgvContext("trajectory", "--no_visual", "--save_trajectory"):
            self.pipeline_execution()
            assert_allclose(self.trajectory, self.reference_trajectory)


class TestPipeline2Tank(Pipeline2TankTest):
    @pytest.fixture(autouse=True)
    def load_trajectory(self):
        with open("./refs/trajectory_2tank.pickle", "rb") as f:
            self.reference_trajectory = pickle.load(f)

    def test_run_animate(self):
        with ArgvContext("run_animate"):
            self.pipeline_execution()

    def test_run_raw(self):
        with ArgvContext("run_animate", "--no_visual"):
            self.pipeline_execution()

    def test_run_raw_sql(self):
        with ArgvContext("run_raw_sql", "--no_visual", "--ctrl_mode", "SQL"):
            self.pipeline_execution()

    def test_trajectory(self, load_trajectory):
        with ArgvContext("trajectory", "--no_visual", "--save_trajectory"):
            self.pipeline_execution()
            assert_allclose(self.trajectory, self.reference_trajectory)
