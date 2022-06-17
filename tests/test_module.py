import os, sys
import pickle5 as pickle
from unittest_data_generator import PresetPipeline
import pytest

PARENT_DIR = os.path.abspath(__file__ + "/../..")
sys.path.insert(0, PARENT_DIR)


class TestPresetPipeline3WRobot(PresetPipeline):
    @pytest.fixture(autouse=True)
    def load_trajectory(self):
        with open("./refs/trajectory_3wrobot.pickle", "rb") as f:
            self.reference_trajectory = pickle.load(f)

    def test_run_animate(self):
        self.pipeline_execution()

    def test_run_raw(self):
        args = {"is_visualization": False}
        self.pipeline_execution(args)

    def test_trajectory_(self, load_trajectory):
        args = {"is_visualization": False}
        self.pipeline_execution(args, save_trajectory=True)
        self.trajectory = pytest.approx(self.reference_trajectory)
