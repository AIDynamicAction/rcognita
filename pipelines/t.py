import os, sys

PARENT_DIR = os.path.abspath(__file__ + "/../../")
sys.path.insert(0, PARENT_DIR)
CUR_DIR = os.path.abspath(__file__ + "/..")
sys.path.insert(0, CUR_DIR)
from rcognita.rl_tools import CriticRLStabQuadratic

a = CriticRLStabQuadratic(5, 5, 5, 5,)
print(a)
