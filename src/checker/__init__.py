from utils import read_yaml
from checker.evaluator import Evaluator
from checker.visualizer import Visualizer

try:
    model_dict = read_yaml("./checker/models.yaml")
except:
    pass
