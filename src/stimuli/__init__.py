from .experiment1 import EXPERIMENT as experiment1
from .experiment2 import EXPERIMENT as experiment2
from .experiment3 import EXPERIMENT as experiment3

EXPERIMENTS = [experiment1, experiment2, experiment3]

generate_experiment1 = experiment1.generate
generate_experiment2 = experiment2.generate
generate_experiment3 = experiment3.generate

__all__ = [
    "EXPERIMENTS",
    "generate_experiment1",
    "generate_experiment2",
    "generate_experiment3",
]
