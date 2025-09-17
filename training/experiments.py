from dataclasses import dataclass
from enum import Enum

class ClassifierModel(Enum):
    RIDGE = "ridge"
    LOGISTIC = "logistic"
    SVM = "svm"

@dataclass
class Experiment:
    model_type: ClassifierModel
    rocket_params: dict
    ridge_params: dict

experiments = {
    "ridge_default_rocket_default": 
        Experiment(
            model_type=ClassifierModel.RIDGE,
            rocket_params={"num_kernels": 10000},
            ridge_params={"alpha": 1.0}
    ),
    "ridge_alpha_0.1_rocket_default": 
        Experiment(
            model_type=ClassifierModel.RIDGE,
            rocket_params={"num_kernels": 10000},
            ridge_params={"alpha": 0.1}
    ),
}