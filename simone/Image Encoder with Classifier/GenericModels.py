from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
from Logger import Logger
from typing import Any


#####
#####
#####    HOW TO IMPLEMENT THE LOGGER IN THIS 
#####
#####


# In main define:
# TrainingConfig 
# list[Experiment]
# example of an Exeriment entry:  MLP, [MLPConfig(input_layer=512, .........., OptimizerConfig(torch.optim.Adam,{'lr':0.001, ....}), ...) , MLPConfig......   ]


def dict_to_str(d: dict)-> str:
    out = ""
    for k, v in d.items():
        out+=k+":"
        if type(v) is dict:
            out+="("+dict_to_str(v)+")"
        else:
            out+=str(v)
        out+=","
    return out

@dataclass
class TrainingConfig:
    epochs: int = 300
    seed: int = 42
    early_stop: int = 20
    num_workers: int = 6
    device: torch.device = torch.device('cpu')
    tqdm_progress: bool = True
    loggers: list[type[Logger]]| None = None
    project_name: str = "ArchitecturalStyles"
    @property
    def to_dict(self) -> dict:
        return {"device": self.device,
                "num_workers": self.num_workers,
                "epochs": self.epochs,
                "seed": self.seed,
                "early_stop": self.early_stop, }


@dataclass
class ModelConfig(ABC):
    pass
    @property
    @abstractmethod
    def to_dict(self) -> dict:
        pass



@dataclass
class Metrics:
    top1: float = 0.0
    top2: float = 0.0
    top3: float = 0.0
    F1_macro: float = 0.0
    eval_loss: float = 0.0
    train_loss: float | None = None
    @property
    def to_dict(self) -> dict:
        if self.train_loss is None: 
            return {"top1": self.top1, "top2": self.top2, "top3": self.top3, "F1_macro": self.F1_macro, "eval_loss": self.eval_loss,}
        else:
            return {"top1": self.top1, "top2": self.top2, "top3": self.top3, "F1_macro": self.F1_macro, "eval_loss": self.eval_loss,"train_loss": self.train_loss}
         


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, config: ModelConfig, input_size: int, num_classes: int, train_config: TrainingConfig):
        self.input_size=input_size
        self.num_classes=num_classes
        self.conf = config
        self.train_conf = train_config
        if train_config.seed is not None and train_config.seed != 0:
            torch.manual_seed(train_config.seed)

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None, loggers: list[Logger]| None = None) -> BaseModel:
        """Pay attention to the fact that X and y will be torch.Tensor and not Numpy array so converesion may be needed"""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Pay attention to the fact that X and y will be torch.Tensor and not Numpy array so converesion may be needed"""
        pass

    @abstractmethod
    def evaluate(self, X, y, loggers: list[Logger] | None = None) -> Metrics:
        """Pay attention to the fact that X and y will be torch.Tensor and not Numpy array so converesion may be needed"""
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @staticmethod
    @abstractmethod
    def model_name() -> str:
        return "Base Model"



@dataclass
class Experiment:
    model_cls: type[BaseModel]
    configs: list[Any]

    @property
    def to_dicts(self) -> list[dict]:
        mdl_str = self.model_cls.model_name()
        out = []
        for conf in self.configs:
            out.append({"Model":mdl_str} | conf.to_dict)
        return out