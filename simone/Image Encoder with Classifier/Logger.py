import wandb
from abc import ABC, abstractmethod
from typing import Any, Sequence
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np


class Logger(ABC):
    """Abstract logger class that defines the logger interface.
    This Logger is meant for tracking AI training runs and hyperparameters tests.
    It should be used as reference to implement Loggers using API for tools like:
    Weights and Biases, MLFlow, Tensorboard"""

    def __init__(self,project_name: str, run_name: str, config: dict, class_names: Sequence[str]):
        self.project_name = project_name
        self.run_name= run_name
        self.config= config
        self.class_names= class_names

    @abstractmethod
    def log_metric(self, key: str, value, step: int | None = None):
        """Logs a single metric value (e.g., loss, accuracy)."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        """Logs multiple metric value (e.g., loss, accuracy)."""
        pass
    @abstractmethod
    def log_summary(self, metrics: dict[str, Any],  ):
        """Logs a final summary of the run"""

    @abstractmethod
    def log_confusionMatrix(self, preds: list[int],y_true: list[int],  title: str  = 'Confusion Matrix'):
        """Builds the confusion matrix and logs it"""
        pass

    @abstractmethod
    def log_per_class_perf(self,preds: list[int],y_true: list[int]):
        """Logs a table that contains performance measure for each class"""
        pass

    @abstractmethod
    def log_hyperparameters(self, params: dict):
        """Logs a dictionary of hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(self, file_path: str):
        """Logs a file artifact (e.g., a model checkpoint or a plot)."""
        pass
        
    @abstractmethod
    def finish(self):
        """Ends the logging run."""
        pass



class WandBLogger(Logger):
    def __init__(self, project_name: str, run_name: str, config: dict, class_names: Sequence[str],):
        super().__init__(project_name=project_name,run_name=run_name,config=config,class_names=class_names)
        self.run = wandb.init(project=project_name, name=run_name, config=config)
        self.class_names=class_names

    def log_metric(self, key: str, value, step: int | None = None):
        self.run.log({key: value}, step=step)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None):
        self.run.log(metrics, step=step)

    def log_summary(self, metrics: dict[str, Any],  ):
        for k,v in metrics.items():
            self.run.summary[k]=v


    def log_confusionMatrix(self, preds: list[int] ,y_true: list[int],  title: str  = 'Confusion Matrix'):
        conf_matrix = wandb.plot.confusion_matrix(preds=preds,y_true=y_true,class_names=self.class_names, title=title)
        self.run.log({title.lower().replace(' ','_'): conf_matrix})

    def log_per_class_perf(self,preds: list[int],y_true: list[int]):
        df=pd.DataFrame(classification_report(y_true, preds, target_names=self.class_names, digits=4 , zero_division=0,output_dict=True))
        df=df.transpose()
        df.columns=["Precision","Recall","F1-score","Support"]
        df = df.reset_index().rename(columns={"index": "Class"})
        table=wandb.Table(dataframe=df)
        self.run.log({"class_performance": table})

    def log_hyperparameters(self, params: dict):
        # W&B handles this at init, but you could add updates
        self.run.config.update(params)

    def log_artifact(self, file_path: str):
        artifact = wandb.Artifact(name='model', type='model')
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)
        
    def finish(self):
        self.run.finish()

class FileLogger(Logger):

    def __init__(self,project_name: str, run_name: str, config: dict, class_names: Sequence[str],conf_matrix_dir: str = './confusion_matrices', class_metrics_dir= "./per_class_metrics", csv_path = "./runs.csv"):
        super().__init__(project_name=project_name,run_name=run_name,config=config,class_names=class_names)
        self.run_name= run_name.lower().replace(' ','_')
        self.config = config
        self.class_names=class_names
        print(f"--- Starting run {run_name} ---")
        self.step=0
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.conf_matrix_dir=os.path.join(script_dir,conf_matrix_dir)
        
        self.class_metrics_dir=os.path.join(script_dir,class_metrics_dir)
        if not os.path.isdir(self.conf_matrix_dir):
            os.makedirs(self.conf_matrix_dir)
        if not os.path.isdir(self.class_metrics_dir):
            os.makedirs(self.class_metrics_dir)

        self.csv_path=csv_path
        

    def log_metric(self, key: str, value, step: int | None = None, to_csv: bool = False):
        pass
        # if to_csv:
        #     self.log_metrics(metrics={key:value}, step=step,to_csv=True )
        #     return
        # self.step= step if step is not None else self.step+1
        # print(f"{self.run_name} ({self.step}) --- {key} : {value}")
        
    def log_summary(self, metrics: dict[str, Any],  ):
        row= {"Run":self.run_name}
        row = row | metrics
        df = pd.DataFrame([row])
        if os.path.exists(self.csv_path):
            old = pd.read_csv(self.csv_path)
            out = pd.concat([old, df], ignore_index=True)
        else:
            out = df.copy()
        out.to_csv(self.csv_path, index=False)
        
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None, to_csv: bool = True ):
        pass
        # self.step= step if step is not None else self.step+1
        # row= {"Run":self.run_name}
        # row = row | metrics
        # df = pd.DataFrame([row])
        # if to_csv:
        #     if os.path.exists(self.csv_path):
        #         old = pd.read_csv(self.csv_path)
        #         out = pd.concat([old, df], ignore_index=True)
        #     else:
        #         out = df.copy()
        #     out.to_csv(self.csv_path, index=False)
        
        # print(df)

    def log_confusionMatrix(self, preds: list[int],y_true: list[int],  title: str  = 'Confusion Matrix', to_file: bool =True):
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, preds,
            display_labels=self.class_names,
            normalize="true",           # percentuali per riga
            cmap="Blues",
            include_values=False,       # evita numeri sopra le celle se hai tante classi
            xticks_rotation=90,
            ax=ax,
            colorbar=True
        )
        ax.set_title(title)
        plt.tight_layout()
        
        if to_file:
            try:
                plt.savefig(os.path.join(self.conf_matrix_dir, self.run_name+'-'+title.lower().replace(" ",'_')))
            except :
                plt.show()
        else:
            plt.show()
        plt.close()

    def log_per_class_perf(self,preds: list[int],y_true: list[int],  to_file: bool = True):
        df=pd.DataFrame(classification_report(y_true, preds, target_names=self.class_names, digits=4 , zero_division=0,output_dict=True))
        df=df.transpose()
        df.columns=["Precision","Recall","F1-score","Support"]
        if to_file:
            try:
                df.to_csv(os.path.join(self.class_metrics_dir, self.run_name+'.csv'), index=True)
            except :
                print("Couldn't save")   
        else:
            print(df)
        
    def log_hyperparameters(self, params: dict):
        print(self.config)

    def log_artifact(self, file_path: str):
        pass
        
    def finish(self):
        pass
