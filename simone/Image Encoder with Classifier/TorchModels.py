from __future__ import annotations
from GenericModels import *
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
import logging
import gc
import copy
from typing import Any, Optional, Callable

@dataclass
class LRSchedulerConfig:
    lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler]
    params: dict[str, Any]   # e.g. {"lr": 1e-3, "weight_decay": 1e-4}
    def to_dict(self) -> dict:
        return {
            "LR_Scheduler": str(self.lr_scheduler_cls),
            "params": self.params
            }

@dataclass
class OptimizerConfig:
    optimizer_cls: type[torch.optim.Optimizer]
    params: dict[str, Any]   # e.g. {"lr": 1e-3, "weight_decay": 1e-4}
    def to_dict(self) -> dict:
        return {
            "optimizer": str(self.optimizer_cls),
            "params": self.params
            }
          
@dataclass
class HiddenLayer:
    layer_size: int
    activation_layer_type: type[torch.nn.Module]

    @property
    def to_tuple(self) -> tuple:
        return (self.layer_size,str(self.activation_layer_type))

@dataclass
class MLPConfig(ModelConfig):
    # input_layer: int     excluded as dependent on encoder
    hidden_layers: list[HiddenLayer]
    """Tuple containing (layer_size, activation function cls)"""
    # output_layer: int    excluded as dependent on dataset 
    #activation_layer_type: type[torch.nn.Module] 
    batch_size : int 
    loss_func: type[torch.nn.Module] 
    optimizer: OptimizerConfig    
    lr_sched : Optional[LRSchedulerConfig] = None 
    dropout_rate: float = 0.0
    loss_weights_func : Optional[Callable[[torch.Tensor],torch.Tensor]] = None
    @property
    def to_dict(self) -> dict:
        return {
            "hidden_layers": [hl.to_tuple for hl in self.hidden_layers],
            "dropout_rate":self.dropout_rate,
            "loss_func":self.loss_func,
            "optimizer":self.optimizer.to_dict(),
            "batch_size": self.batch_size,
            "weighted_loss": self.loss_weights_func is not None
        }  | {"lr_sched":self.lr_sched.to_dict(),} if self.lr_sched is not None else {} 





class MLP(BaseModel):

    def __init__(self, config: MLPConfig, input_size: int, num_classes: int, train_config: TrainingConfig):
        super().__init__(config, input_size, num_classes, train_config)
        self.device = train_config.device
        self.conf=config
        layers = []
        next_in_features = input_size
        dropout_rate= config.dropout_rate
        # Hidden layers
        for hl in config.hidden_layers:
            layers.append(nn.Linear(next_in_features, hl.layer_size))
            if dropout_rate > 0.0 :
                layers.append(nn.Dropout(dropout_rate)) 
            layers.append(hl.activation_layer_type())
            
            next_in_features = hl.layer_size
        self.loss_func = self.conf.loss_func()
        # Output layer
        layers.append(nn.Linear(next_in_features, self.num_classes))
        
        # Pack everything into Sequential
        self.model = nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None, loggers : list[Logger] | None= None) -> MLP:
        
        self.model.train()
        self.model.to(self.device)
        X= X.to(self.device)
        y= y.to(self.device)

        ds = TensorDataset( X, y)
        if self.device.type=='cpu':
            dl = DataLoader( ds, batch_size = self.conf.batch_size, shuffle = True, num_workers = self.train_conf.num_workers,) 
        else:
            dl = DataLoader( ds, batch_size = self.conf.batch_size, shuffle = True, num_workers = 0,) # num_workers=0 because on GPU can be problematic 
                
        max_acc= 0.0
        best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()} 
        epoch_last_max = 0
        optimizer= self.conf.optimizer.optimizer_cls(self.model.parameters(),**self.conf.optimizer.params)    # Optimizer init
        if self.conf.lr_sched is not None :
            lr_scheduler = self.conf.lr_sched.lr_scheduler_cls(optimizer, **self.conf.lr_sched.params)     # optional lr scheduler init
        self.loss_func= self.conf.loss_func() if self.conf.loss_weights_func is None else self.conf.loss_func(weight= self.conf.loss_weights_func(y)) # loss function init
        logging.info(f"Starting training of {self.model_name} with config:{self.conf.to_dict}")
        pbar=tqdm(range(0, self.train_conf.epochs),disable=(not self.train_conf.tqdm_progress))
        
        for e in pbar:
            running_loss = 0.0
            
            for x, labels in dl:
                x = x.to(self.device)
                out = self.model(x)
                out = out.to(self.device)
                labels = labels.to(self.device)
                loss = self.loss_func(out,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / len(dl)


            if X_val != None and y_val != None:
                metrics=self.evaluate(X_val,y_val)
                metrics.train_loss= avg_loss
                if loggers is not None:
                    for logger in loggers:
                        logger.log_metrics(metrics=metrics.to_dict, step=e)

                self.model.train()
                self.model.to(self.device)
        
                if metrics.top1 > max_acc:
                    max_acc=metrics.top1
                    epoch_last_max = e
                    best_perf= metrics
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                
                pbar.set_description(f"Eval Accuracy: {metrics.top1:.4f}  Eval Loss: {metrics.eval_loss:.4f}  Train Loss: {metrics.train_loss:.4f}")

            else:
                pbar.set_description(f"Train Loss: {metrics.train_loss:.4f}")

            if self.conf.lr_sched is not None :
                lr_scheduler.step()
        

            if self.train_conf.early_stop>0 and e-epoch_last_max > self.train_conf.early_stop:
                logging.info(f"Training stopped early ({e}/{self.train_conf.epochs})")
                break
        
        logging.info(f"Training finished, Eval Accuracy: {best_perf.top1:.4f}  Eval Loss: {best_perf.eval_loss:.4f}  Train Loss: {best_perf.train_loss:.4f}") # type:ignore

        self.model.to("cpu")
        self.model.load_state_dict(best_state ,assign= True)
        self.model.eval()
        self.model.to("cpu")
        torch.cuda.empty_cache()
        del dl
        gc.collect()
        return self
        

    def predict(self, X) -> np.ndarray:  
        y_pred = self.predict_probs(X).argmax(axis=-1)
        y_pred = np.array(y_pred, dtype=int)
        return y_pred

    def predict_probs(self, X) -> np.ndarray :
        logits= torch.Tensor(self.predict_logits(X))
        probs = logits.softmax(dim=-1)
        return probs.cpu().numpy()
    
    def predict_logits(self, X) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        X=X.to(self.device)

        ds = TensorDataset(X)
        if self.device.type=='cpu':
            dl = DataLoader( ds, self.conf.batch_size, shuffle = False, num_workers=self.train_conf.num_workers,) 
        else:
            dl = DataLoader(ds, self.conf.batch_size, shuffle = False, num_workers = 0,) # num_workers=0 because on GPU can be problematic 
        
        logits_collector=[]

        with torch.no_grad():
            for x, in dl:
                x = x.to(self.device)
                class_logits = self.model(x)
                logits_collector.append( class_logits.cpu().numpy())
        logits = np.concatenate(logits_collector, axis=0)
        self.model.to("cpu")
        return logits
    
    def evaluate(self, X, y, loggers: list[Logger] | None = None) -> Metrics:
        logits = self.predict_logits(X)
        
        loss = self.loss_func(torch.from_numpy(logits).to(self.device), y.to(self.device))
        probs = torch.Tensor(logits).softmax(dim=-1).numpy()
        y_pred = probs.argmax(axis=-1)
        y_pred = np.array(y_pred, dtype=float)

        top1 = float(accuracy_score(y, y_pred))
        macro_f1 = float(f1_score(y, y_pred, average="macro"))
        top2 = top_k_accuracy_score(y, probs, k=min(2, self.num_classes), labels=np.arange(self.num_classes))
        top3 = top_k_accuracy_score(y, probs, k=min(3, self.num_classes), labels=np.arange(self.num_classes))
        avg_loss = loss.item() 
        metrics = Metrics(top1,top2,top3,macro_f1,avg_loss)
        if loggers is not None:
            for logger in loggers:
                logger.log_confusionMatrix(preds= np.array(y_pred,dtype=int).tolist(),y_true=np.array(y.cpu().numpy(),dtype=int).tolist())
                logger.log_per_class_perf(preds= np.array(y_pred,dtype=int).tolist(), y_true=np.array(y.cpu().numpy(),dtype=int).tolist())
                logger.log_summary(metrics=metrics.to_dict)

        return metrics

    def save(self, path):
        torch.save(self.model,path)

    def load(self, path):
        self.model=torch.load(path)

    @staticmethod
    def model_name() -> str:
        return "MLP"





# right now it is 1/freq_i  then normalized 
def loss_w_func(y : torch.Tensor) -> torch.Tensor:
    a = y.bincount()
    w = 1/a
    return w / w.mean()




