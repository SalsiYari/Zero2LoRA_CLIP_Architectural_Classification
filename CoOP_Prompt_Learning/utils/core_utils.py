import os
import random
import torch
from tqdm.auto import tqdm

def set_seed(seed: int = 42):
    random.seed(seed)                                       
    os.environ["PYTHONHASHSEED"] = str(seed)                
    torch.manual_seed(seed)                                 
    if torch.cuda.is_available():                    
        torch.cuda.manual_seed_all(seed)                    
        torch.backends.cudnn.deterministic = True           
        torch.backends.cudnn.benchmark = False              

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()

def maybe_tqdm(iterable, enabled: bool, **kwargs):
    """Restituisce tqdm(iterable) se enabled=True, altrimenti l'iterable nudo."""
    return tqdm(iterable, **kwargs) if enabled else iterable
