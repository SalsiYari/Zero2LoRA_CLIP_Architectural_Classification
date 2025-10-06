from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from dataclasses import dataclass
from typing import Any

@dataclass
class ImageEncoderConfig:
    encoder_cls: type[ImageEncoder]
    params: dict[str, Any]   # e.g. {"lr": 1e-3, "weight_decay": 1e-4}

    def to_dict(self) -> dict:
        return {
            "optimizer": str(self.encoder_cls),
            "params": self.params
            }

class ImageEncoder(torch.nn.Module, ABC):
    """Abstract base class for image encoders. Already extends torch.nn.Module"""

    def __init__(self):
        """Already calls the torch.nn.Module (super()) __init__()"""
        super().__init__()

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the image encoder.
        
        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded image features tensor
        """
        pass

    @property 
    @abstractmethod
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""
        pass

    @property
    @abstractmethod
    def model_dtype(self) -> torch.dtype:
        """Return the data type of the model parameters."""
        pass

    @property
    @abstractmethod
    def image_input_size(self) -> tuple[int, int]: 
        """Return the expected input image size as (height, width)."""
        pass
        
    @property
    @abstractmethod
    def output_emb_size(self) -> torch.Size:
        """Return the size of the output embedding (excluding batch dimension)."""
        pass

    @property
    @abstractmethod
    def preprocessing(self): # type: ignore
        """Return the preprocessing function/transform for input images. (ignore the return type error)"""
        pass

    @property
    @abstractmethod
    def get_model_name(self)-> str:
        """Return the name of the model, mainly for printing"""
        pass 
    # def to(self,device: ):
    #     super().to(device)


    
