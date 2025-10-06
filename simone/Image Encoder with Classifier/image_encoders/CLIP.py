import torch
from torch import nn
from image_encoders.ImageEncoder import ImageEncoder
import open_clip
import warnings
from typing import Tuple


class CLIPImageEncoder(ImageEncoder):
    """CLIP-based image encoder implementation."""

    def __init__(self, CLIP_name: str, Pretrained: str):
        super().__init__()
        warnings.filterwarnings("error", ".*quick_gelu.*", UserWarning)
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                CLIP_name, Pretrained
            )
        except UserWarning:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                CLIP_name, Pretrained, force_quick_gelu=True
            )
        self.model_name = CLIP_name
        self.pretrained_name = Pretrained
        self.eval()
        


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP model.
        
        Args:
            images: Preprocessed image tensor
            
        Returns:
            L2-normalized image features
        """
        with torch.no_grad():
            img_features = self.model.encode_image(images)  # type:ignore
            img_features = img_features / torch.linalg.norm(
                img_features, dim=-1, keepdim=True
            )
        return img_features    

    @property
    def num_parameters(self) -> int:
        """Return number of parameters in the visual encoder."""
        return sum(p.numel() for p in self.model.visual.parameters()) +  sum(p.numel() for p in self.model.visual.buffers()) # type:ignore

    @property
    def model_dtype(self) -> torch.dtype:
        """Return the data type of model parameters."""
        return next(self.model.parameters()).dtype  

    @property
    def image_input_size(self) -> Tuple[int, int]: 
        """Return expected input image size (height, width)."""
        if hasattr(self.model.visual, 'image_size'):
            size = self.model.visual.image_size # type:ignore
            # Handle both single int and tuple cases
            if isinstance(size, int):
                return (size, size)
            return tuple(size) if len(size) == 2 else (size[0], size[0]) # type:ignore
        else:
            # Default fallback - you might want to adjust this (224,244 is the default for CLIP in the paper)
            return (224, 224)
        
    @property
    def output_emb_size(self) -> torch.Size:
        """Return the output embedding size."""
        # Get model device and dtype
        device = next(self.model.parameters()).device
        dtype = self.model_dtype
        
        # Create a dummy input tensor
        height, width = self.image_input_size
        test_img = torch.zeros(1, 3, height, width, device=device, dtype=dtype)
        
        # Get output shape
        with torch.no_grad():
            out = self.forward(test_img)
        return out.shape[1:]  # Exclude batch dimension
    
    @property
    def preprocessing(self): #type:ignore
        """Return the preprocessing transform."""
        return self.preprocess



    @property
    def preprocessing(self): # type:ignore
        return self.preprocess
    
    @property
    def get_model_name(self)-> str:
        """Return the name of the model, mainly for printing"""
        return self.model_name+", "+self.pretrained_name