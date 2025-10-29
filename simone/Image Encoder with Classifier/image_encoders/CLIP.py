import torch
from torch import nn
#from image_encoders.ImageEncoder import ImageEncoder
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
    
# From https://github.com/MaxZanella/CLIP-LoRA/blob/main/loralib/layers.py
#  ------------------------------------------------------------------------------------------
#  This code is reconstructed based on loralib (https://github.com/microsoft/LoRA) by Baijiong Lin.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        fan_in_fan_out: bool = False,
        dropout_rate:float = 0,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            #self.scaling = self.lora_alpha / self.r
            self.scaling = self.lora_alpha/math.sqrt(self.r) # 
        # Mark the weight as unmerged
        self.merged = False
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        self.fan_in_fan_out = fan_in_fan_out
        # define params that require LoRA {'param_name': 'lora_name'}
        self.params_with_lora = {}

    def register_lora_param(self):
        r"""Register LoRA matrix"""
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').size()) == 2
            self.register_parameter(f'{lora_name}_lora_A', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((self.r, eval(f'self.{param_name}').size()[1])))
                )
            self.register_parameter(f'{lora_name}_lora_B', 
                nn.Parameter(eval(f'self.{param_name}').new_zeros((eval(f'self.{param_name}').size()[0], self.r)))
                )
                
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.zeros_(eval(f'self.{lora_name}_lora_B'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).view(eval(f'self.{param_name}').shape))

    
   
    
    def merge_lora_param(self):
        r"""p_new = p + scaling * B @ A and keep differentiable to A and B"""
        for param_name, lora_name in self.params_with_lora.items():
            p = set_param(self, param_name, mode='get')
            # detach() is very important here
            
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            set_param(self, param_name, param=p_new, mode='update')

    def add_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling
    
    def sub_lora_data(self):
        r"""NOT differentiable"""
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling
            
    
    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
            # Make sure that the weights are not merged
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
            # Merge the weights and mark it
                self.add_lora_data()
            self.merged = True 


class LinearLoRA(nn.Linear, LoRALayer):
    # LoRA implemented in a Linear layer
    def __init__(
        self, 
        existing_linear: nn.Linear,
        r: int = 0, 
        lora_alpha: int = 1, 
        fan_in_fan_out: bool = False,
        dropout_rate = 0.,
        **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features, 
            out_features=existing_linear.out_features)
        
        self.load_state_dict(existing_linear.state_dict())
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        # Actual trainable parameters
        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def train(self, mode: bool = True):
        super().train(mode)     
        self.lora_train(mode)

    def forward(self, x: torch.Tensor, **kwargs):
        # The original implementation had a bug/inefficiency here.
        # This corrected version is much more memory-friendly.

        # Compute the original linear transformation with frozen weights
        original_output = F.linear(x, self.transpose(self.weight), self.bias)

        # Compute and add the LoRA adjustment
        if self.r > 0 and not self.merged:
            
            # Apply dropout to the input if in training mode
            if self.training and self.dropout is not None:
                x_dropout = self.dropout(x)
            else:
                x_dropout = x

            # Get the LoRA matrices. Note: lora_A is already (r, in_features)
            lora_A = self.w_lora_A 
            lora_B = self.w_lora_B
            
            # Efficient computation: (x @ A.T) @ B.T
            # This keeps the intermediate matrix size small (batch_size, r)
            lora_adjustment = (x_dropout @ lora_A.transpose(0, 1)) @ lora_B.transpose(0, 1) * self.scaling
            
            result = original_output + lora_adjustment
        else:
            result = original_output
            
        return result
    # def forward(self, x: torch.Tensor, **kwargs):
        
    #     if self.dropout is None: # do as before
    #         if self.r > 0 and not self.merged:
    #             self.merge_lora_param()
    #             result = nn.Linear.forward(self, x, **kwargs)
    #             self.sub_lora_data()
    #             return result
    #         else:
    #             return nn.Linear.forward(self, x, **kwargs)
            
    #     # Compute the original linear transformation
    #     original_output = nn.Linear.forward(self, x)

    #     if self.training and self.dropout.p > 0:
    #         x = self.dropout(x)
        
    #     if self.r > 0 and not self.merged:
    #         lora_adjustment = torch.matmul(x,self.merge_BA('weight').transpose(0, 1)) * self.scaling 
    #         result = original_output + lora_adjustment
    #     else:
    #         result = original_output
    #     return result


def replace_linear_with_lora(model : torch.nn.Transformer, target_names: list[str], r: int=8, alpha: int=32, lora_dropout: float=0.1):
    for name, mod in model.named_modules():
        if any(target in name for target in target_names) and isinstance(mod, torch.nn.Linear):
            lora = LinearLoRA(mod, r, alpha, dropout_rate=lora_dropout)
            lora.train()
            # Need to use set_param or navigate the hierarchy to replace
            parent_name, child_name = name.rsplit('.', 1)
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, lora)





class CLIPLoRAImageEncoder(ImageEncoder):
    """CLIP-based image encoder implementation."""

    def __init__(self, CLIP_name: str, Pretrained: str, weights_path: str , r: int = 16, alpha: int = 32):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_name, Pretrained
        )
        self.model_name = CLIP_name + "-LoRA"
        self.pretrained_name = Pretrained
        replace_linear_with_lora(self.model.visual.transformer, target_names=["out_proj", "c_fc", "c_proj"], r=r, alpha=alpha, lora_dropout=0.5) # type:ignore
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))
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