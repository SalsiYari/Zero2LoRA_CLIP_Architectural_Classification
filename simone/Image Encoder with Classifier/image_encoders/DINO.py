from image_encoders.ImageEncoder import ImageEncoder
import torch
from transformers import AutoImageProcessor, AutoModel
import os
import torch.hub
from typing import cast 
from torchvision.transforms import v2
# pip install accelerate
from accelerate import infer_auto_device_map, dispatch_model



class HFProcessorToTensor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        batch = self.processor(image, return_tensors="pt")
        return batch["pixel_values"].squeeze(0)  # remove batch dim (1, C, H, W) -> (C, H, W)


class DinoV3ImageEncoder(ImageEncoder):

    def __init__(self, model_name:str, models_directory = "DINOV3_checkpoints/", repo_clone_dir = "DINOV3_repo/"):
        super().__init__()
        pretrained_name="facebook/"+model_name
        self.preprocess = AutoImageProcessor.from_pretrained(pretrained_name, use_fast=True)
        self.model=AutoModel.from_pretrained(pretrained_name, device_map="auto")
        self.model_name = model_name
        self.model.eval()
        

    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the image encoder.
        
        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded image features tensor

        """
        with torch.no_grad():
            return self.model(images).pooler_output 

    @property 
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
        
    @property    
    def model_dtype(self) -> torch.dtype:
        """Return the data type of the model parameters."""
        return next(self.model.parameters()).dtype 

    @property    
    def image_input_size(self) -> tuple[int, int]: 
        """Return expected input image size (height, width)."""
        image = torch.zeros(1,3,800,800)
        ret = self.preprocess(images=image, return_tensors="pt")
        return ret.pixel_values.shape[-2:]
        
    @property    
    def output_emb_size(self) -> torch.Size:
        """Return the output embedding size."""
        # Get model device and dtype
        device = next(self.model.parameters()).device #type:ignore
        dtype = self.model_dtype
        
        # Create a dummy input tensor
        height, width = self.image_input_size
        test_img = torch.zeros(1, 3, height, width, device=device, dtype=dtype)
        
        # Get output shape
        with torch.no_grad():
            out = self.forward(test_img)
        return out.shape[1:]  # Exclude batch dimension

    @property    
    def preprocessing(self): # type: ignore
        """Return the preprocessing function/transform for input images. (ignore the return type error)"""
        return v2.Compose([HFProcessorToTensor(self.preprocess),])
        
    @property
    def get_model_name(self)-> str:
        """Return the name of the model, mainly for printing"""
        return self.model_name

    # THIS IS NOT FULLY TESTED, CAN?T INSURE WORKING
    def to(self,device: torch.device=torch.device("cpu")):
       
        if device.type == 'cuda':
            if next(self.model.parameters()).device.type == 'cuda':
                return self
            
            super().to("cpu")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Calculate model size in bytes
            bits = torch.finfo(self.model_dtype).bits if self.model_dtype.is_floating_point else torch.iinfo(self.model_dtype).bits
            model_size = self.num_parameters * (bits // 8)

            # Check available GPU memory
            cuda_mem_list = {}
            for i in range(torch.cuda.device_count()):
                free_mem, _ = torch.cuda.mem_get_info(i)
                cuda_mem_list[i] = f"{int(free_mem / 1024 ** 3)}GiB"

            # Compute total free memory
            total_free = sum(int(v[:-3]) for v in cuda_mem_list.values()) * (1024**3)
            if model_size > total_free:
                raise RuntimeError(f"Model too large: needs {model_size/1e9:.2f} GB, but GPUs only have {total_free/1e9:.2f} GB total")

            
            # Dispatch automatically
            device_map = infer_auto_device_map(self.model, max_memory=cuda_mem_list)
            self.model = dispatch_model(self.model, device_map=device_map)

        else:
            super().to(device)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            for attr_name in dir(self.model):
                attr = getattr(self.model, attr_name)
                if torch.is_tensor(attr):
                    setattr(self.model, attr_name, attr.cpu())


        return self


