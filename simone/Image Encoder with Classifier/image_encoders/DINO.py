from image_encoders.ImageEncoder import ImageEncoder
import torch
from transformers import AutoImageProcessor, AutoModel
import os
import torch.hub
from typing import cast 
from torchvision.transforms import v2



class DinoV3ImageEncoder(ImageEncoder):

    def __init__(self, model_name:str, models_directory = "DINOV3_checkpoints/", repo_clone_dir = "DINOV3_repo/"):
        super().__init__()
        # pretrained_name="facebook/"+model_name
        # self.preprocess = AutoImageProcessor.from_pretrained(pretrained_name, use_fast=True)
        # self.model=AutoModel.from_pretrained(pretrained_name)
        self.model_name = model_name
        weights_path = models_directory+model_name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir,weights_path)
        repo_clone_dir = os.path.join(script_dir,repo_clone_dir)
        assert os.path.isfile(weights_path) , print(f"Model not found in the {models_directory}.\nCheck if the file {weights_path}  exists.\n",
                                                    "Maybe you need to dowload it, check the download guide at https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-models") 
        assert os.path.isdir(repo_clone_dir), print(f"Could not find the DINOV3 github repo clone in {repo_clone_dir}\n",
                                                    "Please change the path or clone the repository with 'git clone https://github.com/facebookresearch/dinov3.git' inside the directory")
        
        self.model =torch.hub.load(repo_or_dir=repo_clone_dir, model=model_name,trust_repo='True', source="local", weights=weights_path) # type: ignore
        self.model : torch.nn.Module= cast( torch.nn.Module, self.model)
        self.dataset_name = "lvd1689m" if model_name.find("lvd1689m") != -1 else "sat493m"
        self.model.eval()
        

    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through the image encoder.
        
        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Encoded image features tensor

        """
        with torch.no_grad():
            return self.model(images) 

    @property 
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters()) +  sum(p.numel() for p in self.model.buffers())
        
    @property    
    def model_dtype(self) -> torch.dtype:
        """Return the data type of the model parameters."""
        return next(self.model.parameters()).dtype 

    @property    
    def image_input_size(self) -> tuple[int, int]: 
        """Return expected input image size (height, width)."""
        if hasattr(self.model, 'image_size'):
            size = self.model.image_size # type:ignore
            # Handle both single int and tuple cases
            if isinstance(size, int):
                return (size, size)
            return tuple(size) if len(size) == 2 else (size[0], size[0]) # type:ignore
        else:
            # Default fallback - you might want to adjust this (224,244 is the default for CLIP in the paper)
            return (-1, -1)
        
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
        resize_size: int = 224  # Hyperparam ??
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)

        if self.dataset_name=="lvd1689m":
            normalize = v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        else:
            normalize = v2.Normalize(
                mean=(0.430, 0.411, 0.296),
                std=(0.213, 0.156, 0.143),
            ) 
        return v2.Compose([to_tensor, resize, to_float, normalize])

    @property
    def get_model_name(self)-> str:
        """Return the name of the model, mainly for printing"""
        return self.model_name

    def to(self,device="cpu"):
        if "cuda" not in device:
            super().to(device)
            return
        cuda_list=[]
        if "cuda" in device:
            model_size= torch.finfo(self.model_dtype).bits if self.model_dtype.is_floating_point else torch.finfo(self.model_dtype).bits
            model_size= round(model_size/8)
            model_size*=self.num_parameters
            curr_free_mem=torch.cuda.mem_get_info("cuda")[0]
            if torch.cuda.device_count() == 1 or model_size < curr_free_mem :
                super().to(device)
            else:
                # split model in two devices
                # set flags/parameters to use in the forward to move params
                pass
    




# sistema image_size in modo che sia compatibile con la parte di preprocessing
#