###########
########### This file is an example of a simple training.
########### Keep in mind in this case this may seem like I added a lot of complexity for nothing.
########### In reality all of this was tought to enable you to test multiple configurations at a time.
########### In fact it allows for a list of ImageEncoders, DatasetConfig, Experiment (which also allows a list of model configs)
###########
########### Check the ohter file (main.py), in which I actually use it's full potential
###########
########### 
###########



from __future__ import annotations
from image_encoders.ImageEncoder import ImageEncoder, ImageEncoderConfig    # My wrapper for Image Encoders
from image_encoders.CLIP import CLIPImageEncoder                            # Inherits from ImageEncoder and it support all open_clip models
from GenericModels import BaseModel, ModelConfig, TrainingConfig, Metrics, Experiment   # Parent class for all Classification models
from TorchModels import MLP, MLPConfig, LRSchedulerConfig, OptimizerConfig, HiddenLayer, loss_w_func  # File that contains all torch Model (inherit from GenericModels), import the MLP
from DatasetsUtils import SplittedDatasetConfig                             # Configuration for dataset, handles both the ImageFolder and the Embeddings Dataset
from Logger import FileLogger                               # Used to log metrics and performance (FileLogger for files WandbLogger for W&B)
import os
import gc
import torch
from dataclasses import dataclass
from torchvision.datasets import ImageFolder, VisionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2                       # For the preprocessing/tranforms
import logging                                              # logging in console that can be turned on/off (I use logging.info)
import numpy as np          
from tqdm import tqdm 
import copy



# For console logging, remove if you don't want console output (no prints are used in the code, only logging.info)
logging.basicConfig(
    level=logging.INFO, # Set the minimum level of messages to show
    format="%(asctime)s [%(levelname)s] %(message)s", # Include timestamp and level
)





# Configuration for the training, you can specify loggers to use but here I don't.
# I stopped using FileLogger because i don't like how I implemented and generalized it now.
train_config = TrainingConfig(
    epochs = 300,                           # Max numbers of epochs to run (the one also showed in the progress bar)
    seed = 42,                              # To remove randomness, right now not well used and I don't insure consistency over runs
    early_stop = 60,                        # After how many epochs to stop training after not seeing any improvement in evaluation accuracy
    num_workers = 4,                        # num_workers for DataLoaders
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu"),    # device to use for all (embedding computing and training)
    tqdm_progress = True,                   # enable/disable progress bars
    loggers= [],#[FileLogger,WandBLogger],  # where you would specify the loggers for training/eval metrics and performance (list because can use multiple)
    project_name= "ArchitecturalStyles") # project_name is for loggers


script_dir = os.path.dirname(os.path.abspath(__file__))  # current absolute bath (doesn't work in notebooks, use  os.path.dirname(os.path.abspath('')) )


torch.manual_seed(train_config.seed)   # as i said to remove randomness over runs  but here it's not sufficient 
np.random.seed(train_config.seed)

#################################
#                               #
#                               #
#            Settings           #
#                               #
#                               #
#################################

# Directory in which all the datasets are stored
DATASETS_BASE_PATH: str = os.path.join(script_dir,"../datasets/")

# Directory where to cache the embeddings (computing emebeddings takes like 20 mins, one classifier training less than 1 minute)
EMBEDDINGS_CACHE_DIR = os.path.join(script_dir,"./.cache/")



# Configuration of the Datasest, handles both the Image dataset and the loading of the embeddings Tensors for each dataset
dataset_config =  SplittedDatasetConfig(name="architectural-styles-dataset-spl-warp-lanczos",dataset_root_path=os.path.join(DATASETS_BASE_PATH,"architectural-styles-dataset-spl-warped"), preprocessing = None,)# desired_splits={"train": 0.84,"valid": 0.08,"test": 0.08} ,oversampling=False),


# Configuration to use to create the encoder, see ImageEncoder API if you want more details
enc_config = ImageEncoderConfig(CLIPImageEncoder, {"CLIP_name":"ViT-bigG-14-worldwide","Pretrained":"metaclip2_worldwide"})

# Configurations specific to MLP
scheduler = LRSchedulerConfig( torch.optim.lr_scheduler.CosineAnnealingLR , {"T_max":150})      # Learning rate sheduler, decreases the learning rate over the run
optimizer = OptimizerConfig(torch.optim.AdamW,{"lr":0.009, "weight_decay":0.137})               # Optimizer used for the training
hidden_layers = hidden_layers=[HiddenLayer(1280,torch.nn.ReLU),HiddenLayer(480,torch.nn.GELU)]  # Hidden layers of the MLP, specify theirs size and activation function

# Configuration of the Experiment (One experiment allows to test the classifier with a list of configurations)
classifier_exp = Experiment(MLP, [MLPConfig(hidden_layers=hidden_layers, loss_func=torch.nn.CrossEntropyLoss, optimizer= optimizer, dropout_rate=0.53, batch_size=256 ,loss_weights_func=loss_w_func, lr_sched=scheduler)])




#################################################################
#                                                               #
#                                                               #
#      Computation of all the embeddings not already cached     #
#                                                               #
#       Cache dir structure:                                    #
#            <EMBEDDINGS_CACHE_DIR>/                            #
#                        encoder_name[0]/                       #  
#                            dataset_name[0]/                   #
#                                train/                         #
#                                    imgemb.pt                  #
#                                    labels.pt                  #
#                                test/                          #
#                                    imgemb.pt                  #
#                                    labels.pt                  #
#                                valid/                         #
#                                    imgemb.pt                  #
#                                    labels.pt                  #
#                            dataset_name[1]/                   #
#                                .....                          #
#                        encoder_name[1]/                       #
#                            ......                             #
#                                                               #
#                                                               #
#                                                               #
#################################################################

# returns the base root of the caching folder for the encoder 
def get_emb_cache_path(enc_name: str) -> str:
    return os.path.join(script_dir,EMBEDDINGS_CACHE_DIR,enc_name)


# Function to compute the embeddings of a VisionDataset using any ImageEncoder 
def compute_embeddings(ds: VisionDataset,
        img_enc : ImageEncoder,                   
        device: torch.device = torch.device('cpu'),
        batch_size: int = 32,
        num_workers: int = 4,
        tqdm_progress: bool = True
    ) :

    num_workers = train_config.num_workers
    tqdm_progress = train_config.tqdm_progress
    device = train_config.device

    logging.info(f"Computing embeddings with encoder {img_enc.get_model_name}...")

    # if a transform is defined the ImageEncoder's preprocessing needs to be appended
    if ds.transform is not None and img_enc.preprocessing is not None:            
        ds.transform = v2.Compose([ds.transform,v2.ToPILImage(),img_enc.preprocessing])  
    else:
        ds.transform = img_enc.preprocessing

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl = tqdm(dl,disable=(not tqdm_progress))
    
    img_enc = img_enc.to(device)
    embeddings = []     # stores the tensors of the embeddings in a list of batches of embeddings [ (batch_size, out_emb_size), (batch_size, out_emb_size) ]
    img_labels = []     # same as embeddings but for the labels
    enc_dtype = img_enc.model_dtype

    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(device, dtype = enc_dtype)
            embeddings.append( img_enc( imgs))   # Uses forward to compute embeddings and append to list
            img_labels.append(labels)
            
    img_embedding = torch.cat(embeddings,dim=0)  # Go from list[ (batch_size, out_size), ... ] to Tensor(num_samples, out_size)
    img_labels = torch.cat(img_labels, dim=0).long() # Same but also convert to long (integer)
    
    # This cleans the memory of both the GPU and the RAM
    img_enc = img_enc.to('cpu')
    torch.cuda.empty_cache()
    del dl
    gc.collect()
    logging.info(f"Finished computing embeddings with encoder: {img_enc.get_model_name}")
    return img_embedding, img_labels


# Load the encoder
enc = enc_config.encoder_cls(**enc_config.params)
# get the cache root for the encoder and check it's existance
root_path = get_emb_cache_path(enc.get_model_name)
if not os.path.isdir(root_path):     # if the dir doesn't exist make it 
    os.makedirs(root_path)

# builds the internal cache paths used for the dataset
dataset_config.build_cache_paths(root_path)
for split, (img_path, lbl_path) in dataset_config.emb_paths.items():        
    
    # check if each split's cache folder exists or create it 
    if not os.path.isdir(os.path.dirname(img_path)):
        os.makedirs(os.path.dirname(img_path))

    # if either the imgage embedding or labels cache files don't exist for this dataset, encoder, split combination
    # then compute the embeddings
    if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
        ds = dataset_config.splits_img_ds[split]
        
        logging.info(f"Computing the embeddings for Dataset {dataset_config.name} , encoder {enc.get_model_name} , split {split}")
        img_emb, labels=compute_embeddings(ds,enc,train_config.device)
        
        logging.info("Embeddings computed")
        torch.save(img_emb,img_path)
        torch.save(labels,lbl_path)


logging.info("All the embeddings are cached")






#########################
#                       #
#                       #
#       Training        #
#                       #
#                       #
#########################


# Basically classifier = MLP( config= MLPConfig, input_size = encoder_output_size, num_classes, TrainingConfig)
classifier = classifier_exp.model_cls(config = classifier_exp.configs[0], input_size = enc.output_emb_size[0], num_classes = dataset_config.num_classes, train_config = train_config )



dataset_config.load_embeddings(get_emb_cache_path(enc.get_model_name))


class_names=dataset_config.classes

# Train the model using the train and valid split (valid is only used for validation, so computing performance measures)
# Also saves the best state, not the last
classifier.fit(
    X = dataset_config.emb_splits["train"][0],
    y = dataset_config.emb_splits["train"][1],
    X_val = dataset_config.emb_splits["valid"][0],
    y_val = dataset_config.emb_splits["valid"][1],
    loggers=[]
)

# Evaluate the model
logging.info(classifier.evaluate(X=dataset_config.emb_splits["valid"][0],y=dataset_config.emb_splits["valid"][1], loggers=[]).to_dict)



