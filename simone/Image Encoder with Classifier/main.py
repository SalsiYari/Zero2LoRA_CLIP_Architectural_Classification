# LIST ALL DATASETS 
# LIST ALL IMAGE ENCODERS
# LIST ALL CLASSIFIERS
# MAKE A FILE PATH FOR CACHED EMBEDDINGS/LABELS for (DATASET,SPLIT,ENCOER) 
# CHECKE IF EMBEDDINGS/LABELS ARE ALREADY CACHED 
# IF NOT COMPUTE THE EMBEDDINGS AND SAVE THEM 
# FOR EACH DATASET
#    FOR EACH ENCODER 
#    LOAD THE EMBEDDINGS/LABELS
#    FOR EACH CLASSIFIER
#        FOR EACH SETTINGS (LOSS_FUNC, OPTIMIZER, LEARNING_RATE, .....)
#            TRAIN ON THE EMBEDDINGS/LABELS
#            (SAVING THE BEST MODEL FOR EACH COMBINATION USING A DIFFERENT PATH THAT EXMPLAINS THE SETTINGS)
#            ALSO MAKE A CSV FILE TO COMPARE THE RESULTS FOR EACH MODEL COMBINATION
#########################################
#                                       #
#                                       #
#               IMPORTS                 #
#                                       #
#                                       #
#########################################

from __future__ import annotations
from image_encoders.ImageEncoder import ImageEncoder, ImageEncoderConfig    # My wrapper for Image Encoders
from image_encoders.CLIP import CLIPImageEncoder                             # Inherits from ImageEncoder and it support all open_clip models
from image_encoders.DINO import DinoV3ImageEncoder
from GenericModels import BaseModel, ModelConfig, TrainingConfig, Metrics, Experiment   # Parent class for all Classification models
from TorchModels import MLP, MLPConfig, LRSchedulerConfig, OptimizerConfig, HiddenLayer, loss_w_func  # File that contains all torch Model (inherit from GenericModels), import the MLP
from DatasetsUtils import SplittedDatasetConfig, DatasetConfig              # Configuration for dataset, handles both the ImageFolder and the Embeddings Dataset
from Logger import FileLogger,WandBLogger                                   # Used to log metrics and performance (FileLogger for files WandbLogger for W&B)
import torch, torchvision
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder, VisionDataset
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm 
import numpy as np          
from dataclasses import dataclass
import gc
import pandas as pd


# remove if you don't want almost any print to console
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
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu"),    # device to use for all (embedding computing and training)
    tqdm_progress = True,                   # enable/disable progress bars
    loggers= [],#[WandBLogge,FileLogger],  # do not use FileLogger before fixing it, files don't get saved and the way the lcoations are defined is bad (find better generalization/definition)
    project_name= "ArchitecturalStyles") # project_name is for loggers



script_dir = os.path.dirname(os.path.abspath(__file__))

# to remove randomness over runs  but it's not sufficient to define it here as the randomness is used each training so it's not accurate right now
torch.manual_seed(train_config.seed)   
np.random.seed(train_config.seed)
#############################
#                           #
#                           #
#   Settings for the files  #
#                           #
#                           #
#############################

# Directory in which all the datasets are stored
DATASETS_BASE_PATH: str = os.path.join(script_dir,"../datasets/")
# Dataset names are actually the datasets' folders names


DATASETS_CONFIGS: list[DatasetConfig]=[
   # SplittedDatasetConfig(name="ArchiStyles", dataset_root_path=os.path.join(DATASETS_BASE_PATH,"ArchiStyles"), ),
    # SplittedDatasetConfig(name="architectural-styles-dataset-spl",dataset_root_path=os.path.join(DATASETS_BASE_PATH,"architectural-styles-dataset-spl"), preprocessing = None,),
    # SplittedDatasetConfig(name="architectural-styles-dataset-spl-black-pad",dataset_root_path=os.path.join(DATASETS_BASE_PATH,"architectural-styles-dataset-spl-black-pad"), preprocessing = None,),
    # SplittedDatasetConfig(name="architectural-styles-dataset-spl-warped",dataset_root_path=os.path.join(DATASETS_BASE_PATH,"architectural-styles-dataset-spl-warped"), preprocessing = None,),
    SplittedDatasetConfig(name="architectural-styles-dataset-spl-warp-lanczos",dataset_root_path=os.path.join(DATASETS_BASE_PATH,"architectural-styles-dataset-spl-warped"), preprocessing = None,),

]




# Directory used to cache the images embeddings so that once they are computed you don't need to recompute them
#
# This was done because before each training it needed to be done and it required a lot of time compared to the
# actual training time of the classifier, especially for the bigger image encoders
EMBEDDINGS_CACHE_DIR = os.path.join(script_dir,"./.cache/")



#################################################
#                                               #
#                                               #
#  Definition of all the image encoders and     #
#  Classifiers combinations that will be used   #
#                                               #
#                                               #
#################################################

# I decided to use ImageEncoderConfig because otherwise all the encoders would be initialized and loaded using a lot of memory
ENCODERS_CONFIGS : list[ImageEncoderConfig] = [     # List of the Encoders to try
    #ImageEncoderConfig(CLIPImageEncoder, {"CLIP_name":"ViT-L-14","Pretrained":"openai"}),
    #ImageEncoderConfig(CLIPImageEncoder, {"CLIP_name":"ViT-bigG-14-worldwide","Pretrained":"metaclip2_worldwide"}),
    ImageEncoderConfig(DinoV3ImageEncoder,{"model_name": "dinov3-vit7b16-pretrain-lvd1689m" }),
]

@dataclass
class encoder_info:             # Used to store some basic information about each ImageEncoder
    model_name: str
    output_emb_size : torch.Size


encoders_infos : list[encoder_info] = []        # This gets filled during the check of the embeddings caching. It is used to not have to load the encoder just to get it's name

scheduler_config = LRSchedulerConfig( torch.optim.lr_scheduler.CosineAnnealingLR , {"T_max":150})   # Specific to MLP, it's a configuration for learning rate scheduler

L = HiddenLayer         # to use less space   # defines the hidden layer size and it's activation function

CLASSIFIER_CONFIGS: list[Experiment] = [        # List of the classifiers and classifiers configs to try
    Experiment(MLP, [ # In this example they are Linear models as there are no hidden layers
                MLPConfig( hidden_layers = [],  loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.0311, "weight_decay":0.5867}), dropout_rate= 0.0, batch_size= 32 , loss_weights_func= loss_w_func, lr_sched= scheduler_config),
                MLPConfig( hidden_layers = [],  loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.0320, "weight_decay":0.5638}), dropout_rate= 0.0, batch_size= 32 , loss_weights_func= loss_w_func, lr_sched= scheduler_config),
                MLPConfig( hidden_layers = [],  loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.0486, "weight_decay":0.5635}), dropout_rate= 0.0, batch_size= 64 , loss_weights_func= loss_w_func, lr_sched= scheduler_config),
                ]),
    # Experiment(MLP, [
    #             MLPConfig( hidden_layers = [L(896,torch.nn.ReLU), L(112,torch.nn.ReLU)],  loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.0035, "weight_decay":0.187}), dropout_rate= 0.25, batch_size= 32 , loss_weights_func= loss_w_func, lr_sched= scheduler_config),
    #             MLPConfig( hidden_layers = [L(1280,torch.nn.ReLU),L(1120,torch.nn.ReLU)], loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.0061, "weight_decay":0.138}), dropout_rate= 0.5,  batch_size= 64 , loss_weights_func= loss_w_func, lr_sched= scheduler_config ),
    #             MLPConfig( hidden_layers = [L(1280,torch.nn.ReLU),L(480,torch.nn.GELU)],  loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.009,  "weight_decay":0.137}), dropout_rate= 0.53, batch_size= 256, loss_weights_func= loss_w_func, lr_sched= scheduler_config),
    #             MLPConfig( hidden_layers = [L(1152,torch.nn.ReLU),L(144,torch.nn.ReLU)],  loss_func= torch.nn.CrossEntropyLoss, optimizer= OptimizerConfig( torch.optim.AdamW,{"lr":0.0071, "weight_decay":0.116}), dropout_rate= 0.33, batch_size= 64 , loss_weights_func= loss_w_func, lr_sched= scheduler_config),
    #              ]),
    # Experiment for different Classifier ......
]

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
logging.info("Computing all the embeddings for all dataset-encoder combinations that are not already cached")
logging.info("This may take a long time...")


def get_emb_cache_path(enc_name: str) -> str:                # returns the base root of the caching folder for the encoder 
    return os.path.join(script_dir,EMBEDDINGS_CACHE_DIR,enc_name)



def compute_embeddings(ds: VisionDataset,                 # Function to compute the embeddings of a VisionDataset using any ImageEncoder 
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

    if ds.transform is not None and img_enc.preprocessing is not None:            # if a transform is defined the ImageEncoder's preprocessing needs to be appended
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
    
    # Next lines clean the memory of both the GPU and the RAM
    img_enc = img_enc.to(torch.device('cpu'))
    torch.cuda.empty_cache()
    del dl
    gc.collect()
    logging.info(f"Finished computing embeddings with encoder: {img_enc.get_model_name}")
    return img_embedding, img_labels



# For each ImageEncoder: load it, fill the encoders_infos 
# For each DatasetConfig 
# Check if the embeddings are cached for each split, if not compute the emebddings
for enc_conf in ENCODERS_CONFIGS:                # for each encoder
    enc= enc_conf.encoder_cls(**enc_conf.params)
    encoders_infos.append( encoder_info(enc.get_model_name,enc.output_emb_size))
    root_path= get_emb_cache_path(enc.get_model_name)
    if not os.path.isdir(root_path):     # if the dir doesn't exist make it 
        os.makedirs(root_path)

    for ds_config in  DATASETS_CONFIGS:   # for each dataset
   
        ds_config.build_cache_paths(root_path)

        for split, (img_path, lbl_path) in ds_config.emb_paths.items():    # check if each split folder exists and contains it's img_emb and labels files
            
            if not os.path.isdir(os.path.dirname(img_path)):   # If not existing create the directory
                os.makedirs(os.path.dirname(img_path))

            # if either the imgage embedding or labels cache files don't exist for this dataset, encoder, split combination
            # then compute the embeddings
            if not os.path.isfile(img_path) or not os.path.isfile(lbl_path):
                ds = ds_config.splits_img_ds[split]
                
                logging.info(f"Computing the embeddings for Dataset {ds_config.name} , encoder {enc.get_model_name} , split {split}")
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

logging.info("Starting training")
for dconf in DATASETS_CONFIGS:          # for each dataset

    for enc_conf, enc_info in zip(ENCODERS_CONFIGS,encoders_infos):   # for each encoder
        
        dconf.load_embeddings(get_emb_cache_path(enc_info.model_name))   # load the embeddings of the dataset, encoder combination

        for exp in CLASSIFIER_CONFIGS:   # for each classifier
            for cfg in exp.configs:         # for each classifier config

                # initialize the classifier
                classifier = exp.model_cls(cfg, input_size= enc_info.output_emb_size[0],num_classes= dconf.num_classes,train_config = train_config )

                
                logging.info(f"Starting training for {classifier.model_name()} with dataset {dconf.name} and encoder: {enc_info.model_name}")


                class_names=dconf.classes
                loggers=[]
                if train_config.loggers is not None:      # Initialize the loggers 
                    for logger in train_config.loggers:
                        run = enc_info.model_name + "+"+classifier.model_name()
                        loggers.append(logger(train_config.project_name, run_name= run, config=train_config.to_dict | cfg.to_dict | {"Dataset":dconf.to_dict()}, class_names=class_names))

                # train the model
                # use both the train and eval splits. The eval split is used only for evaluation and so to measure performance metrics and for early stopping
                classifier.fit(
                    X = dconf.emb_splits["train"][0],
                    y = dconf.emb_splits["train"][1],
                    X_val = dconf.emb_splits["valid"][0],
                    y_val = dconf.emb_splits["valid"][1],
                    loggers=loggers
                )

                # evaluation with print
                logging.info(classifier.evaluate(X=dconf.emb_splits["valid"][0],y=dconf.emb_splits["valid"][1], loggers=loggers).to_dict)

                if train_config.loggers is not None:   # end the loggers
                    for logger in loggers:
                        logger.finish()

                test_metrics = classifier.evaluate(X=dconf.emb_splits["test"][0],y=dconf.emb_splits["test"][1], loggers=[]).to_dict
                test_metrics['Model'] = enc_info.model_name
                test_metrics['classifier']= 'Multi Layer Perceptron'
                test_metrics|= classifier.conf.to_dict["optimizer"]['params']
                test_metrics['batch_size']=classifier.conf.to_dict['batch_size']
                df = pd.DataFrame([test_metrics])
                CSV_PATH = "benchmark.csv"
                if os.path.exists(CSV_PATH):
                    old = pd.read_csv(CSV_PATH)
                    out = pd.concat([old, df], ignore_index=True)
                else:
                    out = df.copy()
                out.to_csv(CSV_PATH, index=False)

logging.info("Everything finished, script end")



