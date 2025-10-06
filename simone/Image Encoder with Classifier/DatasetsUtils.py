from __future__ import annotations
from torch.utils.data import Dataset, Subset, random_split, TensorDataset
from dataclasses import dataclass
from typing import Optional, Callable
from torchvision.datasets import ImageFolder,VisionDataset
from torchvision.transforms import functional
import numpy as np
import torch
import os
from abc import ABC,abstractmethod
from PIL import Image, ImageOps
import dill
import hashlib

class DatasetConfig(ABC):
    name : str
    path: str 
    num_classes: int 
    classes: list[str] 
    splits_img_ds : dict[str, ImageFolder] 
    emb_paths: dict[str, tuple[str,str]] 
    emb_splits: dict[str, tuple[torch.Tensor, torch.Tensor]] 
    preprocessing: Optional[Callable]
    train_augmentation: Optional[Callable]

    def __init__(self, name : str, dataset_root_path: str, preprocessing: Optional[Callable], train_augmentation: Optional[Callable] ):
        self.name = name
        self.path = dataset_root_path
        self.splits_img_ds={}
        self.emb_splits = {}
        self.emb_paths = {}
        assert os.path.isdir(self.path), f"The Dataset {self.name} was not found in: {self.path}"
        self.preprocessing=preprocessing
        self.train_augmentation=train_augmentation

    def build_cache_paths(self, cache_root):
        base = remove_chars(self.name)
        if self.preprocessing is not None:
            base += remove_chars(hashlib.sha256(dill.dumps(self.preprocessing, protocol=4)).hexdigest())
        for spl in self.splits_img_ds.keys():
            self.emb_paths[spl] = (os.path.join(cache_root,base,spl,"imgemb.pt"),os.path.join(cache_root,base,spl,"labels.pt"))
    
    @abstractmethod
    def load_embeddings(self, cache_root_path: str):
        pass

    def to_dict(self) -> dict:
        return {"Name":self.name, 
                "preprocessing": self.preprocessing.__name__ if self.preprocessing is not None else None,
                "train_augmentation": self.train_augmentation.__name__ if self.train_augmentation is not None else False}


class  SplittedDatasetConfig(DatasetConfig):
    """Imports a ImageFolder dataset that is already splitted:
    
    preprocessing: Transforms applied on all images (all splits) before computing the embeddings
    train_augmentation_transform: Transform for data augmentation (should be random transforms),
                                 applied on the training embeddings to add "noise" to the embeddings
                                 to augment the dataset.
    """
    def __init__(self, name : str, dataset_root_path: str, splits : list[str] = ["train", "valid", "test"], preprocessing: Optional[Callable]= None, train_augmentation: Optional[Callable]= None):
        super().__init__(name, dataset_root_path, preprocessing, train_augmentation)
        
        for split in splits:
            split_path = os.path.join(self.path,split)
            assert os.path.isdir(split_path), f"The {split} split was not found for Dataset {self.name} : {split_path}"
            if preprocessing is not None:
                ds  = ImageFolder(split_path, transform=preprocessing)
            else:
                ds  = ImageFolder(split_path)
            self.classes = ds.classes
            self.num_classes = len(ds.classes)
            self.splits_img_ds[split] = ds

    def load_embeddings(self, root_path: str):
        self.build_cache_paths(root_path)
        for split, (img_path, lbl_path) in self.emb_paths.items():
            img_emb=torch.load(img_path)
            lbl_emb=torch.load(lbl_path)
            if split == "train" and self.train_augmentation is not None:
            #    self.emb_splits[split]= (self.tensTransform(img_emb), lbl_emb)
                print("ATTENTO EHFUOBRUEOBO")
            else:
                self.emb_splits[split]= (img_emb, lbl_emb) 



class  RawDatasetConfig(DatasetConfig):
    """Imports a ImageFolder dataset that is not splitted:
    The entire dataset is the one on which the embedddings are computed.
    The one to get splitted is the TensorDataset for the embeddings to then go into training the classifier.
    The train augmentation is performed on the train split of the TensorDataset
    
    teain, test, valid splits: fraction of the dataset to use for each
    preprocessing: Transforms applied on all images (all splits) before computing the embeddings
    train_augmentation_transform: Transform for data augmentation (should be random transforms),
                                 applied on the training embeddings to add "noise" to the embeddings
                                 to augment the dataset.
    """
    desired_splits : dict[str, float]
    train_augmentation_transform: Optional[Callable]= None
 
    def __init__(self, name : str, dataset_root_path: str, preprocessing: Optional[Callable]= None, desired_splits: dict[str, float]={"train" : 1.0}, train_augmentation: Optional[Callable]= None, oversampling: bool=True ):
        super().__init__(name,dataset_root_path, preprocessing, train_augmentation)
        if preprocessing is not None:
            ds = ImageFolder(self.path, preprocessing)
        else: 
            ds = ImageFolder(self.path)
        self.classes = ds.classes
        self.num_classes = len(ds.classes)
        self.splits_img_ds[""] = ds
        self.desired_splits = desired_splits
        self.oversampling = oversampling
        
    
    def load_embeddings(self, root_path: str):
        self.build_cache_paths(root_path)
        img_emb=torch.load(self.emb_paths[""][0])
        lbl_emb=torch.load(self.emb_paths[""][1])
        ds = (img_emb, lbl_emb)
        self.splitBalancedAugmentation(ds)


    def to_dict(self) -> dict:
        return super().to_dict() |{
            "splits": self.desired_splits,
            "oversampling": self.oversampling
        }

    def splitBalancedAugmentation(self, tds: tuple[torch.Tensor, torch.Tensor]) :
        if  sum(self.desired_splits.values()) > 1.000001 :
            raise Exception(f"Sum of splits greater than 1.0")

        if  "train" not in self.desired_splits.keys() :
            raise Exception(f"The Dataset doesn't have a train split")

        # 
        # extract from the dataset the class of each sample and memorize for each class it's samples' indexes
        # class_indexes has dims (num_classes, ?) where ? is the number of samples for each class
        #
        classes_indexes: list[list[int]] = []
        for i in range(self.num_classes):
            classes_indexes.append([])

        i = 0
        for t in tds[1]:
            classes_indexes[t].append(i)
            i+= 1
            
        #
        # Count the number of samples per class
        #
        class_samples_count: list[int] = []
        for cl_ind in classes_indexes:

            class_samples_count.append(len(cl_ind))
            
        max_num = max(class_samples_count)

        #
        #   Compute the number of samples for each split
        #
        #valid_num = int(max_num*ds_config.valid_split)
        #train_num = max_num - test_num - valid_num

        nums = {}
        for spl, val in self.desired_splits.items():
            nums[spl] = int(val * max_num)
        
        nums_no_train = sum([v if k!="train" else 0 for k,v in nums.items()])
        nums["train"] = max_num - nums_no_train
        #
        # check if any class has less samples than the sum of the test + valid splits
        #
        for count in class_samples_count:
            if count < nums_no_train:
                raise Exception("Dataset Split exception: The number of samples for one of the classes is lower than test + valid")

        #
        #   Generate the permutations to use for test valid train splits
        #
        randoms_classes : list[np.ndarray] = []
        
        for num in class_samples_count:
            randoms_classes.append(np.random.permutation(num))

        perms = {} 
        tot = 0
        for k,v in nums.items():
            if k != "train":
                perms[k] = np.concatenate([np.array(inds)[r[tot : tot+v]] for r, inds in zip(randoms_classes, classes_indexes)])
                tot+= v
        if self.oversampling :
            for num, i in zip(class_samples_count, range(self.num_classes)):
                to_resample = nums["train"] - (num - nums_no_train )
                randoms_classes[i]=np.append(randoms_classes[i], np.random.choice(randoms_classes[i][ nums_no_train : ] , to_resample))

        perms["train"]= np.concatenate([np.array(inds)[r[nums_no_train:]] for r, inds in zip(randoms_classes, classes_indexes)])

        for k,v in perms.items():
            self.emb_splits[k]= (tds[0][v.tolist()], tds[1][v.tolist()])# if  k != " train" and seslf.train_augmentation is not None else TransformedTensorDataset(Subset(tds,v.tolist()),self.train_augmentation_transform)

    



def remove_chars(s: str) -> str:
    return s.replace(" ","").replace("/","").replace("\\","").replace(".","")



def exif_fix(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def trim_white_edges(img: Image.Image, tol: int = 250, edge_white_frac: float = 0.995) -> Image.Image:
    """Opzionale: rimuove margini bianchi dai bordi esterni."""
    im = img.convert("RGB")
    arr = np.asarray(im)
    h, w, _ = arr.shape
    white = (arr[...,0] >= tol) & (arr[...,1] >= tol) & (arr[...,2] >= tol)

    top, bottom, left, right = 0, h-1, 0, w-1
    while top < bottom and white[top, :].mean() >= edge_white_frac:   top += 1
    while bottom > top and white[bottom, :].mean() >= edge_white_frac: bottom -= 1
    while left < right and white[:, left].mean() >= edge_white_frac:   left += 1
    while right > left and white[:, right].mean() >= edge_white_frac:  right -= 1

    if right <= left or bottom <= top:
        return im
    return im.crop((left, top, right+1, bottom+1))

def warp_to_square_noresize(img: Image.Image, do_trim=False, tol=250) -> Image.Image:
    """
    Warp a quadrato SENZA ridimensionare a 224.
    Risultato: lato = max(width, height) originale.
    """
    im = exif_fix(img).convert("RGB")
    if do_trim:
        im = trim_white_edges(im, tol=tol)
    w, h = im.size
    s = max(w, h)
    return im.resize((s, s), resample=Image.Resampling.BICUBIC)



def letterbox_black_clip(img: Image.Image, trim_edges_first=True, tol=250) -> Image.Image:
    """
    - Rimuove i bordi bianchi esterni.
    - Crea un canvas quadrato nero con lato = max(w, h) dell'immagine.
    - Incolla l'immagine originale centrata, SENZA scalarla.
    """
    im = exif_fix(img).convert("RGB")
    if trim_edges_first:
        im = trim_white_edges(im, tol=tol)

    w, h = im.size
    s = max(w, h)  # lato del canvas quadrato

    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    off_x = (s - w) // 2
    off_y = (s - h) // 2
    canvas.paste(im, (off_x, off_y))
    return canvas



