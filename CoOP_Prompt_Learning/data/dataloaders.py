import os
import random
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision import datasets

def _targets_list(ds):
    return getattr(ds, "targets", [y for _, y in ds.samples])

def _prettify_classnames(names):
    return [n.replace("_", " ").replace("-", " ").strip() for n in names]

def _remap_dataset_to(train_ctoi, other_ds):
    other_ctoi = other_ds.class_to_idx
    mapping = {other_ctoi[name]: train_ctoi[name] for name in other_ctoi.keys()}
    class Remap(torch.utils.data.Dataset):
        def __init__(self, base, map_idx):
            self.base = base
            self.map = map_idx
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            return x, self.map[y]
        @property
        def classes(self):
            return self.base.classes
    return Remap(other_ds, mapping)

def _balanced_subset_indices(targets, limit: int, seed: int) -> List[int]:
    num_samples = len(targets)
    classes = sorted(set(targets))
    C = len(classes)
    if limit <= 0 or limit >= num_samples:
        return list(range(num_samples))
    per_class = {c: [] for c in classes}
    for idx, y in enumerate(targets):
        per_class[int(y)].append(idx)
    rng = random.Random(seed)
    for c in classes:
        rng.shuffle(per_class[c])
    n_base = max(1, limit // C)
    sel = []
    for c in classes:
        sel.extend(per_class[c][:n_base])
    remaining = limit - len(sel)
    if remaining > 0:
        pool = [i for c in classes for i in per_class[c][n_base:]]
        rng.shuffle(pool)
        sel.extend(pool[:remaining])
    rng.shuffle(sel)
    return sel

def build_imagefolder_splits(
    data_root: str,
    preprocess_train,
    preprocess_val,
    batch_size_train: int = 64,
    batch_size_eval: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    limit_train: int = 0,
):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "valid")
    test_dir = os.path.join(data_root, "test")
    for d in [train_dir, val_dir, test_dir]:
        assert os.path.isdir(d), f"Manca la cartella: {d}"

    train_full = ImageFolder(train_dir, transform=preprocess_train)
    val_full = ImageFolder(val_dir, transform=preprocess_val)
    test_full = ImageFolder(test_dir, transform=preprocess_val)

    val_full = _remap_dataset_to(train_full.class_to_idx, val_full)
    test_full = _remap_dataset_to(train_full.class_to_idx, test_full)

    assert set(train_full.classes) == set(val_full.classes)
    assert set(train_full.classes) == set(test_full.classes)

    if limit_train and limit_train > 0:
        targets = _targets_list(train_full)
        indices = _balanced_subset_indices(targets, limit_train, seed)
        train_ds = Subset(train_full, indices)
    else:
        train_ds = train_full

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True, generator=g,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_full, batch_size=batch_size_eval, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_full, batch_size=batch_size_eval, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    classnames = _prettify_classnames(train_full.classes)
    return train_loader, val_loader, test_loader, classnames


def build_dataloaders(
    data_root: str,
    preprocess,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    ds = datasets.ImageFolder(root=data_root, transform=preprocess)
    classnames = ds.classes
    n_total = len(ds)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, classnames
