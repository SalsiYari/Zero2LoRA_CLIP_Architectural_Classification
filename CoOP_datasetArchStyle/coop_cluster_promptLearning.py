#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP + Prompt Learning & Prompt Ensemble su template forniti
-> Dataset: torchvision.datasets.ImageFolder (cartelle = classi)
-> Prompt learning: impara pesi sui template (globali o per-classe)
-> Prompt ensemble: uniform (media semplice) o learned (pesi appresi)
-> No modifica a CLIP: backbone congelato, si imparano solo i pesi dei template
"""

import argparse
import math
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

try:
    import open_clip
except ImportError as e:
    raise SystemExit(
        "Manca la dipendenza 'open_clip_torch'. Installa con:\n"
        "  pip install open_clip_torch\n"
        f"Errore originale: {e}"
    )

from tqdm import tqdm

DATA_ROOT = "/work/cvcs2025/carpaneto_salsi_serafini/ArchiStyles-2"
# =======================
#  TEMPLATE (come nostro standard)
# =======================
TEMPLATES: List[str] = [
    "a photo of {} architecture",
    "a photograph of {} architecture",
    "an image of {} architecture",
    "a picture of {} architecture",
    "a photo of the {} architectural style",
    "the {} architectural style",
    "a photo of a {} style building",
    "a {} style building",
    "a building in the {} style",
    "{}-style building",
    "a {} building",
    "a building of {} architecture",
    "a representation of {} architecture",
    "the facade of a {} building",
    "an example of {} architecture",
]


# ===========================================================
#  Utils
# ===========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================
#  Prompt Ensemble Module
# ===========================================================
class PromptEnsemble(nn.Module):
    """
    Mantiene le embedding testuali per ciascun template e classe, e
    impara i pesi dell'ensemble sui template (prompt learning).
    - Se per_class=True, i pesi sono [num_classes, num_templates]
    - Altrimenti sono [num_templates] globali
    """

    def __init__(
        self,
        model,
        classnames: List[str],
        templates: List[str],
        tokenizer,
        device: torch.device,
        per_class: bool = False,
        init_uniform: bool = True,
    ):
        super().__init__()
        self.model = model
        self.classnames = classnames
        self.templates = templates
        self.tokenizer = tokenizer
        self.device = device
        self.per_class = per_class

        with torch.no_grad():
            self.text_features_templates = self._build_text_features().to(device)
            # shape: [num_templates, num_classes, d]
            self.text_features_templates = nn.functional.normalize(
                self.text_features_templates, dim=-1
            )

        num_templates = len(self.templates)
        num_classes = len(self.classnames)

        if per_class:
            self.template_logits = nn.Parameter(torch.zeros(num_classes, num_templates))
        else:
            self.template_logits = nn.Parameter(torch.zeros(num_templates))

        # inizializzazione: opzionale bias verso uniforme
        if init_uniform:
            with torch.no_grad():
                if per_class:
                    self.template_logits[:] = 0.0  # softmax -> uniforme
                else:
                    self.template_logits[:] = 0.0

    @torch.no_grad()
    def _build_text_features(self) -> torch.Tensor:
        """
        Costruisce le feature testuali per ogni template e classe, senza grad.
        Ritorna tensore [num_templates, num_classes, d]
        """
        self.model.eval()
        all_feats = []
        for tmpl in self.templates:
            prompts = [tmpl.format(name.replace("_", " ")) for name in self.classnames]
            tokenized = self.tokenizer(prompts)
            tokenized = tokenized.to(self.device)
            feats = self.model.encode_text(tokenized)
            all_feats.append(feats)  # [num_classes, d]
        return torch.stack(all_feats, dim=0)

    def get_text_features_weighted(self) -> torch.Tensor:
        """
        Combina le embedding testuali lungo l'asse template con pesi softmax.
        Ritorna [num_classes, d]
        """
        if self.per_class:
            # template_weights: [num_classes, num_templates] -> softmax su dim template
            w = nn.functional.softmax(self.template_logits, dim=-1)  # [C, T]
            # text_features_templates: [T, C, d] -> permuta a [C, T, d]
            t = self.text_features_templates.permute(1, 0, 2)  # [C, T, d]
            # weighted sum su T
            weighted = (w.unsqueeze(-1) * t).sum(dim=1)  # [C, d]
        else:
            # template_weights global: [T]
            w = nn.functional.softmax(self.template_logits, dim=0)  # [T]
            weighted = (w.view(-1, 1, 1) * self.text_features_templates).sum(dim=0)  # [C, d]
        # normalizza
        weighted = nn.functional.normalize(weighted, dim=-1)
        return weighted  # [num_classes, d]

    @torch.no_grad()
    def get_text_features_uniform(self) -> torch.Tensor:
        """
        Ensemble uniforme: media semplice delle embedding dei template.
        [T, C, d] -> media su T -> [C, d]
        """
        feats = self.text_features_templates.mean(dim=0)
        feats = nn.functional.normalize(feats, dim=-1)
        return feats  # [num_classes, d]


# ===========================================================
#  Training / Eval
# ===========================================================
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


def train(
    model,
    ensemble: PromptEnsemble,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    weight_decay: float,
    logit_scale_learnable: bool = False,
) -> dict:
    """
    ALLEna SOLO i pesi del prompt ensemble (template_logits).
    Opzionale: rendere il logit_scale di CLIP apprendibile (di default non lo è)
    """
    model.eval()  # backbone congelato
    params = [ensemble.template_logits]
    if logit_scale_learnable:
        # sblocca un clone di logit_scale (come parametro separato)
        # Nota: in open_clip, model.logit_scale è un parametro -> meglio non modificarlo direttamente
        logit_scale_param = nn.Parameter(model.logit_scale.detach().clone())
        params.append(logit_scale_param)
    else:
        logit_scale_param = None

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best = {"val_acc": 0.0, "epoch": -1, "state_dict": None}

    for epoch in range(1, epochs + 1):
        # -------------------
        # Train 
        # -------------------
        ensemble.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{epochs}", ncols=100)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = nn.functional.normalize(image_features, dim=-1)
                if logit_scale_param is None:
                    logit_scale = model.logit_scale.exp()
                else:
                    logit_scale = logit_scale_param.exp()

            text_features = ensemble.get_text_features_weighted()  # [C, d]

            # logits = s * <img, text>
            logits = logit_scale * image_features @ text_features.t()

            loss = nn.functional.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = images.size(0)
            total += bs
            running_loss += loss.item() * bs
            running_acc += accuracy(logits.detach(), targets) * bs
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{running_acc/total:.4f}")

        # -------------------
        # Val
        # -------------------
        val_res = evaluate(model, ensemble, val_loader, device, mode="learned", logit_scale_param=logit_scale_param)
        if val_res["acc"] > best["val_acc"]:
            best.update(
                {
                    "val_acc": val_res["acc"],
                    "epoch": epoch,
                    "state_dict": {
                        "template_logits": ensemble.template_logits.detach().cpu(),
                        "logit_scale": (logit_scale_param.detach().cpu() if logit_scale_param is not None else model.logit_scale.detach().cpu()),
                    },
                }
            )

        print(
            f"Epoch {epoch:02d} | train_loss={running_loss/total:.4f}  "
            f"train_acc={running_acc/total:.4f}  val_acc(learned)={val_res['acc']:.4f}"
        )

    return best


@torch.no_grad()
def evaluate(
    model,
    ensemble: PromptEnsemble,
    loader: DataLoader,
    device: torch.device,
    mode: str = "learned",  # "learned" o "uniform"
    logit_scale_param: torch.Tensor = None,
) -> dict:
    assert mode in {"learned", "uniform"}
    model.eval()
    ensemble.eval()
    tot = 0
    corr = 0

    if logit_scale_param is None:
        logit_scale = model.logit_scale.exp()
    else:
        logit_scale = logit_scale_param.exp()

    if mode == "learned":
        text_features = ensemble.get_text_features_weighted()  # [C, d]
    else:
        text_features = ensemble.get_text_features_uniform()  # [C, d]

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        image_features = model.encode_image(images)
        image_features = nn.functional.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()

        preds = logits.argmax(dim=-1)
        corr += (preds == targets).sum().item()
        tot += images.size(0)

    return {"acc": corr / tot}


# ===========================================================
#  Main
# ===========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="CLIP Prompt Learning + Prompt Ensemble (templates forniti)")
    ap.add_argument("--data", type=str, default=DATA_ROOT, help="Percorso al dataset ImageFolder (train+val insieme)")
    ap.add_argument("--model", type=str, default="ViT-B-32", help="Modello CLIP open_clip (es. ViT-B-32)")
    ap.add_argument("--pretrained", type=str, default="openai", help="Checkpoint pretrained (es. 'openai')")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Split di validazione")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-2, help="LR per i pesi dei template")
    ap.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", action="store_true", help="Impara pesi sui template per-classe (invece che globali)")
    ap.add_argument("--eval-uniform", action="store_true", help="Stampa anche accuracy con ensemble uniforme")
    ap.add_argument("--logit-scale-learnable", action="store_true", help="Rende apprendibile il logit scale (opzionale)")
    ap.add_argument("--save", type=str, default="prompt_ensemble_weights.pt", help="Dove salvare i pesi migliori")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Carica CLIP + preprocess + tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # freeze CLIP

    # Data
    train_loader, val_loader, classnames = build_dataloaders(
        data_root=args.data,
        preprocess=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Classi ({len(classnames)}): {classnames}")

    # Prompt Ensemble (LA MODIFICA CHIAVE rispetto a CoOp originale)
    ensemble = PromptEnsemble(
        model=model,
        classnames=classnames,
        templates=TEMPLATES,
        tokenizer=tokenizer,
        device=device,
        per_class=args.per_class,
        init_uniform=True,
    ).to(device)

    # Train solo pesi template (e opzionalmente logit scale)
    best = train(
        model=model,
        ensemble=ensemble,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.wd,
        logit_scale_learnable=args.logit_scale_learnable,
    )
    
    # Salva migliori pesi
    if best["state_dict"] is not None:
        torch.save(
            {
                "classnames": classnames,
                "templates": TEMPLATES,
                "per_class": args.per_class,
                **best["state_dict"],
            },
            args.save,
        )
        print(f"Salvato: {args.save} (best val_acc={best['val_acc']:.4f} @ epoch {best['epoch']})")
    else:
        print("Nessun best trovato (controlla training).")
    

    # Valutazioni finali (learned + opzionale uniform)
    # Ricostruisci i pesi migliori in RAM per la valutazione finale
    if os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location=device)
        with torch.no_grad():
            ensemble.template_logits.copy_(ckpt["template_logits"].to(device))
        if args.logit_scale_learnable and "logit_scale" in ckpt:
            logit_scale_param = nn.Parameter(ckpt["logit_scale"].to(device))
        else:
            logit_scale_param = None
    else:
        logit_scale_param = None

    res_learned = evaluate(model, ensemble, val_loader, device, mode="learned", logit_scale_param=logit_scale_param)
    print(f"[VAL] Prompt Ensemble (LEARNED)  acc = {res_learned['acc']:.4f}")

    if args.eval_uniform:
        res_uniform = evaluate(model, ensemble, val_loader, device, mode="uniform", logit_scale_param=logit_scale_param)
        print(f"[VAL] Prompt Ensemble (UNIFORM)  acc = {res_uniform['acc']:.4f}")


if __name__ == "__main__":
    main()
