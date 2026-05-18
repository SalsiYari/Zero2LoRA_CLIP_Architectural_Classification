import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.prompt_ensemble import PromptEnsemble
from utils.core_utils import accuracy


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
    
    model.eval()  
    params = [ensemble.template_logits]
    if logit_scale_learnable:
        logit_scale_param = nn.Parameter(model.logit_scale.detach().clone())
        params.append(logit_scale_param)
    else:
        logit_scale_param = None

    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    best = {"val_acc": 0.0, "epoch": -1, "state_dict": None}

    for epoch in range(1, epochs + 1):
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

            text_features = ensemble.get_text_features_weighted()  

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
    mode: str = "learned",  
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
        text_features = ensemble.get_text_features_weighted()  
    else:
        text_features = ensemble.get_text_features_uniform()  

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
