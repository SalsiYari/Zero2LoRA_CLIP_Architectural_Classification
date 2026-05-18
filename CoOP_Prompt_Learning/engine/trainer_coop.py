from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm.auto import tqdm, trange

from configs.default import BAR_FORMAT
from utils.core_utils import maybe_tqdm
from models.coop import PromptLearner, encode_text_with_embeddings_eval, encode_text_with_embeddings_train


def build_zeroshot_text_features(model, tokenizer, classnames, template, device, use_amp: bool):
    prompts = [template.format(c) for c in classnames]
    text_tokens = tokenizer(prompts).to(device)
    amp_ctx = torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp))
    with torch.no_grad(), amp_ctx:
        text_features = model.encode_text(text_tokens)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    text_features = text_features.to(device=device, dtype=next(model.parameters()).dtype)
    return text_features


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    text_features: torch.Tensor,
    device: torch.device,
    desc: str = "eval",
    use_tqdm: bool = True,
    position: int = 0,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.eval()
    top1, n = 0, 0
    all_preds, all_targets = [], []

    iterator = maybe_tqdm(
        data_loader,
        enabled=use_tqdm,
        desc=desc,
        leave=False,
        position=position,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
    )

    amp_ctx = torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp))

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device)

        with amp_ctx:
            img_feat = model.encode_image(images)
            img_feat = F.normalize(img_feat, dim=-1)

            tf = text_features.to(device=img_feat.device, dtype=img_feat.dtype)  
            logits = img_feat @ tf.t()
            scale = model.logit_scale.exp().to(logits.dtype)
            logits = logits * scale

        preds = logits.argmax(dim=-1)
        top1 += (preds == targets).sum().item()
        n += targets.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    return {"top1": top1 / max(1, n), "macro_f1": f1_score(y_true, y_pred, average="macro")}


def train_coop(
    model,
    tokenizer,
    classnames: List[str],
    train_loader: DataLoader,               
    val_loader: Optional[DataLoader],       
    device: torch.device,                   
    n_ctx: int = 16,                        
    mode: str = "shared",                   
    template_ctx: str = "a photo of a {}",
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    use_tqdm: bool = True,
    use_amp: bool = True,
) -> Tuple["PromptLearner", Dict[str, float]]:      

    prompt_learner = PromptLearner(                 
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        n_ctx=n_ctx,
        template_ctx=template_ctx,
        mode=mode,
        device=device,
    ).to(device)                                    

    for p in model.parameters():                    
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW([prompt_learner.ctx], lr=lr, weight_decay=weight_decay)       
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))          

    best = {"top1": 0.0, "macro_f1": 0.0}               

    epoch_iter = (                                      
        trange(1, epochs + 1, desc="Epochs", dynamic_ncols=True, bar_format=BAR_FORMAT)
        if use_tqdm
        else range(1, epochs + 1)
    )

    amp_ctx = torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp)) 

    for epoch in epoch_iter:
        prompt_learner.train()                         
        model.eval()                                    

        running_loss, num_samples = 0.0, 0

        batch_iter = maybe_tqdm( train_loader , enabled=use_tqdm , desc=f"Train {epoch}/{epochs}" , leave=False , position=1 , dynamic_ncols=True , bar_format=BAR_FORMAT )

        for images, targets in batch_iter:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:                                                            
                tokens, x = prompt_learner()
                txt_feat = encode_text_with_embeddings_train(model, tokens, x)  

                img_feat = model.encode_image(images)
                img_feat = F.normalize(img_feat, dim=-1)

                logits = img_feat @ txt_feat.t()
                scale = model.logit_scale.exp().to(logits.dtype)
                logits = logits * scale

                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

            if use_tqdm:
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]["lr"]
                )

        train_loss = running_loss / max(1, num_samples)

        prompt_learner.eval()
        with torch.no_grad():
            tokens, x = prompt_learner()
            txt_feat_val = encode_text_with_embeddings_eval(model, tokens, x)

        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                txt_feat_val,
                device=device,
                desc=f"Val {epoch}/{epochs}",
                use_tqdm=use_tqdm,
                position=2,
                use_amp=use_amp,
            )
        else:
            val_metrics = {"top1": float("nan"), "macro_f1": float("nan")}

        msg = (
            f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  "
            f"val_top1={val_metrics['top1']:.4f}  val_macroF1={val_metrics['macro_f1']:.4f}"
        )
        (tqdm.write if use_tqdm else print)(msg)

        if isinstance(val_metrics["top1"], float) and val_metrics["top1"] > best["top1"]:
            best = val_metrics.copy()

    return prompt_learner, best
