#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =========================================================
#      -------------------- NOTE --------------------
# LoRA fine-tuning per OpenCLIP — Cluster Ready
# Dataset: ImageFolder (train/valid/test)
# Obiettivo: CE su (image @ text_features) con text side fisso!!
# =========================================================

import os, sys, math, random, argparse, datetime
from typing import List, Tuple, Dict, Optional

# Mitiga la frammentazione dell'allocator CUDA (se non già impostato esternamente)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score

try:
    import open_clip
except ImportError:
    print("[FATAL] open_clip_torch non trovato. Installa il pacchetto nell'ambiente del cluster.")
    raise

from tqdm.auto import tqdm, trange

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

# ------------------------- Utils -------------------------
def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    try:
        # Su Ampere abilita TF32 per maggiore velocità/stabilità
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass

def maybe_tqdm(iterable, enabled: bool, **kwargs):
    return tqdm(iterable, **kwargs) if enabled else iterable

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _prettify_classnames(names):
    return [n.replace("_"," ").replace("-"," ").strip() for n in names]

def _remap_dataset_to(train_ctoi, other_ds):
    other_ctoi = other_ds.class_to_idx
    mapping = {other_ctoi[name]: train_ctoi[name] for name in other_ctoi.keys()}
    class Remap(torch.utils.data.Dataset):
        def __init__(self, base, m): self.base, self.m = base, m
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            x,y = self.base[i]; return x, self.m[y]
        @property
        def classes(self): return self.base.classes
    return Remap(other_ds, mapping)

def build_imagefolder_splits(
    data_root: str,
    preprocess_train,
    preprocess_val,
    batch_size_train: int = 128,
    batch_size_eval: int = 256,
    num_workers: int = 4,
    seed: int = 42,
):
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "valid")
    test_dir  = os.path.join(data_root, "test")
    for d in [train_dir, val_dir, test_dir]:
        assert os.path.isdir(d), f"Manca la cartella: {d}"

    train_full = ImageFolder(train_dir, transform=preprocess_train)
    val_full   = ImageFolder(val_dir,   transform=preprocess_val)
    test_full  = ImageFolder(test_dir,  transform=preprocess_val)

    # ordini coerenti
    val_full  = _remap_dataset_to(train_full.class_to_idx, val_full)
    test_full = _remap_dataset_to(train_full.class_to_idx, test_full)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_full, batch_size=batch_size_train, shuffle=True,  generator=g,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=(num_workers>0))
    val_loader   = DataLoader(val_full,  batch_size=batch_size_eval,  shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=(num_workers>0))
    test_loader  = DataLoader(test_full, batch_size=batch_size_eval,  shuffle=False,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=(num_workers>0))

    classnames = _prettify_classnames(train_full.classes)
    return train_loader, val_loader, test_loader, classnames

# ----------------- Zero-shot text features ----------------
@torch.no_grad()
def build_zeroshot_text_features(model, tokenizer, classnames, template, device):
    """
    Prepara i text features sul device finale e poi li normalizza.
    Il dtype definitivo viene comunque allineato dinamicamente al momento del matmul.
    """
    prompts = [template.format(c) for c in classnames]
    text_tokens = tokenizer(prompts).to(device)
    with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
        text_features = model.encode_text(text_tokens)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    # Rimaniamo in dtype "nativo" del modello; eventuali cast coerenti avvengono in train/eval
    text_features = text_features.to(device=device)
    return text_features

# ------------------ LoRA for nn.Linear -------------------
class LoRALinear(nn.Module):
    """
    Wrapper: y = x W^T + (x A^T B^T) * scale + b
    Allinea device/dtype al layer base per evitare mismatch su cluster.
    """
    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scale = alpha / max(1, r)
        self.has_bias = base_linear.bias is not None

        dev  = base_linear.weight.device
        dtype= base_linear.weight.dtype

        # Manteniamo i pesi originali come Parametri non addestrabili
        self.weight = nn.Parameter(base_linear.weight.detach().to(dev, dtype=dtype), requires_grad=False)
        self.bias   = (nn.Parameter(base_linear.bias.detach().to(dev, dtype=dtype), requires_grad=False)
                       if self.has_bias else None)

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=dev, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=dev, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            # Ricontrollo device/dtype in caso di autocast mixed-precision
            A = self.lora_A.to(device=self.weight.device, dtype=self.weight.dtype)  # [r, in]
            B = self.lora_B.to(device=self.weight.device, dtype=self.weight.dtype)  # [out, r]
            x_adapt = self.dropout(x)
            # (x @ A^T) -> [*, r]
            low = F.linear(x_adapt, A)
            # (low @ B^T) -> [*, out]
            delta = F.linear(low, B)
            out = out + delta * self.scale
        return out

def replace_linear_with_lora(module: nn.Module, name_prefix: str, r: int, alpha: int, dropout: float, only_if_name_contains=("mlp",)):
    for child_name, child in list(module.named_children()):
        full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
        if isinstance(child, nn.Linear):
            if (only_if_name_contains is None) or any(tok in full_name for tok in only_if_name_contains):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            replace_linear_with_lora(child, full_name, r, alpha, dropout, only_if_name_contains)

def collect_lora_params(model: nn.Module):
    return [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n)]

def save_lora_state(model: nn.Module, path: str):
    lora_state = {k: v.cpu() for k, v in model.state_dict().items() if ("lora_A" in k or "lora_B" in k)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(lora_state, path)

# --------------------- Evaluation ------------------------
@torch.inference_mode()  # <--- riduce overhead rispetto a no_grad
def evaluate(model, data_loader, text_features, device, desc="eval",
             use_tqdm=True, position=0, chunk_size: Optional[int] = None):
    """
    Eval con micro-batching sul solo encode dell'immagine.
    Evita di caricare l'intero batch in GPU: si spostano solo i sotto-batch.
    """
    model.eval()
    # normalizza/valida chunk_size
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    top1, n = 0, 0
    all_preds, all_targets = [], []

    iterator = maybe_tqdm(
        data_loader, enabled=use_tqdm,
        desc=desc, leave=False, position=position, dynamic_ncols=True, bar_format=BAR_FORMAT
    )
    tf = text_features  # già su device

    def encode_chunked(images_cpu: torch.Tensor) -> torch.Tensor:
        B = images_cpu.size(0)
        # Caso senza chunking: sposta tutto in device una volta sola
        if (chunk_size is None) or (chunk_size >= B):
            xb = images_cpu.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                feats = model.encode_image(xb)
            return F.normalize(feats, dim=-1)

        # Caso con chunking: sposta in device solo i pezzi
        outs = []
        for s in range(0, B, chunk_size):
            xb = images_cpu[s:s+chunk_size].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                f = model.encode_image(xb)
            outs.append(F.normalize(f, dim=-1))
        return torch.cat(outs, dim=0)

    for images_cpu, targets in iterator:
        # targets sono piccoli: ok portarli subito sul device
        targets = targets.to(device, non_blocking=True)

        img_feat = encode_chunked(images_cpu)
        # -------- FIX DTYPE MISMATCH --------
        # Allinea sempre il dtype delle feature immagine a quello delle text features
        img_feat = img_feat.to(dtype=tf.dtype)

        logits = (img_feat @ tf.t())
        scale  = model.logit_scale.exp().to(logits.dtype)
        logits = logits * scale

        preds = logits.argmax(dim=-1)
        top1 += (preds == targets).sum().item()
        n    += targets.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    return {"top1": top1 / max(1, n), "macro_f1": f1_score(y_true, y_pred, average="macro")}

# ---------------------- Training -------------------------
def train_lora(
    model,
    train_loader,
    val_loader,
    text_features,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    use_tqdm: bool = True,
    accum_steps: int = 1,              # gradient accumulation
    grad_clip: float | None = None,    # gradient clipping (es. 1.0)
    enable_grad_checkpoint: bool = False,  # grad checkpointing se supportato
    eval_chunk: Optional[int] = None,  # <--- passaggio chunk eval
):
    # (opzionale) attiva grad checkpointing del visual tower per ridurre memoria
    if enable_grad_checkpoint:
        try:
            model.visual.set_grad_checkpointing(True)  # type: ignore[attr-defined]
            print("[INFO] grad checkpointing attivo sul visual tower")
        except Exception:
            print("[INFO] grad checkpointing non disponibile per questo modello/tag")

    # freeze completo
    for p in model.parameters():
        p.requires_grad_(False)

    # solo LoRA trainabile
    lora_params = collect_lora_params(model)
    assert len(lora_params) > 0, "Nessun parametro LoRA trovato!"
    for p in lora_params:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best = {"top1": 0.0, "macro_f1": 0.0}
    epoch_iter = trange(1, epochs + 1, desc="Epochs", dynamic_ncols=True, bar_format=BAR_FORMAT) if use_tqdm else range(1, epochs + 1)

    tf = text_features  # già sul device

    for epoch in epoch_iter:
        model.train()
        running_loss, num_samples = 0.0, 0

        batch_iter = maybe_tqdm(
            train_loader, enabled=use_tqdm,
            desc=f"Train {epoch}/{epochs}", leave=False, position=1, dynamic_ncols=True, bar_format=BAR_FORMAT
        )

        optimizer.zero_grad(set_to_none=True)

        step = 0
        for step, (images, targets) in enumerate(batch_iter, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                img_feat = model.encode_image(images)
                img_feat = F.normalize(img_feat, dim=-1)
                # -------- FIX DTYPE MISMATCH --------
                img_feat = img_feat.to(dtype=tf.dtype)

                logits = (img_feat @ tf.t())
                scale  = model.logit_scale.exp().to(logits.dtype)
                logits = logits * scale
                loss   = F.cross_entropy(logits, targets)

            # gradient accumulation: normalizza la loss
            loss_to_backward = loss / max(1, accum_steps)
            scaler.scale(loss_to_backward).backward()

            # logging loss "reale" (non divisa)
            running_loss += loss.detach().float().item() * images.size(0)
            num_samples  += images.size(0)

            do_step = (step % accum_steps == 0)

            if do_step:
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if use_tqdm:
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=optimizer.param_groups[0]["lr"],
                    acc=accum_steps
                )

        # se il numero di batch non è multiplo di accum_steps, fai un ultimo step
        if step > 0 and (step % accum_steps) != 0:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, num_samples)

        # Val
        model.eval()
        val_metrics = evaluate(model, val_loader, tf, device=device,
                               desc=f"Val {epoch}/{epochs}", use_tqdm=use_tqdm, position=2,
                               chunk_size=eval_chunk)
        msg = (f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  "
               f"val_top1={val_metrics['top1']:.4f}  val_macroF1={val_metrics['macro_f1']:.4f}")
        (tqdm.write if use_tqdm else print)(msg)

        if val_metrics["top1"] > best["top1"]:
            best = val_metrics.copy()

        # aiuta la frammentazione tra epoche
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best

# ------------- Load model with tag fallback --------------
def load_openclip_with_fallback(model_name: str, desired_tag: str, device):
    print(f"[INFO] Carico OpenCLIP: {model_name} ({desired_tag})")
    try:
        return open_clip.create_model_and_transforms(model_name, pretrained=desired_tag, device=device)
    except Exception as e:
        print(f"[WARN] Tag '{desired_tag}' non disponibile per {model_name}. Provo fallback...")
        for fb in ["metaclip_fullcc", "laion2b_s39b_b160k", "openai", "laion400m_e32"]:
            try:
                print(f"[INFO] -> provo '{fb}'")
                return open_clip.create_model_and_transforms(model_name, pretrained=fb, device=device)
            except Exception:
                pass
        raise e

# --------------------------- Main ------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA su OpenCLIP (immagine) — Cluster Ready")
    parser.add_argument("--data", type=str, default="/work/cvcs2025/carpaneto_salsi_serafini/ArchiStyles-2",
                        help="Cartella con train/valid/test")
    parser.add_argument("--outdir", type=str, default="./runs_lora", help="Dove salvare i pesi LoRA")
    parser.add_argument("--model", type=str, default="ViT-bigG-14")
    parser.add_argument("--pretrained", type=str, default="metaclip_fullcc",
                        help="Esempi: 'metaclip_fullcc', 'laion2b_s39b_b160k', 'openai'")
    parser.add_argument("--quick-test", action="store_true",
                        help="Usa un modello compatibile più piccolo (ViT-B-32) per test veloci")
    parser.add_argument("--template-zero", type=str, default="a photo of a {}")
    parser.add_argument("--skip-zero-shot", action="store_true",
                        help="Non stampare la baseline (calcolo comunque i text features)")
    # LoRA
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--apply-to", type=str, default="mlp", choices=["mlp","all_linear"],
                        help="mlp = solo fc1/fc2; all_linear = tutte le Linear nel visual")
    # Train (memory-safe defaults)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-train", type=int, default=2)
    parser.add_argument("--batch-eval", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=8,
                        help="Accumulo gradiente per simulare batch effettivo più grande")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Clipping del gradiente (max-norm)")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Abilita gradient checkpointing sul visual tower")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tqdm", action="store_true")
    # --------- micro-batching per la eval ----------
    parser.add_argument("--eval-chunk", type=int, default=16,
                        help="Micro-batch per l'encode in eval (riduce picco VRAM). 0 o negativo = disabilitato.")
    args = parser.parse_args()

    # Heuristic sicura per workers (evita warning su cluster piccoli)
    if args.num_workers > 0:
        cpu_cnt = os.cpu_count() or 4
        suggested = min(4, cpu_cnt)
        if args.num_workers > suggested:
            print(f"[WARN] num_workers={args.num_workers} > suggerito={suggested}. Imposto a {suggested}.")
            args.num_workers = suggested

    # Quick test: override del modello/tag
    if args.quick_test:
        print("[INFO] QUICK TEST attivo: uso ViT-B-32 per velocità.")
        args.model = "ViT-B-32"
        # Tag prudente; i fallback coprono altri casi
        if args.pretrained == "metaclip_fullcc":
            args.pretrained = "openai"

    set_seed(args.seed)

    print(f"[INFO] Inizio: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Modello
    model, preprocess_train, preprocess_val = load_openclip_with_fallback(args.model, args.pretrained, device)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    # Dataloaders
    train_loader, val_loader, test_loader, classnames = build_imagefolder_splits(
        data_root=args.data,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        batch_size_train=args.batch_train,
        batch_size_eval=args.batch_eval,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"[INFO] classi ({len(classnames)}): {classnames}")

    # Text features (servono sempre)
    text_feat_zero = build_zeroshot_text_features(model, tokenizer, classnames, args.template_zero, device)

    if not args.skip_zero_shot:
        print("\n[BASELINE] Zero-shot su valid/test...")
        val_zero  = evaluate(model, val_loader,  text_feat_zero, device,
                             desc="Zero-shot (val)", use_tqdm=not args.no_tqdm,
                             chunk_size=args.eval_chunk)
        test_zero = evaluate(model, test_loader, text_feat_zero, device,
                             desc="Zero-shot (test)", use_tqdm=not args.no_tqdm,
                             chunk_size=args.eval_chunk)
        print(f"[ZERO-SHOT] val_top1={val_zero['top1']:.4f}  val_macroF1={val_zero['macro_f1']:.4f}")
        print(f"[ZERO-SHOT] test_top1={test_zero['top1']:.4f}  test_macroF1={test_zero['macro_f1']:.4f}")

    # libera la cache prima del training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Iniezione LoRA
    only_tokens = None if args.apply_to == "all_linear" else ("mlp",)
    print(f"\n[LoRA] Iniezione: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, target={args.apply_to}")
    replace_linear_with_lora(model.visual, name_prefix="visual", r=args.lora_r,
                             alpha=args.lora_alpha, dropout=args.lora_dropout,
                             only_if_name_contains=only_tokens)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in collect_lora_params(model))
    print(f"[LoRA] Trainable LoRA params: {n_train:,} / {n_total:,} ({100*n_train/n_total:.3f}%)")

    # Training
    print("\n[LoRA] Training...")
    best_val = train_lora(
        model, train_loader, val_loader, text_feat_zero, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, use_tqdm=not args.no_tqdm,
        accum_steps=args.accum_steps, grad_clip=args.grad_clip, enable_grad_checkpoint=args.grad_checkpoint,
        eval_chunk=args.eval_chunk,
    )

    # Valutazione finale
    print("\n[LoRA] Valutazione finale su valid/test...")
    val_lora  = evaluate(model, val_loader,  text_feat_zero, device, desc="LoRA (val)",
                         use_tqdm=not args.no_tqdm, chunk_size=args.eval_chunk)
    test_lora = evaluate(model, test_loader, text_feat_zero, device, desc="LoRA (test)",
                         use_tqdm=not args.no_tqdm, chunk_size=args.eval_chunk)
    print(f"[LoRA] val_top1={val_lora['top1']:.4f}  val_macroF1={val_lora['macro_f1']:.4f}")
    print(f"[LoRA] test_top1={test_lora['top1']:.4f}  test_macroF1={test_lora['macro_f1']:.4f}")

    # Salvataggio
    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"lora_{args.model}_{args.pretrained}_r{args.lora_r}_a{args.lora_alpha}.pt")
    save_lora_state(model, save_path)
    print(f"\n[LoRA] Salvato stato LoRA in: {save_path}")

    print("\n[NOTE] L'avviso QuickGELU di open_clip è innocuo; deriva da differenze tra il config del modello e il tag pretrain. "
          "Se ti dà noia, prova un tag alternativo (es. 'laion2b_s39b_b160k') o il flag --quick-test.")

if __name__ == "__main__":
    main()
