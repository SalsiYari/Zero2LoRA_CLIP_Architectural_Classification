import os
import argparse
import torch
import torch.nn as nn
import open_clip

# Import post-refactoring
from configs.default import DATA_ROOT, TEMPLATES
from utils.core_utils import set_seed, get_device
from data.dataloaders import build_dataloaders
from models.prompt_ensemble import PromptEnsemble
from engine.trainer_ensemble import train, evaluate


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

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  

    train_loader, val_loader, classnames = build_dataloaders(
        data_root=args.data,
        preprocess=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Classi ({len(classnames)}): {classnames}")

    ensemble = PromptEnsemble(
        model=model,
        classnames=classnames,
        templates=TEMPLATES,
        tokenizer=tokenizer,
        device=device,
        per_class=args.per_class,
        init_uniform=True,
    ).to(device)

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
