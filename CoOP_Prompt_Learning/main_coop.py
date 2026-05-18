import os
import sys
import argparse
import torch
import torch.nn.functional as F
import open_clip
from sklearn.metrics import classification_report, confusion_matrix

# Import post-refactoring
from configs.default import DATA_ROOT, BAR_FORMAT
from utils.core_utils import set_seed, maybe_tqdm
from data.dataloaders import build_imagefolder_splits
from engine.trainer_coop import train_coop, build_zeroshot_text_features, evaluate
from models.coop import encode_text_with_embeddings_eval


def main():
    parser = argparse.ArgumentParser(description="CoOp per CLIP con dataset ImageFolder")
    parser.add_argument("--data", type=str, default=DATA_ROOT, help="cartella con train/valid/test")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="nome modello OpenCLIP (es. ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="tag pretrained per OpenCLIP")
    parser.add_argument("--n-ctx", type=int, default=16, help="numero di token di contesto CoOp")                                               
    parser.add_argument("--mode", type=str, default="shared", choices=["shared", "per_class"], help="token condivisi o per-classe")             
    parser.add_argument("--template-zero", type=str, default="a photo of a {}", help="template testuale per baseline zero-shot")                
    parser.add_argument("--template-ctx", type=str, default="a photo of a {}", help="template usato dentro il prompt CoOp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)                                                                                       
    parser.add_argument("--weight-decay", type=float, default=0.0)                                                                              
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="0 => Se il codice fa un fallback prendi SLURM_CPUS_PER_TASK o 2")
    parser.add_argument("--batch-train", type=int, default=64)                                                                                  
    parser.add_argument("--batch-eval", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=0, help="usa solo N campioni per overfit di prova (0=full)")
    parser.add_argument("--epochs-overfit", type=int, default=0, help="se >0, overfitta su train (opz. con --limit-train) e salta val/test")
    parser.add_argument("--no-tqdm", action="store_true", help="disattiva le barre di progresso")                                                
    parser.add_argument("--no-amp", action="store_true", help="disattiva mixed precision (utile per debug)")                                     

    args, unknown = parser.parse_known_args()                                                                
    if unknown:                                                                                              
        print(f"[NOTE] Argomenti extra ignorati: {unknown}")                
    if "ipykernel" in sys.modules:
        sys.argv = sys.argv[:1]

    set_seed(args.seed)                                                                   
    use_tqdm = not args.no_tqdm                                                             
    use_amp = not args.no_amp                                                               

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    if args.num_workers <= 0:                                                               
        args.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))

    print(f"[INFO] Carico OpenCLIP: {args.model} ({args.pretrained})")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(        
        args.model, pretrained=args.pretrained, device=device                                   
    )
    tokenizer = open_clip.get_tokenizer(args.model)                                         
    model.eval()                                                                            

    args.data = args.data or DATA_ROOT                                                      
    train_loader, val_loader, test_loader, classnames = build_imagefolder_splits(           
        data_root=args.data,
        preprocess_train=preprocess_train,
        preprocess_val=preprocess_val,
        batch_size_train=args.batch_train,
        batch_size_eval=args.batch_eval,
        num_workers=args.num_workers,
        seed=args.seed,
        limit_train=args.limit_train,
    )
    print(f"[INFO] classi ({len(classnames)}): {classnames}")

    if args.epochs_overfit and args.epochs_overfit > 0:
        print("\n[DEBUG] Modalità tiny-overfit su train (salto val/test)")
        prompt_learner, _ = train_coop(                                               
            model=model,
            tokenizer=tokenizer,
            classnames=classnames,
            train_loader=train_loader,
            val_loader=None,                    
            device=device,
            n_ctx=args.n_ctx,                   
            mode=args.mode,                     
            template_ctx=args.template_ctx,     
            epochs=args.epochs_overfit,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_tqdm=use_tqdm,
            use_amp=use_amp,                    
        )
        print("[DEBUG] Fine tiny-overfit.")
        return                                  

    print("\n[BASELINE] Valutazione zero-shot su val/test...")
    text_feat_zero = build_zeroshot_text_features(model, tokenizer, classnames, args.template_zero, device, use_amp)        
     
    val_zero = evaluate(
        model,
        val_loader,
        text_feat_zero,
        device=device,
        desc="Zero-shot (val)",
        use_tqdm=use_tqdm,
        position=0,
        use_amp=use_amp,
    )
    test_zero = evaluate(
        model,
        test_loader,
        text_feat_zero,
        device=device,
        desc="Zero-shot (test)",
        use_tqdm=use_tqdm,
        position=0,
        use_amp=use_amp,
    )
    print(
        f"[ZERO-SHOT] val_top1={val_zero['top1']:.4f}  val_macroF1={val_zero['macro_f1']:.4f}"
    )
    print(
        f"[ZERO-SHOT] test_top1={test_zero['top1']:.4f}  test_macroF1={test_zero['macro_f1']:.4f}"
    )

    print("\n[CoOp] Training dei token di contesto...")
    prompt_learner, best_val = train_coop(
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_ctx=args.n_ctx,
        mode=args.mode,
        template_ctx=args.template_ctx,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_tqdm=use_tqdm,
        use_amp=use_amp,
    )

    print("\n[CoOp] Valutazione finale su val/test...")
    with torch.no_grad():
        tokens, x = prompt_learner()
        text_feat_coop = encode_text_with_embeddings_eval(model, tokens, x)

    val_coop = evaluate(
        model,
        val_loader,
        text_feat_coop,
        device=device,
        desc="CoOp (val)",
        use_tqdm=use_tqdm,
        position=0,
        use_amp=use_amp,
    )
    test_coop = evaluate(
        model,
        test_loader,
        text_feat_coop,
        device=device,
        desc="CoOp (test)",
        use_tqdm=use_tqdm,
        position=0,
        use_amp=use_amp,
    )
    print(f"[CoOp] val_top1={val_coop['top1']:.4f}  val_macroF1={val_coop['macro_f1']:.4f}")
    print(f"[CoOp] test_top1={test_coop['top1']:.4f}  test_macroF1={test_coop['macro_f1']:.4f}")

    print("\n===== CONFRONTO (test set) =====")
    print(
        f"Zero-shot  -> Top-1: {test_zero['top1']:.4f} | Macro-F1: {test_zero['macro_f1']:.4f}"
    )
    print(
        f"CoOp       -> Top-1: {test_coop['top1']:.4f} | Macro-F1: {test_coop['macro_f1']:.4f}"
    )

    y_true, y_pred_zero, y_pred_coop = [], [], []
    iter_report = maybe_tqdm(
        test_loader,
        enabled=use_tqdm,
        desc="Report (test)",
        leave=False,
        position=0,
        dynamic_ncols=True,
        bar_format=BAR_FORMAT,
    )

    amp_ctx = torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp))

    for images, targets in iter_report:
        images = images.to(device)
        targets = targets.to(device)

        with amp_ctx:
            img_feat = model.encode_image(images)
            img_feat = F.normalize(img_feat, dim=-1)

            tf_zero = text_feat_zero.to(device=img_feat.device, dtype=img_feat.dtype)
            tf_coop = text_feat_coop.to(device=img_feat.device, dtype=img_feat.dtype)

            logits_zero = img_feat @ tf_zero.t()
            logits_coop = img_feat @ tf_coop.t()

            scale = model.logit_scale.exp().to(img_feat.dtype)
            logits_zero = logits_zero * scale
            logits_coop = logits_coop * scale

        y_true.append(targets.cpu())
        y_pred_zero.append(logits_zero.argmax(dim=-1).cpu())
        y_pred_coop.append(logits_coop.argmax(dim=-1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred_zero = torch.cat(y_pred_zero).numpy()
    y_pred_coop = torch.cat(y_pred_coop).numpy()

    print("\n[ZERO-SHOT] Classification report (test):")
    print(classification_report(y_true, y_pred_zero, target_names=classnames))
    print("[ZERO-SHOT] Confusion matrix:\n", confusion_matrix(y_true, y_pred_zero))

    print("\n[CoOp] Classification report (test):")
    print(classification_report(y_true, y_pred_coop, target_names=classnames))
    print("[CoOp] Confusion matrix:\n", confusion_matrix(y_true, y_pred_coop))


if __name__ == "__main__":
    main()
