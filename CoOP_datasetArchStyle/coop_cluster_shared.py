# =========================================================
# CoOp (Context Optimization) per OpenCLIP — HPC Ready
# Dataset: ImageFolder (usa TUTTO il train, con opzione tiny-overfit)
# Con barre di caricamento (tqdm) per training ed evaluation
# =========================================================
'''
NOTE!!
Se lo lanci interattivo con srun, vedi tutto nel terminale (puoi fare python coop.py ... | tee run.log per salvarlo).
Se lo lanci con sbatch, i log vanno nei file indicati da SLURM (--output/--error). Nel mio esempio erano logs/%x_%j.out e logs/%x_%j.err.

Default principali
--model ViT-B-32
--pretrained laion2b_s34b_b79k
--n-ctx 16 (prompt condiviso --mode shared)
template: "a photo of a {}" (sia zero-shot che CoOp)
--epochs 10, --lr 1e-3, --weight-decay 0.0, --seed 42
Batch: --batch-train 64, --batch-eval 128
Workers: --num-workers 0 → auto: usa SLURM_CPUS_PER_TASK (altrimenti 2)
AMP: attivo (disattivi con --no-amp)
Tiny-overfit di debug: disattivo (attivi con --limit-train N --epochs-overfit E)
tqdm: attivo (disattivi con --no-tqdm)

Con ViT-B/32, immagini 224px, modello gelato e solo prompt che impara:
8 GB → funziona con --batch-train 16–32
11–12 GB → --batch-train 32–64
16 GB (consigliato) → --batch-train 64–128 comodo
24 GB → --batch-train 128–256 sereno
48 GB → batch molto grandi

'''
import os
import sys
import random
import argparse
from typing import List, Tuple, Optional, Dict
'''
mi permette di usare
List[T]:           lista di elementi di tipo T.
Tuple[A, B]:       tupla (immutabile) con elementi A, B (o più).
Optional[T]:        o T o None.
Dict[K, V]:         dizionario chiave/valore.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import open_clip
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from tqdm.auto import tqdm, trange

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
'''
{l_bar}: parte sinistra (descrizione + percentuale).
{bar}: la barra grafica.
{n_fmt}/{total_fmt}: step corrente/totale formattati.
{elapsed}: tempo trascorso.
{remaining}: tempo stimato rimanente.
{rate_fmt}: velocità (iter/sec).
Sarà passato come tqdm(..., bar_format=BAR_FORMAT).
'''

# === CONFIG: path dataset di default (puoi sovrascriverlo con --data) ===
DATA_ROOT = "/work/cvcs2025/carpaneto_salsi_serafini/ArchiStyles-2"

# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)                                       #fissa il seme per il modulo random di Python
    os.environ["PYTHONHASHSEED"] = str(seed)                #fissa la variabile d'ambiente PYTHONHASHSEED -> rende deterministico l'ordine degli hash
    torch.manual_seed(seed)                                 #fissa il seme per CPU
    if torch.cuda.is_available():                    #se c'è una GPU
        torch.cuda.manual_seed_all(seed)                    #fissa il seme per tutte le GPU
        torch.backends.cudnn.deterministic = True           #usa algoritmi deterministici per cuDNN (potrebbe essere più lento ma rende riproducibile)
        torch.backends.cudnn.benchmark = False              #disabilita il benchmark di cuDNN (no autotuning per configurazioni di input variabili)


def maybe_tqdm(iterable, enabled: bool, **kwargs):
    """Restituisce tqdm(iterable) se enabled=True, altrimenti l'iterable nudo."""
    return tqdm(iterable, **kwargs) if enabled else iterable


# =========================================================
# ImageFolder builder (usa TUTTO il train, con opzione di sotto-campionamento)
# =========================================================

'''funzione che restituisce la lista delle etichette dal dataset ImageFolder'''
def _targets_list(ds):
    return getattr(ds, "targets", [y for _, y in ds.samples])                   #return List[int]                        
    #ImageFolder ha l'attributo targets (lista di etichette) ma non tutti i dataset lo hanno, quindi come fallback usa ds.samples 
    #[...] list comprehension, itera ds.samples (ovvero lista di tuple (path, label)) e prende solo y costruendo la lista delle etichette

'''funzione che sostituisce _ e - con spazio e fa strip() ovvero converte i nomi cartella in label più leggibili'''
def _prettify_classnames(names):
    return [n.replace("_", " ").replace("-", " ").strip() for n in names]               #return List[str]
    #.strip() rimuove spazi bianchi iniziali/finali (non quelli in mezzo)

'''funzione che rimappa gli indici di classe di 'other_ds' affinché coincidano con quelli di 'train_ctoi' (class_to_idx del train)'''
def _remap_dataset_to(train_ctoi, other_ds):
    other_ctoi = other_ds.class_to_idx
    mapping = {other_ctoi[name]: train_ctoi[name] for name in other_ctoi.keys()}        #dict {chiave = indice vecchio (other_ds) , valore = indice nuovo (train)}
                                                                                        #       Chiave: other_ctoi[name], Valore: train_ctoi[name].
    '''classe wrapper che rimappa le etichette di un dataset'''
    class Remap(torch.utils.data.Dataset):
        def __init__(self, base, map_idx):
            self.base = base                                    #base è il dataset originale (es. val_full o test_full)
            self.map = map_idx                                  #map è il dizionario di mapping {old_idx: new_idx}  

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            return x, self.map[y]

        @property
        def classes(self):
            return self.base.classes

    return Remap(other_ds, mapping)


def _balanced_subset_indices(targets, limit: int, seed: int) -> List[int]:      #targets: List[int] di etichette ; limit: numero massimo di campioni da selezionare ; seed: seme per random
    """Restituisce indici (quasi) bilanciati per sotto-campionare il train.
    Se limit < num_classes, prende almeno 1 per classe finché possibile.
    """
    num_samples = len(targets)
    classes = sorted(set(targets))
    C = len(classes)
    if limit <= 0 or limit >= num_samples:                                      #se limit non positivo o >= numero totale campioni, restituisce tutti gli indici
        return list(range(num_samples))

    # costruisci lista di indici per classe
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

    # distribuisci gli avanzi
    remaining = limit - len(sel)
    if remaining > 0:
        pool = [i for c in classes for i in per_class[c][n_base:]]
        rng.shuffle(pool)
        sel.extend(pool[:remaining])

    rng.shuffle(sel)
    return sel

'''Definisce una funzione che costruisce train/val/test DataLoader da una struttura ImageFolder.'''
def build_imagefolder_splits(
    data_root: str,
    preprocess_train,               #funzione/transform per preprocessare le immagini di train
    preprocess_val,                 #funzione/transform per preprocessare le immagini di val/test
    batch_size_train: int = 64,
    batch_size_eval: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    limit_train: int = 0,  # 0 = usa tutto
):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "valid")
    test_dir = os.path.join(data_root, "test")
    for d in [train_dir, val_dir, test_dir]:
        assert os.path.isdir(d), f"Manca la cartella: {d}"

    train_full = ImageFolder(train_dir, transform=preprocess_train)                 #crea classei ImageFolder che mappa sottocartelle in etichetta(int) e carica immagini con transform
    val_full = ImageFolder(val_dir, transform=preprocess_val)
    test_full = ImageFolder(test_dir, transform=preprocess_val)

    # ordini di classe coerenti tra split
    val_full = _remap_dataset_to(train_full.class_to_idx, val_full)                 # -> _remap_dataset_to()
    test_full = _remap_dataset_to(train_full.class_to_idx, test_full)               # -> _remap_dataset_to()

    # Controllo coerenza classi
    assert set(train_full.classes) == set(val_full.classes)                         #controlla che le classi siano le stesse nei tre split
    assert set(train_full.classes) == set(test_full.classes)

    # Sotto-campionamento opzionale del train per tiny-overfit/debug
    if limit_train and limit_train > 0:                                             #se limit_train , costruisce un sottoinsieme bilanciato
        targets = _targets_list(train_full)                                         # -> _targets_list() restituisce la lista delle etichette dal dataset ImageFolder
        indices = _balanced_subset_indices(targets, limit_train, seed)              # -> _balanced_subset_indices() restituisce una lista di indici bilanciati per sotto-campionare il train
        train_ds = Subset(train_full, indices)                                      #Subset è una classe di torch.utils.data che crea un sottoinsieme del dataset originale usando gli indici dati  
    else:
        train_ds = train_full

    # dataloaders
    g = torch.Generator().manual_seed(seed)                 #rende deterministico lo shuffle del dataloader
    train_loader = DataLoader(                              #crea il dataloader per il train
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,                                           #mischia ad ogni epoch usando g come generatore di numeri casuali
        generator=g,
        num_workers=num_workers,                                #thread/processi per caricare i dati (prefetch)
        pin_memory=True,                                        #se True, i tensori vengono copiati in memoria pinneata (page-locked) prima di essere trasferiti alla GPU (più veloce il trasferimento host->gpu)
    )
    val_loader = DataLoader(
        val_full,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(                                   
        test_full,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    classnames = _prettify_classnames(train_full.classes)       # -> _prettify_classnames() --> sostituisce _ e - con spazio per convertire i nomi cartella in label più leggibili
    return train_loader, val_loader, test_loader, classnames


# =========================================================
# Zero-shot helper (testo in autocast + dtype allineato)
# =========================================================

def build_zeroshot_text_features(model, tokenizer, classnames, template, device, use_amp: bool):
    prompts = [template.format(c) for c in classnames]
    text_tokens = tokenizer(prompts).to(device)
    amp_ctx = torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp))
    with torch.no_grad(), amp_ctx:
        text_features = model.encode_text(text_tokens)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    text_features = text_features.to(device=device, dtype=next(model.parameters()).dtype)
    return text_features


# =========================================================
# CoOp Prompt Learner =====================================
# =========================================================

'''implementa la funzione forward che restituisce i token e gli embedding testuali'''
class PromptLearner(nn.Module):                             #nn.Module = base class, definisce un modulo allenabile di rete neurale in PyTorch
    def __init__( self, model, tokenizer, classnames: List[str],n_ctx: int = 16, template_ctx: str = "a photo of a {}", mode: str = "shared",  init_std: float = 0.02,device: Optional[torch.device] = None ):
        super().__init__()
        assert mode in ("shared", "per_class")              #controlla che mode sia uno dei due valori ammessi --> 'shared' | 'per_class'
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.mode = mode
        self.device = device or next(model.parameters()).device

        # -dimensione embedding testuale:-
        # --> Obiettivo: ricavare la larghezza canale (dimensione D) delle embedding testuali usate dal transformer testuale di CLIP
        self.text_width = getattr(model.token_embedding, "embedding_dim", None)  #model.token_embedding.embedding_dim --> dimensione D delle embedding testuali se esiste 
        if self.text_width is None:
            # fallback robusto
            self.text_width = model.text_projection.shape[0]        #[ D ]
            '''In OpenCLIP tipico: text_projection ha shape [transformer_width, embed_dim]; 
            la prima dimensione corrisponde al transformer width (D)'''
            

        self.ctx_dtype = torch.float32                          #dtype usato per i token di contesto CoOp (float32 per stabilità numerica) 
        self.model_text_dtype = model.text_projection.dtype     #model.text_projection.dtype  --> dtype usato dal modello CLIP per le embedding testuali (es. float16 se modello in fp16)

        self.prompts_for_classes = [
            ("X " * n_ctx + template_ctx.format(c)).strip() for c in classnames             #creo --> [n_ctx placeholder] + template_ctx.format(nome_classe) per ogni classe
        ]
        self.register_buffer("tokenized_prompts", tokenizer(self.prompts_for_classes))      #il tokanizer aggiunge BOS, EOS, padding e converte in tensore di token (int) --> BOS [n_ctx placeholder] + template_ctx.format(nome_classe)

        if mode == "shared":
            ctx = torch.empty(n_ctx, self.text_width, dtype=self.ctx_dtype)                 #creo tensore vuoto per i token di contesto condivisi
            nn.init.normal_(ctx, std=init_std)                                              #inizializzo i token di contesto con distribuzione normale (mean=0, std=init_std)   
            self.ctx = nn.Parameter(ctx)  # [n_ctx, D]
        else:
            ctx = torch.empty(
                len(classnames), n_ctx, self.text_width, dtype=self.ctx_dtype               #creo tensore vuoto per i token di contesto specifici per classe [Numero_classi, n_ctx, D]
            )
            nn.init.normal_(ctx, std=init_std)
            self.ctx = nn.Parameter(ctx)  # [C, n_ctx, D]

        '''(1) calcola una volta per tutte dove sta l’EOT in ogni prompt, 
        (2) lo salva nel modulo come buffer (quindi si sposta tra CPU/GPU e finisce nello state_dict, ma non viene ottimizzato), 
        (3) lo fa senza costruire grafi di autograd'''
        with torch.no_grad():                                       
            toks = self.tokenized_prompts               #self.tokenized_prompts è un buffer registrato in __init__ con forma [C, L] (ovvero classi, lunghezza_prompt_massima_tipicamente77)                             
            #toks=contiene gli ID interi dei token per ciascun prompt (inclusi BOS, EOT e padding a destra con 0).
            self.register_buffer("eot_indices", toks.argmax(dim=-1))    #toks.argmax(dim=-1) restituisce l'indice del token con il valore massimo lungo l'ultima dimensione (-1) ovvero la lunghezza del prompt (L)
        '''PAD = 0, BOS = 49406, EOT = 49407 (ultimo ID del vocabolario).
            I token “normali” hanno ID < 49407.
            Quindi il più grande nella sequenza reale è proprio EOT: l’argmax punta alla sua posizione (indice tra 0..L-1).
         Risultato: tensore di forma [C] (uno per classe), dtype=torch.long, con gli indici dove CLIP si aspetta di prendere la representazione finale
            della frase (lo “pooled” testuale in CLIP è il vettore alla posizione EOT, non una media).'''



    '''definisce la computazione che avviene ogni volta che il modulo viene chiamato'''
    def forward(self):                                      #no input usa lo stato interno
        tokens = self.tokenized_prompts.to(self.device)           #[C, L]  #sposta i token su device (CPU o GPU)      
        x = self.model.token_embedding(tokens).to(self.model_text_dtype)  # [C, L, D]         #model_text_dtype = dtype usato dal modello CLIP per le embedding testuali (es. float16 se modello in fp16)

        if self.mode == "shared":
            ctx = (                                                 #ctx = token di contesto condivisi formato da 
                self.ctx.to(self.model_text_dtype)                  # [n_ctx, D]  #sposto i token di contesto su device e li converto al dtype del modello CLIP
                .unsqueeze(0)                                       # [1, n_ctx, D]  #aggiungo una dimensione in testa per il batch (classe)
                .expand(x.size(0), -1, -1)                          # [C, n_ctx, D]  #espando la dimensione del batch per copiare i token di contesto per ogni classe
            )  # [C, n_ctx, D]
        else:
            ctx = self.ctx.to(self.model_text_dtype)  # [C, n_ctx, D] 

        x[:, 1 : 1 + self.n_ctx, :] = ctx  # inserisci dopo BOS     #sostituisco i token di contesto (placeholder) con quelli apprendibili
        return tokens, x  # [C, L, D]


# =========================================================
# Text encoding (compatibile con diverse versioni di OpenCLIP)
# =========================================================
'''funzione implementata per gestire modelli OpenCLIP diversi (es. con o senza attn_mask, con o senza batch_first)'''
def _text_transformer_forward(model, x_embed: torch.Tensor) -> torch.Tensor:
    """
    x_embed: [B, L, D] -> output [B, L, D]
    Gestisce attn_mask e modelli con MultiheadAttention batch_first.
    """
    tr = model.transformer
    attn_mask = getattr(model, "attn_mask", None)

    batch_first = False
    try:
        if hasattr(tr, "resblocks") and len(tr.resblocks) > 0:
            attn_mod = getattr(tr.resblocks[0], "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "batch_first"):
                batch_first = bool(attn_mod.batch_first)
    except Exception:
        batch_first = False

    x_in = x_embed if batch_first else x_embed.permute(1, 0, 2)  # [B,L,D] or [L,B,D]

    try:
        if attn_mask is not None:
            x_out = tr(x_in, attn_mask=attn_mask)
        else:
            x_out = tr(x_in)
    except TypeError:
        x_out = tr(x_in)

    x_out = x_out if batch_first else x_out.permute(1, 0, 2)
    return x_out    #x_out rappresenta gli embedding testuali finali [B, L, D] ?

'''funzione che usa _text_transformer_forward per calcolare gli embedding testuali con CoOp'''
@torch.no_grad()
def encode_text_with_embeddings_eval(
    model, tokens: torch.Tensor, x_embed: torch.Tensor
) -> torch.Tensor:
    x = x_embed + model.positional_embedding.to(x_embed.dtype)
    x = _text_transformer_forward(model, x)
    x = model.ln_final(x)
    eot = tokens.argmax(dim=-1)                                         #trova l'indice del token EOT per ogni prompt (argmax lungo la dimensione della lunghezza del prompt); l’EOT “on the fly”, invece di leggere il buffer.
    x = x[torch.arange(x.shape[0]), eot] @ model.text_projection
    x = F.normalize(x, dim=-1)
    return x

'''funzione che usa _text_transformer_forward per calcolare gli embedding testuali con CoOp (usata in training loop)'''
def encode_text_with_embeddings_train(                                  
    model, tokens: torch.Tensor, x_embed: torch.Tensor
) -> torch.Tensor:
    x = x_embed + model.positional_embedding.to(x_embed.dtype)
    x = _text_transformer_forward(model, x)
    x = model.ln_final(x)
    eot = tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0]), eot] @ model.text_projection
    x = F.normalize(x, dim=-1)
    return x


# =========================================================
# Valutazione (dtype allineati) con progress bar
# =========================================================

'''valuta il modello con i text_features dati (zero-shot o CoOp) ritornando un dizionario con top1 e macro_f1'''
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

            tf = text_features.to(device=img_feat.device, dtype=img_feat.dtype)  # guardia dtype/device
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


# ==================================================================================================================
# Training loop (CoOp) con AMP + GradScaler + progress bar
# ==================================================================================================================

def train_coop(
    model,
    tokenizer,
    classnames: List[str],
    train_loader: DataLoader,               #dataloader di train (obbligatorio)
    val_loader: Optional[DataLoader],       #può essere dataloader o None (se None, salta val)
    device: torch.device,                   
    n_ctx: int = 16,                        #numero di token di contesto CoOp (16 default)
    mode: str = "shared",                   #'shared' = "unified context" | 'per_class'="class-specific" (shared default)
    template_ctx: str = "a photo of a {}",
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    use_tqdm: bool = True,
    use_amp: bool = True,
) -> Tuple["PromptLearner", Dict[str, float]]:      #return Tuple[prompt_learner, best_metrics]

    prompt_learner = PromptLearner(                 # -> PromptLearner() -> implementa la funzione forward che restituisce i token apprendibili e costruisce i prompt(embedding testuali) per ogni classe
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        n_ctx=n_ctx,
        template_ctx=template_ctx,
        mode=mode,
        device=device,
    ).to(device)                                    #spostima prompt_learner su GPU

    for p in model.parameters():                    #FREEZA tutto CLIP: nessun gradiente calcolato per i pesi del modello base, allena solo i token del prompr
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW([prompt_learner.ctx], lr=lr, weight_decay=weight_decay)       #ottimizzatore AdamW che aggiorna solo i parametri ctx del prompt_learner implica che ctx è un nn.Parameter ovvero un tensore apprendibile dentro una nn.Module
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_amp))          #oggetto che scala la loss per evitare underflow in mixed precision (abilitato solo se device è cuda e use_amp=True)

    best = {"top1": 0.0, "macro_f1": 0.0}               #dizionario per tenere traccia delle migliori metriche di validazione

    epoch_iter = (                                      #costruisce l'iteratore epoche 
        trange(1, epochs + 1, desc="Epochs", dynamic_ncols=True, bar_format=BAR_FORMAT)
        if use_tqdm
        else range(1, epochs + 1)
    )

    amp_ctx = torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp)) #crea contesto per autocast (automatic mixed precision) se device è cuda e use_amp=True

    for epoch in epoch_iter:
        prompt_learner.train()                         #imposta prompt_learner in modalità train (abilita dropout ecc. se presenti)
        model.eval()                                    #imposta modello CLIP in modalità eval (disabilita dropout, batchnorm ecc.)

        running_loss, num_samples = 0.0, 0

        batch_iter = maybe_tqdm( train_loader , enabled=use_tqdm , desc=f"Train {epoch}/{epochs}" , leave=False , position=1 , dynamic_ncols=True , bar_format=BAR_FORMAT )  #costruisce l'iteratore batch con o senza tqdm

        for images, targets in batch_iter:                        #itero sui batch del dataloader di train    
            images = images.to(device, non_blocking=True)           #sposto immagini su device (GPU) con non_blocking=True per permettere il trasferimento asincrono se le immagini sono in pinned memory
            targets = targets.to(device)                            #sposto targets su device (GPU)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:                                                            #(automatic mixed precision) se non fosse per questo, con modelli grandi come ViT-L/14 si esaurirebbe la memoria GPU
                tokens, x = prompt_learner()
                txt_feat = encode_text_with_embeddings_train(model, tokens, x)  # [C, D]        # -> encode_text_with_embeddings_train() -> usa _text_transformer_forward per calcolare gli embedding testuali con CoOp

                img_feat = model.encode_image(images)                                       # [B, D]
                img_feat = F.normalize(img_feat, dim=-1)                                    #normalizza gli embedding immagine (L2 norm) lungo la dimensione delle feature (D)

                logits = img_feat @ txt_feat.t()                                            # [B, C]  #calcola i logits come prodotto scalare tra img_feat [B, D] e txt_feat trasposto [D, C] --> [B, C]               
                scale = model.logit_scale.exp().to(logits.dtype)                            #lo esponenzio per ottenere la scala effettiva  
                logits = logits * scale

                loss = F.cross_entropy(logits, targets)                                     #calcola la loss di cross-entropy tra i logits predetti e le etichette target

            scaler.scale(loss).backward()                                                   #scala la loss e calcola i gradienti (backward) in mixed precision
            scaler.step(optimizer)                                                   #aggiorna i parametri dell'ottimizzatore usando i gradienti scalati                
            scaler.update()                                                             #aggiorna lo scaler per la prossima iterazione (adatta il fattore di scala in base a overflow/underflow)

            running_loss += loss.item() * images.size(0)                                   #accumula la loss totale (loss per batch * numero di campioni nel batch)
            num_samples += images.size(0)                                                   #accumula il numero totale di campioni visti    

            if use_tqdm:                                                                    #se uso tqdm, aggiorno la barra di progresso con loss e lr attuali
                batch_iter.set_postfix(                     
                    loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]["lr"]
                )

        train_loss = running_loss / max(1, num_samples)                    #calcola la loss media per epoca: loss totale / numero totale campioni

        # Val di fine epoca
        prompt_learner.eval()                                               #imposta prompt_learner in modalità eval (disabilita dropout ecc. se presenti)
        with torch.no_grad():                                               #(non serve in val calcolo grad!!)
            tokens, x = prompt_learner()                                    #ottengo i token e gli embedding testuali con i prompt aggiornati       
            txt_feat_val = encode_text_with_embeddings_eval(model, tokens, x)       # [C, D]   # -> encode_text_with_embeddings_eval() -> usa _text_transformer_forward per calcolare gli embedding testuali con CoOp

        if val_loader is not None:
            val_metrics = evaluate(                                         #valuta le performance del modello sul validation set
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


# ==================================================================================================================
# Main
# ==================================================================================================================

def main():
    parser = argparse.ArgumentParser(description="CoOp per CLIP con dataset ImageFolder")
    #creo un argument parser per passare i parametri da linea di comando --> description appare nell'help con (-h o --/help)
    parser.add_argument("--data", type=str, default=DATA_ROOT, help="cartella con train/valid/test")
    # --data argomento di tipo string, default DATA_ROOT, help appare nell'help
    parser.add_argument("--model", type=str, default="ViT-B-32", help="nome modello OpenCLIP (es. ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="tag pretrained per OpenCLIP")
    parser.add_argument("--n-ctx", type=int, default=16, help="numero di token di contesto CoOp")                                               #quanti token di contesto apprendere
    parser.add_argument("--mode", type=str, default="shared", choices=["shared", "per_class"], help="token condivisi o per-classe")             # tokens-shared default, per_class opzione alternativa
    parser.add_argument("--template-zero", type=str, default="a photo of a {}", help="template testuale per baseline zero-shot")                #template da migliorare 
    parser.add_argument("--template-ctx", type=str, default="a photo of a {}", help="template usato dentro il prompt CoOp")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)                                                                                       #learning rate                      
    parser.add_argument("--weight-decay", type=float, default=0.0)                                                                              #peso per la regolarizzazione L2 optimizer AdamW (se usato)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="0 => Se il codice fa un fallback prendi SLURM_CPUS_PER_TASK o 2")
    parser.add_argument("--batch-train", type=int, default=64)                                                                                  #batch size per il train
    parser.add_argument("--batch-eval", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=0, help="usa solo N campioni per overfit di prova (0=full)")
    parser.add_argument("--epochs-overfit", type=int, default=0, help="se >0, overfitta su train (opz. con --limit-train) e salta val/test")
    parser.add_argument("--no-tqdm", action="store_true", help="disattiva le barre di progresso")                                                #se specificato disattiva le barre di progresso                    
    parser.add_argument("--no-amp", action="store_true", help="disattiva mixed precision (utile per debug)")                                     #se specificato disattiva mixed precision -> messaggi di errore più chiari
    #parse_known_args() ignora argomenti sconosciuti 
    # Notebook-safe
    args, unknown = parser.parse_known_args()                                                                #restituisce una tupla (args, unknown) dove args sono gli argomenti noti e unknown quelli sconosciuti   
    if unknown:                                                                                              #se unknown non è vuoto non si blocca ma stampa un avviso
        print(f"[NOTE] Argomenti extra ignorati: {unknown}")                
    if "ipykernel" in sys.modules:
        sys.argv = sys.argv[:1]

    set_seed(args.seed)                                                                   # -> set_seed() 
    use_tqdm = not args.no_tqdm                                                             #se non specificato --no-tqdm allora use_tqdm = True         
    use_amp = not args.no_amp                                                               #se non specificato --no-amp allora use_amp = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # auto num_workers se lanciato con --cpus-per-task
    if args.num_workers <= 0:                                                               #se num_workers <= 0 legge la variabile d'ambiente SLURM_CPUS_PER_TASK (impostata con --cpus-per-task di srun/sbatch) tipico in cluster SLURM, altrimenti usa 2
        args.num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))

    print(f"[INFO] Carico OpenCLIP: {args.model} ({args.pretrained})")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(        #crea e carica modello args.model col checkpoint args.pretrained
        args.model, pretrained=args.pretrained, device=device                                   #restituisce il modello e due trasform (le funzioni di preprocess per train e val/test)
    )
    tokenizer = open_clip.get_tokenizer(args.model)                                         #Restituisce il tokenizer di testo compatibile col modello
    model.eval()                                                                            #imposta il modello in modalità eval (disabilita dropout, batchnorm ecc.)                 

    # --- Dataloaders dal dataset ImageFolder ---
    args.data = args.data or DATA_ROOT                                                      #se non specificato --data usa DATA_ROOT
    train_loader, val_loader, test_loader, classnames = build_imagefolder_splits(           # -> build_imagefolder_splits()
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

    # Se richiesto, modalità "tiny-overfit" su train (debug correttezza)
    if args.epochs_overfit and args.epochs_overfit > 0:
        print("\n[DEBUG] Modalità tiny-overfit su train (salto val/test)")
        prompt_learner, _ = train_coop(                            # -> train_coop()                   
            model=model,
            tokenizer=tokenizer,
            classnames=classnames,
            train_loader=train_loader,
            val_loader=None,                    #forza a non fare validazione periodica
            device=device,
            n_ctx=args.n_ctx,                   #HYPERP: numero di token di contesto CoOp
            mode=args.mode,                     #HYPERP: 'shared' | 'per_class' 
            template_ctx=args.template_ctx,     #HYPERP: template usato dentro il prompt CoOp
            epochs=args.epochs_overfit,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_tqdm=use_tqdm,
            use_amp=use_amp,                    #HYPERP: se usare mixed precision (se no messaggi di errore più chiari ma più lenta l'esecuzione)
        )
        print("[DEBUG] Fine tiny-overfit.")
        return                                  #esce senza fare la valutazione finale su val/test : chiude main()

    # --- Zero-shot baseline sia val e test ---
    print("\n[BASELINE] Valutazione zero-shot su val/test...")
    text_feat_zero = build_zeroshot_text_features(model, tokenizer, classnames, args.template_zero, device, use_amp)        # -> build_zeroshot_text_features()
    #text_feat_zero: tensore [num_classi, dim_embedding_testuale] 
     
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

    # --- CoOp training ---
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

    # --- CoOp evaluation ---
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

    # --- Report comparativo + diagnosi ---
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
