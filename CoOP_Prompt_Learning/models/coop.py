from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptLearner(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        classnames: List[str],
        n_ctx: int = 16,
        template_ctx: str = "a photo of a {}",
        mode: str = "shared",  
        init_std: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert mode in ("shared", "per_class")
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.n_ctx = n_ctx
        self.mode = mode
        self.device = device or next(model.parameters()).device

        self.text_width = getattr(model.token_embedding, "embedding_dim", None)
        if self.text_width is None:
            self.text_width = model.text_projection.shape[0]
        self.ctx_dtype = torch.float32
        self.model_text_dtype = model.text_projection.dtype

        self.prompts_for_classes = [
            ("X " * n_ctx + template_ctx.format(c)).strip() for c in classnames
        ]
        self.register_buffer("tokenized_prompts", tokenizer(self.prompts_for_classes))

        if mode == "shared":
            ctx = torch.empty(n_ctx, self.text_width, dtype=self.ctx_dtype)
            nn.init.normal_(ctx, std=init_std)
            self.ctx = nn.Parameter(ctx)  
        else:
            ctx = torch.empty(
                len(classnames), n_ctx, self.text_width, dtype=self.ctx_dtype
            )
            nn.init.normal_(ctx, std=init_std)
            self.ctx = nn.Parameter(ctx)  

        with torch.no_grad():
            toks = self.tokenized_prompts
            self.register_buffer("eot_indices", toks.argmax(dim=-1))

    def forward(self):
        tokens = self.tokenized_prompts.to(self.device)
        x = self.model.token_embedding(tokens).to(self.model_text_dtype)  

        if self.mode == "shared":
            ctx = (
                self.ctx.to(self.model_text_dtype)
                .unsqueeze(0)
                .expand(x.size(0), -1, -1)
            )  
        else:
            ctx = self.ctx.to(self.model_text_dtype)  

        x[:, 1 : 1 + self.n_ctx, :] = ctx  
        return tokens, x  

def _text_transformer_forward(model, x_embed: torch.Tensor) -> torch.Tensor:
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

    x_in = x_embed if batch_first else x_embed.permute(1, 0, 2)  

    try:
        if attn_mask is not None:
            x_out = tr(x_in, attn_mask=attn_mask)
        else:
            x_out = tr(x_in)
    except TypeError:
        x_out = tr(x_in)

    x_out = x_out if batch_first else x_out.permute(1, 0, 2)
    return x_out    

@torch.no_grad()
def encode_text_with_embeddings_eval(
    model, tokens: torch.Tensor, x_embed: torch.Tensor
) -> torch.Tensor:
    x = x_embed + model.positional_embedding.to(x_embed.dtype)
    x = _text_transformer_forward(model, x)
    x = model.ln_final(x)
    eot = tokens.argmax(dim=-1)
    x = x[torch.arange(x.shape[0]), eot] @ model.text_projection
    x = F.normalize(x, dim=-1)
    return x

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
