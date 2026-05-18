from typing import List
import torch
import torch.nn as nn

class PromptEnsemble(nn.Module):
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
            self.text_features_templates = nn.functional.normalize(
                self.text_features_templates, dim=-1
            )

        num_templates = len(self.templates)
        num_classes = len(self.classnames)

        if per_class:
            self.template_logits = nn.Parameter(torch.zeros(num_classes, num_templates))
        else:
            self.template_logits = nn.Parameter(torch.zeros(num_templates))

        if init_uniform:
            with torch.no_grad():
                if per_class:
                    self.template_logits[:] = 0.0  
                else:
                    self.template_logits[:] = 0.0

    @torch.no_grad()
    def _build_text_features(self) -> torch.Tensor:
        self.model.eval()
        all_feats = []
        for tmpl in self.templates:
            prompts = [tmpl.format(name.replace("_", " ")) for name in self.classnames]
            tokenized = self.tokenizer(prompts)
            tokenized = tokenized.to(self.device)
            feats = self.model.encode_text(tokenized)
            all_feats.append(feats)  
        return torch.stack(all_feats, dim=0)

    def get_text_features_weighted(self) -> torch.Tensor:
        if self.per_class:
            w = nn.functional.softmax(self.template_logits, dim=-1)  
            t = self.text_features_templates.permute(1, 0, 2)  
            weighted = (w.unsqueeze(-1) * t).sum(dim=1)  
        else:
            w = nn.functional.softmax(self.template_logits, dim=0)  
            weighted = (w.view(-1, 1, 1) * self.text_features_templates).sum(dim=0)  
        weighted = nn.functional.normalize(weighted, dim=-1)
        return weighted  

    @torch.no_grad()
    def get_text_features_uniform(self) -> torch.Tensor:
        feats = self.text_features_templates.mean(dim=0)
        feats = nn.functional.normalize(feats, dim=-1)
        return feats
