import torch
import torch.nn as nn

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** -0.5))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * self.g / norm.clamp(min=self.eps)

def remove_weight_norm(model):
    for name, module in model.named_modules():
        try:
            if hasattr(module, 'weight_g'):
                # print(f"Removing weight norm from {name}")
                torch.nn.utils.remove_weight_norm(module)
        except Exception:
            pass

def apply_scalenorm_to_transformer(transformer):
    """
    Experimental: Replaces LayerNorm with ScaleNorm in the transformer.
    Note: This may require retraining or fine-tuning, but can help with 
    stability in half-precision if the pre-trained weights are compatible.
    """
    for name, module in transformer.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, nn.LayerNorm):
                setattr(module, attr_name, ScaleNorm(child.normalized_shape[0], child.eps))
