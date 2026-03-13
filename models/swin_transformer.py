"""
models/swin_transformer.py
Swin Transformer model builder.
Uses timm if available, otherwise falls back to a lightweight custom implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def build_swin_model(num_classes: int, pretrained: bool = True, model_size: str = "tiny"):
    """
    Build Swin Transformer.
    model_size: 'tiny' | 'small' | 'base'
    """
    try:
        import timm
        name_map = {
            "tiny":  "swin_tiny_patch4_window7_224",
            "small": "swin_small_patch4_window7_224",
            "base":  "swin_base_patch4_window7_224",
        }
        model_name = name_map.get(model_size, "swin_tiny_patch4_window7_224")
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        logger.info(f"Built {model_name} via timm (pretrained={pretrained}, classes={num_classes})")
        return model
    except Exception as e:
        logger.warning(f"timm failed ({e}) — using lightweight fallback.")
        return LightweightSwinTransformer(num_classes=num_classes)


# ── Lightweight fallback (no extra dependencies) ──────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        n  = self.norm1(x)
        x  = x + self.attn(n, n, n)[0]
        x  = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x  = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class LightweightSwinTransformer(nn.Module):
    """
    Lightweight Swin Transformer — no timm required.
    img_size=64 for fast testing, 224 for real training.
    """
    def __init__(
        self,
        img_size=224, patch_size=4, in_chans=3,
        num_classes=4, embed_dim=96,
        depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        mlp_ratio=4.0, drop_rate=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        H = W = img_size // patch_size

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        dim = embed_dim
        for i, (d, nh) in enumerate(zip(depths, num_heads)):
            self.stages.append(nn.Sequential(*[SwinBlock(dim, nh, mlp_ratio, drop_rate) for _ in range(d)]))
            if i < len(depths) - 1:
                self.merges.append(PatchMerging(dim))
                dim *= 2
            else:
                self.merges.append(None)

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x   = self.patch_embed(x)
        H   = W = int(x.shape[1] ** 0.5)
        for stage, merge in zip(self.stages, self.merges):
            x = stage(x)
            if merge is not None:
                x = merge(x, H, W)
                H = W = H // 2
        return self.norm(x).mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))
