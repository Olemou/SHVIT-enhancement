import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_


# -----------------------------
# Basic Modules
# -----------------------------
class MultiScaleFeature(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Project to even feature space for splitting
        self.expand = nn.Conv2d(in_channels, 8, kernel_size=1)  # expand to 8 channels
        split_channels = 4  # split into 4+4

        def feature_branch(kernel_size):
            return nn.Sequential(
                nn.Conv2d(split_channels, split_channels, kernel_size=1),
                nn.BatchNorm2d(split_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(split_channels, split_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(split_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(split_channels, split_channels, kernel_size=1),
                nn.BatchNorm2d(split_channels)
            )

        self.branch1 = feature_branch(kernel_size=3)
        self.branch2 = feature_branch(kernel_size=7)

        self.fuse = nn.Sequential(
            nn.Conv2d(8, in_channels, kernel_size=1),  # compress back to original channels
            nn.BatchNorm2d(in_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        # Step 1: expand channels
        x_proj = self.expand(x)  # shape: (B, 8, H, W)

        # Step 2: split into two branches
        x1, x2 = torch.chunk(x_proj, 2, dim=1)

        # Step 3: process each branch
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)

        # Step 4: concatenate and fuse
        out = torch.cat([out1, out2], dim=1)
        out = self.fuse(out)

        # Step 5: residual connection
        if residual.shape != out.shape:
            # Project residual if needed
            residual = nn.functional.interpolate(residual, size=out.shape[2:], mode='bilinear', align_corners=False)

        out += residual
        return self.relu(out)
    
    
class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02, p_dropout=0.3):
        """
        a: input features
        b: output features
        bias: whether linear has bias
        std: initialization std
        p_dropout: dropout probability (0.0 = no dropout)
        """
        super().__init__()
        self.add_module('bn', nn.BatchNorm1d(a))
        if p_dropout > 0:
            self.add_module('dropout', nn.Dropout(p_dropout))
        self.add_module('l', nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            nn.init.constant_(self.l.bias, 0)



class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class SHSA(nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.ReLU(), Conv2d_BN(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, qk_dim, pdim, type):
        super().__init__()
        if type == "s":
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
            self.ffn = Residual(FFN(dim, int(dim * 2)))
        else:
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
            self.mixer = nn.Identity()
            self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


# -----------------------------
# SPARO Block
# -----------------------------
class SPARO(nn.Module):
    def __init__(self, dim, L, attn_dim, value_dim=None, num_heads=None, share_kv=True):
        super().__init__()
        if value_dim is None:
            value_dim = attn_dim

        self.L = L
        self.attn_dim = attn_dim
        self.value_dim = value_dim
        self.num_heads = num_heads if num_heads is not None else L
        self.num_subheads = L // self.num_heads
        self.share_kv = share_kv
        self.attn_map = None
        self.scale = self.attn_dim ** -0.5

        self.q_emb = nn.Parameter(torch.randn(1, self.num_heads, self.num_subheads, 1, self.attn_dim))
        self.k = nn.Linear(dim, self.num_heads * self.attn_dim)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.zeros_(self.k.bias)

        if not share_kv:
            self.v = nn.Linear(dim, self.num_heads * self.value_dim)
            nn.init.xavier_uniform_(self.v.weight)
            nn.init.zeros_(self.v.bias)
        else:
            assert self.value_dim == self.attn_dim
            self.v = self.k

        self.proj = nn.Linear(self.value_dim, self.value_dim)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, eos_indices=None):
        # x: [B, N, D]
        B, N = x.shape[:2]
        q = self.q_emb  # [1, nh, L/nh, 1, H]
        k = self.k(x).view(B, N, self.num_heads, 1, self.attn_dim).expand(
            -1, -1, -1, self.num_subheads, -1).permute(0, 2, 3, 1, 4)
        v = self.v(x).view(B, N, self.num_heads, 1, self.value_dim).expand(
            -1, -1, -1, self.num_subheads, -1).permute(0, 2, 3, 1, 4)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, nh, L/nh, 1, N]

        if eos_indices is not None:
            attn_mask = torch.arange(N, device=x.device)[None, None, None, None, :] > eos_indices[:, None, None, None, None]
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.L, -1)
        self.attn_map = attn.view(B, self.L, N)
        return x


# -----------------------------
# SHViT with SPARO last block
# -----------------------------
class SHViTEnhanced(nn.Module):
    def __init__(self, in_chans=3, num_classes=200,
                 embed_dim=[128, 224, 320], partial_dim=[32, 48, 68],
                 qk_dim=[16, 16, 16], depth=[2, 4, 5], types=["i", "s", "s"],
                 distillation=False):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Sequential(
            Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), nn.ReLU(),
            Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
        )
        self.multiscale = MultiScaleFeature(in_channels=3)

        # Build blocks
        self.blocks1 = nn.Sequential(*[BasicBlock(embed_dim[0], qk_dim[0], partial_dim[0], types[0]) for _ in range(depth[0])])
        self.downsample1 = PatchMerging(embed_dim[0], embed_dim[1])
        self.blocks2 = nn.Sequential(*[BasicBlock(embed_dim[1], qk_dim[1], partial_dim[1], types[1]) for _ in range(depth[1])])
        self.downsample2 = PatchMerging(embed_dim[1], embed_dim[2])
        #self.blocks3 = nn.Sequential(*[BasicBlock(embed_dim[2], qk_dim[2], partial_dim[2], types[2]) for _ in range(depth[2]-1)])

        # SPARO last block
        self.sparo = SPARO(embed_dim[-1], L=16, attn_dim=embed_dim[-1], value_dim=embed_dim[-1], num_heads=4, share_kv=True)

        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.multiscale(x)
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.downsample1(x)
        x = self.blocks2(x)
        x = self.downsample2(x)
        #x = self.blocks3(x)

        # SPARO expects [B, N, C] sequence
        B, C, H, W = x.shape

        x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_seq = self.sparo(x_seq)             # [B, L, value_dim]


        x = x_seq.mean(dim=1)                  # [B, value_dim]

        # Removed erroneous F.adaptive_avg_pool2d call
        if self.distillation:
            x1, x2 = self.head(x), self.head_dist(x)
            x = (x1 + x2) / 2 if not self.training else (x1, x2)
        else:
            x = self.head(x)
        return x
