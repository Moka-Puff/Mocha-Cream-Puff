import math
import torch
import torch.nn as nn

NORM_EPS = 1e-6

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, kernel_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=NORM_EPS)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=NORM_EPS)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.q(x).reshape(
            b, n, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_kv = x.transpose(1, 2).reshape(b, c, h, w)
            x_kv = self.sr(x_kv)
            x_kv = x_kv.flatten(2).transpose(1, 2)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        kv = self.kv(x_kv).reshape(
            b, -1, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, h, w):
        x = self.fc1(x)
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.attn = EfficientSelfAttention(
            dim=dim, num_heads=num_heads, sr_ratio=sr_ratio
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.mlp = MixFFN(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        return x


class MiTB5(nn.Module):
    def __init__(
        self,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        depths=(3, 6, 40, 3),
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.1,
    ):
        super().__init__()
        assert len(embed_dims) == len(num_heads) == len(mlp_ratios) == len(depths) == len(sr_ratios) == 4
        self.embed_dims = embed_dims
        self.depths = depths

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(3, embed_dims[0], kernel_size=7, stride=4),
            OverlapPatchEmbed(embed_dims[0], embed_dims[1], kernel_size=3, stride=2),
            OverlapPatchEmbed(embed_dims[1], embed_dims[2], kernel_size=3, stride=2),
            OverlapPatchEmbed(embed_dims[2], embed_dims[3], kernel_size=3, stride=2),
        ])

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.blocks = nn.ModuleList()
        current = 0
        for i in range(4):
            stage = nn.ModuleList([
                TransformerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    sr_ratio=sr_ratios[i],
                    drop_path=dpr[current + j],
                )
                for j in range(depths[i])
            ])
            self.blocks.append(stage)
            current += depths[i]

        self.norms = nn.ModuleList([
            nn.LayerNorm(dim, eps=NORM_EPS) for dim in embed_dims
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        outputs = []
        for i in range(4):
            x, h, w = self.patch_embeds[i](x)
            for block in self.blocks[i]:
                x = block(x, h, w)
            x = self.norms[i](x)
            b, n, c = x.shape
            x = x.transpose(1, 2).reshape(b, c, h, w)
            outputs.append(x)
        return outputs
