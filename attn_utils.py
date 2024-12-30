from torch import nn
from timm.models.layers import DropPath
from einops import rearrange

from nn_utils import MLP
from irpe import build_rpe_with_config

class iRPE_SA(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attn_drop,
        proj_drop,
    ):
        super().__init__()

        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Hardcoding to only return the k RPE
        _, self.k_rpe, _ = build_rpe_with_config(head_dim, num_heads)

    def forward(self, x):
        """
        x has shape [batch, window, token, embed]
        """

        qkv = self.qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b w t (h e) -> b w h t e', h=self.num_heads), qkv)

        attn = q @ k.transpose(-2, -1)
        attn *= self.scale

        # Since iRPE doesn't play well with our custom window dimension, temporarily fix that
        temp_q = rearrange(q, 'b w h t e -> (b w) h t e')
        attn_bias = self.k_rpe(temp_q)
        B = x.shape[0]
        attn_bias = rearrange(attn_bias, '(b w) h t e -> b w h t e', b=B)
        attn += attn_bias

        # TODO investigate masking
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        out = attn @ v

        out = rearrange(out, 'b w h t e -> b w t (h e)')

        out = self.proj(out)

        out = self.proj_drop(out)

        return out

class SABlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio,
        norm_layer,
    ):
        super().__init__()

        # TODO investigate adding back in regularization
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        projection_dropout = 0.
        attn_dropout = 0.

        self.norm1 = norm_layer(embed_dim)
        self.sa = iRPE_SA(embed_dim, num_heads, attn_dropout, projection_dropout)

        self.norm2 = norm_layer(embed_dim)
        mlp_projection_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            embed_dim,
            mlp_projection_dim,
            projection_dropout,
        )


    def forward(self, x):
        """
        x has shape [batch, window, token, embed]
        """
        # Perform spatial information mixing w/ regularization
        x = x + self.drop_path(self.sa(self.norm1(x)))

        # Perform channel information mixing w/ regularization
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
