from torch import nn
from einops import rearrange

class PatchEmbed(nn.Module):
    def __init__(self,
        patch_size,
        in_chans,
        embed_dim
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # do the linear projection of each patch
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, projection_dim, proj_drop):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, projection_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(projection_dim, embed_dim)

        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x has shape [batch, window, token, embed]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
