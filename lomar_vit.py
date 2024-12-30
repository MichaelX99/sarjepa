from torch import nn
import torch
from functools import partial
from einops import rearrange
import math
import numpy as np
from torch.nn import functional as F

from nn_utils import PatchEmbed
from attn_utils import SABlock
from lomar_utils import lomar_masking

class GF(nn.Module):
    def __init__(self, nbins=9, pool=7, kensize=5, img_size=224, patch_size=16):
        super(GF, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.img_size = img_size
        self.patch_size = patch_size
        self.k = kensize

        def creat_kernel(r=1):

            M_13 = np.concatenate([np.ones([r+1, 2*r+1]), np.zeros([r, 2*r+1])], axis=0)
            M_23 = np.concatenate([np.zeros([r, 2 * r + 1]), np.ones([r+1, 2 * r + 1])], axis=0)

            M_11 = np.concatenate([np.ones([2*r+1, r+1]), np.zeros([2*r+1, r])], axis=1)
            M_21 = np.concatenate([np.zeros([2 * r + 1, r]), np.ones([2 * r + 1, r+1])], axis=1)


            return torch.from_numpy((M_13)).float(), torch.from_numpy((M_23)).float(), torch.from_numpy((M_11)).float(), torch.from_numpy((M_21)).float()

        M13, M23, M11, M21 = creat_kernel(self.k)

        weight_x1 = M11.view(1, 1, self.k*2+1, self.k*2+1)
        weight_x2 = M21.view(1, 1, self.k*2+1, self.k*2+1)

        weight_y1 = M13.view(1, 1, self.k*2+1, self.k*2+1)
        weight_y2 = M23.view(1, 1, self.k*2+1, self.k*2+1)

        self.register_buffer("weight_x1", weight_x1)
        self.register_buffer("weight_x2", weight_x2)
        self.register_buffer("weight_y1", weight_y1)
        self.register_buffer("weight_y2", weight_y2)


    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect") + 1e-2
        gx_1 = F.conv2d(
            x, self.weight_x1, bias=None, stride=1, padding=0, groups=1
        )
        gx_2 = F.conv2d(
            x, self.weight_x2, bias=None, stride=1, padding=0, groups=1
        )
        gy_1 = F.conv2d(
            x, self.weight_y1, bias=None, stride=1, padding=0, groups=1
        )
        gy_2 = F.conv2d(
            x, self.weight_y2, bias=None, stride=1, padding=0, groups=1
        )
        gx_rgb = torch.log((gx_1) / (gx_2))
        gy_rgb = torch.log((gy_1) / (gy_2))
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)

        return norm_rgb

class LOMARViT(nn.Module):
    def __init__(self,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        decoder_embed_dim,
        decoder_depth,
        decoder_num_heads,
        mlp_ratio,
        norm_layer,
        in_chans,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # TODO looks like the SARJEPA authors are NOT using any sort of positional encoding

        self.vit_blocks = nn.ModuleList(
            [
                SABlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    norm_layer=norm_layer,
                ) for _ in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.encoder_to_decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                SABlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    norm_layer=norm_layer
                ) for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # TODO probs need to update this hardcoded image size of 256 as well as filter num of 4
        self.decoder_proj = nn.Linear(decoder_embed_dim, 256 * 4)

        nbins = 9
        cell_sz = 8
        img_size = 224 # TODO fix this
        self.sarfeature1 = GF(nbins=nbins,pool=cell_sz,kensize=5,
                                  img_size=img_size,patch_size=patch_size)
        self.sarfeature2 = GF(nbins=nbins,pool=cell_sz,kensize=9,
                                  img_size=img_size,patch_size=patch_size)
        self.sarfeature3 = GF(nbins=nbins,pool=cell_sz,kensize=13,
                                  img_size=img_size,patch_size=patch_size)
        self.sarfeature4 = GF(nbins=nbins,pool=cell_sz,kensize=17,
                                  img_size=img_size,patch_size=patch_size)

    def forward(self, x, window_size=7, num_window=4, mask_ratio=0.8):
        assert x.shape[-2] == x.shape[-1], 'Only square images are supported in this implementation'

        window_coords, window_mask_coords, encoding = self.forward_vit(x, window_size, num_window, mask_ratio)

        pred = self.forward_decoder(encoding)

        loss = self.compute_loss(x, window_coords, window_mask_coords, pred)

        return loss

    def forward_vit(self, x, window_size, num_windows, mask_ratio):
        B = x.shape[0]

        # Patch embed the image
        x = self.patch_embed(x)

        window_coords, window_mask_coords, masked_x = lomar_masking(x, window_size, num_windows, mask_ratio, self.mask_token)

        # append the cls token to the begining
        # TODO investigate if we actually need this, feels like NO
        cls_token = self.cls_token.expand(B, num_windows, -1, -1)
        encoding = torch.cat((cls_token, masked_x), dim=2)

        # TODO investigate masking
        for block in self.vit_blocks:
            encoding = block(encoding)

        return window_coords, window_mask_coords, encoding

    def forward_decoder(self, x):
        pred = self.encoder_to_decoder_proj(self.norm(x))

        for block in self.decoder_blocks:
            pred = block(pred)

        pred = self.decoder_norm(pred)

        pred = self.decoder_proj(pred)

        # TODO look into the class token
        pred = pred[:, :, 1:, :]

        return pred

    def compute_loss(self, x, window_coords, window_mask_coords, pred):
        # Compute the speckle resistant feature extraction
        f1 = self.sarfeature1(x)
        f2 = self.sarfeature2(x)
        f3 = self.sarfeature3(x)
        f4 = self.sarfeature4(x)

        # Patchify each of the extracted features except keep all the spatial information and squash it into the last channel dimension
        f1 = rearrange(f1, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        f2 = rearrange(f2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        f3 = rearrange(f3, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
        f4 = rearrange(f4, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)

        # Combine all the features into a single multiscale feature that will be sliced to get our label
        multiscale_f = torch.cat([f1, f2, f3, f4], dim=-1)

        # Replicate the multiscale features for each window
        num_window = window_coords.shape[0]
        multiscale_f = torch.stack([multiscale_f for _ in range(num_window)], dim=1)

        # Extract from the multiscale features the same patches that were processed during the encoder and decoder
        flattened_label = rearrange(multiscale_f, 'b w t e -> b (w t) e')
        max_num_tokens = f1.shape[1]
        window_offsets = torch.arange(num_window).unsqueeze(1) * max_num_tokens
        linearized_window_coords = window_coords + window_offsets
        linearized_window_coords = linearized_window_coords.reshape(-1)
        windowed_label = flattened_label[:, linearized_window_coords, :]

        # Extract only the patches from the windowed_label that had been masked
        window_size = windowed_label.shape[1] // 4 # TODO make 4 (aka the number of features) not be hardcoded
        window_offsets = torch.arange(num_window).unsqueeze(1) * window_size
        label_idx = window_mask_coords + window_offsets
        label_idx = label_idx.reshape(-1)
        masked_label = windowed_label[:, label_idx, :]

        # Extract only the patches from the prediction that had been masked
        masked_pred = rearrange(pred, 'b w t e -> b (w t) e')
        masked_pred = masked_pred[:, label_idx, :]

        # Compute the mean L2 error over all images in the batch and all patches that had been masked
        loss = (masked_pred - masked_label)**2
        loss = loss.mean()

        return loss


def build_lomar_vit_tiny():
    model = LOMARViT(
                patch_size=16, embed_dim=192, depth=12, num_heads=3,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                in_chans=1,
            )

    return model
