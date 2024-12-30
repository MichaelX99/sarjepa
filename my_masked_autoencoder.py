from torch import nn
from functools import partial
import torch
import math
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

class MaskedAutoencoderViT(nn.Module):
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

        self.patch_size = patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward_encoder(self, x, window_size, num_windows, mask_ratio):
        assert x.shape[-2] == x.shape[-1], 'Only square images are supported in this implementation'

        # Patch embed the image
        x = self.patch_embed(x)

        # Make a grid of all the token indices that are possible in the image
        H = W = int(math.sqrt(x.shape[1]))
        all_inds = torch.arange(0, H*W).view(H, W)

        # Filter out the edge indices that can not be chosen as centroids given the window needs to fully be within the image
        window_pad = window_size // 2
        selectable_inds = all_inds[window_pad:-window_pad,window_pad:-window_pad]

        # Select the window centroid indices
        # Since torch does not have an equivalent to numpy.random.choice to sample N elements from a list without replacement we instead randomly generate indices from a multinomial distribution with uniform weights and take the values that those sampled inds correspond to
        selectable_inds = selectable_inds.reshape(-1)
        ind_weights = torch.ones(selectable_inds.shape[0])
        sampled_idx = ind_weights.multinomial(num_windows, replacement=False)
        window_centroids = selectable_inds[sampled_idx]

        window_offsets = torch.arange(
            torch.ceil(torch.tensor(-1 * window_size / 2)),
            torch.ceil(torch.tensor(window_size / 2)),
            dtype=torch.int8,
            device=x.device,
        )
        window_offsets = torch.vstack([window_offsets for _ in range(window_size)])

        squaring_factor = torch.arange(
            torch.ceil(torch.tensor(-1 * window_size / 2)),
            torch.ceil(torch.tensor(window_size / 2)),
            dtype=torch.int8,
            device=x.device,
        ) * H

        squaring_factor = torch.vstack([squaring_factor for _ in range(window_size)])
        squaring_factor = squaring_factor.T

        window_offsets += squaring_factor
        window_offsets = rearrange(window_offsets, 'h w -> 1 (h w)')

        # Combine the centers with the offsets to get the final token indice sets for each of the windows
        window_coords = window_centroids.unsqueeze(1)
        window_coords = window_coords.repeat(1, window_size**2)

        window_coords += window_offsets
        print(window_coords)
        #exit()

        #####################
        # TODO
        #####################
        # I think the window coords logic might be wrong? the masked tokens in the window doesnt look quite right cause there are white patches at the end however that might be due to the faulty hack for token selection logic i have going so once i get the selection method reevaluate if this is required

        # Select which of the tokens within each window that will be masked
        num_tokens_per_window_to_mask = int(mask_ratio * window_size**2)
        # TODO get that fucking random select method
        masked_inds = window_coords[:, :num_tokens_per_window_to_mask]

        # Replicate the projected image for each unique window
        windowed_x = torch.stack([x for _ in range(num_windows)], dim=1)

        # TODO maybe figure out if there is a vectorizable way to set all the mask tokens
        for ind in range(num_windows):
            windowed_x[:, ind, masked_inds[ind]] = self.mask_token

        import matplotlib.pyplot as plt

        for ind in range(num_windows):
            temp = windowed_x[0, ind, :, 0]
            temp = temp.reshape(H, W)
            plt.figure(str(ind))
            plt.imshow(temp.detach().numpy(), cmap='gray')
        plt.show()


        exit()

        # generate the sampled and mask patches from the small windows
        x, ids_restore, mask_indices = self.generate_window_patches(x, rand_left_locations, rand_top_locations, window_size, mask_ratio)

        # append the cls token to the begining
        cls_token = self.cls_token.expand(N * num_window, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        return None, None, None



    def forward(self, x, window_size=7, num_window=4, mask_ratio=0.8):
        pred, mask_indices, ids_restore = self.forward_encoder(x, window_size, num_window, mask_ratio)




def main():
    model = MaskedAutoencoderViT(
                patch_size=16, embed_dim=192, depth=12, num_heads=3,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                in_chans=3,
            )

    x = torch.zeros(4, 3, 224, 224)

    out = model(x)

if __name__ == "__main__":
    main()
