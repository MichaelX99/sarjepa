import math
import torch
from einops import rearrange

def sample_window_centroids(H, W, window_size, num_windows):
    # Make a grid of all the token indices that are possible in the image
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

    return window_centroids

def form_full_window_indices(H, window_centroids, window_size):
    window_offsets = torch.arange(
        torch.ceil(torch.tensor(-1 * window_size / 2)),
        torch.ceil(torch.tensor(window_size / 2)),
        dtype=torch.int8,
    )
    window_offsets = torch.vstack([window_offsets for _ in range(window_size)])

    squaring_factor = torch.arange(
        torch.ceil(torch.tensor(-1 * window_size / 2)),
        torch.ceil(torch.tensor(window_size / 2)),
        dtype=torch.int8,
    ) * H

    squaring_factor = torch.vstack([squaring_factor for _ in range(window_size)])
    squaring_factor = squaring_factor.T

    window_offsets += squaring_factor
    window_offsets = rearrange(window_offsets, 'h w -> 1 (h w)')

    # Combine the centers with the offsets to get the final token indice sets for each of the windows
    window_coords = window_centroids.unsqueeze(1)
    window_coords = window_coords.repeat(1, window_size**2)
    window_coords += window_offsets

    return window_coords

def perform_windowing(x, num_windows, window_coords):
    # Replicate the projected image for each unique window
    stacked_x = torch.stack([x for _ in range(num_windows)], dim=1)

    # TODO explain the logic here
    row_offsets = torch.arange(num_windows) * stacked_x.shape[2]
    row_offsets = row_offsets.unsqueeze(1)
    linearized_coords = window_coords + row_offsets
    linearized_coords = linearized_coords.reshape(-1)

    linearized_x = rearrange(stacked_x, 'b w t e -> b (w t) e')

    windowed_x = linearized_x[:, linearized_coords, :]

    windowed_x = rearrange(windowed_x, 'b (w t) e -> b w t e', w=num_windows)

    return windowed_x

def perform_masking(windowed_x, mask_ratio, mask_value):
    window_size = int(math.sqrt(windowed_x.shape[2]))
    num_windows = windowed_x.shape[1]

    # Select which of the tokens within each window that will be masked
    num_tokens_per_window_to_mask = int(mask_ratio * window_size**2)
    ind_weights = torch.ones((num_windows, window_size**2))
    mask_idx = ind_weights.multinomial(num_tokens_per_window_to_mask, replacement=False)

    # I may be stupid, but I cant figure out how to perform 2D slicing...
    row_offset = torch.arange(mask_idx.shape[0]) * window_size**2
    row_offset = row_offset.unsqueeze(1)
    linearized_mask_idx = mask_idx + row_offset
    linearized_mask_idx = linearized_mask_idx.reshape(-1)

    linearized_masked_x = rearrange(windowed_x, 'b w t e -> b (w t) e')
    linearized_masked_x[:, linearized_mask_idx, :] = mask_value
    
    masked_x = rearrange(linearized_masked_x, 'b (w t) e -> b w t e', w=num_windows)

    return masked_x, mask_idx

def lomar_masking(x, window_size, num_windows, mask_ratio, mask_value):
        H = W = int(math.sqrt(x.shape[1]))

        # Find the centroids of each local window we take from the input
        window_centroids = sample_window_centroids(H, W, window_size, num_windows) # shape [num_windows]

        # Get all the patch indices that fall within each window
        window_coords = form_full_window_indices(H, window_centroids, window_size) # shape [num_windows, window_size**2]

        # Extract the local windows from the embedding
        windowed_x = perform_windowing(x, num_windows, window_coords) # shape [batch_size, num_windows, window_size**2, embedding dim]

        # Randomly mask certain token indices according to the requested ratio
        # masked_x has shape [batch_size, num_windows, window_size**2 embedding_dim]
        # window_mask_inds has shape [num_windows, mask_ratio*(window_size**2)]
        masked_x, window_mask_coords = perform_masking(windowed_x, mask_ratio, mask_value)

        return window_coords, window_mask_coords, masked_x
