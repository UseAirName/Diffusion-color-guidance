import torch
import torchvision.transforms
import torchvision.transforms.functional as tf

from .color_map import color_map
from diffusers import DiffusionPipeline


def latent_color_loss(c_x: torch.Tensor,
                      z_0_t: torch.Tensor,
                      pipeline: DiffusionPipeline,
                      mean_error: float,
                      resolution: int

                      ) -> torch.Tensor:
    """
    Loss between conditional color map and estimated color map at timestep t
    :param c_x: conditional color map
    :param z_0_t: estimated real latent at timestep t
    :param pipeline: diffusion pipeline (here for the decoder VAE)
    :param mean_error: estimated shifting of the mean
    :param resolution: resolution of the color map
    :return: guidance loss value
    """
    # Decoded estimated image
    x_0_t = pipeline.vae.decode(z_0_t / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    x_0_t = x_0_t * 0.5 + 0.5
    x_0_t = x_0_t.clamp(0, 1).squeeze(0)

    # Estimate color maps
    c_0_t, cmap = color_map(x_0_t - 0 * mean_error * torch.ones_like(x_0_t), resolution)

    # Resize color maps
    c_x = tf.resize(c_x, [resolution, resolution],
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=False)
    c_0_t = tf.resize(c_0_t, [resolution, resolution],
                      interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=False)

    # Compute loss between color maps
    loss = torch.nn.functional.mse_loss(c_x, c_0_t, reduction="sum")
    return loss
