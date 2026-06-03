import torch
import torch_dct as dct
import torchvision.transforms.functional as tf

from PIL import Image
from typing import Literal


def color_map(x: torch.Tensor | Image.Image,
              size: int,
              method: Literal["gaussian", "identity"] = "identity"
              ):
    """
    Color map computed with 2D-DCT
    :param x: image tensor or PIL
    :param size: thresholding parameter in the DCT
    :param method: /
    :return: tensor of the color map
    """
    if type(x) is Image.Image:
        x = tf.to_tensor(x)
    # 2d DCT of x
    X = dct.dct_2d(x)
    # Filter
    if method == "gaussian":
        tsh_filter = torch.zeros_like(X)
    elif method == "identity":
        tsh_filter = torch.zeros_like(X)
        for i in range(size):
            tsh_filter[:, i, i] = 1
    else:
        raise RuntimeError("Non implemented method: ", method)
    X = torch.matmul(tsh_filter, X)
    X = torch.matmul(X, tsh_filter)
    # inverse 2d DCT
    x_out = dct.idct_2d(X)
    return x_out, x_out


def color_map_vis(c):
    c = c.to("cpu")
    x = torch.ones((3, 768, 768))
    x[:, :, :384] *= c[0].view(-1, 1, 1)
    x[:, :, 384:] *= c[1].view(-1, 1, 1)
    return x
