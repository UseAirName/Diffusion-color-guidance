import torch
import torchvision.transforms.functional as tf

from torchvision.transforms import InterpolationMode as InterpMode


RGB2YUV = torch.tensor([
    [0.29900, 0.58700, 0.11400],
    [-.14713, -.28886, 0.43600],
    [0.61500, -.51498, -.10001]
])

YUV2RGB = torch.tensor([
    [1, 0.00000, 1.13983],
    [1, -.39465, -.58060],
    [1, 2.03211, 0.00000]
])


def crop_image(img, width=512, height=512):
    img_h, img_w = img.shape[-2:]
    if img_h < height or img_w < width:
        return None, False
    else:
        img = tf.center_crop(img, [width, height])
    return img, True


def image2cmap(image: torch.Tensor, c_size: int):
    cmap = tf.resize(image, [c_size, c_size], antialias=True)
    return cmap


def rgb_to_yuv(img):
    prod = torch.zeros(img.size(0), img.size(1), img.size(2))

    for x in range(img.size(1)):
        for y in range(img.size(2)):
            prod[:, x, y] = torch.matmul(RGB2YUV, img[:, x, y].reshape(-1, 1)).reshape(1, -1)

    return prod


def yuv_to_rgb(img):
    prod = torch.zeros(img.size(0), img.size(1), img.size(2))

    for x in range(img.size(1)):
        for y in range(img.size(2)):
            prod[:, x, y] = torch.matmul(YUV2RGB, img[:, x, y].reshape(-1, 1)).reshape(1, -1)

    return prod


def down_quant_up_sample(img, nb_bits=3):
    down1 = tf.resize(img[1].unsqueeze(0), [img[1].size(0) // 2, img[1].size(1) // 2], antialias=True)
    down2 = tf.resize(img[2].unsqueeze(0), [img[2].size(0) // 2, img[2].size(1) // 2], antialias=True)

    quant_0, size0 = quantify_cmap(img[0], nb_bits=nb_bits)
    quant_1, size1 = quantify_cmap(down1, nb_bits=nb_bits)
    quant_2, size2 = quantify_cmap(down2, nb_bits=nb_bits)

    size_total = size0 + size1 + size2

    up1 = tf.resize(quant_1, [img[1].size(0), img[1].size(1)], interpolation=InterpMode.NEAREST)
    up2 = tf.resize(quant_2, [img[2].size(0), img[2].size(1)], interpolation=InterpMode.NEAREST)

    to_ret = torch.zeros(img.size(0), img.size(1), img.size(2))
    to_ret[0] = quant_0
    to_ret[1] = up1
    to_ret[2] = up2

    return to_ret, size_total


def cmap_compression(img, nb_bits):
    yuv_img = rgb_to_yuv(img)
    q_img, size = down_quant_up_sample(yuv_img, nb_bits)
    rgb_img = yuv_to_rgb(q_img)
    return rgb_img, size


def normalize(vect, norm):
    return vect * (norm / torch.norm(vect))


def quantify_clip(clip_latent, nb_bits, clamp=1):
    """
    Quantify a Clip latent.
    :param clip_latent: Clip latent
    :param nb_bits: Number of bits to aim
    :param clamp: Max to clamp to
    :return: Quantified latent and, optional, the size of the compressed latent
    """
    q = 1 / (2 ** nb_bits)
    eps = q / 4
    clamped_lat = (torch.clamp(clip_latent, min=-clamp + eps, max=clamp - eps) + clamp) / (2 * clamp)
    quant_lat = torch.floor(clamped_lat / q) * q + q / 2
    quant_lat = torch.nan_to_num(quant_lat, nan=0.0, posinf=clamp - (q / 2), neginf=(q / 2) - clamp)

    quant_lat = (quant_lat * 2 * clamp) - clamp

    size = 768 * nb_bits
    quant_lat = quant_lat/quant_lat.norm() * clip_latent.norm()
    return quant_lat, size


def quantify_cmap(cmap, nb_bits):
    """
    Quantify a color map.
    :param cmap: A color map, 2 or 3-dimensional
    :param nb_bits: Number of bits per coefficient
    :return: Quantified color map and, optional, the size of the compressed color map
    """
    q = 1 / (2 ** nb_bits)
    quant_cmap = torch.floor(cmap / q) * q + q / 2

    if len(cmap.shape) == 3:
        size = cmap.size(0) * cmap.size(1) * cmap.size(2) * nb_bits

    elif len(cmap.shape) == 2:
        size = cmap.size(0) * cmap.size(1) * nb_bits

    else:
        raise ValueError(f'Color map shape is {cmap.shape}, expecting 2 or 3 dimensions.')

    return quant_cmap, size
