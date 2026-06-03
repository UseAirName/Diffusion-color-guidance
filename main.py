import torchvision.utils
import yaml
import torch
import pickle
import os.path
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf

from PIL import Image
from config import CFG
from pathlib import Path
from datetime import datetime
from diffusion.loader import load_diffusion_model, load_clip_model
from diffusion.color_map import color_map, color_map_vis
from diffusion.transforms import quantify_clip, cmap_compression
from diffusion.guided_diffusion import decode


def format_image(image_size, image_pil):
    if image_pil.width < image_pil.height:
        image_pil = image_pil.resize((image_size, int(image_pil.height * image_size / image_pil.width)))
    else:
        image_pil = image_pil.resize((int(image_pil.width * image_size / image_pil.height), image_size))
    return image_pil


def main():
    ### EDIT config.py to adjust the experiment ###

    # Remove warnings from resize antialiasing
    warnings.filterwarnings("ignore")

    # Read config
    exp_name = CFG["exp_name"]
    print("####################################################################")
    print("Launching experiment: " + exp_name)
    data_path = CFG["data_dir"]
    print("Dataset read at:      " + data_path)
    output_path = CFG["save_dir"]
    crop_size = CFG["image_dim"]
    nb_color = CFG["nb_color"]

    # Opening dataset
    image_files = list(Path(data_path).glob("*"))[:]

    # Creating output folder
    now = datetime.now()
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    str_now = now.strftime("%Y-%m-%d_%H:%M")
    output_path = output_path + exp_name + "_" + str_now + "/"
    os.mkdir(output_path)

    # Exporting configuration file
    print("Saving files in:      ", output_path)
    with open(output_path + "lambdas.yaml", "w") as file:
        yaml.dump(CFG, file)

    # Loading models
    cuda_device = torch.device("cuda")
    print("Loading huggingface models ...")
    pipe = load_diffusion_model(cuda_device, CFG["cache_dir"])
    clip_model, clip_preproc = load_clip_model(cuda_device, CFG["cache_dir"])

    # Generating images
    for image_file in image_files:
        print("####################################################################")
        print("Decoding image:       " + str(image_file))
        image_pil = Image.open(image_file).convert("RGB")
        image_pil = format_image(crop_size, image_pil)
        image_tensor = tf.to_tensor(image_pil)

        # Center crop of the image
        image_tensor = tf.center_crop(image_tensor, [crop_size, crop_size])

        # Computing the color map and rate (without the lossless entropy coding)
        c_map, c_map_vis = color_map(image_tensor, nb_color)
        c_map, cmap_bits = cmap_compression(c_map, nb_bits=CFG["cmap_b"])
        c_map = c_map.to(cuda_device)
        save_name = output_path + os.path.basename(os.path.splitext(image_file)[0])
        # Save original image tensor
        tf.to_pil_image(image_tensor).save(save_name + "_0_original.png")
        # Save quantized color map
        tf.to_pil_image(c_map_vis.clamp(0, 1)).save(save_name + "_1_colormap.png")

        with torch.no_grad():
            # Compute clip embeddings
            image_preproc = clip_preproc(tf.to_pil_image(image_tensor)).unsqueeze(0).to(cuda_device)
            clip_embeddings = clip_model.encode_image(image_preproc)
            clip_embeddings, clip_bits = quantify_clip(clip_embeddings, CFG["clip_b"])
            out_dict = decode(
                pipeline=pipe,
                clip_embeds=clip_embeddings,
                clip_guidance=CFG["clip_scale"],
                c_map=c_map,
                c_map_guidance=CFG["color_scale"],
                steps=CFG["steps"],
                image_dim=crop_size,
                color_resolution=nb_color,
                prompt=CFG["prompt"],
                negative_prompt=CFG["neg_prompt"],
                device=cuda_device,
                repeat_guidance=CFG["repeat"],
                log_n_image=CFG["log_n_image"],
                guide_diffusion=CFG["guide_diffusion"],
                return_dict=True
            )
            image_output, loss_values, latent_output, inter_latents, grid_image = \
                out_dict["image"], out_dict["loss"], out_dict["latent"], out_dict["latents"], out_dict["images"]

            image_output.save(save_name + "_2_decoded.png")
            sns.lineplot(loss_values)

            plt.savefig(save_name + "_3_loss_color.png")
            plt.clf()
            if CFG["log_n_image"]:
                grid_image.save(save_name + "_4_logged.png")

if __name__ == "__main__":
    main()
