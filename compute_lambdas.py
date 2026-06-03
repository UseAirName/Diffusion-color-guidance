import os
import torch
import pickle
import numpy as np
import torchvision.transforms.functional as tf

from PIL import Image
from config import CFG
from pathlib import Path
from diffusion.loader import load_diffusion_model, load_clip_model


def encode(x, pipeline, device):
    x = tf.to_tensor(x).to(device)
    z = pipeline.vae.encode(x.unsqueeze(0), return_dict=False)[0].sample()
    return z


def decode(z, pipeline, scale=True):
    z = z / pipeline.vae.config.scaling_factor if scale else z
    x = pipeline.vae.decode(z, return_dict=False)[0]
    return x


def main():
    device = torch.device("cuda")
    pipeline = load_diffusion_model(device, CFG["cache_dir"])
    output_path = os.path.join(CFG["save_dir"], "lambda_test")
    images_path = CFG["data_dir"]
    images_files = list(Path(images_path).glob("*"))[:50]
    file = open(os.path.join(output_path, "lambda_ts.txt"), "rb")
    lambda_ts = pickle.load(file)
    file.close()
    mean_ts = []
    std_ts = []
    with torch.no_grad():
        for t, alpha_t, lambda_t in lambda_ts:
            error_t = []
            for image_file in images_files:
                image_pil = Image.open(image_file).convert("RGB")
                image_pil = image_pil.resize((768, 768))
                x_0 = tf.to_tensor(image_pil)
                z_0 = encode(image_pil, pipeline, device)
                for k in range(5):
                    eps = torch.randn_like(z_0)
                    z_noised = z_0 + lambda_t * ((1-alpha_t)/alpha_t)**0.5 * eps
                    x_noised = decode(z_noised, pipeline, scale=False).squeeze(0).to("cpu")
                    error = x_noised - x_0
                    error = error.flatten()
                    error_t.append(error.mean().cpu())
            mean_ts.append(np.array(error_t).mean())
            std_ts.append(np.array(error_t).std())
        print("JOB is done: writing files")
        file = open(os.path.join(output_path, "mean_shift_latent.txt"), "wb")
        pickle.dump([[lambda_ts[i][0], mean_ts[i]] for i in range(len(lambda_ts))], file)
        file.close()
        file = open(os.path.join(output_path, "std_shift_latent.txt"), "wb")
        pickle.dump([[lambda_ts[i][0], std_ts[i]] for i in range(len(lambda_ts))], file)
        file.close()


def main2():
    device = torch.device("cuda")
    pipeline = load_diffusion_model(device, CFG["cache_dir"])
    clip_model, clip_preproc = load_clip_model(device, CFG["cache_dir"])
    vae_scaling_factor = pipeline.vae.config.scaling_factor
    output_path = os.path.join(CFG["save_dir"], "lambda_test")
    images_path = CFG["data_dir"]
    images_files = list(Path(images_path).glob("*"))[:50]
    ts = []
    alpha_ts = []
    std_ts = []
    pipeline.scheduler.set_timesteps(num_inference_steps=30, device=device)
    timesteps = pipeline.scheduler.timesteps[:]
    with torch.no_grad():
        for t in timesteps:
            ts.append(t.cpu())
            alpha_t = pipeline.scheduler.alphas_cumprod[t]
            alpha_ts.append(alpha_t)
            error_ts = []
            for image_file in images_files:
                image_pil = Image.open(image_file).convert("RGB")
                image_pil = image_pil.resize((768, 768))
                z_0 = encode(image_pil, pipeline, device)
                prompt = ""
                neg_prompt = 'lowres, worst quality, low quality, jpg, blurry, bad, watermark, signature, compressed'
                image_preproc = clip_preproc(tf.to_pil_image(tf.to_tensor(image_pil))).unsqueeze(0).to(device)
                clip_embeds = clip_model.encode_image(image_preproc)

                clip_embeds = pipeline._encode_image(
                    image=None,
                    device=device,
                    batch_size=1,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    noise_level=0,
                    generator=None,
                    image_embeds=clip_embeds,
                )
                prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
                    prompt=prompt,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=neg_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    lora_scale=None,
                )
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                for k in range(5):
                    eps = torch.randn_like(z_0)
                    z_t = alpha_t ** 0.5 * z_0 * vae_scaling_factor + (1 - alpha_t) ** 0.5 * eps
                    z_t_input = torch.cat([z_t] * 2)
                    eps_theta = pipeline.unet(
                        z_t_input,
                        t,
                        class_labels=clip_embeds,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=None,
                        return_dict=False
                    )[0]
                    eps_theta_uncond, eps_theta_text = eps_theta.chunk(2)
                    eps_theta = eps_theta_uncond + 6 * (eps_theta_text - eps_theta_uncond)
                    error = eps_theta - eps
                    error = error.flatten()
                    error_ts.append(error.mean().cpu())
            std_ts.append(np.array(error_ts).std())
    file = open(os.path.join(output_path, "lambda_ts.txt"), "wb")
    pickle.dump([(ts[i], alpha_ts[i], std_ts[i]) for i in range(len(ts))], file)
    file.close()


main()
