import torch

from PIL import Image
from typing import List
from diffusers import DiffusionPipeline

from .lmbd import decoder_mean_of_lambdas, latent_mean_lambdas, get_lmbda, get_mean
from .guidance_loss import latent_color_loss


def diffusion_step(z_t: torch.Tensor,
                   t: torch.Tensor,
                   clip_embeds: torch.Tensor,
                   prompt_embeds: torch.Tensor,
                   conditional_scale: float,
                   pipeline: DiffusionPipeline
                   ) -> torch.Tensor:
    """
    Step of the diffusion model with conditional guidance
    :param z_t: sample at timestep t
    :param t: timestep
    :param clip_embeds: image embeddings
    :param prompt_embeds: prompt embeddings
    :param conditional_scale: scaling of the conditional diffusion
    :param pipeline: diffusion pipeline
    :return: epsilon_theta (z_t, t, c)
    """
    # Initialize input of unconditional guidance
    z_t_input = torch.cat([z_t] * 2)
    z_t_input = pipeline.scheduler.scale_model_input(z_t_input, t)

    # Apply diffusion model epsilon_theta(z_t, t), epsilon_theta(z_t, t, c)
    eps_theta = pipeline.unet(
        z_t_input,
        t,
        class_labels=clip_embeds,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=None,
        return_dict=False
    )[0]

    # Split variance and mean models, split unconditional and conditional values
    eps_theta_uncond, eps_theta_text = eps_theta.chunk(2)

    # Apply unconditional guidance with scaling
    eps_theta = eps_theta_uncond + conditional_scale * (eps_theta_text - eps_theta_uncond)

    # Return epsilon estimation of the model
    return eps_theta


def guided_diffusion_step(z_t: torch.Tensor,
                          t: torch.Tensor,
                          clip_embeds: torch.Tensor,
                          prompt_embeds: torch.Tensor,
                          c_map: torch.Tensor,
                          conditional_scale: float,
                          guidance_scale: float,
                          resolution: int,
                          pipeline: DiffusionPipeline,
                          eps_error,
                          lambda_mean,
                          lambda_std
                          ) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute a step estimation of the diffusion model with unconditional guidance and classifier guidance
    :param z_t: sample
    :param t: timestep
    :param clip_embeds: image embeddings
    :param prompt_embeds: prompt embeddings
    :param c_map: conditional color map
    :param conditional_scale: scaling of the conditional diffusion
    :param guidance_scale: guidance scale for color guidance
    :param resolution: resolution of the color map
    :param pipeline: diffusion pipeline
    :param eps_error: error on epsilon prediction
    :param lambda_mean: shifting of the mean on decoding
    :param lambda_std: std on decoding
    :return: epsilon estimation after unconditional guidance
    """
    # Compute the gradient of the classifier loss
    # Grad_{z_t} l(z_0_t, sigma)
    with torch.enable_grad():
        # Gradient on z_t
        z_t_var = z_t.detach().requires_grad_(True)

        # Initialize input of unconditional guidance
        z_t_input = torch.cat([z_t_var] * 2)
        z_t_input = pipeline.scheduler.scale_model_input(z_t_input, t)

        # Apply diffusion model epsilon_theta(z_t, t), epsilon_theta(z_t, t, c)
        eps_theta = pipeline.unet(
            z_t_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=clip_embeds,
            cross_attention_kwargs=None,
            return_dict=False
        )[0]

        # Split variance and mean models, split unconditional and conditional values
        eps_theta_uncond, eps_theta_text = eps_theta.chunk(2)

        # Apply unconditional guidance
        eps_theta = eps_theta_uncond + conditional_scale * (eps_theta_text - eps_theta_uncond)

        # Compute z_0_t from epsilon_theta
        alpha_t = pipeline.scheduler.alphas_cumprod[t]
        lmbda_t = get_lmbda(t.item(), eps_error)
        lmbda_t = lmbda_t * ((1 - alpha_t) / alpha_t) ** 0.5
        mean_error = get_mean(lmbda_t, lambda_mean)

        z_0_t = (z_t_var - (1 - alpha_t) ** 0.5 * eps_theta) / alpha_t ** 0.5

        # Compute the guidance loss
        loss = latent_color_loss(c_map, z_0_t, pipeline, mean_error, resolution)

        # Compute the gradient of the loss
        grad = torch.autograd.grad(loss, z_t_var)[0].detach()

    std_error = get_mean(lmbda_t, lambda_std)

    # Compute fine guidance delta
    delta_epsilon = guidance_scale * grad * alpha_t ** 0.5 / (2 * std_error)
    return eps_theta, delta_epsilon, loss


def decode(pipeline: DiffusionPipeline,
           clip_embeds: torch.Tensor,
           clip_guidance: float,
           c_map: torch.Tensor,
           c_map_guidance: float,
           steps: int,
           image_dim: int,
           color_resolution: int,
           device: torch.device,
           prompt: str,
           negative_prompt: str,
           eps_error=latent_mean_lambdas("./lambdas/latent_std_lambda.txt"),
           lambda_mean=decoder_mean_of_lambdas("./lambdas/mean_lambda.txt"),
           lambda_std=decoder_mean_of_lambdas("./lambdas/std_lambda.txt"),
           repeat_guidance: int = 5,
           log_n_image: int = 0,
           guide_diffusion: bool = True,
           return_dict: bool = True
           ) -> [Image.Image, List[float]]:
    """
    Decode image from its clip embeddings using a diffusion model pipeline
    :param pipeline: diffusion pipeline
    :param clip_embeds: clip embeddings of the image
    :param clip_guidance: scale of the clip guidance
    :param c_map: color_map guide
    :param c_map_guidance: guidance scale for the color map
    :param steps: number of diffusion steps
    :param image_dim: resolution of the image
    :param color_resolution: resolution of the color_map
    :param device: torch device
    :param prompt: prompt added to the image embeds
    :param negative_prompt: prompt added to the image embeds
    :param repeat_guidance: number of repeated guidance per step
    :param eps_error: error on epsilon prediction by the diffusion model
    :param lambda_mean: shifting of the mean on decoding
    :param lambda_std: std on decoding
    :param log_n_image: log n intermediate images
    :param guide_diffusion: set to "False" to not guide with color
    :param return_dict: output type dict
    :return: decoded image from clip_embeds and sigma
    """
    do_classifier_guidance = True

    # Preparing logging images
    log_n_image = log_n_image + 1 if log_n_image else log_n_image
    logged_images = []

    # Record loss values
    values2save = {"latents": [],
                   "latent": [],
                   "loss": []}

    # Set number of timesteps
    pipeline.scheduler.set_timesteps(steps, device=device)
    timesteps = pipeline.scheduler.timesteps[:]

    # Initialize the starting noise
    z_t = None

    # Cast color map to device
    c_map = c_map.to(dtype=pipeline.dtype, device=device)

    # compute text embeddings
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_guidance,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Adjust clip embeddings
    clip_embeds = pipeline._encode_image(
        image=None,
        device=device,
        batch_size=1,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_guidance,
        noise_level=torch.tensor([0]),
        generator=None,
        image_embeds=clip_embeds,
    )

    z_t = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=pipeline.unet.config.in_channels,
        height=image_dim,
        width=image_dim,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=None,
        latents=z_t,
    )

    # Timesteps progress bar
    progress_bar = pipeline.progress_bar(timesteps)

    for i, t in enumerate(progress_bar):
        z_tm1, eps_theta = None, None

        # Do not guide on last step
        if i == len(progress_bar) - 1:
            guide_diffusion = False

        # Guided diffusion
        if guide_diffusion:
            # Repeat guidance on step t
            for k in range(repeat_guidance):
                # Edit eps_theta
                eps_theta, delta_epsilon, loss = guided_diffusion_step(z_t=z_t,
                                                                       t=t,
                                                                       clip_embeds=clip_embeds,
                                                                       prompt_embeds=prompt_embeds,
                                                                       c_map=c_map,
                                                                       conditional_scale=clip_guidance,
                                                                       guidance_scale=c_map_guidance,
                                                                       resolution=color_resolution,
                                                                       pipeline=pipeline,
                                                                       eps_error=eps_error,
                                                                       lambda_mean=lambda_mean,
                                                                       lambda_std=lambda_std)
                # Modify eps_theta
                eps_theta = eps_theta + delta_epsilon
                # Save loss value
                values2save["loss"].append(loss.item())
                loss_desc = "loss_color :" + "{:.3f}".format(loss.item())
                progress_bar.set_description(desc=loss_desc + "| ")

                # Step in the scheduler
                z_tm1 = pipeline.scheduler.step(eps_theta, t, z_t)[0]

                # Rewind one step back
                if k != repeat_guidance - 1:
                    pipeline.scheduler._step_index -= 1
                    tm1 = timesteps[i + 1]
                    alpha_t, alpha_tm1 = pipeline.scheduler.alphas_cumprod[t], pipeline.scheduler.alphas_cumprod[tm1]
                    eps = torch.randn_like(z_tm1)
                    z_t = (alpha_t / alpha_tm1) ** 0.5 * z_tm1 + (1 - alpha_t / alpha_tm1) ** 0.5 * eps
        else:
            eps_theta = diffusion_step(z_t=z_t,
                                       t=t,
                                       clip_embeds=clip_embeds,
                                       prompt_embeds=prompt_embeds,
                                       conditional_scale=clip_guidance,
                                       pipeline=pipeline)
            # Step in the scheduler
            z_tm1 = pipeline.scheduler.step(eps_theta, t, z_t)[0]

        if int(i * log_n_image / len(progress_bar)) != int((i + 1) * log_n_image / len(progress_bar)) \
                and i != len(progress_bar) - 1 \
                and log_n_image:
            tm1 = timesteps[i + 1]
            alpha_tm1 = pipeline.scheduler.alphas_cumprod[tm1]
            z_0_t = (z_tm1 - (1 - alpha_tm1) ** 0.5 * eps_theta) / alpha_tm1 ** 0.5
            image_t = pipeline.vae.decode(z_0_t / pipeline.vae.config.scaling_factor, return_dict=False)[0]
            image_t = pipeline.image_processor.postprocess(image_t, output_type="pil")[0]
            logged_images.append(image_t)
        z_t = z_tm1
        values2save["latents"].append(z_tm1)

    # Decode z_0
    image = pipeline.vae.decode(z_t / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    values2save["image"] = image
    values2save["latent"] = z_t / pipeline.vae.config.scaling_factor

    # Save logged images as grid
    grid = None
    if log_n_image:
        grid = Image.new("RGB", size=(len(logged_images) * image_dim, image_dim))
        for i, img in enumerate(logged_images):
            grid.paste(img, box=(i * image_dim, 0, (i + 1) * image_dim, image_dim))
            values2save["images"] = grid

    pipeline.maybe_free_model_hooks()

    if return_dict:
        return values2save
    else:
        return image


def encode_image(pipeline: DiffusionPipeline, image):
    z = pipeline.vae.encode(image, return_dict=False)[0].sample()
    return z
