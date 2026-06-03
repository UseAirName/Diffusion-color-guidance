import torch
import open_clip

from diffusers import StableUnCLIPImg2ImgPipeline, DPMSolverMultistepScheduler, DiffusionPipeline


def load_diffusion_model(device: torch.device, cache_dir: str) -> DiffusionPipeline:
    """
    Load diffusion model
    """
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",
                                                           torch_dtype=torch.float32,
                                                           cache_dir=cache_dir).to(device)
    scheduler_parameters = {
        'beta_schedule': 'scaled_linear',
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'prediction_type': 'v_prediction',
        "dynamic_thresholding_ratio": 0.995,
        "num_train_timesteps": 1000,
        "steps_offset": 1,
        "thresholding": False,
        "use_karras_sigmas": True,
        'algorithm_type': 'dpmsolver++'
    }
    pipeline.scheduler = DPMSolverMultistepScheduler(**scheduler_parameters)
    return pipeline


def load_clip_model(device: torch.device, cache_dir: str):
    """
    Load clip model and preprocessing
    """
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14',
                                                                           pretrained='laion2b_s32b_b79k',
                                                                           device=device,
                                                                           cache_dir=cache_dir)

    return clip_model, clip_preprocess
