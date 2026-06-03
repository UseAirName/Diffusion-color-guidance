CFG = {
    "exp_name": "default_config",  # name of the experiment output folder
    "data_dir": "./test_images/",  # path to your image folder
    "cache_dir": "./cache/",  # path to cache models
    "save_dir": "./output_folder/",  # path for the output folder
    "clip_scale": 4.1,
    "color_scale": 6.1,
    "steps": 50,
    "image_dim": 768,
    "nb_color": 25,
    "log_n_image": 1,  # number of intermediate image logged
    "repeat": 4,
    "lambda_vals": "./lambdas/lambda_ts.txt",
    "mean_shift_vals": "./lambdas/mean_shift_latent.txt",
    "std_shift_vals": "./lambdas/std_shift_latent.txt",
    "prompt": "",
    "neg_prompt": "painting, drawing, blurry",
    "cmap_b": 8,  # number of bits to encode color channels
    "clip_b": 5,  # number of bits to encode clip vector components
    "guide_diffusion": True  # Leave true to guide the diffusion with color
}
