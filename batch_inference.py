#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import logging
import gc
import math
import time
import random
from pathlib import Path

import diffusers
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import ptp_utils
from run_videop2p import NullInversion, make_controller
from save_video import save_video
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.data.dataset import TuneAVideoDataset
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import save_videos_grid, ddim_inversion


POS_PROMPT = (
    " ,best quality, extremely detailed, HD, ultra, 8K, HQ, masterpiece, trending on artstation, art, smooth")
NEG_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, deformed body, bloated, ugly, blurry, low res, unaesthetic"
)

data_root = '/workspace/DynEdit'
method_name = 'video-p2p'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    seed=6,
    # TODO define arguments
    pretrained_model_path='/pretrained/stable-diffusion-v1-5',
    mixed_precision="fp16",
    trainable_modules=("attn1.to_q", "attn2.to_q", "attn_temp"),
    learning_rate=3e-5,
    n_sample_frames=24,
    fps=12,
    train_batch_size=1,
    max_train_steps=500, 
    gradient_accumulation_steps=1,
    num_inference_steps=50,  # see run_videop2p.NUM_DDIM_STEPS
    num_inv_steps=50,  # see run_videop2p.NUM_DDIM_STEPS
    guidance_scale=12.5,
))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


logger = get_logger(__name__, log_level="INFO")


def train_model(
    accelerator: Accelerator,
    weight_dtype,
    video_path,
    prompt,
):
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(config.pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(config.trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")
    unet.enable_gradient_checkpointing()

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Get the training dataset
    train_dataset = TuneAVideoDataset(
        video_path, prompt, 512, 512, n_sample_frames=config.n_sample_frames,
    )

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # if global_step % checkpointing_steps == 0:
                #     if accelerator.is_main_process:
                #         save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                #         accelerator.save_state(save_path)
                #         logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            config.pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(config.pretrained_model_path, subfolder="scheduler")
        )
        pipeline.scheduler.set_timesteps(config.num_inference_steps, device=accelerator.device)
        # pipeline.save_pretrained(output_dir)

    accelerator.end_training()
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return pipeline, latents


def main():
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    logger.info(accelerator.state, main_process_only=False)
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    data_config = OmegaConf.load(config.config_file)
    set_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/frames/{row.video_id}'
        # TODO load video

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)
        
        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        pipeline, _ = train_model(accelerator, weight_dtype, video_path, row.prompt)
        # try:
        #     pipeline.disable_xformers_memory_efficient_attention()
        # except AttributeError:
        #     print("Attribute disable_xformers_memory_efficient_attention() is missing")
        pipeline = pipeline.to(torch.float32)
        null_inversion = NullInversion(pipeline, torch.float32)
        _, x_t, uncond_embeddings = null_inversion.invert(
            video_path, row.prompt, offsets=(0,0,0,0), verbose=True, n_sample_frame=config.n_sample_frames)
        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']
            if edit['type'].startswith('compound'):
                edit['type'] = edit['type'].replace('compound:', '')
            blend_words = []
            equilizer_params={'words':[], 'values':[]}
            for edit_type, src, tgt in zip(
                edit['type'].split(','), edit['src_words'].split(','), edit['tgt_words'].split(',')
            ):
                if edit_type == 'stylization':
                    continue
                elif edit_type == 'foreground' or edit_type == 'background':
                    if len(blend_words) == 0:
                        blend_words.append([src,])
                        blend_words.append([tgt,])
                    else:
                        blend_words[0].append(src)
                        blend_words[1].append(tgt)
                else:
                    raise ValueError(f"Unknown edit type {edit_type}")
                equilizer_params['words'].append(tgt)
                equilizer_params['values'].append(4)
            if len(blend_words) == 0:
                blend_words = None

            controller = make_controller(
                prompts=[row['prompt'], edit['prompt']],
                is_replace_controller=(not 'stylization' in edit['type']),
                cross_replace_steps={'default_': 0.2,},
                self_replace_steps=0.5,
                blend_words=blend_words,
                equilizer_params=equilizer_params,
                mask_th = (.3, .3),
                tokenizer=pipeline.tokenizer,
            )
            ptp_utils.register_attention_control(pipeline, controller)

            with torch.no_grad():
                sequence = pipeline(
                    [row['prompt'], edit['prompt']],
                    generator=generator,
                    latents=x_t,
                    uncond_embeddings_pre=uncond_embeddings,
                    controller=controller,
                    video_length=config.n_sample_frames,
                    fast=False,
                    num_inference_steps=config.num_inference_steps,
                ).videos
            video = rearrange(sequence[1], "c t h w -> t c h w")
            print(f'Video id: {row.video_id}')
            print(f'latents', x_t[0].dtype, x_t[0].max())
            print(f'null text', uncond_embeddings[0].dtype, uncond_embeddings[0].max())
            print(f'sequence', sequence[1].dtype, sequence[1].max())
            print(f'video', video.dtype, video.max())
            save_video(output_dir / f'{i}.mp4', video, config.fps)
        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()