from typing import List

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import is_wandb_available
from tqdm import tqdm
from transformers import CLIPTokenizer

from training.config import RunConfig
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.xti_attention_processor import XTIAttenProc
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call

if is_wandb_available():
    import wandb


class ValidationHandler:

    def __init__(self, cfg: RunConfig, placeholder_token_id: int, weights_dtype: torch.dtype):
        self.cfg = cfg
        self.placeholder_token_id = placeholder_token_id
        self.weight_dtype = weights_dtype

    def infer(self, accelerator: Accelerator,
              tokenizer: CLIPTokenizer,
              text_encoder: NeTICLIPTextModel,
              unet: UNet2DConditionModel, vae: AutoencoderKL,
              prompts: List[str],
              num_images_per_prompt: int,
              seeds: List[int],
              step: int):
        """ Runs inference during our training scheme. """
        pipeline = self.load_stable_diffusion_model(accelerator, tokenizer, text_encoder, unet, vae)
        prompt_manager = PromptManager(tokenizer=pipeline.tokenizer,
                                       text_encoder=pipeline.text_encoder,
                                       timesteps=pipeline.scheduler.timesteps,
                                       placeholder_token=self.cfg.data.placeholder_token,
                                       placeholder_token_id=self.placeholder_token_id)
        joined_images = []
        for prompt in prompts:
            images = self.infer_on_prompt(pipeline=pipeline,
                                          prompt_manager=prompt_manager,
                                          prompt=prompt,
                                          num_images_per_prompt=num_images_per_prompt,
                                          seeds=seeds)
            prompt_image = Image.fromarray(np.concatenate(images, axis=1))
            joined_images.append(prompt_image)
        final_image = Image.fromarray(np.concatenate(joined_images, axis=0))
        final_image.save(self.cfg.log.exp_dir / f"val-image-{step}.png")
        self.log_with_accelerator(accelerator, joined_images, step=step)
        del pipeline
        torch.cuda.empty_cache()
        text_encoder.text_model.embeddings.mapper.train()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        return final_image

    def infer_on_prompt(self, pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        prompt: str,
                        seeds: List[int],
                        num_images_per_prompt: int = 1) -> List[Image.Image]:
        prompt_embeds = self.compute_embeddings(prompt_manager=prompt_manager, prompt=prompt)
        all_images = []
        for idx in tqdm(range(num_images_per_prompt)):
            generator = torch.Generator(device='cuda').manual_seed(seeds[idx])
            images = sd_pipeline_call(pipeline,
                                      prompt_embeds=prompt_embeds,
                                      generator=generator,
                                      num_images_per_prompt=1).images
            all_images.extend(images)
        return all_images

    @staticmethod
    def compute_embeddings(prompt_manager: PromptManager, prompt: str) -> torch.Tensor:
        with torch.autocast("cuda"):
            with torch.no_grad():
                prompt_embeds = prompt_manager.embed_prompt(prompt)
        return prompt_embeds

    def load_stable_diffusion_model(self, accelerator: Accelerator,
                                    tokenizer: CLIPTokenizer,
                                    text_encoder: NeTICLIPTextModel,
                                    unet: UNet2DConditionModel,
                                    vae: AutoencoderKL) -> StableDiffusionPipeline:
        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(self.cfg.model.pretrained_model_name_or_path,
                                                           text_encoder=accelerator.unwrap_model(text_encoder),
                                                           tokenizer=tokenizer,
                                                           unet=unet,
                                                           vae=vae,
                                                           torch_dtype=self.weight_dtype)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(self.cfg.eval.num_denoising_steps, device=pipeline.device)
        pipeline.unet.set_attn_processor(XTIAttenProc())
        text_encoder.text_model.embeddings.mapper.eval()
        return pipeline

    def log_with_accelerator(self, accelerator: Accelerator, images: List[Image.Image], step: int):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log({"validation": [wandb.Image(image, caption=f"{i}: {self.cfg.eval.validation_prompts[i]}")
                                            for i, image in enumerate(images)]})
