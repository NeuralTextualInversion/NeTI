import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPTokenizer

sys.path.append(".")
sys.path.append("..")

import constants
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from models.xti_attention_processor import XTIAttenProc
from checkpoint_handler import CheckpointHandler
from utils import vis_utils


@dataclass
class InferenceConfig:
    # Specifies which checkpoint iteration we want to load
    iteration: Optional[int] = None
    # The input directory containing the saved models and embeddings
    input_dir: Optional[Path] = None
    # Where the save the inference results to
    inference_dir: Optional[Path] = None
    # Specific path to the mapper you want to load, overrides `input_dir`
    mapper_checkpoint_path: Optional[Path] = None
    # Specific path to the embeddings you want to load, overrides `input_dir`
    learned_embeds_path: Optional[Path] = None
    # List of prompts to run inference on
    prompts: Optional[List[str]] = None
    # Text file containing a prompts to run inference on (one prompt per line), overrides `prompts`
    prompts_file_path: Optional[Path] = None
    # List of random seeds to run on
    seeds: List[int] = field(default_factory=lambda: [42])
    # If you want to run with dropout at inference time, this specifies the truncation indices for applying dropout.
    # None indicates that no dropout will be performed. If a list of indices is provided, will run all indices.
    truncation_idxs: Optional[Union[int, List[int]]] = None
    # Whether to run with torch.float16 or torch.float32
    torch_dtype: str = "fp16"

    def __post_init__(self):
        assert bool(self.prompts) != bool(self.prompts_file_path), \
            "You must provide either prompts or prompts_file_path, but not both!"
        self._set_prompts()
        self._set_input_paths()
        self.inference_dir.mkdir(exist_ok=True, parents=True)
        if type(self.truncation_idxs) == int:
            self.truncation_idxs = [self.truncation_idxs]
        self.torch_dtype = torch.float16 if self.torch_dtype == "fp16" else torch.float32

    def _set_input_paths(self):
        if self.inference_dir is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify inference_dir"
            self.inference_dir = self.input_dir / f"inference_{self.iteration}"
        if self.mapper_checkpoint_path is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify mapper_checkpoint_path"
            self.mapper_checkpoint_path = self.input_dir / f"mapper-steps-{self.iteration}.pt"
        if self.learned_embeds_path is None:
            assert self.input_dir is not None, "You must pass an input_dir if you do not specify learned_embeds_path"
            self.learned_embeds_path = self.input_dir / f"learned_embeds-steps-{self.iteration}.bin"

    def _set_prompts(self):
        if self.prompts_file_path is not None:
            assert self.prompts_file_path.exists(), f"Prompts file {self.prompts_file_path} does not exist!"
            self.prompts = self.prompts_file_path.read_text().splitlines()


@pyrallis.wrap()
def main(infer_cfg: InferenceConfig):
    train_cfg, mapper = CheckpointHandler.load_mapper(infer_cfg.mapper_checkpoint_path)
    pipeline, placeholder_token, placeholder_token_id = load_stable_diffusion_model(
        pretrained_model_name_or_path=train_cfg.model.pretrained_model_name_or_path,
        mapper=mapper,
        learned_embeds_path=infer_cfg.learned_embeds_path,
        torch_dtype=infer_cfg.torch_dtype
    )
    prompt_manager = PromptManager(tokenizer=pipeline.tokenizer,
                                   text_encoder=pipeline.text_encoder,
                                   timesteps=pipeline.scheduler.timesteps,
                                   unet_layers=constants.UNET_LAYERS,
                                   placeholder_token=placeholder_token,
                                   placeholder_token_id=placeholder_token_id,
                                   torch_dtype=infer_cfg.torch_dtype)
    for prompt in infer_cfg.prompts:
        output_path = infer_cfg.inference_dir / prompt.format(placeholder_token)
        output_path.mkdir(exist_ok=True, parents=True)
        for truncation_idx in infer_cfg.truncation_idxs:
            print(f"Running with truncation index: {truncation_idx}")
            prompt_image = run_inference(prompt=prompt,
                                         pipeline=pipeline,
                                         prompt_manager=prompt_manager,
                                         seeds=infer_cfg.seeds,
                                         output_path=output_path,
                                         num_images_per_prompt=1,
                                         truncation_idx=truncation_idx)
            if truncation_idx is not None:
                save_name = f"{prompt.format(placeholder_token)}_truncation_{truncation_idx}.png"
            else:
                save_name = f"{prompt.format(placeholder_token)}.png"
            prompt_image.save(infer_cfg.inference_dir / save_name)


def run_inference(prompt: str,
                  pipeline: StableDiffusionPipeline,
                  prompt_manager: PromptManager,
                  seeds: List[int],
                  output_path: Optional[Path] = None,
                  num_images_per_prompt: int = 1,
                  truncation_idx: Optional[int] = None) -> Image.Image:
    with torch.autocast("cuda"):
        with torch.no_grad():
            prompt_embeds = prompt_manager.embed_prompt(prompt,
                                                        num_images_per_prompt=num_images_per_prompt,
                                                        truncation_idx=truncation_idx)
    joined_images = []
    for seed in seeds:
        generator = torch.Generator(device='cuda').manual_seed(seed)
        images = sd_pipeline_call(pipeline,
                                  prompt_embeds=prompt_embeds,
                                  generator=generator,
                                  num_images_per_prompt=num_images_per_prompt).images
        seed_image = Image.fromarray(np.concatenate(images, axis=1)).convert("RGB")
        if output_path is not None:
            save_name = f'{seed}_truncation_{truncation_idx}.png' if truncation_idx is not None else f'{seed}.png'
            seed_image.save(output_path / save_name)
        joined_images.append(seed_image)
    joined_image = vis_utils.get_image_grid(joined_images)
    return joined_image


def load_stable_diffusion_model(pretrained_model_name_or_path: str,
                                learned_embeds_path: Path,
                                mapper: Optional[NeTIMapper] = None,
                                num_denoising_steps: int = 50,
                                torch_dtype: torch.dtype = torch.float16) -> Tuple[StableDiffusionPipeline, str, int]:
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = NeTICLIPTextModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch_dtype,
    )
    if mapper is not None:
        text_encoder.text_model.embeddings.set_mapper(mapper)
    placeholder_token, placeholder_token_id = CheckpointHandler.load_learned_embed_in_clip(
        learned_embeds_path=learned_embeds_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        text_encoder=text_encoder,
        tokenizer=tokenizer
    ).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_denoising_steps, device=pipeline.device)
    pipeline.unet.set_attn_processor(XTIAttenProc())
    return pipeline, placeholder_token, placeholder_token_id


if __name__ == '__main__':
    main()
