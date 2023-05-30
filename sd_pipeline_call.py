from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionPipeline


@torch.no_grad()
def sd_pipeline_call(
        pipeline: StableDiffusionPipeline,
        prompt_embeds: torch.FloatTensor,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
    """ Modification of the standard SD pipeline call to support NeTI embeddings passed with prompt_embeds argument."""

    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 2. Define call parameters
    batch_size = 1
    device = pipeline._execution_device

    neg_prompt = get_neg_prompt_input_ids(pipeline, negative_prompt)
    negative_prompt_embeds, _ = pipeline.text_encoder(
        input_ids=neg_prompt.input_ids.to(device),
        attention_mask=None,
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        pipeline.text_encoder.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs.
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            if do_classifier_free_guidance:
                latent_model_input = latents
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred_uncond = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1),
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                ###############################################################
                # NeTI logic: use the prompt embedding for the current timestep
                ###############################################################
                embed = prompt_embeds[i] if type(prompt_embeds) == list else prompt_embeds
                noise_pred_text = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embed,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if output_type == "latent":
        image = latents
        has_nsfw_concept = None
    elif output_type == "pil":
        # 8. Post-processing
        image = pipeline.decode_latents(latents)
        # 9. Run safety checker
        image, has_nsfw_concept = pipeline.run_safety_checker(image, device, pipeline.text_encoder.dtype)
        # 10. Convert to PIL
        image = pipeline.numpy_to_pil(image)
    else:
        # 8. Post-processing
        image = pipeline.decode_latents(latents)
        # 9. Run safety checker
        image, has_nsfw_concept = pipeline.run_safety_checker(image, device, pipeline.text_encoder.dtype)

    # Offload last model to CPU
    if hasattr(pipeline, "final_offload_hook") and pipeline.final_offload_hook is not None:
        pipeline.final_offload_hook.offload()

    if not return_dict:
        return image, has_nsfw_concept

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def get_neg_prompt_input_ids(pipeline: StableDiffusionPipeline,
                             negative_prompt: Optional[Union[str, List[str]]] = None):
    if negative_prompt is None:
        negative_prompt = ""
    uncond_tokens = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
    uncond_input = pipeline.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return uncond_input
