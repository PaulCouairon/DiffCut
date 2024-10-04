from typing import Optional, Tuple, Literal, List
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F

from diffusers import AutoPipelineForText2Image, DDIMScheduler

def build_ldm_from_cfg(model_key: str,
                       device: int = 0):
    print('Loading SD model')
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')

    pipe = AutoPipelineForText2Image.from_pretrained(model_key, torch_dtype=torch.float16).to(device)

    pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            )
        
    print('SD model loaded')
    return pipe, device

class LdmExtractor(nn.Module):

    LDM_CONFIGS = {
        "SSD-1B": ("segmind/SSD-1B", "XL"),
        "SSD-vega": ("segmind/Segmind-Vega", "XL"),
        "SD1.4": ("CompVis/stable-diffusion-v1-4", None)
    }

    def __init__(
        self,
        model_name: str = "SSD-1B",
        device: int = 0, 
    ):

        super().__init__()

        self.model_name = model_name
        model_key, sd_version = self.LDM_CONFIGS[self.model_name]

        self.text_encoders = []
        self.pipe, self.device = build_ldm_from_cfg(model_key, device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoders.append(self.pipe.text_encoder)

        if sd_version == "XL":
            self.text_encoders.append(self.pipe.text_encoder_2)

        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(50)
    
    def register_hooks(self):
        hook_handles = []
        if self.model_name == "SD1.4":
            attn_block = self.unet.down_blocks[-2].attentions[-1].transformer_blocks[-1].attn1
        else:
            attn_block = self.unet.down_blocks[-1].attentions[-1].transformer_blocks[-1].attn1

        def hook_self_attn(mod, input, output):
            self._features = output.detach()
        hook_handles.append(attn_block.register_forward_hook(partial(hook_self_attn)))
        return hook_handles

    def do_classifier_free_guidance(self, guidance_scale):
        return guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @torch.no_grad()
    def get_text_embeds(self, prompt, num_images_per_prompt=1, guidance_scale=1.0, img_size=1024):
        do_classifier_free_guidance = self.do_classifier_free_guidance(guidance_scale)
        batch_size = len(prompt)

        prompt_embeds_tuple = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance)

        if len(prompt_embeds_tuple) == 2:
            prompt_embeds, negative_prompt_embeds = prompt_embeds_tuple
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            return prompt_embeds, None

        else:
            (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ) = prompt_embeds_tuple

            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self.pipe._get_add_time_ids(
                (img_size, img_size), (0, 0), (img_size, img_size), dtype=prompt_embeds.dtype, \
                    text_encoder_projection_dim=self.text_encoders[1].config.projection_dim)
            negative_add_time_ids = add_time_ids

            if self.do_classifier_free_guidance(guidance_scale):
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            prompt_embeds = prompt_embeds.to(self.device)
            add_text_embeds = add_text_embeds.to(self.device)
            add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            return prompt_embeds, added_cond_kwargs

    @torch.no_grad()
    def encode_to_latent(self, input_image):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            input_image = 2 * input_image - 1
            posterior = self.vae.encode(input_image).latent_dist
            latent_image = posterior.mean * self.vae.config.scaling_factor
        return latent_image

    def forward(self,
                img,
                num_images_per_prompt: int = 1,
                guidance_scale: float = 1.,
                step: Tuple[int, ...] = 50,
                img_size: int = 1024,
                ):

        batch_size = img.shape[0]
        images = F.interpolate(img, size=(img_size, img_size), mode='bilinear')
        batch_size = images.shape[0]

        rng = torch.Generator(device=self.device).manual_seed(42)

        prompt_embeds, added_cond_kwargs = self.get_text_embeds([""] * batch_size, num_images_per_prompt, guidance_scale, img_size)

        latent_image = self.encode_to_latent(images)

        noise = torch.randn(1, 4, img_size//8, img_size//8, generator=rng, device=self.device)
        noise = noise.expand_as(latent_image)

        hook_handles = self.register_hooks()

        t = torch.tensor([step], device=self.device).expand(batch_size)
        
        noisy_latent_image = self.pipe.scheduler.add_noise(latent_image, noise, t)

        if self.do_classifier_free_guidance(guidance_scale):
            noisy_latent_image = torch.cat([noisy_latent_image] * 2)

        t = t.repeat(noisy_latent_image.shape[0] // t.shape[0])

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                noisy_latent_image = self.pipe.scheduler.scale_model_input(noisy_latent_image, t)
                self.unet(noisy_latent_image, t, encoder_hidden_states=prompt_embeds, \
                            added_cond_kwargs=added_cond_kwargs).sample
                
        for hook_handle in hook_handles:
            hook_handle.remove()

        return self._features
