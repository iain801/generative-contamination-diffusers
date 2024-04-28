
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from PIL.Image import Image


def generate(pipeline : StableDiffusionPipeline, 
             prompt : str,
             **kwargs
            ) -> Image:
    return batch_generate(pipeline, [prompt], **kwargs)[0]

@torch.no_grad() 
def batch_generate(pipeline : StableDiffusionPipeline, 
                   prompts : list[str], 
                   steps : int = 30, 
                   cfg : float = 9,
                   lora_scale : float = 0.9, 
                   seed : int = 0,
                  ) -> list[Image]:
    return pipeline(
        prompts, 
        num_inference_steps=steps, 
        cross_attention_kwargs={"scale": lora_scale}, 
        guidance_scale=cfg,
        generator=torch.manual_seed(seed)
    ).images
    