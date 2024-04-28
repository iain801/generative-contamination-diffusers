import os
import torch

from diffusers import DiffusionPipeline, AutoPipelineForText2Image

from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import cast_training_params

from peft import LoraConfig, get_peft_model

from accelerate import Accelerator

from typing import Literal
import logging

def load_diffuser_from_pretrained(pretrained_path : str, adapter_path : str = None, accelerator : Accelerator = None, adapter_name : str = "adapter", device : str = "cpu", compile : bool = False) -> DiffusionPipeline:
    
    pipe : DiffusionPipeline = AutoPipelineForText2Image.from_pretrained(
        pretrained_path, torch_dtype=torch.bfloat16
        ).to(device)
    
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, 'text_encoder_2'):
        pipe.text_encoder_2.requires_grad_(False)
    
    if accelerator: 
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        pipe.to(accelerator.device, dtype=weight_dtype)
        
    if adapter_path:
        adapter_dir, adapter_filename = os.path.split(adapter_path)
        pipe.load_lora_weights(adapter_dir, weight_name=adapter_filename, adapter_name=adapter_name)

    if compile:
        logging.basicConfig(level=logging.WARNING)
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
        
    return pipe
    
def attach_new_lora(pipeline : DiffusionPipeline, adapter_name : str, lora_target : Literal["unet","encoder","both"] = "both",
                    unet_rank : int = 32, unet_alpha : int = 16, unet_dropout : float = 0, 
                    encoder_rank : int = 8, encoder_alpha : int = 8, encoder_dropout : float = 0, 
                    seed : int = 0):
    
    adapt_unet = lora_target == "unet" or lora_target == "both"
    adapt_te = lora_target == "encoder" or lora_target == "both"
    
    lora_layers = []
    
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline, 'text_encoder_2'):
        pipeline.text_encoder_2.requires_grad_(False)
    
    # New adapter for encoder
    if adapt_unet:
        unet = pipeline.unet
        unet_modules = [name for name, module in unet.named_modules() if isinstance(module, torch.nn.Linear)]
        
        unet_cfg = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=unet_modules,
            r=unet_rank,
            lora_alpha=unet_alpha,
            lora_dropout=unet_dropout,
        )
            
        peft_unet = get_peft_model(unet, unet_cfg)

        peft_unet_state_dict = {name.replace('.default.weight','.weight') : value for name, value in peft_unet.base_model.model.state_dict(keep_vars=True).items()}
        peft_unet_alphas = {name : unet_alpha for name, item in peft_unet_state_dict.items()}
        pipeline.load_lora_into_unet(peft_unet_state_dict, peft_unet_alphas, unet, adapter_name=adapter_name)
        
        lora_layers.extend(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
        cast_training_params(pipeline.unet, dtype=torch.float32)

    
    # New adapter for encoder
    if adapt_te:
        encoder = pipeline.text_encoder
        encoder_modules = [name for name, module in encoder.named_modules() if isinstance(module, torch.nn.Linear)]

        encoder_cfg = LoraConfig(
            init_lora_weights="loftq",
            target_modules=encoder_modules,
            r=encoder_rank,
            lora_alpha=encoder_alpha,
            lora_dropout=encoder_dropout,
        )

        peft_encoder = get_peft_model(encoder, encoder_cfg)
        peft_encoder_state_dict = {name.replace('.default.weight','.weight') : value for name, value in peft_encoder.base_model.state_dict(keep_vars=True).items()}
        peft_encoder_alphas = {name : encoder_alpha for name, item in peft_encoder_state_dict.items()}
        pipeline.load_lora_into_transformer(peft_encoder_state_dict, peft_encoder_alphas, encoder, adapter_name=adapter_name)
        
        lora_layers.extend(filter(lambda p: p.requires_grad, pipeline.text_encoder.parameters()))
        cast_training_params(pipeline.text_encoder, dtype=torch.float32)

        
    if adapt_te and hasattr(pipeline, 'text_encoder_2'):
        encoder_2 = pipeline.text_encoder_2
        encoder_2_modules = [name for name, module in encoder_2.named_modules() if isinstance(module, torch.nn.Linear)]

        encoder_2_cfg = LoraConfig(
            init_lora_weights="loftq",
            target_modules=encoder_2_modules,
            r=encoder_rank,
            lora_alpha=encoder_alpha,
            lora_dropout=encoder_dropout,
        )

        peft_encoder_2 = get_peft_model(encoder_2, encoder_2_cfg)
        peft_encoder_2_state_dict = {name.replace('.default.weight','.weight') : value for name, value in peft_encoder_2.base_model.state_dict(keep_vars=True).items()}
        peft_encoder_2_alphas = {name : encoder_alpha for name, item in peft_encoder_2_state_dict.items()}
        pipeline.load_lora_into_transformer(peft_encoder_2_state_dict, peft_encoder_2_alphas, encoder_2, adapter_name=adapter_name)
        
        lora_layers.extend(filter(lambda p: p.requires_grad, pipeline.text_encoder_2.parameters()))
        cast_training_params(pipeline.text_encoder_2, dtype=torch.float32)
        
    return lora_layers
    
    
def save_lora_weights(pipeline : DiffusionPipeline, adapter_name : str, output_path : str):

    unet_state_dict = pipeline.unet.state_dict(prefix="unet.")
    unet_lora_state_dict = {key.replace(f'.{adapter_name}.weight','.weight') : value for key, value in unet_state_dict.items() if 'lora' in key and adapter_name in key}
    
    te_state_dict = pipeline.text_encoder.state_dict(prefix="text_encoder.")
    te_lora_state_dict = {key.replace(f'.{adapter_name}.weight','.weight') : value for key, value in te_state_dict.items() if 'lora' in key and adapter_name in key}
    
    if hasattr(pipeline, 'text_encoder_2'):
        te2_state_dict = pipeline.text_encoder_2.state_dict(prefix="text_encoder_2.")
        te_lora_state_dict.update({key.replace(f'.{adapter_name}.weight','.weight') : value for key, value in te2_state_dict.items() if 'lora' in key and adapter_name in key})
    
    pipeline.save_lora_weights(output_path, unet_lora_state_dict, te_lora_state_dict)
    
    
def get_lora_state_dict(pipeline : DiffusionPipeline, adapter_path : str):

    adapter_dir, adapter_filename = os.path.split(adapter_path)
    return pipeline.lora_state_dict(adapter_dir, weight_name=adapter_filename)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


