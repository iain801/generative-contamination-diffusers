from utils import *
from generation import *

import os

from tqdm.auto import tqdm
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data import DataLoader


from datasets import load_dataset
from accelerate import Accelerator
from accelerate import PartialState
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset, Dataset
    
if __name__ == "__main__":
        
    pipe = load_diffuser_from_pretrained(
        "models/stable-diffusion-2-1-base", 
        adapter_path="sddata/finetune/lora/celeba/pytorch_lora_weights.safetensors",
        adapter_name="celeba",
        device="cuda:1",
        compile=False
    )
    
    distributed_state = PartialState()
    pipe.to(distributed_state.device)
    
    prompts = load_dataset('data/celebhq-caption-10k')['train']['text']
        
    batch_size = 20
    
    out_dir = "outputs/sd21-celeba-itr1/"
    

    # Pre-encode all prompts
    # encoded_prompts = pipe.encode_prompt(prompts, pipe.device, len(prompts), True)
    prompt_loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)
    

    outputs = []
    for batch in tqdm(prompt_loader, desc="Generating images", disable=not distributed_state.is_main_process):
        batch_outputs = batch_generate(pipe, batch)
        outputs.extend(batch_outputs)
        print()    
    
    os.makedirs(out_dir, exist_ok=True)
        
    gen_dataset = Dataset.from_dict({
        "image": outputs,
        "text": prompts
    }, split='train')
    
    gen_dataset.to_parquet(out_dir + "dataset.parquet")
    
    os.makedirs(out_dir + "images/", exist_ok=True)
    for i, img in enumerate(outputs[:100]):
        img.save(out_dir + f'images/{i}.png')