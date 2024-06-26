export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/opt/ml/model"
export HUB_MODEL_ID="sd15-pokemon-lora"
export DATASET_NAME="reach-vb/pokemon-blip-captions"

accelerate launch --config-file="sagemaker_config.yaml" train_lora.py 
  # --pretrained_model_name_or_path=$MODEL_NAME \
  # --dataset_name=$DATASET_NAME \
  # --mixed_precision="bf16" \
  # --dataloader_num_workers=8 \
  # --resolution=512 \
  # --center_crop=True \
  # --random_flip=True \
  # --train_batch_size=6 \
  # --gradient_accumulation_steps=1 \
  # --max_train_steps=15000 \
  # --learning_rate=1e-04 \
  # --max_grad_norm=1 \
  # --lr_scheduler="cosine" \
  # --lr_warmup_steps=0 \
  # --output_dir=${OUTPUT_DIR} \
  # --report_to=wandb \
  # --checkpointing_steps=500 \
  # --validation_prompt="A pokemon with blue eyes." \
  # --seed=1337 \
  # --resume_from_checkpoint=checkpoint-3500