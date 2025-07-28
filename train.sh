export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-inpainting"
export OUTPUT_DIR="runs/dual_encoder"
export DATA_DIR="../BrushData" # replace with you BrushData path

accelerate launch --mixed_precision="fp16" --main_process_port 0 train.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir $DATA_DIR \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --checkpointing_steps 50000 \
 --validation_steps 50000 \
 --report_to wandb \
 --resume_from_checkpoint latest \
 --num_train_epochs 1 \
 --ues_dual_encoders \
 --use_half_zero