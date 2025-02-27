#!/bin/bash

dataset_name=banking77
num_train_samples=800

python finetuning/run_classification.py \
    --model_name_or_path meta-llama/Llama-3.1-8B \
    --dataset_name $dataset_name \
    --shuffle_train_dataset \
    --shuffle_seed 42 \
    --seed 42 \
    --data_seed 42 \
    --test_file finetuning/data/${dataset_name}_test.csv \
    --text_column_names text \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --weight_decay 0.01 \
    --save_strategy no \
    --max_train_samples $num_train_samples \
    --num_train_epochs 30 \
    --peft \
    --learning_rate 1e-3 \
    --fp16 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --eval_accumulation_steps 128 \
    --gradient_checkpointing \
    --output_dir /data/datasets/hf_cache/icl/outputs/allshot/banking77/llama3_banking77_examples=${num_train_samples}_run=0_peft=True_init=random_seed=42 \
    --run_name llama3_banking77_examples=${num_train_samples}_run=0_peft=True_init=random_seed=42
