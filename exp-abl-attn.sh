#!/bin/bash
#SBATCH --job-name=ablation_attn
#SBATCH --gres=gpu:L40S:1   
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --array=0-4

datasets=("banking77" "clinic150" "nlu" "trec" "trecfine")
max_blocks=(16 20 22 21 20)
subset_blocks=(6 7 7 7 7)
index=$SLURM_ARRAY_TASK_ID

model="togethercomputer/LLaMA-2-7B-32K"
# model="meta-llama/Llama-3.1-8B"
dataset=${datasets[$index]}
n_runs=10
output_dir="./attn"
output_file="./logs/${dataset}_attn.txt"
> $output_file
block_size=50

for multiplier in 1; do
    max_block=$((max_blocks[$index] * multiplier))

    echo "Running with max_blocks=$max_block, using full selection" >> $output_file

    echo "full attention" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window $block_size \
        --n-windows $max_block  \
        --block-select-method all \
        --n-selected-blocks $max_block \
        --attn-prev-blocks 2 \
        --attn-sink-blocks -1 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "sink + prev + self" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window $block_size \
        --n-windows $max_block  \
        --block-select-method all \
        --n-selected-blocks $max_block \
        --attn-prev-blocks 2 \
        --attn-sink-blocks 1 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait
    
    echo "sink + self" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window $block_size \
        --n-windows $max_block  \
        --block-select-method all \
        --n-selected-blocks $max_block \
        --attn-prev-blocks 0 \
        --attn-sink-blocks 1 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "self" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window $block_size \
        --n-windows $max_block  \
        --block-select-method all \
        --n-selected-blocks $max_block \
        --attn-prev-blocks 0 \
        --attn-sink-blocks 0 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "all done" >> $output_file
done
