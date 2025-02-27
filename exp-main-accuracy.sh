#!/bin/bash
#SBATCH --job-name=main
#SBATCH --gres=gpu:A100_80GB:1   
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --array=0-4  

# Define parameter sets
datasets=("banking77" "clinic150" "nlu" "trec" "trecfine")
max_blocks=(16 20 22 21 20)
subset_blocks=(6 7 7 7 7)

# Get the index of the current job
index=$SLURM_ARRAY_TASK_ID

# Set parameters for the current job
dataset=${datasets[$index]}
n_runs=10
output_dir="./long"
output_file="./logs/${dataset}_long.txt"
> $output_file
block_size=50

# To run llama-2 30k accuracy, use model="togethercomputer/LLaMA-2-7B-32K" and multiplier in 1
for multiplier in 3 2 1; do
    max_block=$((max_blocks[$index] * multiplier))
    subset_block=$((subset_blocks[$index] * multiplier))

    echo "Running with max_blocks=$max_block, subset_blocks=$subset_block" >> $output_file

    echo "full attention & full selection" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model meta-llama/Llama-3.1-8B \
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

    echo "retrieval ICL" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model meta-llama/Llama-3.1-8B \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window 1 \
        --n-windows $((max_block * block_size))  \
        --use-retrieval \
        --block-select-method bm25 \
        --n-selected-blocks $((subset_block * block_size)) \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "DBSA" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model meta-llama/Llama-3.1-8B \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window $block_size \
        --n-windows $max_block  \
        --block-select-method bm25 \
        --n-selected-blocks $subset_block \
        --attn-prev-blocks 2 \
        --attn-sink-blocks 1 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "all done for parameter set $((index + 1)) with multiplier $multiplier" >> $output_file
done
