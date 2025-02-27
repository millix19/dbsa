#!/bin/bash
#SBATCH --partition=general          
#SBATCH --job-name=icl_abl_group
#SBATCH --gres=gpu:L40S:1    # for oom, use A100_80GB // L40S
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=user@andrew.cmu.edu  # Replace with your email address
#SBATCH --array=0-4  # Array job with 5 tasks (0 to 4)

# Define parameter sets
datasets=("banking77" "clinic150" "nlu" "trec" "trecfine")
max_blocks=(16 20 22 21 20)
subset_blocks=(6 7 7 7 7)

# Get the index of the current job
index=$SLURM_ARRAY_TASK_ID

# Set parameters for the current job
# "togethercomputer/LLaMA-2-7B-32K", "meta-llama/Llama-3.1-8B"
model="togethercomputer/LLaMA-2-7B-32K"
token="hfg token"
dataset=${datasets[$index]}
max_block=$((max_blocks[$index]))
subset_block=$((subset_blocks[$index]))
block_size=50
n_runs=10
output_dir="./abl-group"
output_file="./logs/${dataset}_abl_group.txt"
> $output_file

# start exp
source ~/miniconda3/bin/activate icl
#:<<END -- END

echo "--- block grouping: random ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model $model \
    --token $token \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window $block_size \
    --n-windows $max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $subset_block \
    --attn-prev-blocks 4 \
    --attn-sink-blocks 2 \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait

echo "--- block grouping: bm25 ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model $model \
    --token $token \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window $block_size \
    --n-windows $max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $subset_block \
    --block-group-method bm25-clustering \
    --attn-prev-blocks 4 \
    --attn-sink-blocks 2 \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait

echo "--- block ordering: bm25-swap ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model $model \
    --token $token \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window $block_size \
    --n-windows $max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $subset_block \
    --block-group-method bm25-clustering-swap \
    --attn-prev-blocks 4 \
    --attn-sink-blocks 2 \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait