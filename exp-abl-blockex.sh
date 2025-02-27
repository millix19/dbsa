#!/bin/bash
#SBATCH --partition=preempt          
#SBATCH --job-name=icl_abl_blockex
#SBATCH --gres=gpu:L40S:1   
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL  # Send email at job start, end, and failure
#SBATCH --mail-user=user@andrew.cmu.edu  # Replace with your email address
#SBATCH --requeue  # Allows the job to be requeued after preemption
#SBATCH --array=0-4  # Array job with 5 tasks (0 to 4)

# Usage
# cd long-context-icl
# conda activate icl
# sbatch exp.sh

# max fit on 32k for all datasets
# banking77 838
# clinic 1169
# nlu 1309
# trec 1129

# Define parameter sets
datasets=("banking77" "clinic150" "nlu" "trec" "trecfine")
max_blocks=(16 20 22 21 20)
subset_blocks=(6 7 7 7 7)

# Get the index of the current job
index=$SLURM_ARRAY_TASK_ID

# Set parameters for the current job
dataset=${datasets[$index]}
max_block=$((max_blocks[$index]))
subset_block=$((subset_blocks[$index]))
n_runs=10
output_dir="./abl-blockex"
output_file="./logs/${dataset}_abl_blockex.txt"
> $output_file
block_size=50

# start exp
source ~/miniconda3/bin/activate icl

echo "--- block VS example: block retrieval without cache ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model togethercomputer/LLaMA-2-7B-32K \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window $block_size \
    --n-windows $max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $subset_block \
    --use-retrieval \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait

echo "--- block VS example: block retrieval with cache ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model togethercomputer/LLaMA-2-7B-32K \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window $block_size \
    --n-windows $max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $subset_block \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait

ex_max_block=$((max_block * block_size))
ex_subset_block=$((subset_block * block_size))


echo "--- block VS example: example retrieval without cache ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model togethercomputer/LLaMA-2-7B-32K \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window 1 \
    --n-windows $ex_max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $ex_subset_block \
    --use-retrieval \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait

echo "--- block VS example: example retrieval with cache ---" >> $output_file
python run_evaluation.py \
    --dataset $dataset \
    --model togethercomputer/LLaMA-2-7B-32K \
    --subsample-test-set 250 \
    --n-runs $n_runs \
    --n-shots-per-window 1 \
    --n-windows $ex_max_block  \
    --block-select-method bm25 \
    --n-selected-blocks $ex_subset_block \
    --overwrite \
    --fp16 \
    --output-dir $output_dir >> $output_file
wait