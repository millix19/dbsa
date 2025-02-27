#!/bin/bash
#SBATCH --job-name=ablations
#SBATCH --gres=gpu:L40S:1   
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --array=0-4  

# Define parameter sets
datasets=("banking77" "clinic150" "nlu" "trec" "trecfine")

# Get the index of the current job
index=$SLURM_ARRAY_TASK_ID

# Set parameters for the current job
model="togethercomputer/LLaMA-2-7B-32K"
# model="meta-llama/Llama-3.1-8B"
dataset=${datasets[$index]}
n_runs=10
output_dir="./ablations"
output_file="./logs/${dataset}_ablations_short.txt"
> $output_file

for multiplier in 1; do
    echo "retrieval ICL" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window 1 \
        --n-windows 1000  \
        --use-retrieval \
        --attn-sink-blocks 1 \
        --block-select-method bm25 \
        --n-selected-blocks 100 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "DBSA retrieval ICL" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window 1 \
        --n-windows 1000  \
        --block-select-method bm25 \
        --n-selected-blocks 100 \
        --attn-prev-blocks 100 \
        --attn-sink-blocks 20 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "block retrieval ICL" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window 20 \
        --n-windows 50  \
        --use-retrieval \
        --attn-sink-blocks 1 \
        --block-select-method bm25 \
        --n-selected-blocks 5 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "DBSA block retrieval ICL" >> $output_file
    python run_evaluation.py \
        --dataset $dataset \
        --model $model \
        --subsample-test-set 250 \
        --n-runs $n_runs \
        --n-shots-per-window 20 \
        --n-windows 50  \
        --block-select-method bm25 \
        --n-selected-blocks 5 \
        --attn-prev-blocks 5 \
        --attn-sink-blocks 1 \
        --overwrite \
        --fp16 \
        --output-dir $output_dir >> $output_file
    wait

    echo "all done for parameter set $((index + 1)) with multiplier $multiplier" >> $output_file
done
