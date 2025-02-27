#!/bin/bash
#SBATCH --partition=preempt          
#SBATCH --job-name=efficiency
#SBATCH --gres=gpu:L40S:1   
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=50G

##### testing for second stage inference efficiency #######

# model="togethercomputer/LLaMA-2-7B-32K"
model="meta-llama/Llama-3.1-8B"

echo "Running cached fixed ICL evaluation"
python run_evaluation.py \
--dataset banking77 \
--model $model \
--subsample-test-set 250 \
--n-runs 5 \
--n-shots-per-window 50 \
--n-windows 16  \
--block-select-method all \
--n-selected-blocks 16 \
--attn-prev-blocks 1 \
--attn-sink-blocks -1 \
--attn-implementation flash_attention_2 \
--overwrite \
--fp16 \
--output-dir ./test

wait
echo "Running retICL no cache evaluation"
python run_evaluation.py \
--dataset banking77 \
--model $model \
--subsample-test-set 250 \
--n-runs 5 \
--n-shots-per-window 1 \
--n-windows 800  \
--block-select-method bm25 \
--n-selected-blocks 250 \
--use-retrieval \
--attn-implementation flash_attention_2 \
--overwrite \
--fp16 \
--output-dir ./test

# uncomment zero shot lines in experiment_manager.py
wait 
echo "Running equivalent of finetune zero-shot"
python run_evaluation.py \
--dataset banking77 \
--model $model \
--subsample-test-set 250 \
--n-runs 5 \
--n-shots-per-window 1 \
--n-windows 1  \
--block-select-method all \
--n-selected-blocks 1 \
--attn-implementation flash_attention_2 \
--overwrite \
--fp16 \
--output-dir ./test

# to test first stage pre-encoding efficiency, change to --attn-implementation flex_attention
wait
echo "Running DBSA"
python run_evaluation.py \
--dataset banking77 \
--model $model \
--subsample-test-set 250 \
--n-runs 5 \
--n-shots-per-window 50 \
--n-windows 16  \
--block-select-method bm25 \
--n-selected-blocks 2 \
--attn-prev-blocks 2 \
--attn-sink-blocks 1 \
--attn-implementation flash_attention_2 \
--overwrite \
--fp16 \
--output-dir ./test

wait
echo "all done"