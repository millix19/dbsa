#!/bin/bash
#SBATCH --job-name=efficiency
#SBATCH --gres=gpu:A100_80G:1   
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=50G

##### testing for second stage inference efficiency #######

model="meta-llama/Llama-3.1-8B"

echo "Running cached fixed ICL evaluation"
python run_evaluation.py \
--dataset banking77 \
--model $model \
--subsample-test-set 250 \
--n-runs 5 \
--n-shots-per-window 50 \
--n-windows 48  \
--block-select-method all \
--n-selected-blocks 48 \
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
--model meta-llama/Llama-3.1-8B \
--subsample-test-set 250 \
--n-runs 1 \
--n-shots-per-window 750 \
--attn-implementation flash_attention_2 \
--use-retrieval \
--overwrite \
--n-windows 1 \
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

##### to test first stage pre-encoding efficiency, change to --attn-implementation flex_attention #######
wait
echo "Running DBSA"
python run_evaluation.py \
--dataset banking77 \
--model $model \
--subsample-test-set 250 \
--n-runs 5 \
--n-shots-per-window 50 \
--n-windows 48  \
--block-select-method bm25 \
--n-selected-blocks 15 \
--attn-prev-blocks 2 \
--attn-sink-blocks 1 \
--attn-implementation flash_attention_2 \
--overwrite \
--fp16 \
--output-dir ./test

wait
echo "all done"