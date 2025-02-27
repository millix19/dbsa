import math
import logging
import random
from typing import List, Dict
from collections import Counter
from typing import Optional, Union

import numpy as np
import torch
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from rank_bm25 import BM25Okapi
# from langchain_huggingface import HuggingFaceEmbeddings
from k_means_constrained import KMeansConstrained

from constants import TEXT_BETWEEN_SHOTS, N_TOKENS, PROMPTS
from datasets_loader import LABEL_TOKENS

from logits_processor import RestrictiveTokensLogitsProcessor
from utils import n_tokens_in_prompt, encode_labels, encode_stop_seq, synchronize_examples_across_dfs, retrieve_context, create_retriever, create_example_retriever
from utils import *
from encoding_utils import *
import time
import psutil


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

STOP_SEQUENCE = '\n'


class ExperimentManager:
    def __init__(self, test_df: pd.DataFrame, train_df: pd.DataFrame, model, tokenizer,
                 labels: List[str], random_seed: int = 42, subsample_test_set: int = 250,
                 n_shots_per_window: int = None, test_on_train: bool=False,
                 pcw_base_model: bool = False, context_size: int = 4096, is_classification: bool = True,
                 stride_size: int = -1, examples_stride: int = -1, window_shuffle_seed: int = None,
                 use_retrieval: bool = False, sort_by_label=False, block_select_method=None,
                 block_order_method=None, block_group_method=None, n_selected_blocks=None,
                 attn_prev_blocks=0, attn_sink_blocks=-1):

        if test_on_train:                 # remove examples from train and test if they're only in one 
            train_df, test_df = synchronize_examples_across_dfs(train_df, test_df)

        if subsample_test_set < len(test_df):
            if test_on_train:
                _logger.warning("You're testing on train data; only do this as a sanity check!")
                self.full_test_df = test_df.copy()
            else:
                np.random.seed(random_seed)
                test_df = test_df.sample(subsample_test_set)
        
        self.subsample_test_set = subsample_test_set
        self.sync_shuffle = test_on_train
        self.test_df = test_df
        self.train_df = train_df
        self.model = model
        self.base_random_seed = random_seed
        self.n_shots_per_window = n_shots_per_window
        self.tokenizer = tokenizer
        self.label_distribution_prompt = dict()
        self.label_map = None
        self.reverse_label_map = None
        self.is_classification = is_classification
        self.pcw_base_model = pcw_base_model
        self.context_size = context_size
        self.stride_size = stride_size 
        self.examples_stride = examples_stride
        self.window_shuffle_seed = window_shuffle_seed
        self.use_retrieval = use_retrieval
        self.sort_by_label = sort_by_label
        self.block_select_method = block_select_method
        self.block_order_method = block_order_method
        self.block_group_method = block_group_method
        self.n_selected_blocks = n_selected_blocks
        self.attn_prev_blocks = attn_prev_blocks
        self.attn_sink_blocks = attn_sink_blocks
        
        np.random.seed(random_seed)
        self.random_orders = [np.random.permutation(list(self.train_df.index)) for i in range(20)]
        self.times_shuffled = 0
        if is_classification:
            _logger.info(f"Setting up labels and logt processor for {len(labels)} possible labels")
            self._initialize_labels_and_logit_processor(labels)
        else:
            self._fix_labels_wrt_tokneizer(labels)
            self.logit_processor = None


    def synchronize_examples():
        pass
    
    def _initialize_labels_and_logit_processor(self, labels: List[str]) -> None:
        map_labels, pad, shorten_label_tokens = self._fix_labels_wrt_tokneizer(labels)
        labels_tokens_array = np.array(
            [i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in shorten_label_tokens])
        labels_tokens_array = self.pad_contained_labels_with_stop_seq(shorten_label_tokens, labels_tokens_array)
        self.logit_processor = RestrictiveTokensLogitsProcessor(restrictive_token_ids=labels_tokens_array,
                                                                eos_token_id=self.tokenizer.eos_token_id)
        self.possible_labels = set(map_labels.values())

    def _fix_labels_wrt_tokneizer(self, labels):
        _logger.info(f"Provided labels: {labels}")
        labels_tokens = encode_labels(self.tokenizer, labels)
        labels_tokens_array = self.minimize_labels_tokens(labels_tokens)
        _logger.info(f"Provided labels average n_tokens: {np.round(np.mean([len(lt) for lt in labels_tokens]), 3)}")
        # we fix the labels accordingly in the test set:
        shorten_label_tokens = [t[t != self.tokenizer.eos_token_id].tolist() for t in labels_tokens_array]
        _logger.info(
            f"shortened labels average n_tokens: {np.round(np.mean([len(lt) for lt in shorten_label_tokens]), 3)}")
        # Moving the test set label tokens to their shorter version:
        map_labels = {old_label: self.tokenizer.decode(t).lstrip() for old_label, t in
                      zip(labels, shorten_label_tokens)}
        self.label_map = lambda a: map_labels[a]
        inv_map = {v: k for k, v in map_labels.items()}
        self.reverse_label_map = inv_map
        self.test_df[LABEL_TOKENS] = self.test_df[LABEL_TOKENS].map(map_labels)
        pad = len(max(shorten_label_tokens, key=len))
        self.max_n_tokens = pad

        return map_labels, pad, shorten_label_tokens

    def minimize_labels_tokens(self, labels_tokens: List[List[int]]) -> npt.NDArray[int]:
        """
         Minimize the number of tokens per label to be the shortest possible unique one.
        """
        pad = len(max(labels_tokens, key=len))
        labels_tokens_array = np.array([i + [self.tokenizer.eos_token_id] * (pad - len(i)) for i in labels_tokens])
        for i, tokens in enumerate(labels_tokens):
            for j in range(len(tokens)):
                labels_with_shared_beginnings = np.sum(
                    np.all(labels_tokens_array[:, :j] == np.array(tokens[:j]), axis=1))
                if labels_with_shared_beginnings == 1:
                    labels_tokens_array[i, j:] = self.tokenizer.eos_token_id
                    break
        return labels_tokens_array

    def pad_contained_labels_with_stop_seq(self, labels_tokens: List, labels_tokens_array: npt.NDArray[int]) -> npt.NDArray[int]:
        """
        In case we have two labels, where one label contains the other label (for example: "A" and "A B") we need
        to allow the restrictive decoding to produce the output "A". We support it by adding "\n" to the shorter label.
        """
        stop_seq_token_id = encode_stop_seq(self.tokenizer, STOP_SEQUENCE)
        for i, tokens in enumerate(labels_tokens):
            labels_with_shared_beginnings = np.sum(
                np.all(labels_tokens_array[:, :len(tokens)] == np.array(tokens), axis=1))
            if labels_with_shared_beginnings > 1:
                _logger.info(f"label{self.tokenizer.decode(tokens)} is the beginning of one of the other labels,"
                             f"adding stop sequence to its end")
                labels_tokens_array[i, len(tokens)] = stop_seq_token_id
        return labels_tokens_array

    def _set_random_seed(self, random_seed: int) -> None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_few_shots_acc(self, windows_few_shot: List[str]) -> float:
        if self.block_select_method is not None and not self.use_retrieval:
            predicted_labels = self.get_predicted_labels_select_blocks(windows_few_shot, self.logit_processor)
        elif self.use_retrieval:
            predicted_labels = self.get_predicted_labels_retrieval(windows_few_shot, restrictive_logit_preprocessor=self.logit_processor)
        return self.calc_acc(predicted_labels)


    def get_predicted_labels_retrieval(self, contexts, restrictive_logit_preprocessor, use_majority_vote=False):       
        predicted_labels = []
        
        if self.n_selected_blocks is not None:
            # index, context_df = create_example_retriever(contexts)
            context_df = pd.DataFrame(contexts, columns=['prompts'])
            context_df['text'] = context_df['prompts']
            index = create_retriever(context_df)
        else:
            index = create_retriever(self.train_df)
                    
        for q in tqdm(self.test_df[PROMPTS]):
            if self.n_selected_blocks is not None and self.attn_sink_blocks == 1: # fair comparison for ablations
                context = retrieve_context_ablations(train_df=context_df, index=index, curr_example=q, \
                    n_examples=self.n_selected_blocks, split_text = TEXT_BETWEEN_SHOTS, shuffle_seed=self.window_shuffle_seed) 
            elif self.n_selected_blocks is not None: # retrieve from context with possible reordering
                selected_blocks = self.block_selection_method(q, len(contexts), self.n_selected_blocks, self.block_select_method, index, self.block_order_method)
                dps = [context_df.loc[context_df['id'] == block_id, 'text'].values[0] for block_id in selected_blocks] # not dps = list(context_df.loc[context_df['id'].isin(selected_blocks)]['text'])!
                context = TEXT_BETWEEN_SHOTS.join(dps) 
            else: # retrieve from train_df
                context = retrieve_context(train_df=self.train_df, index=index, curr_example=q, \
                    n_examples=self.n_shots_per_window, split_text = TEXT_BETWEEN_SHOTS, shuffle_seed=self.window_shuffle_seed)
            if use_majority_vote:
                dps = context.split(TEXT_BETWEEN_SHOTS)
                intents = [extract_intent(example) for example in dps]
                intent_counts = Counter(intents)
                majority_intent = intent_counts.most_common(1)[0][0]
                predicted_labels.append(majority_intent.lstrip().strip(STOP_SEQUENCE))
                continue

            fewshot_examples = self.tokenizer(context, add_special_tokens=False, return_tensors='pt')
            fewshot_len = fewshot_examples['input_ids'].shape[-1]

            assert q == q.rstrip(), "prompt ends with a space!"
            encoded_task_text = self.tokenizer(TEXT_BETWEEN_SHOTS+q, add_special_tokens=False, return_tensors='pt')
            encoded_task_text['input_ids'] = encoded_task_text['input_ids'][:, 1:] #cut off '' at the beginning (token 29871)
            encoded_task_text['attention_mask'] = encoded_task_text['attention_mask'][:, 1:] #cut off '' at the beginning (token 29871)

            
            encoded_inputs = torch.cat((fewshot_examples['input_ids'], encoded_task_text['input_ids']), dim=-1).to(self.model.device)
            attention_mask = torch.cat((fewshot_examples['attention_mask'], encoded_task_text['attention_mask']), dim=-1).to(self.model.device)
            custom_attention_mask = None
            input_len = encoded_inputs.shape[-1]
            decoded_inputs = self.tokenizer.decode(encoded_inputs[0])
            
            kwargs = dict(input_ids=encoded_inputs,
                          custom_attention_mask=custom_attention_mask,
                          attention_mask=attention_mask,
                          #position_ids=torch.arange(0,input_len).unsqueeze(0).to(self.model.device),
                          do_sample=False,
                          num_beams=1,
                          pad_token_id=self.tokenizer.eos_token_id,
                          max_new_tokens=self.max_n_tokens)
            if restrictive_logit_preprocessor is not None:
                restrictive_logit_preprocessor.update_new_prompt_length_to_skip(encoded_inputs.shape[1])
                kwargs['logits_processor'] = [restrictive_logit_preprocessor]

            with torch.no_grad():
                res = self.model.generate(**kwargs)[0]
                res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
                predicted_label = self.tokenizer.decode(res[encoded_inputs.shape[1]:])
            predicted_labels.append(predicted_label.lstrip().strip(STOP_SEQUENCE))
            if restrictive_logit_preprocessor is not None:
                assert set(predicted_labels).issubset(self.possible_labels)
            else:
                # clip prediction
                predicted_labels[-1] = predicted_labels[-1].split('\n')[0].split('==')[0].split('source:')[0].rstrip() # we assume batch size of 1 anyway...  hardcoded for smcalflow at the moment but can change the split to use the x_prefix and the examplifier delimeters to be more general if we need
        return predicted_labels
    
    # unused
    # Function to encode a block with positional offsets
    def encode_block(self, input_ids, position_offset):
        # Generate custom position IDs
        # position_offset = 0 # reset pos id for each block doesn't work well naively
        # default llama don't allow for position offset, changed modeling_llama.py
        position_ids = torch.arange(
            position_offset, position_offset + input_ids.size(1)
        ).unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)
        return outputs.past_key_values, position_ids 

    # Function to concatenate KV caches
    def concatenate_kv_caches(self, kv_caches):
        concatenated_kv = []
        for layer in range(len(kv_caches[0])):
            keys = torch.cat([kv[layer][0] for kv in kv_caches], dim=2)
            values = torch.cat([kv[layer][1] for kv in kv_caches], dim=2)
            concatenated_kv.append((keys, values))
        return tuple(concatenated_kv)

    # Tokenize all blocks with TEXT_BETWEEN_SHOTS
    def tokenize_all_blocks(self, contexts):
        input_ids_list = []
        for i, block_text in enumerate(contexts):
            # Add TEXT_BETWEEN_SHOTS for blocks after the first
            if i == 0:
                input_ids = self.tokenizer(block_text, add_special_tokens=False, return_tensors='pt')['input_ids'].to(self.model.device)
                # attention_mask = self.tokenizer(block_text, add_special_tokens=False, return_tensors='pt')['attention_mask']
                # assert not torch.any(attention_mask == 0), "Attention mask contains zero values"
            else:
                input_ids = self.tokenizer(TEXT_BETWEEN_SHOTS + block_text, add_special_tokens=False, return_tensors='pt')['input_ids'][:, 1:].to(self.model.device) # 29871 is the token for ''
                # attention_mask = self.tokenizer(block_text, add_special_tokens=False, return_tensors='pt')['attention_mask'][:, 1:]
                # assert not torch.any(attention_mask == 0), "Attention mask contains zero values"
            input_ids_list.append(input_ids)
        return input_ids_list

    # Encode all blocks separately
    def encode_all_blocks(self, input_ids_list):
        kv_caches = []
        position_ids_list = []
        position_offset = 0
        for input_ids in input_ids_list:
            kv_cache, pos_ids = self.encode_block(input_ids, position_offset)
            kv_caches.append(kv_cache)
            position_ids_list.append(pos_ids)
            # Update position offset
            position_offset += input_ids.size(1)
        return kv_caches, position_ids_list
    
    def encode_all_blocks_with_mask(self, tokenized_input_ids, example_boundaries):
        # Create the block attention mask
        example_boundaries = torch.tensor(example_boundaries).to(self.model.device)
        sink_blocks = len(example_boundaries) - 1 if self.attn_sink_blocks == -1 else self.attn_sink_blocks
        if self.model.config._attn_implementation in ["flex_attention"]:
            if self.attn_sink_blocks > len(example_boundaries) - 1: # out of bounds means using streaming LLM, interpret self.attn_sink_blocks as token size
                mask = make_flex_streaming_mask(seq_len=tokenized_input_ids.size(1), num_window_token=self.attn_prev_blocks, num_sink_token=self.attn_sink_blocks)
            else:
                mask = make_flex_custom_block_mask(block_boundaries=example_boundaries, num_prev_blocks=self.attn_prev_blocks, num_anchor_blocks=sink_blocks)
        else:
            mask = make_custom_block_mask(block_boundaries=example_boundaries.to('cpu'), num_prev_blocks=self.attn_prev_blocks, num_anchor_blocks=sink_blocks).to(self.model.device)
            
        with torch.no_grad():
            outputs = self.model(input_ids=tokenized_input_ids, custom_attention_mask=mask)

        # Save KV cache for each block
        kv_caches = []
        for start, end in zip(example_boundaries[:-1], example_boundaries[1:]):
            kv_cache = [(layer[0][:, :, start:end, :], layer[1][:, :, start:end, :]) for layer in outputs.past_key_values]
            kv_caches.append(kv_cache)

        return kv_caches
    
    def concatenate_selected_kv_caches(self, kv_caches, selected_blocks, attention_sink_kv=None):
        selected_kv_caches = [kv_caches[i] for i in selected_blocks]
        if attention_sink_kv is not None:
            selected_kv_caches.insert(0, attention_sink_kv)
        concatenated_kv = []
        for layer in range(len(selected_kv_caches[0])):
            keys = torch.cat([kv[layer][0] for kv in selected_kv_caches], dim=2)
            values = torch.cat([kv[layer][1] for kv in selected_kv_caches], dim=2)
            concatenated_kv.append((keys, values))
        return tuple(concatenated_kv)

    def get_example_boundaries(self, tokenized_input_ids):
        example_boundaries = [0]  # Initialize boundaries
        total_length = 0  # Tracks the cumulative length of all blocks

        # Calculate boundaries for each block
        for input_ids in tokenized_input_ids:
            total_length += len(input_ids)
            example_boundaries.append(total_length)

        return example_boundaries

    def get_predicted_labels_select_blocks(self, contexts: List[str], restrictive_logit_preprocessor):
        predicted_labels = []
        # Tokenize all blocks
        input_ids_list = []
        for i, context in enumerate(contexts):
            if i == 0:
                # Tokenize the first block
                input_ids = self.tokenizer(context, add_special_tokens=False)['input_ids']
            else:
                # Tokenize subsequent blocks with TEXT_BETWEEN_SHOTS and skip the first token
                input_ids = self.tokenizer(TEXT_BETWEEN_SHOTS + context, add_special_tokens=False)['input_ids']
                if "LLaMA-2" in self.model.name_or_path:
                    input_ids = input_ids[1:]
            input_ids_list.append(input_ids)
        # Compute example boundaries
        example_boundaries = self.get_example_boundaries(input_ids_list)
        # flattened input_ids_list
        tokenized_input_ids = self.tokenizer(TEXT_BETWEEN_SHOTS.join(contexts), add_special_tokens=False, return_tensors='pt')['input_ids'].to(self.model.device)
        # Encode all blocks with blockwise attention
        with torch.no_grad():
            kv_caches = self.encode_all_blocks_with_mask(tokenized_input_ids, example_boundaries)
        del tokenized_input_ids
        torch.cuda.empty_cache()

        if len(contexts) == self.n_selected_blocks: # all blocks selected
            selected_blocks = list(range(len(contexts))) # TODO doesn't allow reorder
            combined_kv_cache = self.concatenate_selected_kv_caches(kv_caches, selected_blocks)
            #del kv_caches
            #torch.cuda.empty_cache()

        # create retriever for blocks
        contexts_df = pd.DataFrame(contexts, columns=['text'])
        if self.block_select_method == "dense":
            index = create_retriever(contexts_df, True)
        else:
            index = create_retriever(contexts_df)

        for q in tqdm(self.test_df[PROMPTS]):
            assert q == q.rstrip(), "prompt ends with a space!"
            test_query_input_ids = self.tokenizer(TEXT_BETWEEN_SHOTS + q, add_special_tokens=False, return_tensors='pt')['input_ids'][:, 1:].to(self.model.device)

            selected_blocks = self.block_selection_method(q, len(contexts), self.n_selected_blocks, self.block_select_method, index, self.block_order_method)
            combined_kv_cache = self.concatenate_selected_kv_caches(kv_caches, selected_blocks)

            combined_context_length = sum([len(input_ids_list[i]) for i in selected_blocks]) + test_query_input_ids.size(1)
            attention_mask = torch.ones(1, combined_context_length).to(self.model.device)

            kwargs = dict(input_ids=test_query_input_ids,
                        custom_attention_mask=None,
                        attention_mask=attention_mask,
                        past_key_values=combined_kv_cache,
                        # attention_mask=None, ### uncomment for zero-shot 
                        # past_key_values=None, ### uncomment for zero-shot 
                        use_cache=True,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=self.max_n_tokens)
            if restrictive_logit_preprocessor is not None:
                restrictive_logit_preprocessor.update_new_prompt_length_to_skip(test_query_input_ids.shape[1])
                kwargs['logits_processor'] = [restrictive_logit_preprocessor]

            with torch.no_grad():
                res = self.model.generate(**kwargs)[0]
                res = res[:-1] if res[-1] == self.tokenizer.eos_token_id else res
                predicted_label = self.tokenizer.decode(res[test_query_input_ids.shape[1]:])
            predicted_labels.append(predicted_label.lstrip().strip(STOP_SEQUENCE))
            if restrictive_logit_preprocessor is not None:
                assert set(predicted_labels).issubset(self.possible_labels)
            else:
                predicted_labels[-1] = predicted_labels[-1].split('\n')[0].split('==')[0].split('source:')[0].rstrip()
        return predicted_labels  

    
    def get_predicted_labels(self, windows_few_shots: List[str]) -> List[str]:
        windows_cache = self.model.get_contexts_cache(windows_few_shots)
       
        predicted_labels = []
        raw_labels = []
        
        for q in tqdm(self.test_df[PROMPTS]):
            predicted_label = self.predict_label(TEXT_BETWEEN_SHOTS + q, windows_cache)
            predicted_label = predicted_label.strip()
            predicted_labels.append(predicted_label)
            assert set(predicted_labels).issubset(self.possible_labels)
        return predicted_labels


    def predict_label(self, task_text: str, cache: Dict) -> str:
        assert task_text == task_text.rstrip(), "prompt ends with a space!"
        res = self.model.pcw_generate(task_text=task_text,
                                      contexts_cache=cache,
                                      restrictive_logit_preprocessor=self.logit_processor,
                                      temperature=0,
                                      max_new_tokens=self.max_n_tokens)
        
        return res.lstrip().strip(STOP_SEQUENCE)

    def calc_acc(self, predicted_labels: List) -> float:
        predicted_labels = pd.Series(predicted_labels, index=self.test_df.index, name='outputs')
        multieval = False
        
        if 'labels' in self.test_df:
            multieval = True
            true_labels = self.test_df['labels']
        else:
            # normal eval
            true_labels = self.test_df[LABEL_TOKENS]
            
            try:
                true_labels = true_labels.map(self.label_map)
            except:
                pass # already mapped!
        
        save_state = pd.concat([predicted_labels, true_labels], axis=1)
        save_state['true_numeric_labels'] = self.test_df["label"]
        save_state['true_label_present_in_prompt'] = save_state['true_numeric_labels'].isin(self.label_distribution_prompt)

        if multieval:
            # from chatgpt, proceed w caution
            # Function to check if 'output' is in 'labels'
            def check_output_in_labels(row):
                return row['outputs'] in row['labels']

            # Add a new column to check if 'output' is in 'labels'
            save_state['correct'] = save_state.apply(check_output_in_labels, axis=1)
        else:
            save_state['correct'] = save_state['outputs'] == save_state['label_tokens']

        acc = np.mean(save_state['correct'])
        _logger.info(f"accuracy = {np.round(acc, 3)}")
        print(f"accuracy = {np.round(acc, 3)}")

        if self.is_classification:
            save_state['outputs'] = save_state['outputs'].map(self.reverse_label_map)
            # hacky -- get the shortened text labels that correspond to each of the labels
            textual_prompt_labels = list(map(self.label_map, [self.train_df.loc[self.train_df.index[self.train_df['label'] == i][0]]['label_tokens'] for i in self.label_distribution_prompt]))
            save_state['predicted_label_present_in_prompt'] = save_state['outputs'].isin(textual_prompt_labels)

            save_state['prompt_labels'] = str(self.label_distribution_prompt)

        return acc, save_state

    def run_experiment_across_shots(self, n_shots_to_test: List[int], n_runs: int,
                                    too_long_patience: float = 0.2,
                                    context_window_size: int = 4096):
        accuracies = np.zeros((len(n_shots_to_test), n_runs))
        predictions = [] #np.zeros((len(n_shots_to_test), n_runs))
        for i, n_shots in enumerate(tqdm(n_shots_to_test)):
            predictions_row = []
            _logger.info(f"starting with n = {n_shots}")
            self._set_random_seed(self.base_random_seed + n_shots)
            j = 0
            n_errors = 0
            while j < n_runs:
                few_shots_idx = self.sample_n_shots(n_shots)
                self.label_distribution_prompt = dict(Counter(self.train_df.loc[few_shots_idx, "label"]))
                selected = self.train_df.loc[few_shots_idx]
                if self.sort_by_label:
                    selected = selected.sort_values("label")
                
                if False: # testing unsupervised ICL. performance very bad
                    few_shots_prompts = ["query: " + text for text in selected['text']]
                few_shots_prompts = list(selected[PROMPTS]) 
                if self.window_shuffle_seed:
                    prev_state = random.getstate()
                    random.seed(self.window_shuffle_seed)
                    random.shuffle(few_shots_prompts)
                    random.setstate(prev_state)
                windows_few_shots = self.build_windows_few_shots_text(few_shots_prompts, self.n_shots_per_window, self.block_group_method)
                longest_window_n_tokens = max(n_tokens_in_prompt(self.tokenizer, window)
                                              for window in windows_few_shots)
                n_tokens_between_shots = n_tokens_in_prompt(self.tokenizer, TEXT_BETWEEN_SHOTS)

                # check if too long
                if ((longest_window_n_tokens + n_tokens_between_shots + self.test_df[N_TOKENS].max()
                        + self.max_n_tokens) > context_window_size) and not self.use_retrieval:
                    _logger.warning("Drawn training shots were too long, trying again")
                    n_errors += 1
                    assert n_errors <= too_long_patience * n_runs, "too many long inputs were drawn!"
                    continue
                accuracies[i, j], this_prediction = self.get_few_shots_acc(windows_few_shots)
                this_prediction['prompt_example_indices'] = str(list(few_shots_idx))
                predictions_row.append(this_prediction) 
                j += 1
            predictions.append(predictions_row)
        return accuracies, predictions

    def sample_n_shots(self, n_shots: int) -> npt.NDArray[int]:
        if self.times_shuffled >= len(self.random_orders):
            self.times_shuffled = 0
            self.random_orders = [np.random.permutation(list(self.train_df.index)) for i in range(20)]
            
        few_shots_df = self.train_df.loc[self.random_orders[self.times_shuffled][:n_shots]]
        if self.sync_shuffle:
            self.test_df = self.full_test_df.loc[self.random_orders[self.times_shuffled][:self.subsample_test_set]]
        
        assert few_shots_df.index.is_unique, "few shots samples were not unique!"
        window_size = self.n_shots_per_window or n_shots
        n_windows = int(len(few_shots_df) / window_size)
        self.times_shuffled += 1

        if not self.n_shots_per_window or n_windows == 1:
            return few_shots_df.index

        return self.balance_windows_sizes(n_windows, few_shots_df)

    def balance_windows_sizes(self, n_windows: int, few_shots_df: pd.DataFrame) -> npt.NDArray[int]:
        few_shots_df.sort_values(by=N_TOKENS, inplace=True, ascending=False)
        shape = (self.n_shots_per_window, n_windows)
        indexes = np.array(few_shots_df.index).reshape(shape)
        sizes = few_shots_df.loc[indexes.flatten()].n_tokens.values.reshape(indexes.shape)
        for i in range(1, self.n_shots_per_window):
            order = np.argsort((np.sum(sizes[:i, :], axis=0)))
            sizes[i, :] = sizes[i, order]
            indexes[i, :] = indexes[i, order]
        indexes = indexes.T.flatten()
        return indexes

    @staticmethod
    def build_windows_few_shots_text(few_shots_prompts: List[str], window_size: int, block_group_method: str, swap_ratio: int = 0.1) -> List[str]:
        ## if window_size is not given, default 1 window/block
        if window_size is None:
            window_size = len(few_shots_prompts)
        window_num = math.ceil(len(few_shots_prompts) / window_size)

        ## default grouping method: random grouping or sorted grouping
        if block_group_method is None or block_group_method == "default": # random or sorted, depend on sorted flag
            return [TEXT_BETWEEN_SHOTS.join(few_shots_prompts[i: i + window_size]) for i in
                range(0, len(few_shots_prompts), window_size)]
        
        ## 'balanced' - grouping according to labels
        ## options: [balanced, balanced-shuffle]
        elif "balanced" in block_group_method:
            # get label from prompt, then split into balanced groups using StratifiedKFold
            labellist = []
            for prompt in few_shots_prompts:
                labellist.append(prompt.strip(STOP_SEQUENCE).split('\n')[-1]) # \n{self.y_prefix}{x[LABEL_TOKENS]}(\n)* (ref: datasets_loader.apply_format)
            if block_group_method=="balanced-shuffle":
                window_splits = StratifiedKFold(n_splits=window_num, shuffle=True)
            else:
                window_splits = StratifiedKFold(n_splits=window_num, shuffle=False)
            window_splits.get_n_splits(labellist, labellist)
            windows = []
            for _, ids in window_splits.split(labellist, labellist):
                windows.append(TEXT_BETWEEN_SHOTS.join([few_shots_prompts[i] for i in ids]))
            return windows
        
        ## 'clustering' - grouping with clustering using bm25 or embedding
        ## options: [bm25-clustering, bm25-clustering-swap, embed-clustering, embed-clustering-swap]
        elif "clustering" in block_group_method:
            kmeans_data = None
            if "bm25-clustering" in block_group_method:
                tokenized_corpus = [doc.split(" ") for doc in few_shots_prompts]
                bm25 = BM25Okapi(tokenized_corpus)
                # construct the BM25 matrix
                # treat each unique word as a "query" and get BM25 scores for each document.
                vocab = list(set(word for doc in tokenized_corpus for word in doc))
                bm25_matrix = np.zeros((len(few_shots_prompts), len(vocab)))
                for i, word in enumerate(vocab):
                    scores = bm25.get_scores([word]) # BM25 scores for each doc given the single-term query 
                    bm25_matrix[:, i] = scores
                kmeans_data = bm25_matrix
            elif "embed-clustering" in block_group_method:
                # get embedding using langchain HuggingFaceEmbedding, then cluster into same size kmeans
                # TODO: Try different embedding models listed in https://huggingface.co/spaces/mteb/leaderboard
                embed_model_name = "dunzhang/stella_en_1.5B_v5"
                embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
                embeddings = list(embed_model.embed_documents(few_shots_prompts))
                kmeans_data = embeddings
            else:
                raise ValueError(f"Unknown clustering method: {block_group_method}; choice=['bm25-clustering(-*)', 'embed-clustering(-*)']")
                
            # kmeans clustering    
            kmeans_model = KMeansConstrained(n_clusters=window_num, size_max=window_size) # doc ref: https://joshlk.github.io/k-means-constrained/
            assert kmeans_data is not None, "Data for kmeans clustering is not initialized!"
            labels = kmeans_model.fit_predict(kmeans_data)

            # randomly swap {swap_ratio} examples' labels, i.e. the cluster they're assigned to
            if "swap" in block_group_method:
                ## get the cluster ids (e.g. [0, 1, 2,...])
                cluster_ids = set(labels)
                # Helper function to map each cluster --> indices of points within that cluster
                def map_cluster_to_indices(data_points):
                    cluster_map = {}
                    for idx, point in enumerate(data_points):
                        cluster_map.setdefault(point, []).append(idx)
                    return cluster_map
                # map cluster to indices
                cluster_to_indices = map_cluster_to_indices(labels)
                ## iterate over each cluster and attempt swaps
                for cur_cluster in list(cluster_ids):
                    cur_indices = cluster_to_indices[cur_cluster]
                    # calculate how many points to swap from this cluster
                    num_to_swap = int(len(cur_indices) * swap_ratio)
                    if num_to_swap == 0:
                        continue
                    # get swap_candidates from other clusters
                    swap_candidates = []
                    for candidate_cluster, candidate_indices in cluster_to_indices.items():
                        if candidate_cluster != cur_cluster:
                            swap_candidates.extend(candidate_indices)
                    if len(swap_candidates) == 0:
                        continue
                    # limit num_to_swap to the availability of other cluster points
                    num_to_swap = min(num_to_swap, len(swap_candidates))             
                    # choose random points from this cluster
                    chosen_from_cur_indices = random.sample(cur_indices, num_to_swap)
                    # choose random points from the candidate clusters
                    chosen_from_swap_candidates = random.sample(swap_candidates, num_to_swap)
                    ## swap their labels
                    for i in range(num_to_swap):
                        cur_i = chosen_from_cur_indices[i]
                        candidate_i = chosen_from_swap_candidates[i]
                        # swap labels at these indices
                        labels[cur_i], labels[candidate_i] = labels[candidate_i], labels[cur_i]
                    # Update cluster_to_indices after the swap
                    cluster_to_indices = map_cluster_to_indices(labels)

            windows = []
            for i in range(window_num):
                windows.append(TEXT_BETWEEN_SHOTS.join([few_shots_prompts[j] for j in range(len(few_shots_prompts)) if labels[j] == i]))
            return windows

        else:
            raise ValueError(f"Unknown group method: {block_group_method}; choice=['default', 'bm25-clustering(-*)', 'embed-clustering(-*)']")
        
    @staticmethod 
    def block_selection_method(q: str, total_blocks_num: int, block_num: int, 
                               block_select_method: str, index, block_order_method: str) -> List[int]:
        if block_select_method == 'all':
            return list(range(total_blocks_num))
        #assert block_num <= total_blocks_num, "n_selected_blocks can't be greater than total_blocks_num"
        assert block_num > 0, "n_selected_blocks needs to be specified"
        if block_select_method == 'random': # TODO change to inorder
            return np.random.choice(total_blocks_num, block_num, replace=False)
        if block_select_method in ['bm25', 'dense']: # TODO change to inorder
            retrieved = index.search(query=q, return_docs=False, cutoff=block_num) # ref: utils.retrieve_context
            retrieved_items = sorted(retrieved.items(), key=lambda item: item[1], reverse=True)
            retrieved_indices = [int(key) for key, _ in retrieved_items[:block_num]]
            if 0 in retrieved_indices: # TODO change to custom sink block
                retrieved_indices.remove(0)
            retrieved_indices.insert(0, 0)
            retrieved_indices = retrieved_indices[:block_num]  # 0 + top scores idx from high to low
            if not block_order_method: #default in order
                return sorted(retrieved_indices)
            if block_order_method=='lo2hi':
                return [0] + retrieved_indices[1:][::-1]
            elif block_order_method=='hi2lo':
                return [0] + retrieved_indices[1:]
            elif block_order_method=='mid2lo2hi':
                mid = len(retrieved) // 2
                return [0] + retrieved_indices[1:][mid:] + retrieved_indices[1:][:mid][::-1]
            elif block_order_method == 'reversed':
                return [0] + sorted(retrieved_indices[1:], reverse=True) #TODO: changed!!
            else:
                raise ValueError(f"Unknown ordering method: {block_order_method}; choose 'hi2lo' or 'lo2hi' or 'mid2lo2hi' or 'reversed'")
        else:
            raise ValueError(f"Unknown selection method: {block_select_method}; choose 'all', 'random' or 'bm25' or 'dense")