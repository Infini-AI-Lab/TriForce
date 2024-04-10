from datasets import load_dataset
from tqdm import tqdm
import secrets
import random
import torch
import json
import os

def build_chat_input_lwm(tokenizer, message, prefill=127*1024):
    # chat format:
    # single-turn: You are a helpful assistant. USER: {} \n ASSISTANT:
    book = tokenizer.encode(message)[:prefill-84]
    prompt = "You are a helpful assistant. USER: Please read a part of the book below, and then give me the summary.\n[start of the book]\n" + tokenizer.decode(book, skip_special_tokens=True) + "\n[end of the book]\n\nNow you have read it. Please summarize it for me. First, tell me the title and the author, and then tell the story in 400 words.\n\nASSISTANT: "
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    return input_tokens

def get_dataset(dataset_name, tokenizer=None, datalen=None, task=None):
    if dataset_name == '128k':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(len(dataset))):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    elif dataset_name == 'gs':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(20)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts
    
    elif dataset_name == 'one-shot':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(1)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts

    elif dataset_name == 'demo':
        dataset = load_dataset("narrativeqa")
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[2]]['document']['text'][3:1024*500])
        tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'lwm':
        dataset = load_dataset("narrativeqa")
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        for i in range(20):
            tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[i]]['document']['text'][3:1024*500])
            if tokenized_prompt.shape[-1] != 127*1024:
                print(i, tokenized_prompt.shape)
                continue
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    else:
        raise Exception("Dataset not found")