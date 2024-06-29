"""
A few standard LLM Datasets can be found stored in huggingface datasets. A few prominent ones
+ FineWeb-Edu dataset
+ CommonCrawl
+ WebText
+ RedPajama

In this code, we'll be working with the FineWeb-Edu dataset.
"""
#install tiktoken, datasets, tqdm
#We use multi-processing to write the files into a single shard. to increase the speed of the operation (Parrallel processing per shard)

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm              #progress bar

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"    #hf name (10 billion token sample)
shard_size = int(1e8)       #a shard is 100M which means the dataset is 100 shard

# creating local dir if not exists
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)  #we could have used os.curdir
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

#dataset download
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

#initialize the tokenizer
enc = tiktoken.get_encoding("gpt2")  
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]   #starts with end of text
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Tokens must be between 0 and max BTE tokens"
    tokens_np_uint8 = tokens_np.astype(np.uint8)
    return tokens_np_uint8

def write_datafile(filename, token_np):
    np.save(filename, token_np)


#multiprocessing
nprocs = max(1, os.cpu_count() // 2)  #ensuring 1 if not

with mp.Pool(nprocs) as pool:
    shard_index = 0

    #allocate buffers for holding shards
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    #applies the tokenize function to the data in fw
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        
        #checking for space in current shard
        if token_count + len(tokens) < shard_size:
            
            all_tokens_np[token_count:token_count +len(tokens)] = tokens
            token_count += len(tokens)  #loop through tokens

            #update progress_bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'{shard_index=}')

            progress_bar.update(len(tokens))
        else:
            #write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:6d}")

            #the remainder goes into the other shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]  #till remainder
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None

            #populate next shard with remainder
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]  #put remainder at about the end of shard
            token_count = len(tokens)-remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:6d}")
        write_datafile(filename, all_tokens_np[:token_count])
