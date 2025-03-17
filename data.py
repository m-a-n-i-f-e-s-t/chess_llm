import os
import time
import torch
import torch.distributed as dist
import numpy as np
import glob
from prepare_data import get_tokenizer
from typing import Callable, List, Tuple, Optional, Dict
from utils import rank_print, timing, DistributedInfo
from transformers import AutoTokenizer

GS_BUCKET = os.environ.get("GS_BUCKET")

def download_data(di: DistributedInfo,
                config: dict,
                datasets_dir: str) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """ Downloads data for training.
    """

    tokenizer_name = config['tokenizer'] if config['tokenizer'] is not None else config['model_name']

    CPI_TO_SPLITS ={
        "fen_top3_move": 1053,
        "fen_english_move": 30,
    }
    cpis = config['cpi'].split(',')
    ratios = [float(ratio) for ratio in config['cpi_ratio'].split(',')]
    assert len(cpis) == len(ratios), "Number of CPIs and ratios must be the same"
    cpi_ratios = {cpi: ratios[i] for i, cpi in enumerate(cpis)}

    rank_print(f"downloading data")
    # Each dp_rank downloads its own shard of data
    with timing("Downloading data", rank_print):
        # TODO: this is heavily over-downloading, we should only download the data we need
        all_data_files = {}
        for cpi in cpis:
            os.makedirs(f"{datasets_dir}/{cpi}", exist_ok=True)
            data_files = [[
                f"train{fid:04d}.bin",
                f"mask{fid:04d}.bin"
            ] for fid in range(0, CPI_TO_SPLITS[cpi])] # english move only has 30
            os.environ["GSUTIL_PARALLEL_THREAD_COUNT"] = "8"
            os.environ["GSUTIL_PARALLEL_PROCESS_COUNT"] = "64"
            already_exists = all([os.path.exists(f"{datasets_dir}/{cpi}/{file}") for files in data_files for file in files])
            if di.local_rank == 0:
                if not already_exists:
                    print(f"Rank {di.world_rank} downloading data...")
                    os.system(f"gsutil -m cp {GS_BUCKET}/sfchess-v7/{tokenizer_name}/{cpi}/* {datasets_dir}/{cpi}/")
                    # Use find to locate all .gz files and process them in parallel
                    os.system(f"cd {datasets_dir}/{cpi} && find . -name '*.gz' | parallel --bar 'pigz -d -c {{}} > {{.}}'")
                os.system(f"touch {datasets_dir}/{cpi}/done")
            else:
                while not os.path.exists(f"{datasets_dir}/{cpi}/done"):
                    rank_print(f"sleeping for 60 seconds to wait for data to be downloaded")
                    time.sleep(60)

            full_data_files = [[f"{datasets_dir}/{cpi}/{file}" for file in files] for files in data_files]
            all_data_files[cpi] = full_data_files
        dist.barrier()  # Wait for all ranks to finish downloading

    return all_data_files, cpi_ratios

def get_retokenized_files(datasets_dir: str) -> List[List[str]]:
    """
    Find all retokenized train and mask files in the datasets directory.
    
    Args:
        datasets_dir: Directory containing the datasets
        
    Returns:
        List of [train_file, mask_file] pairs for retokenized files
    """
    # Find all retokenized train files
    train_pattern = os.path.join(datasets_dir, "train*ts.bin")
    train_files = sorted(glob.glob(train_pattern))
    
    # Create corresponding mask files
    retokenized_files = []
    for train_file in train_files:
        # Get the corresponding mask file
        mask_file = train_file.replace("train", "mask")
        
        # Check if both train and mask files exist
        if os.path.exists(mask_file):
            # Extract just the filenames, not the full paths
            train_filename = os.path.basename(train_file)
            mask_filename = os.path.basename(mask_file)
            retokenized_files.append([train_filename, mask_filename])
    
    return retokenized_files

def setup_data(di: DistributedInfo,
               config: dict) -> tuple[str, str, str, Callable, List[List[str]]]:
    """
    Sets up directories and data for training.
    
    Args:
        di: Distributed info
        tokenizer: The tokenizer being used for training
        ds_tokenizer: DeepSeek tokenizer (if applicable)
        config: Training configuration
        retokenize: Whether to use retokenized data
        
    Returns:
        Tuple of (checkpoints_dir, datasets_dir, unrolls_dir, get_batch)
    """
    # Setup data directories (rank 0 only)
    checkpoints_dir = f"/data/checkpoint"
    datasets_dir = f"/data/datasets"
    unrolls_dir = f"/data/unrolls"
    if di.local_rank == 0:
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(unrolls_dir, exist_ok=True)
    rank_print(f"syncing barrier in data")
    dist.barrier()

    # Download the data if needed
    all_data_files, cpi_ratios = download_data(di, config, datasets_dir)
    cpi_probs = {cpi: cpi_ratios[cpi] / sum(cpi_ratios.values()) for cpi in cpi_ratios}
    cpi_names = list(cpi_probs.keys())
    flattened_data_file_tuples = []
    for cpi in cpi_names:
        flattened_data_file_tuples.extend(all_data_files[cpi])

    def split_batch_by_cpi(batch_size: int):
        import math
        cpi_batches = {}
        sorted_cpi_prob_tuples = sorted(cpi_probs.items(), key=lambda x: x[1])
        for cpi, prob in sorted_cpi_prob_tuples[:-1]:
            cpi_batches[cpi] = math.ceil(batch_size * prob)
        cpi_batches[sorted_cpi_prob_tuples[-1][0]] = batch_size - sum(cpi_batches.values())
        return cpi_batches
    

    mixture_type = config['mixture_type']
    if mixture_type == "within_batch":
        cpi_batches = split_batch_by_cpi(config['batch_size'])
        print(f"Batch size per CPI: {cpi_batches}")


    # The rng gen is created so that we can have an explicit control over
    # the batch being sampled regardless of main torch rng state
    rng_gen = torch.Generator(device=f"cpu")
    rng_gen.manual_seed(1337 + di.real_dp_rank)


    def get_batch(iters, with_response_mask=False):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        cpi_names = list(cpi_probs.keys())
        if mixture_type == "across_batch":
            cpi_index = torch.multinomial(torch.tensor(list(cpi_probs.values()), device="cpu"), num_samples=1, generator=rng_gen).item()
            cpi = cpi_names[cpi_index]
            data_files = all_data_files[cpi]
            train_file, mask_file = data_files[torch.randint(0, len(data_files), (1,), generator=rng_gen).item()]
            data = np.memmap(os.path.expanduser(f'{train_file}'), dtype=np.uint32, mode='r')
            ix = torch.randint(len(data) - (config['seq_len']+1), (config['batch_size'],), generator=rng_gen)
            # print(f"[{torch.distributed.get_rank()}] {ix=} {cpi=}")
            src = np.stack([data[i:i+config['seq_len']+1] for i in ix]).astype(np.int64)
            x = torch.from_numpy(src[:, :-1])
            y = torch.from_numpy(src[:,1:  ])
            x, y = x.pin_memory().to(di.local_rank, non_blocking=True), y.pin_memory().to(di.local_rank, non_blocking=True)
            if not with_response_mask:
                return x, y
            else:
                mask_data = np.memmap(os.path.expanduser(f'{mask_file}'), dtype=np.uint16, mode='r')
                response_mask = np.stack([mask_data[i+1:i+config['seq_len']+1] for i in ix]).astype(np.int64)
                n_responses = sum([len(np.unique(row[row > 0])) if len(row[row > 0]) > 0 else 0 for row in response_mask])
                response_mask = torch.from_numpy(response_mask).pin_memory().to(di.local_rank, non_blocking=True)
                n_responses = torch.tensor(n_responses, device=di.local_rank)
                return x, y, response_mask, n_responses
        else: # mixture_type == "within_batch"
            x = torch.zeros((config['batch_size'], config['seq_len']), device='cpu', dtype=torch.int64)
            y = torch.zeros((config['batch_size'], config['seq_len']), device='cpu', dtype=torch.int64)
            idx = 0
            train_files, mask_files, ix = {}, {}, {}
            for cpi in cpi_names:
                data_files = all_data_files[cpi]
                train_file, mask_file = data_files[torch.randint(0, len(data_files), (1,), generator=rng_gen).item()]
                train_files[cpi] = train_file
                mask_files[cpi] = mask_file
                data = np.memmap(os.path.expanduser(f'{train_file}'), dtype=np.uint32, mode='r')
                ix[cpi] = torch.randint(len(data) - (config['seq_len']+1), (cpi_batches[cpi],), generator=rng_gen)
                src = np.stack([data[i:i+config['seq_len']+1] for i in ix[cpi]]).astype(np.int64)
                x[idx:idx+cpi_batches[cpi], :] = torch.from_numpy(src[:, :-1]).pin_memory().to(di.local_rank, non_blocking=True)
                y[idx:idx+cpi_batches[cpi], :] = torch.from_numpy(src[:,1:  ]).pin_memory().to(di.local_rank, non_blocking=True)
                idx += cpi_batches[cpi]
            # print(f"[{torch.distributed.get_rank()}] {ix=}")
            assert idx == config['batch_size'], f"idx: {idx}, config['batch_size']: {config['batch_size']}"
            x, y = x.pin_memory().to(di.local_rank, non_blocking=True), y.pin_memory().to(di.local_rank, non_blocking=True)
            if not with_response_mask:
                return x, y
            else:
                response_mask = torch.zeros((config['batch_size'], config['seq_len']), device=di.local_rank)
                n_responses = 0
                idx = 0
                for cpi in cpi_names:
                    mask_data = np.memmap(os.path.expanduser(f'{mask_files[cpi]}'), dtype=np.uint16, mode='r')
                    np_response_mask = np.stack([mask_data[i+1:i+config['seq_len']+1] for i in ix[cpi]]).astype(np.int64)
                    n_responses += sum([len(np.unique(row[row > 0])) if len(row[row > 0]) > 0 else 0 for row in np_response_mask])
                    response_mask[idx:idx+cpi_batches[cpi], :] = torch.from_numpy(np_response_mask).pin_memory().to(di.local_rank, non_blocking=True)
                    idx += cpi_batches[cpi]
                n_responses = torch.tensor(n_responses, device=di.local_rank, dtype=torch.int64)
                assert idx == config['batch_size'], f"idx: {idx}, config['batch_size']: {config['batch_size']}"
                return x, y, response_mask, n_responses

    
    return checkpoints_dir, datasets_dir, unrolls_dir, get_batch, flattened_data_file_tuples
