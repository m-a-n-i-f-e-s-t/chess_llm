import os
import torch
import torch.distributed as dist
from pathlib import Path
import requests
from tqdm import tqdm
import logging
import sys
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INT_TO_DTYPE = {
    0: torch.float8_e4m3fn,
    1: torch.float8_e4m3fnuz,
    2: torch.float8_e5m2,
    3: torch.float8_e5m2fnuz,
    4: torch.float16,
    5: torch.bfloat16,
    6: torch.float32,
    7: torch.float64,
}

NON_CONVERTIBLE_DTYPES = [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

DTYPE_TO_INT = {v: k for k, v in INT_TO_DTYPE.items()}

def receive_weights(param_server_addr, param_server_port, dtype, save_dir):
    """
    Client process that receives model files from the parameter server
    """
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    save_dir = Path(save_dir) / f"mp{world_size}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # First establish connection with training network
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    try:
        # Get tensor info from parameter server
        url = f"http://{param_server_addr}:{param_server_port}/serve?mp={world_size}&dtype={dtype}&rank={rank}"
        try:
            response = requests.get(url)
            if response.status_code == 503:
                logger.error("Server is busy with another transfer")
                sys.exit(1)
            response.raise_for_status()
            tensor_info = response.json()
            tensor_keys = tensor_info['keys']
            save_path = save_dir / tensor_info['file']
            logger.info(f"Rank {rank}: Expecting {len(tensor_keys)} tensors")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get tensor info: {e}")
            if hasattr(e.response, 'json'):
                logger.error(f"Server response: {e.response.json()['detail']}")
            sys.exit(1)
            
        # Wait for all ranks to get their info

        dist.destroy_process_group()
        
        # Connect to parameter server network
        os.environ['MASTER_ADDR'] = 'deepseek-svc'
        os.environ['MASTER_PORT'] = '29500'
        new_rank = rank + 1  # Rank 0 is the server
        
        dist.init_process_group("nccl", rank=new_rank, world_size=world_size + 1)
        
        # Receive number of tensors (should match keys from HTTP response)
        num_tensors = torch.empty(1, dtype=torch.long, device='cuda')
        dist.recv(num_tensors, src=0)
        assert num_tensors.item() == len(tensor_keys), "Mismatch in expected tensor count"
        
        # Receive tensors and build state dict
        state_dict = {}
        for key in tqdm(tensor_keys, desc=f"Receiving tensors for rank {rank}"):
            # Receive tensor metadata
            shape = torch.empty(8, dtype=torch.int64, device='cuda')  # Max 8D tensor
            dist.recv(shape, src=0)
            # print(f"{key}: shape: {shape}")
            shape = shape[:shape.nonzero().max().item() + 1].tolist()
            
            dtype = torch.empty(1, dtype=torch.long, device='cuda')
            dist.recv(dtype, src=0)
            # print(f"{key}: dtype: {dtype}")
            dtype = INT_TO_DTYPE[dtype.item()]
            
            # Receive tensor data
            if dtype in NON_CONVERTIBLE_DTYPES:
                tensor = torch.empty(shape, dtype=torch.float32, device='cuda')
            else:
                tensor = torch.empty(shape, dtype=dtype, device='cuda')
            dist.recv(tensor, src=0)
            
            state_dict[key] = tensor.to(dtype)
            # logger.info(f"Rank {rank}: Received tensor {key}")
        
        # Save tensors to file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, save_path)
        logger.info(f"Rank {rank}: Saved weights to {save_path}")
        
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-server", default="deepseek-svc")
    parser.add_argument("--param-port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="fp8", choices=["fp8", "bf16"])
    parser.add_argument("--save-dir", type=str, default="/data/r1")
    args = parser.parse_args()
    receive_weights(args.param_server, args.param_port, args.dtype, args.save_dir) 