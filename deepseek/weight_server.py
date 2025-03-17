import os
import torch
import torch.distributed as dist
from pathlib import Path
import time
import threading
import logging
from tqdm import tqdm
from safetensors import safe_open
from enum import Enum
from typing import Set, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerState(Enum):
    IDLE = "idle"
    WAITING_FOR_RANKS = "waiting_for_ranks"
    TRANSFERRING = "transferring"

class TensorInfo(BaseModel):
    file: str
    keys: List[str]

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
class WeightServer:
    def __init__(self, base_weight_dir: str = "/data/r1", dist_port: int = 29500):
        self.dist_port = dist_port
        self.base_weight_dir = Path(base_weight_dir)
        
        # State management
        self.state_lock = threading.Lock()
        self.state = ServerState.IDLE
        self.expected_world_size = None
        self.joined_ranks: Set[int] = set()
        self.transfer_thread = None
        
    @contextmanager
    def state_transaction(self):
        """Context manager for thread-safe state access"""
        with self.state_lock:
            yield
            
    def find_safetensor(self, mp_size: int, dtype: str, rank: int) -> Path:
        """Find safetensor file for given mp_size and rank"""
        weight_path = self.base_weight_dir / f"{dtype}/mp{mp_size}/model{rank}-mp{mp_size}.safetensors"
        if not weight_path.exists():
            logger.error(f"No safetensor found for mp_size={mp_size}, dtype={dtype}, rank={rank} at {weight_path}")
            return None
        return weight_path

    def start_weight_transfer(self, mp_size: int, dtype: str):
        """Initialize distributed group and transfer weights"""
        with self.state_transaction():
            if self.state != ServerState.WAITING_FOR_RANKS:
                logger.error("Invalid state transition to transfer")
                return
            self.state = ServerState.TRANSFERRING
        
        logger.info("Starting weight transfer")
        start = time.time()
        os.environ['MASTER_ADDR'] = 'deepseek-svc'
        os.environ['MASTER_PORT'] = str(self.dist_port)

        dtype_str = dtype
        
        try:
            dist.init_process_group("nccl", rank=0, world_size=mp_size+1)
            
            # For each rank
            for rank in range(1, mp_size+1):
                safetensor_path = self.find_safetensor(mp_size, dtype_str, rank)
                if not safetensor_path:
                    logger.error(f"No safetensor found for mp_size={mp_size}, dtype={dtype_str}, rank={rank}")
                    continue
                
                with safe_open(safetensor_path, framework="pytorch", device="cuda") as f:
                    tensor_keys = sorted(f.keys())
                    
                    num_tensors = torch.tensor([len(tensor_keys)], device='cuda')
                    dist.send(num_tensors, dst=rank)
                    
                    for key in tqdm(tensor_keys, desc=f"Transferring tensors for rank {rank - 1}"):
                        logger.debug(f"Transferring tensor {key} for rank {rank}")
                        tensor = f.get_tensor(key)
                        tensor = tensor.cuda()
                        
                        shape = torch.tensor(tensor.shape, dtype=torch.int64, device='cuda')
                        # Pad shape to 8 dimensions with zeros
                        padded_shape = torch.zeros(8, dtype=torch.int64, device='cuda')
                        padded_shape[:len(shape)] = shape
                        logger.debug(f"{key} shape: {padded_shape}")
                        dist.send(padded_shape, dst=rank)
                        
                        dtype = torch.tensor([DTYPE_TO_INT[tensor.dtype]], device='cuda')
                        logger.debug(f"{key} dtype: {dtype}")
                        dist.send(dtype, dst=rank)
                        
                        if tensor.dtype in NON_CONVERTIBLE_DTYPES:
                            dist.send(tensor.to(torch.float32), dst=rank)
                        else:
                            dist.send(tensor, dst=rank)
                
            logger.info(f"Weight transfer completed successfully in {time.time() - start} seconds")
            
        finally:
            dist.destroy_process_group()
            with self.state_transaction():
                self.reset_state()
            
    def reset_state(self):
        """Reset server state to idle"""
        self.state = ServerState.IDLE
        self.expected_world_size = None
        self.joined_ranks.clear()
        self.transfer_thread = None

# Create FastAPI app and server instance
app = FastAPI()
server = WeightServer()

@app.get("/status")
def get_status():
    """Get current server status"""
    with server.state_transaction():
        return {
            "state": server.state.value,
            "world_size": server.expected_world_size,
            "joined_ranks": list(server.joined_ranks)
        }

@app.get("/clean")
def clean_state(force: bool = False):
    """Reset server state"""
    logger.info(f"Cleaning server state, force={force}")
    with server.state_transaction():
        logger.info(f"Server state: {server.state}")
        if server.state == ServerState.TRANSFERRING:
            if not force:
                raise HTTPException(status_code=400, detail="Cannot clean while transfer is in progress")
            else:
                logger.warning("Force cleaning server state while transfer is in progress. This may cause client errors.")
                # Kill the transfer thread if it exists
                if server.transfer_thread and server.transfer_thread.is_alive():
                    # Try to cleanup NCCL process group
                    try:
                        dist.destroy_process_group()
                    except:
                        logger.warning("Failed to cleanup process group during force clean")
                    
                    # We can't really "kill" a thread in Python, but we can let it know it should stop
                    server.transfer_thread = None
        
        server.reset_state()
    return {"status": "cleaned", "force_applied": force}

@app.get("/serve")
def serve_weights(mp: int, dtype: str, rank: int) -> TensorInfo:
    """Handle weight serving request"""
    with server.state_transaction():
        current_state = server.state
        
        # Handle based on current state
        if current_state == ServerState.TRANSFERRING:
            raise HTTPException(status_code=503, detail="Weight transfer in progress")
            
        elif current_state == ServerState.IDLE:
            # First request initiates new transfer session
            server.state = ServerState.WAITING_FOR_RANKS
            server.expected_world_size = mp
            
        elif current_state == ServerState.WAITING_FOR_RANKS:
            # Verify mp_size matches
            if mp != server.expected_world_size:
                raise HTTPException(status_code=400, detail="Mismatched mp_size")
        
        # Check for duplicate rank
        if rank in server.joined_ranks:
            raise HTTPException(status_code=400, detail=f"Rank {rank} already joined")
        
        # Get tensor info for this rank
        safetensor_path = server.find_safetensor(mp, dtype, rank)
        if safetensor_path is None:
            raise HTTPException(status_code=404, detail=f"No safetensor found for mp_size={mp}, dtype={dtype}, rank={rank}")
        
        with safe_open(safetensor_path, framework="pytorch") as f:
            tensor_keys = sorted(f.keys())
            response_data = TensorInfo(
                file=str(safetensor_path.relative_to(server.base_weight_dir)),
                keys=tensor_keys
            )
        
        # Add rank to joined set
        server.joined_ranks.add(rank)
        
        # If all ranks joined, start transfer
        if len(server.joined_ranks) == mp:  # -1 because rank 0 is server
            logger.info(f"Starting weight transfer for mp_size={mp}, dtype={dtype}")
            server.transfer_thread = threading.Thread(
                target=server.start_weight_transfer,
                args=(mp, dtype)
            )
            server.transfer_thread.start()
        
        return response_data

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server() 