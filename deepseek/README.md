# Weight Distribution

A parameter server at `deepseek-svc:8000` has been manually set up, where the 670B model weights have been downloaded and processed.

## Inference Weight

Inference requires the original fp8 weights split by <world_size>, where the minimum world size is 16. For a batch job to obtain the weights for different ranks, prepend the following command to your main command:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=deepseek-r1-trainer-0.deepseek-r1-network:29500 deepseek/weight_client.py --param-server deepseek-svc --param-port 8000 --dtype fp8 --save-dir /data/r1
```

In the above command, I'm distributing the weights into 2 nodes, each with 8 GPUs, the weights will be saved in `/data/r1/mp16`.

This command scatters about 2.5TB of weights (in fp32 because NCCL doesn't support fp8 yet) to all the GPUs, which takes about 7 minutes.

## Training Weight

Training requires loading bf16 weights (pre-converted from fp8 weights and stored in the parameter server).

```bash
torchrun --nnodes=2 --nproc-per-node=8 --rdzv-backend=c10d --rdzv-endpoint=deepseek-r1-trainer-0.deepseek-r1-network:29500 deepseek/weight_client.py --param-server deepseek-svc --param-port 8000 --dtype bf16 --save-dir /data/r1
```

