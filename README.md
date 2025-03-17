# Chess LLM

This repository contains training code for training LLMs on chess for the [chess-llm project](https://manifestai.com/articles/post-training-r1-for-chess/).

## Disclaimer

This repository is prepared specifically for the cluster sponsored generously by [SF Compute](https://sfcompute.com/). To run the code, we provide a Dockerfile that we used for training as well as the model definition and training scripts.

There're various internal infrastructures such as customized logging, checkpointing, data preprocessing that are not published in this repository. So you might bump into issues when used in another environment.

## How to train

To build the docker image, run

```bash
docker buildx build --platform linux/amd64 .
```

In the container/k8s cluster, you can start training a model with the following command:

```bash
torchrun --nnodes=<num_nodes> --nproc_per_node=<num_gpus_per_node> trainer.py --model-name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B --seq-len=1024 --batch-size=8 --dp-size=<desired_dp_size> --tp-size=<desired_tp_size> --pp-size=<desired_pp_size> --run-name=<run_name>
```

which starts training from the R1-distilled 8B llama model with a sequence length of 1024 and a batch size of 8.

To train the actual 671B deepseek model, use the following command:

```bash
torchrun --nnodes=<num_nodes> --nproc_per_node=<num_gpus_per_node> trainer.py --deepseek --deepseek-config=deepseek/configs/config_671B.json --seq-len=1024 --batch-size=8 --dp-size=<desired_dp_size> --tp-size=<desired_tp_size> --pp-size=<desired_pp_size> --run-name=<run_name>
```

Note that as detailed in the [release](https://manifestai.com/articles/post-training-r1-for-chess/), training the 671B model requires at least 16 nodes, each with 8 H100 GPUs. It's recommended to actually use 32 nodes with 16 GPUs each for training, which allows for a larger batch size and context length.

## How to evaluate

To evaluate the trained model, the model checkpoint should be uploaded to a google cloud storage bucket, which is exposed as an environment variable `GS_BUCKET_CHECKPOINT` in the container.

Then, to evaluate the model, first install stockfish by

```bash
cd evaluation
./install_stockfish.sh
```

and then run

```bash
python evaluation/main.py --model-name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B --tokenizer=deepseek/tokenizer.json --run-name=<run_name> --checkpoint=<checkpoint_iter> --seq-len=<seq_len> --batch-size=<batch_size> --cpi=<cpi> --which=<which> --sf-depth=<sf_depth> --sf-games=<sf_games> --sf-batch-size=<sf_batch_size>
```

