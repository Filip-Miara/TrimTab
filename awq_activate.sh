#!/bin/bash
# Activate the AWQ Python 3.11 environment (with Marlin kernel)
export AWQ_ENV=/mnt/windows/awq_env2
export TORCH_LIB=$AWQ_ENV/lib/python3.11/site-packages/torch/lib
export CUDA_LIB=$AWQ_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_LIB:$LD_LIBRARY_PATH
export HF_HOME=/home/filip/.cache/huggingface

source $AWQ_ENV/bin/activate
echo "AWQ env active (Python 3.11, torch 2.6.0+cu124, Marlin kernel compiled)"
echo "Backup: /run/media/filip/B522-875D/awq_env_backup.tar.gz"
echo "Usage: from transformers import AutoModelForCausalLM, AutoTokenizer"
