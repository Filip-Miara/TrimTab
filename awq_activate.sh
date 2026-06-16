#!/bin/bash
# Activate the AWQ Python 3.11 environment (with Marlin kernel)
export TORCH_LIB=/mnt/windows/awq_env/lib/python3.11/site-packages/torch/lib
export CUDA_LIB=/mnt/windows/awq_env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$TORCH_LIB:$CUDA_LIB:$LD_LIBRARY_PATH

# Workaround for NTFS: set HF_HOME to writable location
export HF_HOME=/mnt/windows/awq_env/hf_cache

source /mnt/windows/awq_env/bin/activate
echo "AWQ env active (Python 3.11, torch 2.6.0+cu124, Marlin kernel compiled)"
echo "Usage: from transformers import AutoModelForCausalLM, AutoTokenizer"
