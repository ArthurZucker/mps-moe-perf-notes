source /raid/arthur/venv-t215/bin/activate
C=/raid/arthur/venv-t215/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$C PATH=$C/bin:$PATH LIBRARY_PATH=$C/lib TORCH_CUDA_ARCH_LIST=10.0
export TORCHINDUCTOR_CACHE_DIR=/raid/arthur/inductor-cache-t215 TORCH_EXTENSIONS_DIR=/raid/arthur/torch-ext-t215
export TRANSFORMERS_MOE_AFFINE_KERNEL=/home/arthur/mps-moe-perf-notes/b200/kernel-src/moe-affine
export HF_HUB_OFFLINE=1
