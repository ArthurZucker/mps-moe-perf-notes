source /home/arthur/tf-pr50997/.venv/bin/activate
C=/home/arthur/tf-pr50997/.venv/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$C PATH=$C/bin:$PATH LIBRARY_PATH=$C/lib TORCH_CUDA_ARCH_LIST=10.0
export TORCHINDUCTOR_CACHE_DIR=/raid/arthur/inductor-cache TORCH_EXTENSIONS_DIR=/raid/arthur/torch-ext
export TRANSFORMERS_MOE_AFFINE_KERNEL=/home/arthur/mps-moe-perf-notes/b200/kernel-src/moe-affine
export HF_HUB_OFFLINE=1
