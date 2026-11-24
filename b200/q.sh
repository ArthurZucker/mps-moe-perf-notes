#!/bin/bash
# Serialized, guarded run on one GPU: refuses if any other process uses it, records before/after state.
# usage: GPU=1 ./q.sh <logname> <command...>
GPU=${GPU:-1}; name=$1; shift
uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
exec 9>/tmp/claude-150039/gpu$GPU.lock; flock 9
busy() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid"; }
while [ "$(busy)" != "0" ]; do echo "GPU$GPU busy, waiting"; sleep 30; done
source /home/arthur/mps-moe-perf-notes/b200/env.sh
echo "== $(date -Is) GPU$GPU idle; $*" > logs/$name.log
NODE=$(nvidia-smi topo -m | awk -v g="GPU$GPU" '$1==g{print $(NF-1)}')
CUDA_VISIBLE_DEVICES=$GPU ${NUMA:+numactl --cpunodebind=$NODE --membind=$NODE} "$@" >> logs/$name.log 2>&1
rc=$?
others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | wc -l)
echo "== $(date -Is) exit $rc; other processes on GPU at end: $others" >> logs/$name.log
