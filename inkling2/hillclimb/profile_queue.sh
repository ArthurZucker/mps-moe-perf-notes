#!/bin/zsh
# Component profiles (in-context GPU time per sub-block) of control vs candidate for each kernel experiment.
cd /Users/arthurzucker/Work/transformers/.claude/worktrees/inkling2-perf
H=benchmark_v2/inkling2_metal/hillclimb
i=0; start=$(date +%s)
for c in casts bundle prev-rev qkv-multi prev-rev2 down-route exact-prev exact-bmv; do
  i=$((i+1)); sleep 120
  PYTHONPATH=src:benchmark_v2/inkling2_metal:$H python ~/Work/hub-kernels/opt/gpu_lock.py python $H/hc_components.py $H/configs/$c.json 96 > $H/profiles/components-$c.log 2>&1
  el=$(( $(date +%s) - start ))
  echo "[$i/8] $c | $(grep -E '^==' $H/profiles/components-$c.log | cut -c1-150)$(grep -E 'refusing|Error' $H/profiles/components-$c.log | head -1) | elapsed ${el}s eta ~$(( el / i * (8 - i) ))s"
done
echo PROFILE_QUEUE_DONE
