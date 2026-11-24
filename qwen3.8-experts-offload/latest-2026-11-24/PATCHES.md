# transformers patches in this directory

Apply to the `stream-experts` checkout (tested at `e6e0218`):

```bash
git -C transformers apply /path/to/patches/0001-offload-pin-ahead-and-lookahead.diff
git -C transformers apply /path/to/patches/0002-qwen4-deterministic-topk.diff
```

## 0001-offload-pin-ahead-and-lookahead.diff

`src/transformers/distributed/offload.py`

- `TRANSFORMERS_EXPERTS_OFFLOAD_LOOKAHEAD=N`: read-ahead depth override.
- `TRANSFORMERS_EXPERTS_OFFLOAD_PIN_AHEAD=1`: when installing read-ahead guesses for
  layer `nxt`, keep every expert of `nxt` already resident. This prevents read-ahead from
  overwriting a slot the GPU's `nxt` pass 1 may read.
- Effect: eliminates the `KeyError ('model.layers.N.mlp.experts', e)` at 6-14 GiB in
  short runs; full 16 GiB verify still fails greedy equivalence (5/128). Not a complete
  correctness fix.

## 0002-qwen4-deterministic-topk.diff

`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`

- `DETERMINISTIC_TOPK=1`: stable `argsort` instead of `torch.topk`, then selected experts
  ordered by id. The same top-k set and the same summation order on every run.
- Effect: does **not** fix the remaining nondeterminism (see `RESULTS.md` section 5); it
  removes one possible source and is a useful diagnostic.

## Not a patch: deterministic expert decode

`tools/bench_short.py` can disable `MetalExperts._fused_decode` after load
(`DETERMINISTIC=1`) to force the non-fused `_qmm` + top-k sum path. This also did not fix
the nondeterminism, but it isolates the fused decode kernel.

## Not in this directory

The final checkpoint builder is a tool, not a transformers patch:
`tools/make_final_qwen38_checkpoint.py`. The final harness is
`tools/run_qwen38_metal_final.sh` and embeds the 32 GB workarounds.
