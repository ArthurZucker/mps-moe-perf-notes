# Same workflow, two models: Inkling-2-Mini vs Qwen3.8-Flash-Next

The Inkling result is now reproducible on `huggingface/transformers@inkling2-metal-perf@fb89fd0`.

| | Inkling-2-Mini | Qwen3.8-Flash-Next |
|---|---|---|
| dense | 4-bit Metal (prequantized? no, current plan) | **prequantized final checkpoint** |
| experts | native NVFP4/MXFP8, 4-bit | 4-bit affine, group 64 |
| model resident | 10.5 GiB current / 18 GiB driver after empty_cache | 3.5 GiB driver after load |
| quantization peak | **37 GB** (on-the-fly dense) | **0** (prequantized) |
| 32 GB decode | **79.14 tok/s** | **30.88 tok/s** (JITEngine), 18.86 transformers buggy, 8.56 correct |
| 48 GB decode | — | **59.59 tok/s** |
| 64 GB decode | — | **69.80 tok/s** |
| output hash, 32→96 GB | — | `07a676778e3a6720` at every budget |

## Why the same workflow does not give Qwen3.8 50 tok/s at 32 GB

Inkling's whole expert bank fits: 4.97 GB of experts + dense, 18 GB driver. Qwen3.8's
experts are ~65 GB; a 32 GB budget holds 9,122 of 24,576 expert records (37%), and decode
reads about 37 MB/token from SSD. The same prequantized dense checkpoint, sibling fusion,
and 4-bit expert kernels are already used; the limit is the expert bank, not the kernels.

The exact same method applied to Qwen3.8 is:

1. final checkpoint with always-on layers prequantized (done, `-final`);
2. fused sibling linears (192 groups);
3. 4-bit Metal experts with `gather_qmm`/`qmv4`;
4. memory ladder with one output hash (done).

It yields 30.88 tok/s at 32 GB and 59.59 at 48 GB.

## If 50 tok/s at 32 GB is required

- 2.5-bit Sakura 352E: **56.4–62.2 tok/s** at 30 GB footprint, not 4-bit/transformers.
- 4-bit: use **48 GB** (59.59 tok/s).
- Otherwise the 32 GB 4-bit ceiling is ~31 tok/s, with a measured ~40 tok/s overlap
  ceiling under the current non-stall path.

Inkling reaches 50+ at 32 GB because the model fits; Qwen3.8 does not. Applying the same
kernel/scheduling workflow cannot change the expert-bank size.
