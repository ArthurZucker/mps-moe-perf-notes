# Correction: the row-table read path is supported by safetensors PRs

The earlier note that "the PLE table is a custom lazy mmap, not supported" was wrong.
The streaming path is exactly what the safetensors work targets:

- **safetensors#996** (`prefetch: load to mps, straight into unified memory`):
  `prefetch(plan, device="mps")` reads rows with the GIL released straight into MPS
  unified memory; `out=` fills existing tensors, e.g. expert-pool slots. This is what
  the transformers stream-experts branch uses for expert misses.
- **safetensors#995** (`Stream sparse rows into device tensors with the GIL released
  (CUDA and Metal)`): the issue for the general row-table path, including PLE/n-gram
  rows.
- **safetensors#994**: a prefetch plan value can be a list of slices.
- **safetensors#884** (MLX): zero-copy Metal loading via DLPack, the MLX counterpart.

The transformers `_RowReader` currently reads PLE/n-gram rows through `safe_open`
slices (`_RowReader.read`), dequantizes FP8 rows on the host, and `cat`s them into one
device transfer. That is a supported streaming path, not an unsupported custom mmap.

This does not change the 32 GB expert-cache bottleneck: PLE rows are already streamed;
the 4-bit routed experts are the resident set. It corrects the record and points the
next read-path work at #995/#996 rather than a new format.
