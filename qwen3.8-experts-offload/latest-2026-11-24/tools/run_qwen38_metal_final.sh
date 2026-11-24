#!/usr/bin/env bash
# Qwen3.8-Flash-Next on Apple Silicon, final-checkpoint path.
#
# Differences from run_qwen38_metal_fixed.sh:
#  - uses a checkpoint whose always-on nn.Linears are already packed (Metal 4-bit, group 32,
#    clipping search). No quantize_dense work happens during the benchmark.
#  - refuses to run if quantize_dense quantizes anything (REQUIRE_NO_QUANT=1).
#  - can enforce a Metal allocator cap (MPS_MEM_GIB) to emulate a smaller Mac.
#  - pins kernels < 0.19.0: transformers 5.23 rejects 0.19.0 (MAX_VERSION is exclusive).
#
#   MODEL_PATH=/path/to/final-ckpt bash run_qwen38_metal_final.sh [expert cache GiB, default 6]
set -euo pipefail
CACHE_GIB=${1:-6}
REPO=${MODEL_PATH:-ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit-final}
WORK=${WORK:-$HOME/qwen38-metal}
mkdir -p "$WORK" && cd "$WORK"

xcode-select -p >/dev/null 2>&1 || { echo "needs Xcode command line tools: xcode-select --install"; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d transformers ] || git clone -q --depth 1 -b stream-experts https://github.com/huggingface/transformers
git -C transformers fetch -q --depth 1 origin stream-experts && git -C transformers reset -q --hard FETCH_HEAD
echo "transformers @ $(git -C transformers log --oneline -1)"
[ -d .venv ] || uv venv -q -p 3.12 .venv
source .venv/bin/activate
uv pip install -q -e ./transformers "torch>=2.15" numpy accelerate ninja \
  "kernels>=0.18.2,<0.19.0" "huggingface_hub[hf_xet]" "safetensors>=0.9.0rc1"

if [ -n "${MODEL_PATH:-}" ]; then
  REPO=$MODEL_PATH
  echo "[1/2] using local checkpoint $REPO"
else
  echo "[1/2] downloading $REPO (resumable)"
  hf download "$REPO"
fi

echo "[2/2] decode benchmark (expert cache ${CACHE_GIB} GiB)"
B=$WORK/transformers/benchmark_v2/experts_offload
export HF_DEACTIVATE_ASYNC_LOAD=1
export TRANSFORMERS_EXPERTS_OFFLOAD_MPS_EVENTS=${TRANSFORMERS_EXPERTS_OFFLOAD_MPS_EVENTS:-0}
export REQUIRE_NO_QUANT=${REQUIRE_NO_QUANT:-1}
mkdir -p "$WORK/.bench"
cat > "$WORK/.bench/mem_bounded_run.py" <<'PY_WRAPPER'
# Run a script with quantize_dense / fuse_sibling_linears bounded in GPU memory (32 GB Macs). The expert pool is
# allocated at load, so quantization runs with bf16 dense + pool already resident; MPS is asynchronous, so without a
# sync the float32 temporaries of many layers pile up past the Metal working set.
import gc, json, os, runpy, sys

import torch
import dense_quant
import transformers.integrations.metal_quantization as mq

# Emulate a 32 GB Mac's Metal working set on a larger host. On a 32 GB M5,
# torch.mps.recommended_max_memory() is ~25 GiB; a fraction of that value gives the
# same allocator ceiling here. Set MPS_MEM_GIB=0 to disable.
_mps_cap = float(os.environ.get("MPS_MEM_GIB", "0") or 0)
if _mps_cap:
    _recommended = torch.mps.recommended_max_memory()
    _fraction = _mps_cap * 2**30 / _recommended
    torch.mps.set_per_process_memory_fraction(float(_fraction))
    print(f"  MPS cap {_mps_cap:.2f} GiB (fraction {_fraction:.5f} of {_recommended / 2**30:.2f} GiB recommended)",
          flush=True)

# Count actual quantization work. A final checkpoint (always-on layers already
# packed) must reach generate with zero calls here; REQUIRE_NO_QUANT=1 makes the
# run fail instead of silently quantizing on the fly.
QUANT_CALLS = [0]


def gib():
    return torch.mps.driver_allocated_memory() / 2**30


def release():
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()


def bounded(fn):
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        QUANT_CALLS[0] += 1
        out = fn(*args, **kwargs)
        torch.mps.synchronize()
        calls += 1
        if calls % 16 == 0:
            torch.mps.empty_cache()
        return out

    return wrapped


def chunked(fn, rows_per_chunk=16384):
    # lm_head (248320 x 2560) alone took the process from 9.7 to 26.4 GB: the clipping search keeps several float32
    # copies of the whole matrix alive. Groups lie within a row, so quantizing row chunks is exact.
    def wrapped(weight, *args, **kwargs):
        if weight.shape[0] <= rows_per_chunk:
            return fn(weight, *args, **kwargs)
        parts = []
        for start in range(0, weight.shape[0], rows_per_chunk):
            parts.append(fn(weight[start : start + rows_per_chunk], *args, **kwargs))
            torch.mps.synchronize()
            torch.mps.empty_cache()
        return tuple(torch.cat(p) for p in zip(*parts))

    return wrapped


dense_quant._clipped_affine = chunked(bounded(dense_quant._clipped_affine))
mq._affine_quantize_tensor = chunked(bounded(mq._affine_quantize_tensor))
fuse = mq.fuse_sibling_linears


def fuse_bounded(*args, **kwargs):
    release()  # the replaced bf16 Linears, before fusion concatenates copies
    print(f"  MPS before fusion: {gib():.1f} GiB", flush=True)
    out = fuse(*args, **kwargs)
    release()  # the unfused members
    print(f"  MPS after fusion: {gib():.1f} GiB", flush=True)
    return out


mq.fuse_sibling_linears = fuse_bounded

_orig_quantize_dense = dense_quant.quantize_dense


def quantize_checked(model, spec):
    before = QUANT_CALLS[0]
    out = _orig_quantize_dense(model, spec)
    n = QUANT_CALLS[0] - before
    print(f"  on-the-fly quantization calls: {n}", flush=True)
    if os.environ.get("REQUIRE_NO_QUANT", "0") == "1" and n:
        raise SystemExit(f"REQUIRE_NO_QUANT=1 but quantize_dense made {n} quantization calls")
    return out


dense_quant.quantize_dense = quantize_checked
import os

# EXPERTS_NOCACHE (default 1): read streamed experts with preadv from F_NOCACHE descriptors instead of safetensors
# slices. Prefill with a small cache reads ~1 GB/s of experts; through the unified buffer cache that churn made macOS
# compress and swap everything else out (traced: 24 GB read in 21 s, swap 1.5 -> 12 GB, process footprint flat).
# Rows land in the pool right after, so caching them in the file cache buys nothing. Same bytes as the slice read.
if os.environ.get("EXPERTS_NOCACHE", "1") == "1":
    import fcntl, json, struct, threading

    import transformers.distributed.offload as offload

    _DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8,
               "I8": torch.int8, "U32": torch.uint32, "I32": torch.int32, "I64": torch.int64,
               "F8_E4M3": torch.float8_e4m3fn}
    _files, _keys, _lock = {}, {}, threading.Lock()

    def _locate(path, key):
        with _lock:
            if (path, key) not in _keys:
                if path not in _files:
                    fd = os.open(path, os.O_RDONLY)
                    fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    with open(path, "rb") as f:
                        size = struct.unpack("<Q", f.read(8))[0]
                        _files[path] = (fd, 8 + size, json.loads(f.read(size)))
                fd, start, header = _files[path]
                info = header[key]
                row_bytes = (info["data_offsets"][1] - info["data_offsets"][0]) // info["shape"][0]
                _keys[(path, key)] = (fd, start + info["data_offsets"][0], row_bytes, _DTYPES[info["dtype"]],
                                      info["shape"][1:])
            return _keys[(path, key)]

    def _row(path, key, index):
        fd, base, row_bytes, dtype, shape = _locate(path, key)
        buf = torch.empty(row_bytes, dtype=torch.uint8)
        if os.preadv(fd, [memoryview(buf.numpy())], base + index * row_bytes) != row_bytes:
            raise OSError(f"short read of row {index} of {key}")
        return buf.view(dtype).reshape(shape)

    _read = offload.ExpertsOffloadStore._read

    def _read_nocache(self, layer, expert):
        if self.key_files is None:
            return _read(self, layer, expert)
        self.reads += 1
        out = {}
        for reader in self.readers[layer]:
            for param, _, rows, dtype, key in reader.sources(expert):
                tensor = _row(self.key_files[key], key, rows.start)
                out[param] = tensor if dtype is None or dtype == tensor.dtype else tensor.to(dtype)
        return out

    offload.ExpertsOffloadStore._read = _read_nocache

# STREAM_EMBED (default 1): stream embed_tokens rows from disk like the n-gram table (one row per token), instead of
# a 1.2 GiB bf16 table pinned (wired) on the GPU. max_memory shrinks by the same bytes so the expert pool stays the
# size asked for.
if os.environ.get("STREAM_EMBED", "1") == "1":
    import transformers

    _from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

    def _from_pretrained_embed_on_disk(repo, *args, **kwargs):
        device_map, max_memory = kwargs.get("device_map"), kwargs.get("max_memory")
        if isinstance(device_map, dict) and device_map.get("experts") == "disk" and max_memory:
            text = transformers.AutoConfig.from_pretrained(repo).get_text_config()
            embed_bytes = text.vocab_size * text.hidden_size * 2
            kwargs["device_map"] = {**device_map, "embed_tokens": "disk"}
            kwargs["max_memory"] = {**max_memory, "mps": max_memory["mps"] - embed_bytes}
            print(f"  embed_tokens streamed from disk ({embed_bytes / 2**30:.2f} GiB off the GPU)", flush=True)
        return _from_pretrained(repo, *args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = _from_pretrained_embed_on_disk

# TILED_QMM (default 1): affine_qmm_t (kernels-community/mlx-quantization-metal-kernels v2, every multi-row
# MetalLinear call: prefill) returns wrong results above 256 output tiles of 32 x 32 on an M5 (e.g. q_proj 12288 x 2560
# from 25 rows, any 1024-wide layer from 512 rows): Qwen3.8 then generates garbage. Calls stay under the limit by
# blocks of 256 rows x 1024 outputs (slices of the packed rows: exact). One-row calls (decode) are unaffected.
if os.environ.get("TILED_QMM", "1") == "1":
    _metal_forward = mq.MetalLinear.forward

    def _tiled_forward(self, input):
        rows = input.numel() // self.in_features
        if self.weight.dtype != torch.uint32 or rows <= 1 or (-(-rows // 32)) * (-(-self.out_features // 32)) <= 256:
            return _metal_forward(self, input)
        kernel = mq._get_metal_kernel()
        x = input.reshape(rows, self.in_features)
        scales, biases = self.scales.to(input.dtype), self.qbiases.to(input.dtype)
        out = torch.empty(rows, self.out_features, dtype=input.dtype, device=input.device)
        for r in range(0, rows, 256):
            for c in range(0, self.out_features, 1024):
                out[r : r + 256, c : c + 1024] = kernel.affine_qmm_t(
                    x[r : r + 256], self.weight[c : c + 1024], scales[c : c + 1024], biases[c : c + 1024],
                    self.group_size, self.bits)
        out = out.reshape(*input.shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    mq.MetalLinear.forward = _tiled_forward

# A local checkpoint with pre-quantized always-on layers: verify_generate.py sizes max_memory as 2 bytes per element
# of every non-expert tensor, but packed uint32 weights take 4, and MetalLinear scales/biases load as float32. Add the
# difference so the expert pool keeps the size asked for, then store scales/biases as bf16 (what the kernels use).
import transformers as _tf

_load = _tf.AutoModelForCausalLM.from_pretrained


def _from_pretrained_prequantized(repo, *args, **kwargs):
    index = os.path.join(str(repo), "model.safetensors.index.json")
    max_memory = kwargs.get("max_memory")
    if os.path.isfile(index) and max_memory:
        from safetensors import safe_open

        weight_map = json.load(open(index))["weight_map"]
        extra = 0
        for file in sorted(set(weight_map.values())):
            with safe_open(os.path.join(str(repo), file), "pt") as f:
                for key in f.keys():
                    if ".experts." in key or ".ngram_embedding." in key:
                        continue
                    if key.endswith((".scales", ".qbiases")) or f.get_slice(key).get_dtype() == "U32":
                        extra += torch.Size(f.get_slice(key).get_shape()).numel() * 2
        if extra:
            kwargs["max_memory"] = {**max_memory, "mps": max_memory["mps"] + extra}
            print(f"  pre-quantized checkpoint: max_memory +{extra / 2**30:.2f} GiB (packed weights, scales)", flush=True)
    model = _load(repo, *args, **kwargs)
    for module in model.modules():
        if isinstance(module, mq.MetalLinear) and module.scales.dtype == torch.float32:
            module.scales.data, module.qbiases.data = module.scales.data.to(torch.bfloat16), module.qbiases.data.to(torch.bfloat16)
    return model


_tf.AutoModelForCausalLM.from_pretrained = _from_pretrained_prequantized

# SHOW_TEXT (default 1): print the start of what the eager reference and the compiled check generate. verify_generate.py
# forces its token counts and never looks at the text, so garbage output went unnoticed through whole benchmarks.
if os.environ.get("SHOW_TEXT", "1") == "1":
    from transformers.generation.utils import GenerationMixin as _GM

    _gen, _calls, _tok = _GM.generate, [0], {}

    def _generate_showing(self, *args, **kwargs):
        out = _gen(self, *args, **kwargs)
        _calls[0] += 1
        if _calls[0] in (1, 3):
            if "tok" not in _tok:
                _tok["tok"] = _tf.AutoTokenizer.from_pretrained(self.name_or_path)
            prompt = (args[0] if args else kwargs["input_ids"]).shape[1]
            text = _tok["tok"].decode(out[0, prompt:])[:160]
            print(f"  {'eager' if _calls[0] == 1 else 'compiled'} text: {text!r}", flush=True)
        return out

    _GM.generate = _generate_showing

# MEM_TRACE=1: every 2 s, append to $WORK/.bench/memtrace.tsv the phase, the MPS memory held by live tensors
# (current) and by the allocator (driver), the process physical footprint and the system swap in use. Each phase
# change is logged at once. Above MEM_TRACE_SWAP_GB of swap (default 10), list the 25 largest live tensors to
# memtrace_tensors.txt and stop the run (exit 3).
import os

if os.environ.get("MEM_TRACE") == "1":
    import ctypes, subprocess, threading, time

    import transformers
    from transformers.generation.utils import GenerationMixin

    bench = os.path.dirname(os.path.abspath(__file__))
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    t0 = time.perf_counter()
    state = {"phase": "start", "peak": 0.0}
    out = open(os.path.join(bench, "memtrace.tsv"), "w", buffering=1)
    out.write(
        "t_s\tphase\tmps_current_gib\tmps_driver_gib\tfootprint_gib\tpeak_footprint_gib\tswap_used_gib"
        "\tproc_disk_read_gib\tproc_pageins\tsys_file_backed_gib\tsys_anonymous_gib\tsys_compressor_gib"
        "\tsys_wired_gib\tsys_free_gib\tsys_swapins\tsys_swapouts\n"
    )
    limit = float(os.environ.get("MEM_TRACE_SWAP_GB", "10"))
    lock = threading.Lock()

    def rusage():
        # rusage_info_v4 as 64-bit words: uuid (0-1), ..., ri_pageins 6, ri_phys_footprint 9, ri_diskio_bytesread 18
        info = (ctypes.c_uint64 * 64)()
        libproc.proc_pid_rusage(os.getpid(), 4, info)
        return info

    def footprint():
        return rusage()[9] / 2**30

    def vm():
        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(text.split("page size of ")[1].split(" ")[0])
        pages = {}
        for line in text.splitlines()[1:]:
            name, _, value = line.partition(":")
            if value.strip().rstrip(".").isdigit():
                pages[name.strip()] = int(value.strip().rstrip("."))
        gib = lambda key: pages.get(key, 0) * page / 2**30
        return (
            f"{gib('File-backed pages'):.2f}\t{gib('Anonymous pages'):.2f}\t{gib('Pages occupied by compressor'):.2f}"
            f"\t{gib('Pages wired down'):.2f}\t{gib('Pages free'):.2f}\t{pages.get('Swapins', 0)}\t{pages.get('Swapouts', 0)}"
        )

    def swap_used():
        s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        return float(s.split("used = ")[1].split("M")[0]) / 1024

    def largest_tensors(path):
        seen = {}
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.device.type == "mps":
                    st = obj.untyped_storage()
                    seen[st.data_ptr()] = (st.nbytes(), tuple(obj.shape), str(obj.dtype))
            except Exception:
                pass
        rows = sorted(seen.values(), reverse=True)
        with open(path, "w") as f:
            f.write(f"phase {state['phase']}: {len(rows)} MPS storages, {sum(r[0] for r in rows) / 2**30:.2f} GiB\n")
            for nbytes, shape, dtype in rows[:25]:
                f.write(f"{nbytes / 2**20:10.1f} MiB  {dtype:16s} {shape}\n")

    def sample(tag=""):
        with lock:
            ru = rusage()
            fp = ru[9] / 2**30
            state["peak"] = max(state["peak"], fp)
            sw = swap_used()
            out.write(
                f"{time.perf_counter() - t0:.1f}\t{state['phase']}{tag}\t"
                f"{torch.mps.current_allocated_memory() / 2**30:.2f}\t{torch.mps.driver_allocated_memory() / 2**30:.2f}\t"
                f"{fp:.2f}\t{state['peak']:.2f}\t{sw:.2f}\t{ru[18] / 2**30:.2f}\t{ru[6]}\t{vm()}\n"
            )
            return sw

    def phase(name):
        state["phase"] = name
        sample(" (start)")

    proc_limit = float(os.environ.get("PROC_MEM_GIB", "0") or 0)

    def watch():
        while True:
            time.sleep(2)
            sw = sample()
            fp = rusage()[9] / 2**30
            if sw > limit:
                largest_tensors(os.path.join(bench, "memtrace_tensors.txt"))
                print(f"MEM_TRACE: swap above {limit} GB in phase {state['phase']}: stopping", flush=True)
                os._exit(3)
            if proc_limit and fp > proc_limit:
                largest_tensors(os.path.join(bench, "memtrace_tensors.txt"))
                print(f"MEM_TRACE: process footprint {fp:.2f} GiB above {proc_limit:.2f} GiB in phase "
                      f"{state['phase']}: stopping", flush=True)
                os._exit(4)

    def traced(name, fn):
        def wrapped(*args, **kwargs):
            phase(name)
            try:
                return fn(*args, **kwargs)
            finally:
                phase(name + " done")

        return wrapped

    transformers.AutoModelForCausalLM.from_pretrained = traced("load", transformers.AutoModelForCausalLM.from_pretrained)
    dense_quant.quantize_dense = traced("quantize", dense_quant.quantize_dense)
    mq.fuse_sibling_linears = traced("fusion", mq.fuse_sibling_linears)
    plan = ["eager ref", "compile warmup", "compiled check"] + [f"decode {i}/4" for i in range(1, 5)] + ["sustained"]
    generate, calls = GenerationMixin.generate, [0]

    def traced_generate(self, *args, **kwargs):
        calls[0] += 1
        name = plan[calls[0] - 1] if calls[0] <= len(plan) else f"generate {calls[0]}"
        inner, puts = kwargs.get("streamer"), [0]

        class Streamer:  # second put = first generated token: prefill is over
            def put(self, value):
                if inner is not None:
                    inner.put(value)
                puts[0] += 1
                if puts[0] == 2:
                    phase(name + ": decode")

            def end(self):
                if inner is not None:
                    inner.end()

        kwargs["streamer"] = Streamer()
        phase(name + ": prefill")
        try:
            return generate(self, *args, **kwargs)
        finally:
            phase(name + " done")
            if name == "compile warmup":
                largest_tensors(os.path.join(bench, "memtrace_tensors_after_warmup.txt"))

    GenerationMixin.generate = traced_generate
    threading.Thread(target=watch, daemon=True).start()

script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(script, run_name="__main__")
PY_WRAPPER
cd "$B" && PYTHONPATH="$WORK/transformers/src:$B" python "$WORK/.bench/mem_bounded_run.py" verify_generate.py "$REPO" 4g32c "$CACHE_GIB" 2>&1 \
  | grep -vE "^W[0-9]{4}|Loading weights|Fetching" | tee "$WORK/result_final.txt"
echo "results in $WORK/result_final.txt"
