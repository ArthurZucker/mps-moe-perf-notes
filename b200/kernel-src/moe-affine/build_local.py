"""Build this kernel repo for the local GPU into build/<variant>/ (the layout `kernels.get_local_kernel` loads), with the
local nvcc. kernel-builder (nix) produces the same layout for every variant when publishing."""
import hashlib, json, os, shutil, subprocess, sys, sysconfig
import torch
from torch.utils import cpp_extension

# torch >= 2.13 headers need C++20
STD = "-std=c++20" if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 13) else "-std=c++17"

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, "csrc", "moe_affine.cu")
digest = hashlib.sha256(open(src, "rb").read()).hexdigest()[:7]
ns = f"_moe_affine_cuda_{digest}"
tv = torch.__version__.split("+")[0].split(".")
variant = f"torch{tv[0]}{tv[1]}-cu{torch.version.cuda.replace('.', '')}-x86_64-linux"
final = os.path.join(here, "build", variant)
out = final + ".tmp"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(os.path.join(here, "torch-ext", "moe_affine"), out)
major, minor = torch.cuda.get_device_capability()
arch = f"{major}{minor}a"
tmp = os.path.join(here, ".obj"); os.makedirs(tmp, exist_ok=True)
inc = [f"-I{p}" for p in cpp_extension.include_paths(device_type="cuda")] + [f"-I{sysconfig.get_paths()['include']}"]
obj = os.path.join(tmp, "moe_affine.o")
nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
site = os.path.dirname(os.path.dirname(torch.__file__)) + "/"
maps = [f"-ffile-prefix-map={site}=", f"-ffile-prefix-map={here}/="]
subprocess.check_call([nvcc, "-c", src, "-o", obj, "-O3", "--use_fast_math", STD, "-Xcompiler", "-fPIC", *[a for m in maps for a in ("-Xcompiler", m)],
                       f"-gencode=arch=compute_{arch},code=sm_{arch}", f"-DOPS_NS={ns}", "-D_GLIBCXX_USE_CXX11_ABI=1",
                       "-DTORCH_API_INCLUDE_EXTENSION_H", *inc])
obj2 = os.path.join(tmp, "host_tier.o")
subprocess.check_call(["g++", "-c", os.path.join(here, "csrc", "host_tier.cpp"), "-o", obj2, "-O3", STD, "-fPIC",
                       "-march=sapphirerapids", *maps, "-D_GLIBCXX_USE_CXX11_ABI=1", "-DTORCH_API_INCLUDE_EXTENSION_H",
                       f"-I{os.path.join(os.environ.get('CUDA_HOME', '/usr/local/cuda'), 'include')}", *inc])
obj3 = os.path.join(tmp, "host_tier2.o")
subprocess.check_call(["g++", "-c", os.path.join(here, "csrc", "host_tier2.cpp"), "-o", obj3, "-O3", STD, "-fPIC",
                       "-march=sapphirerapids", *maps, "-D_GLIBCXX_USE_CXX11_ABI=1", "-DTORCH_API_INCLUDE_EXTENSION_H",
                       f"-I{os.path.join(os.environ.get('CUDA_HOME', '/usr/local/cuda'), 'include')}", *inc])
libs = [f"-L{p}" for p in cpp_extension.library_paths(device_type="cuda")]
so = os.path.join(out, f"{ns}.abi3.so")
subprocess.check_call(["g++", "-shared", obj, obj2, obj3, "-o", so, "-lpthread", *libs, "-lc10", "-lc10_cuda", "-ltorch", "-ltorch_cpu", "-ltorch_cuda",
                       "-lcudart"])
open(os.path.join(out, "_ops.py"), "w").write(f'''import torch
from . import {ns}
ops = torch.ops.{ns}


def add_op_namespace_prefix(op_name: str):
    return f"{ns}::{{op_name}}"
''')
import base64
files = {}
for root, _, names in os.walk(out):
    for n in sorted(names):
        if n.endswith((".pyc", "metadata.json")) or "__pycache__" in root: continue
        fp = os.path.join(root, n)
        files[os.path.relpath(fp, out)] = base64.b64encode(hashlib.sha256(open(fp, "rb").read()).digest()).decode()
json.dump({"name": "moe-affine", "id": ns, "digest": {"algorithm": "sha256", "files": files}, "version": 1, "license": "Apache-2.0", "python-depends": [], "kernel-depends": [],
           "backend": {"type": "cuda", "archs": [f"{major}.{minor}"]}}, open(os.path.join(out, "metadata.json"), "w"), indent=2)
old = final + ".old"
shutil.rmtree(old, ignore_errors=True)
if os.path.exists(final):
    os.rename(final, old)  # running processes keep their loaded .so (by inode); new ones see the new build
os.rename(out, final)
print("built", final)
