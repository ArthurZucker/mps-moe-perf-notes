"""Build the extension for the running Python + torch into the variant folder: python build_ext.py <variant dir>."""
import glob, os, shutil, sys
from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
build = os.path.join(here, ".build", f"py{sys.version_info[0]}{sys.version_info[1]}")
os.makedirs(build, exist_ok=True)
load("_mps_events", [os.path.join(here, "mps_events.mm")], build_directory=build,
     extra_cflags=["-std=c++20", "-O2", "-D_DARWIN_C_SOURCE", "-include", "sys/types.h"],
     extra_ldflags=["-framework", "Metal", "-framework", "Foundation"], verbose=False)
so = glob.glob(os.path.join(build, "_mps_events*.so"))[0]
tag = f"cpython-{sys.version_info[0]}{sys.version_info[1]}-darwin"
shutil.copy(so, os.path.join(sys.argv[1], f"_mps_events.{tag}.so"))
print("built", tag)
