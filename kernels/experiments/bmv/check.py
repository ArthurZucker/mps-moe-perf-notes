import importlib.util, sys, torch
B = "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal"
spec = importlib.util.spec_from_file_location("mmd", B + "/__init__.py", submodule_search_locations=[B])
mmd = importlib.util.module_from_spec(spec); sys.modules["mmd"] = mmd; spec.loader.exec_module(mmd)
xs = [torch.randn(1, 1, 2304, device="mps", dtype=torch.bfloat16) for _ in range(3)]
ws = [torch.randn(n, 2304, device="mps", dtype=torch.bfloat16) * 0.02 for n in (3072, 1024, 1024)]
outs = mmd.bmv_multi(xs[0], ws[0], xs[1], ws[1], xs[2], ws[2])
print("rel err", max(((torch.nn.functional.linear(x, w).float() - o.float()).abs().max() / torch.nn.functional.linear(x, w).float().abs().max()).item() for x, w, o in zip(xs, ws, outs)), [tuple(o.shape) for o in outs])
