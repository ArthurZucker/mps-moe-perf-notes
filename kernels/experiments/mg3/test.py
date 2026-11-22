import importlib.util, sys, statistics, torch
spec = importlib.util.spec_from_file_location("mmd", "/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal/__init__.py",
                                              submodule_search_locations=["/Users/arthurzucker/Work/hub-kernels/metal-moe-decode/build/torch-metal"])
mmd = importlib.util.module_from_spec(spec); sys.modules["mmd"] = mmd; spec.loader.exec_module(mmd)
F = sys.modules["mmd.inkling_fused"]
sys.path.insert(0, "/Users/arthurzucker/Work/transformers/.claude/worktrees/inkling2-perf/src")
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
cfg = Inkling2LoopTextConfig(hidden_size=2304, loop_gate_rank=64)
torch.manual_seed(0)
gates = [M.Inkling2LoopMessageGate(cfg).to("mps") for _ in range(3)]
for g in gates:
    g.down_proj.to(torch.bfloat16)
    for p in g.parameters(): p.data.normal_(0, 0.05)
norm = M.Inkling2LoopRMSNorm(2304, eps=1e-6).to("mps").to(torch.bfloat16); norm.weight.data.uniform_(0.5, 1.5)
H = lambda: torch.randn(1, 1, 2304, device="mps", dtype=torch.bfloat16)
hs, ms = (H(),) * 3, (H(), H(), H())
ref_fn = M.inkling2_loop_qkv_message_gates.__wrapped__ if hasattr(M.inkling2_loop_qkv_message_gates, "__wrapped__") else None
def ref(hs, ms, norms):
    outs = [F.inkling2_message_gate(h, m, g) for h, m, g in zip(hs, ms, gates)]
    return [o if norms is None else n(o) for n, o in zip(norms or [None] * 3, outs)]
with torch.inference_mode():
    for label, hs_, norms in (("sliding (shared h, norms)", hs, (norm,) * 3), ("full (3 h, no norm)", (H(), H(), H()), None)):
        a = ref(hs_, ms, norms); b = F.inkling2_qkv_message_gates(hs_, ms, gates, norms)
        err = max(((x.float() - y.float()).abs().max() / x.float().abs().max()).item() for x, y in zip(a, b))
        def t(fn):
            for _ in range(5): fn()
            r = []
            for _ in range(7):
                torch.mps.synchronize(); s0, e0 = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
                s0.record(); [fn() for _ in range(24)]; e0.record(); e0.synchronize(); r.append(s0.elapsed_time(e0) / 24 * 1e3)
            return statistics.median(r)
        print(f"{label}: rel err {err:.1e} | 3 gates (+norms) {t(lambda: ref(hs_, ms, norms)):.1f} us -> fused {t(lambda: F.inkling2_qkv_message_gates(hs_, ms, gates, norms)):.1f} us", flush=True)
