import torch, time
from torch import nn
class M(nn.Module):
    def forward(self, xs):
        outs=[]
        for i,x in enumerate(xs):
            y=torch.sin(x)
            for _ in range(3): y=y*1.01 + (0.1*i)
            outs.append(y)
        return tuple(outs)
xs=[torch.randn(1<<16, device='mps') for _ in range(8)]
with torch.inference_mode(): refs=M()(xs)
def bench(m):
    for _ in range(3):
        with torch.inference_mode(): o=m(xs)
        torch.mps.synchronize()
    ts=[]
    for _ in range(20):
        t=time.perf_counter()
        with torch.inference_mode(): o=m(xs)
        torch.mps.synchronize(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]*1e3
for p in ['toy_many.pt2','toy_many_concurrent.pt2']:
    m=torch._inductor.aoti_load_package('/Users/transformers/qwen38-metal/aoti-mps/'+p)
    with torch.inference_mode(): o=m(xs)
    torch.mps.synchronize()
    ok=all(torch.equal(a,b) for a,b in zip(o,refs))
    print(p,'exact',ok,'maxdiff',max((a-b).abs().max().item() for a,b in zip(o,refs)),'ms',bench(m))
