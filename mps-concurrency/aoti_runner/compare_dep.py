import torch, time
from torch import nn
class M(nn.Module):
    def forward(self, x, y):
        a=torch.sin(x); a=torch.cumsum(a,dim=0); a=torch.cos(a)
        b=torch.sin(y); b=torch.cumsum(b,dim=0); b=torch.cos(b)
        return a,b
x=torch.randn(1<<16, device='mps'); y=torch.randn(1<<16, device='mps')
with torch.inference_mode(): ra,rb=M()(x,y)
def bench(m):
    for _ in range(3):
        with torch.inference_mode(): a,b=m(x,y)
        torch.mps.synchronize()
    ts=[]
    for _ in range(20):
        t=time.perf_counter()
        with torch.inference_mode(): a,b=m(x,y)
        torch.mps.synchronize(); ts.append(time.perf_counter()-t)
    return sorted(ts)[len(ts)//2]*1e3
for p in ['toy_dep.pt2','toy_dep_concurrent.pt2']:
    m=torch._inductor.aoti_load_package('/Users/transformers/qwen38-metal/aoti-mps/'+p)
    with torch.inference_mode(): a,b=m(x,y)
    torch.mps.synchronize()
    print(p, 'exact', torch.equal(a,ra), torch.equal(b,rb), 'maxdiff', (a-ra).abs().max().item(), (b-rb).abs().max().item(), 'ms', bench(m))
