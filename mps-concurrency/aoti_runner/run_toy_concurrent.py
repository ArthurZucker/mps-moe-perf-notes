import torch, time
from torch import nn
class M(nn.Module):
    def forward(self, x, y):
        a=torch.sin(x); b=torch.cos(y)
        for _ in range(3):
            a=a+0.5; b=b-0.25
        return a,b
x=torch.randn(1<<20, device='mps'); y=torch.randn(1<<20, device='mps')
with torch.inference_mode(): ra,rb=M()(x,y)
torch.mps.synchronize()
m=torch._inductor.aoti_load_package('/Users/transformers/qwen38-metal/aoti-mps/toy_concurrent.pt2')
with torch.inference_mode(): a,b=m(x,y)
torch.mps.synchronize()
print('exact', torch.equal(a,ra), torch.equal(b,rb), (a-ra).abs().max().item(), (b-rb).abs().max().item())
for _ in range(3):
    with torch.inference_mode(): a,b=m(x,y)
    torch.mps.synchronize()
ts=[]
for _ in range(10):
    t=time.perf_counter()
    with torch.inference_mode(): a,b=m(x,y)
    torch.mps.synchronize(); ts.append(time.perf_counter()-t)
print('concurrent ms', sorted(ts)[len(ts)//2]*1e3)
