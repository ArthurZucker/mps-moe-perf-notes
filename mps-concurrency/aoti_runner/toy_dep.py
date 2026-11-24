import torch, os
from torch import nn
class M(nn.Module):
    def forward(self, x, y):
        a = torch.sin(x)
        a = torch.cumsum(a, dim=0)
        a = torch.cos(a)
        b = torch.sin(y)
        b = torch.cumsum(b, dim=0)
        b = torch.cos(b)
        return a, b
x=torch.randn(1<<16, device='mps'); y=torch.randn(1<<16, device='mps')
ep=torch.export.export(M(),(x,y))
print('exported')
p=torch._inductor.aoti_compile_and_package(ep, package_path='aoti-mps/toy_dep.pt2', inductor_configs={'cpp_wrapper': True})
print('compiled',p)
m=torch._inductor.aoti_load_package(p)
with torch.inference_mode(): a,b=m(x,y)
torch.mps.synchronize(); print('ran',a.shape,b.shape)
