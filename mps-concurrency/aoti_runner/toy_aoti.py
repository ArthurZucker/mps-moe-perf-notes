import os, torch
from torch import nn

class M(nn.Module):
    def forward(self, x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        for _ in range(3):
            a = a + 0.5
            b = b - 0.25
        return a, b

x = torch.randn(1<<20, device='mps'); y = torch.randn(1<<20, device='mps')
ep = torch.export.export(M(), (x, y))
print('exported')
p = torch._inductor.aoti_compile_and_package(ep, package_path='aoti-mps/toy.pt2', inductor_configs={'cpp_wrapper': True})
print('compiled', p)
m = torch._inductor.aoti_load_package(p)
with torch.inference_mode(): a,b = m(x,y)
torch.mps.synchronize()
print('ran', a.shape, b.shape, a.sum().item(), b.sum().item())
