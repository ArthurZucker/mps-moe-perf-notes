import torch
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
ep=torch.export.export(M(), (xs,))
print('exported')
p=torch._inductor.aoti_compile_and_package(ep, package_path='aoti-mps/toy_many.pt2', inductor_configs={'cpp_wrapper': True})
print('compiled',p)
m=torch._inductor.aoti_load_package(p)
with torch.inference_mode(): outs=m(xs)
torch.mps.synchronize(); print('ran',len(outs),outs[0].shape)
