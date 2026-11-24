import os, torch
from kernels import get_local_kernel
from pathlib import Path
k = get_local_kernel(Path(os.environ["TRANSFORMERS_MOE_AFFINE_KERNEL"]))
E,H,I,TOPK,G=512,2560,640,10,64
wgu=torch.randint(-2**31,2**31-1,(E,2*I,H//8),dtype=torch.int32,device="cuda");sgu=torch.randn(E,2*I,H//G,device="cuda").bfloat16();bgu=sgu.clone()
wd=torch.randint(-2**31,2**31-1,(E,H,I//8),dtype=torch.int32,device="cuda");sd=torch.randn(E,H,I//G,device="cuda").bfloat16();bd=sd.clone()
x=torch.randn(1,H,device="cuda").bfloat16();a=torch.randn(TOPK,I,device="cuda").bfloat16();w=torch.rand(TOPK,device="cuda")
for i in range(5):
    ids=torch.randperm(E,device="cuda")[:TOPK].int()
    k.gather_qmv4(x,wgu,sgu,bgu,ids,TOPK,G); k.down_route4(a,wd,sd,bd,ids,w,TOPK,G)
torch.cuda.synchronize()
