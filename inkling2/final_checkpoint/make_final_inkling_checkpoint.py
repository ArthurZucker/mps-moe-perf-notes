import json, os, shutil, time, torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, MetalConfig
from transformers.quantizers.quantizer_metal import MetalHfQuantizer
# do not fuse siblings while saving: the checkpoint must keep the loader's original keys
MetalHfQuantizer._process_model_after_weight_loading = lambda self, model, **kwargs: model
metal=MetalConfig(bits=4,group_size=64)
plan={"model.visual.*":None,"*_message_gate":None,"lm_head":metal,"*":metal}
t=time.time()
m=AutoModelForImageTextToText.from_pretrained('thinkingmachines/Inkling-2-Mini',device_map='mps',dtype=torch.bfloat16,quantization_config=plan)
print('loaded',time.time()-t,flush=True)
out='/Users/transformers/qwen38-metal/inkling-final2'
os.makedirs(out,exist_ok=True)
# start from resolved expert plan, add dense Metal plan
cfg=m.config.to_dict()
expert_plan=json.loads(json.dumps(cfg['quantization_config'], default=str)) if isinstance(cfg['quantization_config'],dict) else cfg['quantization_config']
# ensure per-expert entries remain; add None/Metal patterns and '*' last
plan_out=dict(expert_plan)
plan_out['model.visual.*']=None
plan_out['*_message_gate']=None
plan_out['lm_head']={'quant_method':'metal','bits':4,'group_size':64,'modules_to_not_convert':None,'dequantize':False}
plan_out['*']={'quant_method':'metal','bits':4,'group_size':64,'modules_to_not_convert':None,'dequantize':False}
cfg['quantization_config']=plan_out
json.dump(cfg,open(os.path.join(out,'config.json'),'w'),indent=2)
src='/Users/transformers/qwen38-metal/inkling'
for name in os.listdir(src):
    if name.endswith('.safetensors') or name in ('config.json','model.safetensors.index.json'):
        continue
    p=os.path.join(src,name)
    if os.path.isfile(p): shutil.copy2(p,os.path.join(out,name))
sd=m.state_dict()
cpu={k:(v.detach().cpu().contiguous() if isinstance(v,torch.Tensor) else v) for k,v in sd.items()}
del sd; torch.mps.empty_cache()
save_file(cpu,os.path.join(out,'model.safetensors'),metadata={'format':'pt'})
print('saved',out,os.path.getsize(os.path.join(out,'model.safetensors'))/2**30,time.time()-t,flush=True)
