import torch, time, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
path = snapshot_download("ArthurZ/Qwen3.8-Flash-Next-metal-experts-4bit", allow_patterns=["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt"])
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda", dtype=torch.bfloat16)
tc = model.config.get_text_config()
print("num_mtp_layers:", getattr(tc, "num_mtp_layers", None), "mtp_num_hidden_layers:", getattr(tc, "mtp_num_hidden_layers", None))
ids = tok.apply_chat_template([{"role": "user", "content": "Explain the causes of the French Revolution to a high school student."}], add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).input_ids.cuda()
N = 96
with torch.inference_mode():
    model.generate(ids, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize(); t = time.perf_counter()
    ref = model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False); torch.cuda.synchronize(); a = time.perf_counter() - t
    print(f"plain eager: {N / a:.1f} tok/s")
    for k in (1, 2, 3):
        try:
            torch.cuda.synchronize(); t = time.perf_counter()
            out = model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, use_mtp=True, num_assistant_tokens=k, return_dict_in_generate=True)
            torch.cuda.synchronize(); b = time.perf_counter() - t
            seq = out.sequences
            n = min(seq.shape[1], ref.shape[1]); same = (seq[0, :n] == ref[0, :n]).all().item()
            print(f"mtp k={k}: {N / b:.1f} tok/s, identical to plain greedy: {same}")
        except Exception as e:
            import traceback; traceback.print_exc(); break
