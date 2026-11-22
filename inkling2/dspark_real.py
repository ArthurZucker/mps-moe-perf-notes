"""Inkling-2-Mini with its bundled DSpark drafter (`mtp.safetensors`) on MPS: greedy exactness against plain greedy
decoding, mean accepted draft tokens per verify step, and decode tok/s with and without speculation.
    PYTHONPATH=src:benchmark_v2/inkling2_metal python benchmark_v2/inkling2_metal/dspark_real.py [new_tokens]
        [--prompts N] [--compile] [--draft-tokens K] [--profile]
`--compile` runs both paths with a static cache and torch.compile (see speed_real.py); `--draft-tokens` verifies only
the first K drafted tokens of each block."""

import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    CompileConfig,
    Inkling2DSparkConfig,
    Inkling2DSparkModel,
)
from transformers.generation.candidate_generator import DSparkTokenCandidateGenerator


REPO = "thinkingmachines/Inkling-2-Mini"
PROMPTS = [
    "Write a short story about a lighthouse keeper.",
    "Explain how a hash map works, with a small Python example.",
    "What is 17 * 23? Show the steps.",
    "Write a Python function that checks whether a string is a palindrome.",
]
new_tokens = int(next((a for a in sys.argv[1:] if a.isdigit()), 128))
num_prompts = int(sys.argv[sys.argv.index("--prompts") + 1]) if "--prompts" in sys.argv else 1
draft_tokens = int(sys.argv[sys.argv.index("--draft-tokens") + 1]) if "--draft-tokens" in sys.argv else None


def load_drafter(target):
    config = Inkling2DSparkConfig.from_target_config(target.config)
    with torch.device(target.device):
        drafter = Inkling2DSparkModel(config).to(torch.bfloat16)
    drafter.load_state_dict(load_file(hf_hub_download(REPO, "mtp.safetensors"), device=str(target.device)))
    return drafter.eval()


class Stamps:
    """Streamer recording the time each chunk of tokens is emitted (the first chunk is the prompt)."""

    def __init__(self):
        self.times, self.counts = [], []

    def put(self, tokens):
        self.times.append(time.perf_counter())
        self.counts.append(tokens.numel())

    def end(self):
        pass

    def median_step_ms(self):
        # One chunk per decode step (greedy) or per verify step (speculative)
        steps = sorted(b - a for a, b in zip(self.times[2:], self.times[3:]))
        return 1e3 * steps[len(steps) // 2]

    def decode_rate(self):
        # From the first generated token (end of prefill) to the last one
        generated = sum(self.counts[2:])
        return generated / (self.times[-1] - self.times[1])


accepted = []
update_strategy = DSparkTokenCandidateGenerator.update_candidate_strategy


def record_matches(self, input_ids, scores, num_matches):
    accepted.append(int(num_matches))
    return update_strategy(self, input_ids, scores, num_matches)


DSparkTokenCandidateGenerator.update_candidate_strategy = record_matches

draft_ms = []
get_candidates = DSparkTokenCandidateGenerator.get_candidates


def timed_get_candidates(self, *args, **kwargs):
    # Synchronized: `--profile` separates drafting from the verify forward in the step time
    torch.mps.synchronize()
    start = time.perf_counter()
    out = get_candidates(self, *args, **kwargs)
    torch.mps.synchronize()
    draft_ms.append(1e3 * (time.perf_counter() - start))
    return out


if "--profile" in sys.argv:
    DSparkTokenCandidateGenerator.get_candidates = timed_get_candidates


if __name__ == "__main__":
    target = AutoModelForImageTextToText.from_pretrained(REPO, device_map="mps", dtype=torch.bfloat16).eval()
    drafter = load_drafter(target)
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    greedy = {"do_sample": False, "max_new_tokens": new_tokens, "min_new_tokens": new_tokens}
    if "--compile" in sys.argv:
        import inductor_mps_fixes

        inductor_mps_fixes.apply()
        compile_config = CompileConfig()
        compile_config._compile_all_devices = True
        greedy |= {"cache_implementation": "static", "compile_config": compile_config}
    if draft_tokens is not None:
        greedy |= {"num_assistant_tokens": draft_tokens}
    speculative = greedy | {"assistant_model": drafter, "speculation_type": "dspark"}

    def prompt_ids(prompt):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return ids["input_ids"].to("mps")

    def timed_generate(ids, **kwargs):
        stamps = Stamps()
        out = target.generate(ids, streamer=stamps, **kwargs)
        return out[0, ids.shape[1] :], stamps.decode_rate(), stamps.median_step_ms()

    prompts = [prompt_ids(prompt) for prompt in PROMPTS[:num_prompts]]
    with torch.inference_mode():
        # Plain greedy decoding first: with `--compile`, `generate` compiles its own decode step
        for _ in range(2):
            target.generate(prompt_ids("Hi"), **(greedy | {"max_new_tokens": 16, "min_new_tokens": 16}))
        references = [timed_generate(ids, **greedy) for ids in prompts]

        for _ in range(2):
            target.generate(prompt_ids("Hi"), **(speculative | {"max_new_tokens": 32, "min_new_tokens": 32}))
        accepted.clear()

        rates = {"greedy": [], "dspark": []}
        for index, (ids, (new_reference, greedy_rate, greedy_step)) in enumerate(zip(prompts, references)):
            start = len(accepted)
            new_drafted, dspark_rate, dspark_step = timed_generate(ids, **speculative)
            rates["greedy"].append(greedy_rate)
            rates["dspark"].append(dspark_rate)
            steps = accepted[start:]

            length = min(len(new_reference), len(new_drafted))
            mismatch = (new_reference[:length] != new_drafted[:length]).nonzero()
            first = mismatch[0].item() if mismatch.numel() else None
            print(
                f"[{index + 1}/{num_prompts}] exact={first is None and len(new_reference) == len(new_drafted)} "
                f"(first mismatch {first}, {len(new_reference)} vs {len(new_drafted)} tokens) | "
                f"accepted/step {sum(steps) / len(steps):.2f} over {len(steps)} verify steps | "
                f"greedy {greedy_rate:.1f} tok/s ({greedy_step:.1f} ms/step), dspark {dspark_rate:.1f} tok/s "
                f"({dspark_step:.1f} ms/verify step incl. drafting)",
                flush=True,
            )
            if first is not None:
                print("  reference:", repr(tokenizer.decode(new_reference[max(0, first - 5) : first + 10])))
                print("  dspark:   ", repr(tokenizer.decode(new_drafted[max(0, first - 5) : first + 10])))

    if draft_ms:
        print(f"drafting: median {sorted(draft_ms)[len(draft_ms) // 2]:.1f} ms per step")
    histogram = [accepted.count(n) for n in range(drafter.config.block_size)]
    print(f"accepted-length histogram (0..{drafter.config.block_size - 1}): {histogram}")
    print(
        f"mean accepted/step {sum(accepted) / len(accepted):.2f} | greedy {sum(rates['greedy']) / len(rates['greedy']):.1f} "
        f"tok/s | dspark {sum(rates['dspark']) / len(rates['dspark']):.1f} tok/s"
    )
    print(repr(tokenizer.decode(new_drafted)[:300]))
