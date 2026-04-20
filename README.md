---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: Qwen/Qwen3.6-35B-A3B
datasets:
- lordx64/reasoning-distill-opus-4-7-max-sft
tags:
- text-generation
- reasoning
- distillation
- chain-of-thought
- qwen
- qwen3.6
- mixture-of-experts
- moe
- lora
- unsloth
model-index:
- name: Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled
  results: []
---

# Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled

A reasoning-distilled variant of **Qwen3.6-35B-A3B** taught to imitate the chain-of-thought style of **Claude Opus 4.7**, the frontier reasoning model from Anthropic. The goal: port Claude-grade reasoning behavior into a permissively-licensed Mixture-of-Experts model that an individual can actually run.

## Why this model

- **Claude-style reasoning, open weights.** Claude Opus 4.7 is one of the strongest reasoning models available, but only via a proprietary API. This model has been fine-tuned on ~8k high-quality reasoning traces produced by Opus 4.7, teaching the base to *think* before answering — with explicit `<think>…</think>` blocks — in Claude's structure and cadence.
- **Sparse activation, dense knowledge.** The base is a 35B-parameter MoE with **256 experts, 8 routed + 1 shared**, of which only about **3B parameters are active** per token. You get the capacity of a 35B model at the inference cost of a small dense model. Full-quality bf16 inference runs on a single 80GB A100 or H100.
- **Long thinking supported.** 64k token context. The model routinely emits 5–30k tokens of `<think>` reasoning on hard problems before giving the final answer — which is the whole point of reasoning models, and why this one was specifically trained end-to-end with an upstream teacher that also reasons explicitly.
- **Clean base to build on.** LoRA adapter is also published separately (`…-adapter`), so you can apply the distillation to other checkpoints of the same base, or stack further fine-tunes.

## Intended use

Built for hard reasoning: graduate-level STEM, competition math (AIME / MATH), code reasoning with explicit walk-through, multi-step logic puzzles, and agentic planning where explicit `<think>` helps correctness.

For short-turn conversational latency-sensitive workloads the thinking budget can be large; cap `max_new_tokens` or post-process to strip `<think>…</think>` blocks if you only want final answers in production.

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(
    repo, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)

messages = [{"role": "user", "content": "How many positive integers less than 1000 have digits that sum to 20?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=32768, do_sample=False)
print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Recommended backend: **vLLM** for serving — the MoE routing + KV cache benefit significantly from continuous batching.
```
vllm serve lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled \
  --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.9
```

### GGUF (LM Studio / llama.cpp)

Quantized GGUF weights are available for `llama.cpp` and LM Studio:

- [**IQ4_XS** (18.9 GB)](https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-IQ4_XS-GGUF) — fits in ~24 GB RAM/VRAM, default pick for LM Studio

Search `lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled` inside LM Studio's model browser once HF has indexed the GGUF repo (usually within an hour of publication). More quant levels (`Q4_K_M`, `Q5_K_M`, `Q8_0`) can be added on request.

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3.6-35B-A3B` (loaded via `unsloth/Qwen3.6-35B-A3B` for faster finetuning) |
| Teacher | Claude Opus 4.7 (Anthropic) |
| Training dataset | [`lordx64/reasoning-distill-opus-4-7-max-sft`](https://huggingface.co/datasets/lordx64/reasoning-distill-opus-4-7-max-sft) — reasoning traces from Claude Opus 4.7 reformatted into SFT conversations |
| Source dataset | [`lordx64/reasoning-distill-claude-opus-4-7-max`](https://huggingface.co/datasets/lordx64/reasoning-distill-claude-opus-4-7-max) — raw teacher traces (pre-SFT formatting) |
| Dataset size | ~7,800 full conversations, assistant side trained including `<think>…</think>` |
| Method | SFT with Unsloth + TRL `SFTTrainer` + `train_on_responses_only` (loss only on assistant tokens) |
| LoRA config | `r=16, alpha=16, dropout=0.0, targets=["q_proj","k_proj","v_proj","o_proj"]` (attention-only) |
| Hyperparameters | `lr=2e-5`, cosine schedule, `warmup_ratio=0.03`, `weight_decay=0.01`, optimizer `adamw_8bit` |
| Batch | `per_device=1, grad_accum=16, effective=16`, 2 epochs = 978 steps |
| Sequence | 4096 tokens during training (64k usable at inference — base supports it natively) |
| Precision | bf16 on 1× H200 141GB (HF Inference Endpoint, custom container) |
| Trainable | 3.44M params out of 35.1B (0.01%) |

### Why attention-only LoRA on a MoE

The initial plan was full LoRA including the MoE expert FFNs (`gate_proj/up_proj/down_proj`). In the course of this project I filed and upstreamed a shape-mismatch fix to unsloth-zoo's MoE+LoRA grouped-mm path — [unslothai/unsloth-zoo#601](https://github.com/unslothai/unsloth-zoo/pull/601) — without which the expert-LoRA forward crashes on Qwen3.6's 256-expert layout. Even with that fix, single-GPU memory made expert-LoRA impractical for this run. Attention-only captures most of the signal on *style* distillation anyway (the point of this model) while leaving the expert FFNs' learned knowledge intact — a v2 training run with expert LoRA on multi-GPU is a natural next step if the style-only signal isn't enough.

## Evaluation

Evaluated via `lm-evaluation-harness` (v0.4.9) with vLLM backend at 64k context, bf16. Custom eval path strips `<think>…</think>` from generations before the filter pipeline, uses per-task conventional fewshot counts, and runs with `fewshot_as_multiturn=True` so few-shot examples are proper chat turns rather than concatenated prompt text. Raw results JSON is public: [lordx64/qwen3-6-distill-evals](https://huggingface.co/datasets/lordx64/qwen3-6-distill-evals).

| Benchmark | Setup | Score |
|---|---|---|
| **GSM8K CoT** | 8-shot multiturn, limit 300 | **84.3%** (flexible-extract) / 76.7% (strict-match) |
| **MMLU-Pro** | 5-shot multiturn, limit 500 | **74.9%** |
| AIME 2024 | 0-shot, full (30) | _extraction fix in progress — model generates answers but not in a format the AIME extractor recognizes (`\boxed{}` vs plain prose)_ |
| AIME 2025 | 0-shot, full (30) | _same — pending_ |
| GPQA Diamond | 0-shot CoT, full (198) | _same — pending_ |
| MATH-500 | 0-shot, limit 100 | _rerun pending (missing `sympy` / `math_verify` dep in the first run)_ |

### MMLU-Pro subject breakdown

Standard reasoning-model profile: strong on STEM, weaker on law/engineering. All subjects evaluated at limit 500, 5-shot multiturn.

| Subject | Acc | Subject | Acc |
|---|---:|---|---:|
| Biology | 86.0% | Chemistry | 78.8% |
| Psychology | 83.4% | Health | 73.8% |
| Math | 83.6% | Business | 74.4% |
| Economics | 83.0% | Other | 72.6% |
| Physics | 81.0% | Philosophy | 71.3% |
| Computer Science | 79.0% | History | 70.9% |
|  |  | **Engineering** | **54.8%** |
|  |  | **Law** | **55.6%** |

Full per-task JSON with stderr, filter configs, and timings lives in the [evals dataset](https://huggingface.co/datasets/lordx64/qwen3-6-distill-evals/tree/main/reasoning/lordx64__Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled). The remaining tasks will be added to this table after a diagnostic rerun identifies why AIME/GPQA extraction is returning no-match on generated outputs.

## Limitations

- **Reasoning ≠ knowledge.** Distillation transfers *how to reason*, not new facts. Anything the base Qwen3.6-35B-A3B doesn't already know, this model still doesn't know.
- **Attention-only LoRA.** Expert FFNs are untouched from the base — domains where Claude and Qwen3.6 diverge in factual priors may see uneven improvement.
- **Long generations.** The model will genuinely use tens of thousands of tokens on hard problems. Budget your `max_new_tokens` accordingly, and provide `max_model_len ≥ 32k` at inference.
- **Distillation provenance.** Training data was generated with Anthropic's Claude Opus 4.7 via API. Downstream users should confirm compliance with Anthropic's [usage policies](https://www.anthropic.com/legal/usage-policy) for their specific use case.

## Citation

If you use this model, please cite the base and the distillation:

```bibtex
@misc{qwen36_a3b_2026,
  title  = {Qwen3.6-35B-A3B},
  author = {Qwen Team},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/Qwen/Qwen3.6-35B-A3B}},
}

@misc{lordx64_qwen36_distill_2026,
  title  = {Qwen3.6-35B-A3B distilled from Claude Opus 4.7 reasoning},
  author = {lordx64},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/lordx64/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled}},
}
```

## Acknowledgements

- **Unsloth** — 2× faster training of large MoE LoRA; the bug we hit and fixed was in their `unsloth-zoo` patches (credit for rapid review of PR #601).
- **Anthropic** — for the teacher model.
- **Qwen team** — for releasing Qwen3.6 with a permissive Apache-2.0 license, enabling work like this.
- **lm-evaluation-harness (EleutherAI)** — evaluation methodology.
