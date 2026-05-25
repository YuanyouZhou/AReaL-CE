# Standalone Math Evaluation

`math_sglang_eval.py` evaluates a local or Hugging Face checkpoint through an
auto-launched SGLang server. It is independent from AReaL training code, but reuses
the repository virtual environment.

Example:

```bash
uv run python eval/math_sglang_eval.py Qwen/Qwen2.5-Math-7B \
  -ds EleutherAI/hendrycks_math HuggingFaceH4/MATH-500 \
  -N 8 \
  --parallel 256
```

Outputs are written under `eval/runs/<timestamp>/`:

- `metrics.json`: settings, pass@k, average response tokens, sequence entropy, and
  correct/wrong conditional entropies.
- `samples.jsonl`: one JSON record per prompt, including raw dataset row, all sampled
  responses, correctness, and per-prompt metrics.

The script uses `temperature=1.0` and does not set top-p, top-k, min-p, penalties, or
other sampling modifiers.
