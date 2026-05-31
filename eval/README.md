# Standalone Math Evaluation

`eval.py` evaluates a local or Hugging Face checkpoint through an auto-launched SGLang
server. It is independent from AReaL training code, but reuses the repository virtual
environment.

Example:

```bash
uv run python eval/eval.py Qwen/Qwen2.5-Math-7B \
  -ds EleutherAI/hendrycks_math HuggingFaceH4/MATH-500 \
  -N 8 \
  --parallel 64
```

By default, `-N` samples are split into one completion per HTTP request
(`--samples-per-request 1`). This is more stable for SGLang under heavy load than
asking one Chat Completions request to return many long generations. You can raise
`--samples-per-request` after confirming the server is stable.

Outputs are written under `eval/runs/<timestamp>/`:

- `metrics.json`: settings, pass@k, average response tokens, sequence entropy, and
  correct/wrong conditional entropies.
- `samples.jsonl`: one JSON record per prompt, including raw dataset row, all sampled
  responses, correctness, and per-prompt metrics.

`sequence_entropy` is estimated from returned token logprobs as mean sequence negative
log-likelihood in nats. `sequence_diversity_entropy` is the older empirical entropy of
normalized response strings; it often saturates at `log(N)` when every sampled response
is unique, so it should only be used as a diversity diagnostic.

Conditional entropy is computed per prompt as
`H(Y|x,C) = -(1/P(C|x)) * E[1_C log p(Y|x)] + log P(C|x)`, where `C` is
correctness or incorrectness and `P(C|x)` is estimated by the sample fraction for that
prompt. The JSON output includes the uncorrected indicator NLL mean, the
`1/P(C|x)`-corrected NLL term, `log_normalizer`, and estimated condition probability
used for each conditional entropy value. Conditions with one sampled response are
still estimated; only conditions with zero sampled responses contribute `0`.

The script uses `temperature=1.0` and does not set top-p, top-k, min-p, penalties, or
other sampling modifiers.
