#!/usr/bin/env python3
"""Standalone math evaluation with a local SGLang server.

This script intentionally does not import AReaL training modules. It reuses the
repository environment dependencies only: datasets, math-verify, openai,
sglang, torch, tqdm, and transformers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from datasets import concatenate_datasets, load_dataset
from math_verify import parse, verify
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

DatasetName = Literal[
    "EleutherAI/hendrycks_math",
    "HuggingFaceH4/MATH-500",
    "MathArena/aime_2025",
    "MathArena/aime_2026",
]

SUPPORTED_DATASETS: tuple[DatasetName, ...] = (
    "EleutherAI/hendrycks_math",
    "HuggingFaceH4/MATH-500",
    "MathArena/aime_2025",
    "MathArena/aime_2026",
)

HENDRYCKS_MATH_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

BOXED_INSTRUCTION = "\nPlease put your final answer within \\boxed{}."


@dataclass(frozen=True)
class EvalSample:
    dataset: str
    sample_id: str
    prompt: str
    answer: str
    raw: dict[str, Any]


@dataclass
class Generation:
    text: str
    token_count: int
    sequence_logprob: float | None
    sequence_nll: float | None
    correct: bool


@dataclass
class ConditionalEntropy:
    entropy: float
    sample_count: int
    condition_probability: float
    mean_nll: float
    unconditional_indicator_nll_mean: float
    condition_corrected_mean_nll: float
    inverse_condition_probability: float
    log_normalizer: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate math checkpoints through a standalone SGLang server."
    )
    parser.add_argument(
        "checkpoint",
        help="Local path or Hugging Face model id passed to SGLang --model-path.",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
        help="One or more datasets to evaluate.",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=1,
        help="Number of parallel samples per prompt. Must be a positive integer.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 selects a free port.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--parallel",
        type=int,
        default=64,
        help="Maximum number of concurrent prompt requests sent to SGLang.",
    )
    parser.add_argument(
        "--samples-per-request",
        type=int,
        default=1,
        help=(
            "How many completions to request with one OpenAI call. Keep the default "
            "1 for SGLang stability; increase only after checking server capacity."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument(
        "--logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request token logprobs and compute model sequence entropy from them.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "runs"))
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--mem-fraction-static", type=float, default=0.9)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to SGLang and tokenizer loading.",
    )
    args = parser.parse_args()
    if args.N <= 0:
        parser.error("-N must be a positive integer.")
    if args.parallel <= 0:
        parser.error("--parallel must be a positive integer.")
    if args.samples_per_request <= 0:
        parser.error("--samples-per-request must be a positive integer.")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive.")
    if args.request_retries < 0:
        parser.error("--request-retries must be non-negative.")
    if args.retry_base_delay <= 0:
        parser.error("--retry-base-delay must be positive.")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be a positive integer.")
    return args


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def detect_gpu_count() -> int:
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return count
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        count = len([line for line in result.stdout.splitlines() if line.strip()])
        if count > 0:
            return count
    except Exception:
        pass

    raise RuntimeError("No CUDA GPU was detected for SGLang deployment.")


def launch_sglang_server(args: argparse.Namespace, gpu_count: int, port: int):
    log_path = (
        Path(args.server_log)
        if args.server_log is not None
        else Path(args.output_dir) / "sglang_server.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.checkpoint,
        "--host",
        args.host,
        "--port",
        str(port),
        "--tp-size",
        str(gpu_count),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--random-seed",
        str(args.seed),
        "--log-level",
        "warning",
        "--log-level-http",
        "warning",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc, log_file, log_path, cmd


async def wait_for_server(base_url: str, proc: subprocess.Popen, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=10) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"SGLang server exited early with code {proc.returncode}.")
            try:
                response = await client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready within {timeout} seconds.")


def stop_server(proc: subprocess.Popen, log_file) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
    log_file.close()


def load_eval_samples(dataset_name: str) -> list[EvalSample]:
    if dataset_name == "EleutherAI/hendrycks_math":
        pieces = [
            load_dataset(dataset_name, name=subset, split="test")
            for subset in HENDRYCKS_MATH_SUBSETS
        ]
        dataset = concatenate_datasets(pieces)
        return [
            EvalSample(
                dataset=dataset_name,
                sample_id=f"{dataset_name}:{idx}",
                prompt=str(row["problem"]) + BOXED_INSTRUCTION,
                answer=str(row["solution"]),
                raw=dict(row),
            )
            for idx, row in enumerate(dataset)
        ]

    split = "test" if dataset_name == "HuggingFaceH4/MATH-500" else "train"
    dataset = load_dataset(dataset_name, split=split)
    return [
        EvalSample(
            dataset=dataset_name,
            sample_id=f"{dataset_name}:{idx}",
            prompt=extract_prompt(row) + BOXED_INSTRUCTION,
            answer=extract_gold_answer(row),
            raw=dict(row),
        )
        for idx, row in enumerate(dataset)
    ]


def extract_prompt(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "prompt", "query"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"Cannot find prompt field in dataset row keys: {sorted(row)}")


def extract_gold_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "solution", "gt_answer", "ground_truth"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError(f"Cannot find answer field in dataset row keys: {sorted(row)}")


def math_verify_correct(response: str, answer: str) -> bool:
    try:
        return bool(verify(parse(response), parse(answer)))
    except Exception:
        return False


def exact_answer_correct(response: str, answer: str) -> bool:
    return normalize_answer(extract_final_answer(response)) == normalize_answer(answer)


def extract_final_answer(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxed:
        return boxed[-1]

    patterns = (
        r"final answer\s*(?:is|:)?\s*([^\n]+)",
        r"answer\s*(?:is|:|=)?\s*([^\n]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else text.strip()


def normalize_answer(value: str) -> str:
    value = str(value).strip()
    value = extract_final_answer(value) if "\\boxed" in value else value
    value = value.replace("$", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value.lower()


def empirical_entropy(values: list[str]) -> float:
    if len(values) < 2:
        return 0.0
    total = len(values)
    counts = Counter(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def pass_at_k(correct: list[bool], max_k: int) -> dict[str, float]:
    n = len(correct)
    c = sum(correct)
    metrics: dict[str, float] = {}
    for k in range(1, max_k + 1):
        if c == 0:
            metrics[f"pass@{k}"] = 0.0
        elif n - c < k:
            metrics[f"pass@{k}"] = 1.0
        else:
            metrics[f"pass@{k}"] = 1.0 - math.comb(n - c, k) / math.comb(n, k)
    return metrics


def dataset_judge(dataset_name: str):
    if dataset_name in ("EleutherAI/hendrycks_math", "HuggingFaceH4/MATH-500"):
        return math_verify_correct
    return exact_answer_correct


def token_count(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return 0


async def generate_one_prompt(
    client: AsyncOpenAI,
    sample: EvalSample,
    n_samples: int,
    max_tokens: int,
    args: argparse.Namespace,
    server_proc: subprocess.Popen,
    log_path: Path,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    remaining = n_samples
    while remaining > 0:
        chunk_size = min(remaining, args.samples_per_request)
        outputs.extend(
            await generate_completion_chunk(
                client=client,
                sample=sample,
                n_samples=chunk_size,
                max_tokens=max_tokens,
                args=args,
                server_proc=server_proc,
                log_path=log_path,
            )
        )
        remaining -= chunk_size
    return outputs


async def generate_completion_chunk(
    client: AsyncOpenAI,
    sample: EvalSample,
    n_samples: int,
    max_tokens: int,
    args: argparse.Namespace,
    server_proc: subprocess.Popen,
    log_path: Path,
) -> list[dict[str, Any]]:
    for attempt in range(args.request_retries + 1):
        if server_proc.poll() is not None:
            raise RuntimeError(
                "SGLang server exited while evaluating "
                f"{sample.sample_id} with code {server_proc.returncode}. "
                f"Check server log: {log_path}"
            )
        try:
            response = await client.chat.completions.create(
                model="default",
                messages=[{"role": "user", "content": sample.prompt}],
                temperature=1.0,
                n=n_samples,
                max_tokens=max_tokens,
                logprobs=args.logprobs,
                timeout=args.request_timeout,
            )
            outputs: list[dict[str, Any]] = []
            for choice in response.choices:
                content = choice.message.content
                text = content if content is not None else ""
                sequence_logprob = extract_sequence_logprob(choice)
                outputs.append(
                    {
                        "text": text,
                        "sequence_logprob": sequence_logprob,
                        "sequence_nll": (
                            -sequence_logprob
                            if sequence_logprob is not None
                            else None
                        ),
                    }
                )
            if len(outputs) != n_samples:
                raise RuntimeError(
                    f"Expected {n_samples} generations for {sample.sample_id}, "
                    f"got {len(outputs)}."
                )
            return outputs
        except (APIConnectionError, APITimeoutError, APIStatusError, httpx.HTTPError):
            if attempt >= args.request_retries:
                raise
            delay = args.retry_base_delay * (2**attempt)
            await asyncio.sleep(delay)
    raise RuntimeError(f"Failed to generate completions for {sample.sample_id}.")


def extract_sequence_logprob(choice: Any) -> float | None:
    """Return sum of output token logprobs from an OpenAI-compatible choice."""
    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return None

    token_logprobs: list[float] = []
    content = getattr(logprobs, "content", None)
    if content is not None:
        for item in content:
            logprob = getattr(item, "logprob", None)
            if logprob is not None:
                token_logprobs.append(float(logprob))

    if not token_logprobs and hasattr(logprobs, "model_dump"):
        token_logprobs.extend(extract_logprobs_from_dump(logprobs.model_dump()))

    if not token_logprobs:
        return None
    return float(sum(token_logprobs))


def extract_logprobs_from_dump(logprobs: dict[str, Any]) -> list[float]:
    content = logprobs.get("content")
    if isinstance(content, list):
        return [
            float(item["logprob"])
            for item in content
            if isinstance(item, dict) and isinstance(item.get("logprob"), int | float)
        ]

    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(token_logprobs, list):
        return [
            float(logprob)
            for logprob in token_logprobs
            if isinstance(logprob, int | float)
        ]

    return []


async def evaluate_dataset(
    client: AsyncOpenAI,
    dataset_name: str,
    samples: list[EvalSample],
    args: argparse.Namespace,
    tokenizer,
    server_proc: subprocess.Popen,
    log_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(args.parallel)
    judge = dataset_judge(dataset_name)

    async def evaluate_sample(sample: EvalSample) -> dict[str, Any]:
        async with semaphore:
            outputs = await generate_one_prompt(
                client=client,
                sample=sample,
                n_samples=args.N,
                max_tokens=args.max_tokens,
                args=args,
                server_proc=server_proc,
                log_path=log_path,
            )
        generations = [
            Generation(
                text=output["text"],
                token_count=token_count(tokenizer, output["text"]),
                sequence_logprob=output["sequence_logprob"],
                sequence_nll=output["sequence_nll"],
                correct=judge(output["text"], sample.answer),
            )
            for output in outputs
        ]
        correctness = [generation.correct for generation in generations]
        prompt_entropy = sequence_entropy(generations)
        correct_entropy = conditional_sequence_entropy(generations, condition=True)
        wrong_entropy = conditional_sequence_entropy(generations, condition=False)
        return {
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "answer": sample.answer,
            "raw": sample.raw,
            "generations": [asdict(generation) for generation in generations],
            "pass_at_k": pass_at_k(correctness, args.N),
            "sequence_entropy": prompt_entropy,
            "sequence_diversity_entropy": empirical_entropy(
                [normalize_sequence(generation.text) for generation in generations]
            ),
            "correct_conditional_entropy": correct_entropy.entropy,
            "wrong_conditional_entropy": wrong_entropy.entropy,
            "correct_conditional_entropy_detail": asdict(correct_entropy),
            "wrong_conditional_entropy_detail": asdict(wrong_entropy),
        }

    tasks = [evaluate_sample(sample) for sample in samples]
    records = await tqdm.gather(*tasks, desc=dataset_name)
    metrics = aggregate_metrics(dataset_name, records, args.N)
    return metrics, records


def normalize_sequence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sequence_entropy(generations: list[Generation]) -> float:
    nlls = [
        generation.sequence_nll
        for generation in generations
        if generation.sequence_nll is not None
    ]
    if len(nlls) != len(generations):
        raise RuntimeError(
            "SGLang did not return token logprobs for every generation, so model "
            "sequence entropy cannot be computed. Keep --logprobs enabled and "
            "check whether this SGLang/OpenAI endpoint supports chat logprobs."
        )
    return mean([float(nll) for nll in nlls])


def conditional_sequence_entropy(
    generations: list[Generation],
    condition: bool,
) -> ConditionalEntropy:
    selected = [
        generation
        for generation in generations
        if generation.correct is condition and generation.sequence_nll is not None
    ]
    if len(selected) < 2:
        return ConditionalEntropy(
            entropy=0.0,
            sample_count=len(selected),
            condition_probability=len(selected) / len(generations),
            mean_nll=0.0,
            unconditional_indicator_nll_mean=0.0,
            condition_corrected_mean_nll=0.0,
            inverse_condition_probability=0.0,
            log_normalizer=0.0,
        )

    condition_prob = len(selected) / len(generations)
    mean_nll = mean([float(generation.sequence_nll) for generation in selected])
    unconditional_indicator_nll_mean = (
        sum(float(generation.sequence_nll) for generation in selected)
        / len(generations)
    )
    inverse_condition_probability = 1.0 / condition_prob
    condition_corrected_mean_nll = (
        unconditional_indicator_nll_mean * inverse_condition_probability
    )
    # H(Y|x,C) = -sum_{y in C} p(y|x,C) log p(y|x,C)
    # = -(1/P(C|x)) sum_{y in C} p(y|x) log p(y|x) + log P(C|x).
    # The first term is estimated from all samples with the explicit
    # 1/P(C|x) condition-probability correction.
    log_normalizer = math.log(condition_prob)
    return ConditionalEntropy(
        entropy=condition_corrected_mean_nll + log_normalizer,
        sample_count=len(selected),
        condition_probability=condition_prob,
        mean_nll=mean_nll,
        unconditional_indicator_nll_mean=unconditional_indicator_nll_mean,
        condition_corrected_mean_nll=condition_corrected_mean_nll,
        inverse_condition_probability=inverse_condition_probability,
        log_normalizer=log_normalizer,
    )


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate_metrics(
    dataset_name: str,
    records: list[dict[str, Any]],
    n_samples: int,
) -> dict[str, Any]:
    all_generations = [
        generation
        for record in records
        for generation in record["generations"]
    ]
    token_counts = [float(generation["token_count"]) for generation in all_generations]
    pass_metrics = {
        f"pass@{k}": mean([record["pass_at_k"][f"pass@{k}"] for record in records])
        for k in range(1, n_samples + 1)
    }
    correct_entropy_details = [
        record["correct_conditional_entropy_detail"] for record in records
    ]
    wrong_entropy_details = [
        record["wrong_conditional_entropy_detail"] for record in records
    ]
    return {
        "dataset": dataset_name,
        "num_prompts": len(records),
        "num_generations": len(all_generations),
        "accuracy": pass_metrics["pass@1"],
        "pass_at_k": pass_metrics,
        "avg_response_tokens": mean(token_counts),
        "sequence_entropy": mean(
            [float(record["sequence_entropy"]) for record in records]
        ),
        "sequence_diversity_entropy": mean(
            [float(record["sequence_diversity_entropy"]) for record in records]
        ),
        "correct_conditional_entropy": mean(
            [float(record["correct_conditional_entropy"]) for record in records]
        ),
        "wrong_conditional_entropy": mean(
            [float(record["wrong_conditional_entropy"]) for record in records]
        ),
        "correct_conditional_mean_nll": mean(
            [float(detail["mean_nll"]) for detail in correct_entropy_details]
        ),
        "wrong_conditional_mean_nll": mean(
            [float(detail["mean_nll"]) for detail in wrong_entropy_details]
        ),
        "correct_conditional_unconditional_indicator_nll_mean": mean(
            [
                float(detail["unconditional_indicator_nll_mean"])
                for detail in correct_entropy_details
            ]
        ),
        "wrong_conditional_unconditional_indicator_nll_mean": mean(
            [
                float(detail["unconditional_indicator_nll_mean"])
                for detail in wrong_entropy_details
            ]
        ),
        "correct_condition_corrected_mean_nll": mean(
            [
                float(detail["condition_corrected_mean_nll"])
                for detail in correct_entropy_details
            ]
        ),
        "wrong_condition_corrected_mean_nll": mean(
            [
                float(detail["condition_corrected_mean_nll"])
                for detail in wrong_entropy_details
            ]
        ),
        "correct_inverse_condition_probability": mean(
            [
                float(detail["inverse_condition_probability"])
                for detail in correct_entropy_details
            ]
        ),
        "wrong_inverse_condition_probability": mean(
            [
                float(detail["inverse_condition_probability"])
                for detail in wrong_entropy_details
            ]
        ),
        "correct_conditional_log_normalizer": mean(
            [float(detail["log_normalizer"]) for detail in correct_entropy_details]
        ),
        "wrong_conditional_log_normalizer": mean(
            [float(detail["log_normalizer"]) for detail in wrong_entropy_details]
        ),
        "correct_condition_probability": mean(
            [float(detail["condition_probability"]) for detail in correct_entropy_details]
        ),
        "wrong_condition_probability": mean(
            [float(detail["condition_probability"]) for detail in wrong_entropy_details]
        ),
    }


def write_outputs(
    output_dir: Path,
    settings: dict[str, Any],
    metrics: dict[str, Any],
    records_by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "settings": settings,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for dataset_records in records_by_dataset.values():
            for record in dataset_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def amain() -> None:
    args = parse_args()
    gpu_count = detect_gpu_count()
    port = args.port if args.port > 0 else find_free_port(args.host)
    base_url = f"http://{args.host}:{port}"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=args.trust_remote_code,
    )
    proc, log_file, log_path, cmd = launch_sglang_server(args, gpu_count, port)
    try:
        await wait_for_server(base_url, proc, args.startup_timeout)
        client = AsyncOpenAI(
            base_url=f"{base_url}/v1",
            api_key="EMPTY",
            max_retries=0,
            timeout=httpx.Timeout(timeout=None, connect=60),
        )

        records_by_dataset: dict[str, list[dict[str, Any]]] = {}
        dataset_metrics: dict[str, Any] = {}
        for dataset_name in args.dataset:
            samples = load_eval_samples(dataset_name)
            metrics, records = await evaluate_dataset(
                client=client,
                dataset_name=dataset_name,
                samples=samples,
                args=args,
                tokenizer=tokenizer,
                server_proc=proc,
                log_path=log_path,
            )
            dataset_metrics[dataset_name] = metrics
            records_by_dataset[dataset_name] = records

        settings = {
            "checkpoint": args.checkpoint,
            "datasets": args.dataset,
            "samples_per_prompt": args.N,
            "temperature": 1.0,
            "sampling": "standard probability sampling; no top_p/top_k/min_p/penalty overrides",
            "max_tokens": args.max_tokens,
            "parallel": args.parallel,
            "samples_per_request": args.samples_per_request,
            "request_timeout": args.request_timeout,
            "request_retries": args.request_retries,
            "logprobs": args.logprobs,
            "gpu_count": gpu_count,
            "sglang_base_url": base_url,
            "sglang_command": cmd,
            "sglang_log": str(log_path),
            "entropy_definition": (
                "Model sequence-level entropy is estimated as the mean sequence "
                "negative log-likelihood E[-log p(y|x)] from SGLang token logprobs, "
                "in nats. Conditional entropies use H(Y|x,C)=E[-log p(Y|x)|C]+log "
                "p(C|x), with p(C|x) estimated from samples; subsets with fewer than "
                "two samples contribute 0. sequence_diversity_entropy is only the "
                "empirical entropy of normalized response strings and is kept as a "
                "diagnostic diversity metric."
            ),
        }
        write_outputs(
            output_dir=output_dir,
            settings=settings,
            metrics={"datasets": dataset_metrics},
            records_by_dataset=records_by_dataset,
        )
        print(f"Wrote evaluation results to {output_dir}")
    finally:
        stop_server(proc, log_file)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
