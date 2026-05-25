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
from openai import AsyncOpenAI
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
    correct: bool


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
    parser.add_argument("--parallel", type=int, default=256)
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
    tokenizer,
) -> list[str]:
    response = await client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": sample.prompt}],
        temperature=1.0,
        n=n_samples,
        max_tokens=max_tokens,
    )
    outputs: list[str] = []
    for choice in response.choices:
        content = choice.message.content
        outputs.append(content if content is not None else "")
    if len(outputs) != n_samples:
        raise RuntimeError(
            f"Expected {n_samples} generations for {sample.sample_id}, got {len(outputs)}."
        )
    return outputs


async def evaluate_dataset(
    client: AsyncOpenAI,
    dataset_name: str,
    samples: list[EvalSample],
    args: argparse.Namespace,
    tokenizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(args.parallel)
    judge = dataset_judge(dataset_name)

    async def evaluate_sample(sample: EvalSample) -> dict[str, Any]:
        async with semaphore:
            texts = await generate_one_prompt(
                client=client,
                sample=sample,
                n_samples=args.N,
                max_tokens=args.max_tokens,
                tokenizer=tokenizer,
            )
        generations = [
            Generation(
                text=text,
                token_count=token_count(tokenizer, text),
                correct=judge(text, sample.answer),
            )
            for text in texts
        ]
        correctness = [generation.correct for generation in generations]
        correct_texts = [
            normalize_sequence(generation.text)
            for generation in generations
            if generation.correct
        ]
        wrong_texts = [
            normalize_sequence(generation.text)
            for generation in generations
            if not generation.correct
        ]
        return {
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "answer": sample.answer,
            "raw": sample.raw,
            "generations": [asdict(generation) for generation in generations],
            "pass_at_k": pass_at_k(correctness, args.N),
            "sequence_entropy": empirical_entropy(
                [normalize_sequence(generation.text) for generation in generations]
            ),
            "correct_conditional_entropy": empirical_entropy(correct_texts),
            "wrong_conditional_entropy": empirical_entropy(wrong_texts),
        }

    tasks = [evaluate_sample(sample) for sample in samples]
    records = await tqdm.gather(*tasks, desc=dataset_name)
    metrics = aggregate_metrics(dataset_name, records, args.N)
    return metrics, records


def normalize_sequence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
        "correct_conditional_entropy": mean(
            [float(record["correct_conditional_entropy"]) for record in records]
        ),
        "wrong_conditional_entropy": mean(
            [float(record["wrong_conditional_entropy"]) for record in records]
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
            "gpu_count": gpu_count,
            "sglang_base_url": base_url,
            "sglang_command": cmd,
            "sglang_log": str(log_path),
            "entropy_definition": (
                "Empirical sequence-level entropy over complete normalized responses "
                "per prompt, averaged over prompts. Conditional entropies use the "
                "same estimator inside correct and wrong response subsets; subsets "
                "with fewer than two samples contribute 0."
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
