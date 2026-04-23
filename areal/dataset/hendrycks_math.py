from collections.abc import Iterable

from datasets import concatenate_datasets, load_dataset

DEFAULT_HENDRYCKS_MATH_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

_BOXED_ANSWER_INSTRUCTION = "\nPlease put your final answer within \\boxed{}."


def _resolve_subsets(
    name: str | None = None,
    subset: str | None = None,
    subsets: str | Iterable[str] | None = None,
) -> tuple[str, ...]:
    selected = subset if subset is not None else name
    if selected is not None:
        return (selected,)
    if subsets is None:
        return DEFAULT_HENDRYCKS_MATH_SUBSETS
    if isinstance(subsets, str):
        return (subsets,)
    return tuple(subsets)


def _load_hendrycks_math_dataset(
    path: str,
    split: str,
    name: str | None = None,
    subset: str | None = None,
    subsets: str | Iterable[str] | None = None,
):
    selected_subsets = _resolve_subsets(name=name, subset=subset, subsets=subsets)
    if not selected_subsets:
        raise ValueError("At least one hendrycks_math subset must be selected.")
    datasets = [
        load_dataset(path=path, name=dataset_name, split=split)
        for dataset_name in selected_subsets
    ]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def get_hendrycks_math_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    **kwargs,
):
    dataset = _load_hendrycks_math_dataset(path=path, split=split, **kwargs)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["problem"] + sample["solution"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["problem"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["problem", "solution"])

    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_hendrycks_math_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    **kwargs,
):
    dataset = _load_hendrycks_math_dataset(path=path, split=split, **kwargs)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["problem"] + _BOXED_ANSWER_INSTRUCTION,
            }
        ]
        return {"messages": messages, "answer": sample["solution"]}

    dataset = dataset.map(process).remove_columns(["problem"])

    if max_length is not None:

        def filter_length(sample):
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
