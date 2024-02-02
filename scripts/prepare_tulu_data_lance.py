"""
Script for preparing the Tulu V2 data for fine-tuning an OLMo model.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import datasets as ds
import numpy as np
from rich.progress import track

import lance
import pyarrow as pa

from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def main(opts) -> None:
    tokenizer: Tokenizer
    if Path(opts.tokenizer).is_file():
        tokenizer = Tokenizer.from_file(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)
    else:
        tokenizer = Tokenizer.from_pretrained(opts.tokenizer, eos_token_id=opts.eos, pad_token_id=opts.pad)

    dataset = ds.load_dataset("allenai/tulu-v2-sft-mixture", split="train")

    log.info("Tokenizing dataset...")
    dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, max_seq_len=opts.seq_len),
        batched=False,
        remove_columns=["dataset", "id", "messages"],
        num_proc=opts.num_proc,  # type: ignore
    )

    log.info("Filtering dataset...")
    n = len(dataset)  # type: ignore
    dataset = dataset.filter(filter, batched=False, num_proc=opts.num_proc)  # type: ignore
    log.info(f"Filtered out {n - len(dataset):,d} examples")

    log.info("Counting tokens...")
    total_tokens = 0
    for ex in track(dataset):
        assert len(ex["input_ids"]) == opts.seq_len  # type: ignore
        total_tokens += len(ex["input_ids"])  # type: ignore
    log.info(f"Total tokens: {total_tokens:,d}")

    log.info(f"Saving results to '{opts.output_dir}'...")

    # Save input_ids and label_mask in lance format
    all_input_ids = [ex["input_ids"] for ex in dataset]
    all_label_masks = [ex["label_mask"] for ex in dataset]
    input_ids_pa_table = pa.Table.from_arrays([all_input_ids], names=["value"])
    label_masks_pa_table = pa.Table.from_arrays([all_label_masks], names=["value"])

    lance.write_dataset(input_ids_pa_table, "tulu_dataset_input_ids.lance", {"model": "create"})
    lance.write_dataset(label_masks_pa_table, "tulu_dataset_label_masks.lance", {"model": "create"})

    log.info("Done!")


def filter(example):
    return example["n_labels"] > 0


def preprocess(example, tokenizer: Tokenizer, max_seq_len: int):
    input_ids = [tokenizer.eos_token_id]
    label_mask = [False]

    for msg in example["messages"]:
        role_tokens = tokenizer.encode(f"<|{msg['role']}|>\n", add_special_tokens=False)
        label_mask += [False] * len(role_tokens)
        input_ids += role_tokens

        if msg["role"] == "assistant":
            content_tokens = tokenizer.encode(
                msg["content"].strip() + tokenizer.eos_token + "\n", add_special_tokens=False
            )
            label_mask += [True] * len(content_tokens)
            # mask out the last '\n'
            assert content_tokens[-2] == tokenizer.eos_token_id
            label_mask[-1] = False
        else:
            content_tokens = tokenizer.encode(msg["content"].strip() + "\n", add_special_tokens=False)
            label_mask += [False] * len(content_tokens)
        input_ids += content_tokens

    input_ids = input_ids[:max_seq_len]
    label_mask = label_mask[:max_seq_len]

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        label_mask += [False] * pad_len

    assert len(input_ids) == len(label_mask)
    n_labels = sum(label_mask)

    return {"input_ids": input_ids, "label_mask": label_mask, "n_labels": n_labels}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare Tulu V2 dataset")
    parser.add_argument("output_dir", type=str, help="""Directory to save the results to.""")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="""Tokenizer path or identifier.""",
        default="tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json",
    )
    parser.add_argument("-s", "--seq-len", type=int, help="""Max sequence length.""", default=2048)
    parser.add_argument("--eos", type=int, help="""EOS token ID.""", default=50279)
    parser.add_argument("--pad", type=int, help="""PAD token ID.""", default=1)
    parser.add_argument("-j", "--num-proc", type=int, help="""Number of workers.""", default=8)
    return parser


if __name__ == "__main__":
    prepare_cli_environment()
    opts = get_parser().parse_args()
    main(opts)
