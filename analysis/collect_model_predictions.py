#!/usr/bin/env python3
"""
Run GatorTron inference for records where AI is True.

For each qualifying record:
1. Append the nonblank diagnosis and assessment text.
2. Run the fine-tuned GatorTron multilabel classifier on CUDA.
3. Retain every concept code with probability >= 0.5.
4. Save predictions to JSON, keyed by record_id.

Example:
    python collect_model_predictions.py \
        --records records_df.csv \
        --model ../fine_tuned/model_save_gator.pkl \
        --tokenizer UFNLP/gatortron-medium \
        --all-codes ../all_codes.pt \
        --output gatortron_predictions.json \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from models import GatorTronClassifier


THRESHOLD = 0.5
MAX_LENGTH = 512
HIDDEN_SIZE = 2560
BREED_PREFIX = "[-1.0] "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multilabel GatorTron predictions."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("records_df.csv"),
        help="CSV containing record_id, diagnosis, assessment, and AI columns.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../fine_tuned/model_save_gator.pkl"),
        help="Fine-tuned PyTorch checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help=(
            "Local tokenizer directory or Hugging Face model identifier used "
            "during training."
        ),
    )
    parser.add_argument(
        "--all-codes",
        type=Path,
        default=Path("all_codes.pt"),
        help="Torch file containing concept codes in model-output order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gatortron_predictions.json"),
        help="Destination JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    return parser.parse_args()


def normalize_ai_column(series: pd.Series) -> pd.Series:
    """
    Convert common boolean, numeric, and string representations to bool.
    Missing or unrecognized values become False.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype("string").str.strip().str.lower()

    string_map = {
        "true": True,
        "false": False,
        "t": True,
        "f": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "1": True,
        "0": False,
    }

    mapped = normalized.map(string_map)

    # Also support numeric values such as 1.0 and 0.0.
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_bool = numeric.eq(1)

    return mapped.fillna(numeric_bool).fillna(False).astype(bool)


def clean_text(value: Any) -> str:
    """Return stripped text, treating null values as empty strings."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def append_record_text(row: pd.Series) -> str:
    """
    Append diagnosis and assessment in that order.

    A newline is used as a neutral separator. Empty fields are omitted.
    """
    parts = [
        clean_text(row.get("diagnosis")),
        clean_text(row.get("assessment")),
    ]
    text = "\n".join(part for part in parts if part)

    if not text:
        return ""

    # Match the preprocessing used by the existing application.
    if not text.startswith("["):
        text = BREED_PREFIX + text

    return text


def load_concept_codes(path: Path) -> list:
    """Load concept codes while preserving model-output order."""
    loaded = torch.load(path, map_location="cpu")

    if isinstance(loaded, torch.Tensor):
        values = loaded.detach().cpu().tolist()
    elif isinstance(loaded, pd.Series):
        values = loaded.tolist()
    elif isinstance(loaded, (list, tuple)):
        values = list(loaded)
    else:
        raise TypeError(
            f"Unsupported all-codes object type: "
            f"{type(loaded).__name__}"
        )

    return [str(value) for value in values]



class RecordDataset(Dataset):
    def __init__(self, records: pd.DataFrame) -> None:
        self.record_ids = records["record_id"].astype(str).tolist()
        self.texts = records["model_text"].tolist()

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.record_ids[index], self.texts[index]


def make_collate_fn(tokenizer: PreTrainedTokenizerBase):
    def collate(
        batch: list[tuple[str, str]],
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        record_ids, texts = zip(*batch)

        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        return list(record_ids), encoded

    return collate


def load_model(
    checkpoint_path: Path,
    number_of_codes: int,
    device: torch.device,
) -> GatorTronClassifier:
    model = GatorTronClassifier(HIDDEN_SIZE, number_of_codes)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    return model


def predict(
    model: GatorTronClassifier,
    data_loader: DataLoader,
    concept_codes: list[str],
    device: torch.device,
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}

    with torch.inference_mode():
        for record_ids, encoded in tqdm(
            data_loader,
            desc="Running GatorTron inference",
            unit="batch",
        ):
            input_ids = encoded["input_ids"].to(
                device,
                non_blocking=True,
            )

            attention_mask = (input_ids != 0).to(torch.int64)

            logits = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)

            if probabilities.ndim != 2:
                raise RuntimeError(
                    "Expected model output with shape "
                    f"[batch, classes], received {tuple(probabilities.shape)}."
                )

            if probabilities.shape[1] != len(concept_codes):
                raise RuntimeError(
                    "Model output dimension does not match all_codes.pt: "
                    f"{probabilities.shape[1]} outputs versus "
                    f"{len(concept_codes)} concept codes."
                )

            probabilities = probabilities.detach().cpu()

            for record_id, row_probabilities in zip(
                record_ids,
                probabilities,
            ):
                selected_indices = torch.nonzero(
                    row_probabilities >= THRESHOLD,
                    as_tuple=False,
                ).flatten()

                # Sort qualifying codes by probability, highest first.
                selected_indices = selected_indices[
                    torch.argsort(
                        row_probabilities[selected_indices],
                        descending=True,
                    )
                ]

                predictions[record_id] = [
                    concept_codes[index]
                    for index in selected_indices.tolist()
                ]

    return predictions


def main() -> None:
    args = parse_args()

    device = torch.device("cuda")

    record_df = pd.read_csv(args.records)

    required_columns = {
        "record_id",
        "AI",
        "diagnosis",
        "assessment",
    }

    # Allow the lowercase "ai" column used by the existing Streamlit app.
    if "AI" not in record_df.columns and "ai" in record_df.columns:
        record_df = record_df.rename(columns={"ai": "AI"})

    missing_columns = required_columns - set(record_df.columns)
    if missing_columns:
        raise ValueError(
            "Records CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    record_df["AI"] = normalize_ai_column(record_df["AI"])
    prediction_df = record_df.loc[record_df["AI"]].copy()

    if prediction_df["record_id"].isna().any():
        raise ValueError("record_id contains missing values among AI records.")

    prediction_df["record_id"] = prediction_df["record_id"].astype(str)

    duplicate_ids = prediction_df.loc[
        prediction_df["record_id"].duplicated(keep=False),
        "record_id",
    ].unique()

    if len(duplicate_ids) > 0:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(
            "record_id must be unique because it is used as the JSON key. "
            f"Duplicate IDs include: {preview}"
        )

    prediction_df["model_text"] = prediction_df.apply(
        append_record_text,
        axis=1,
    )

    # Empty diagnosis plus empty assessment cannot produce a useful prediction.
    empty_mask = prediction_df["model_text"].eq("")
    skipped_empty_ids = prediction_df.loc[
        empty_mask, "record_id"
    ].tolist()
    prediction_df = prediction_df.loc[~empty_mask].copy()

    concept_codes = load_concept_codes(args.all_codes)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
    )

    # Dynamic batch padding requires a pad token.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token. "
                "Configure the same pad token used during training."
            )
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(
        checkpoint_path=args.model,
        number_of_codes=len(concept_codes),
        device=device,
    )

    dataset = RecordDataset(prediction_df)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    predictions = predict(
        model=model,
        data_loader=data_loader,
        concept_codes=concept_codes,
        device=device,
    )

    # Preserve AI=True records with no clinical text as empty prediction lists.
    for record_id in skipped_empty_ids:
        predictions[record_id] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            predictions,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print(f"Saved predictions for {len(predictions):,} records to {args.output}")


if __name__ == "__main__":
    main()