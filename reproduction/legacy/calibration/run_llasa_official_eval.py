#!/usr/bin/env python3
"""Run LLaSA official zero-shot HAR evaluation with local paths.

This keeps the official prompt, random sampling policy, output parser, and
classification_report metrics, but loads the 13B/7B model once instead of once
per sample.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm


PAPER_TABLE5 = {
    "hhar": {"f1_score": 0.84, "precision": 0.88, "recall": 0.86},
    "motion": {"f1_score": 0.83, "precision": 0.85, "recall": 0.83},
    "shoaib": {"f1_score": 0.81, "precision": 0.84, "recall": 0.81},
    "uci": {"f1_score": 0.72, "precision": 0.82, "recall": 0.75},
}


LAST_AXIS = {"hhar": 2, "motion": 0, "shoaib": 0, "uci": 0}
ANSWER_FORMAT = "The identified class is: "
UNCLEAR_LABEL = "Unclear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--root-data-dir", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--activity-label-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["hhar", "motion", "shoaib", "uci"])
    parser.add_argument("--per-class-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--load-embedded-vision-tower", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_class_maps(dataset_activity_label_dict: dict[str, dict[str, str]]) -> tuple[set[str], dict[str, int]]:
    unique_classes: set[str] = set()
    for val in dataset_activity_label_dict.values():
        unique_classes.update(val.values())
    unique_classes_dict = {unique_class: idx for idx, unique_class in enumerate(unique_classes)}
    unique_classes_dict[UNCLEAR_LABEL] = len(unique_classes_dict)
    return unique_classes, unique_classes_dict


def build_prompt(unique_classes: set[str]) -> str:
    return (
        "The given IMU sensor data can be associated with one of the following classes: "
        f"{unique_classes}. "
        "Write only the name of the identified class in the format "
        f"'{ANSWER_FORMAT}'"
    )


def infer_conv_mode(model_name: str) -> str:
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def normalize_prediction(text: str) -> str | None:
    match = re.search(r"The identified class is:?\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    pred_label = match.group(1)
    pred_label = re.sub(r"[^\w\s]", "", pred_label)
    return pred_label.strip().lower()


def load_embedded_vision_tower(model_path: str, model) -> dict:
    """Load vision tower tensors embedded in the HF checkpoint.

    The public LLaSA HF checkpoint contains model.vision_tower weights, while
    the official loader initializes the tower from ../model_4.pt. This helper
    lets us test the embedded checkpoint path without changing model code.
    """
    from safetensors import safe_open

    model_dir = Path(model_path)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    tower = vision_tower.vision_tower[0]
    tower_dtype = next(tower.parameters()).dtype
    tower_state = {}
    for filename in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                prefix = "model.vision_tower.vision_tower.0."
                if key.startswith(prefix):
                    tower_state[key[len(prefix) :]] = f.get_tensor(key).to(dtype=tower_dtype)
    if not tower_state:
        raise RuntimeError(f"No embedded vision tower tensors found in {model_path}")
    missing, unexpected = tower.load_state_dict(tower_state, strict=False)
    return {
        "loaded_tensors": len(tower_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def eval_one(
    *,
    dataset: str,
    data: np.ndarray,
    labels: np.ndarray,
    label_dict: dict[str, str],
    unique_classes_dict: dict[str, int],
    prompt: str,
    conv_mode: str,
    image_root: Path,
    model_objects: tuple,
    args: argparse.Namespace,
) -> dict:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    tokenizer, model, image_processor, _context_len = model_objects
    pred: list[int] = []
    gt: list[int] = []
    records: list[dict] = []
    label_count_dict = {val: 0 for val in label_dict.values()}
    last_axis = LAST_AXIS[dataset]
    sample_plan: list[tuple[int, str, Path]] = []

    for _idx in tqdm(range(len(data)), desc=dataset):
        sample_index = random.randint(0, len(data))
        image_file = image_root / dataset / f"sensor_image_{sample_index}.npy"
        if not image_file.exists():
            continue

        sample_label = label_dict[str(int(labels[sample_index, 0, last_axis]))]
        if label_count_dict[sample_label] < args.per_class_samples:
            label_count_dict[sample_label] += 1
            sample_plan.append((sample_index, sample_label, image_file))
        else:
            continue

        if all(count >= args.per_class_samples for count in label_count_dict.values()):
            break

    for sample_index, sample_label, image_file in tqdm(sample_plan, desc=f"{dataset} infer"):

        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        image = torch.from_numpy(np.load(image_file))
        images_tensor = process_images([image], image_processor, model.config).to(
            device=model.device, dtype=model.dtype
        )
        input_ids = tokenizer_image_token(
            full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=None,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        gt.append(unique_classes_dict[sample_label])
        parsed = normalize_prediction(output)
        if parsed is None or parsed not in unique_classes_dict:
            pred.append(unique_classes_dict[UNCLEAR_LABEL])
            status = "unparsed" if parsed is None else "out_of_box"
        else:
            pred.append(unique_classes_dict[parsed])
            status = "ok"

        records.append(
            {
                "sample_index": sample_index,
                "image_file": str(image_file),
                "true_label": sample_label,
                "output": output,
                "parsed_label": parsed,
                "status": status,
            }
        )

    target_names = list(unique_classes_dict.keys())
    report = classification_report(
        gt,
        pred,
        labels=list(range(len(unique_classes_dict))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    weighted = report.get("weighted avg", {})
    precision, recall, f1, _support = precision_recall_fscore_support(
        gt, pred, average="weighted", zero_division=0
    )
    paper = PAPER_TABLE5.get(dataset, {})
    metrics = {
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "support": len(gt),
        "sklearn_weighted_avg": weighted,
    }
    gaps = {
        key: float(metrics[key] - paper[key])
        for key in ("f1_score", "precision", "recall")
        if key in paper
    }
    return {
        "dataset": dataset,
        "metrics": metrics,
        "paper_reference": paper,
        "paper_gap_signed": gaps,
        "paper_gap_abs": {key: abs(val) for key, val in gaps.items()},
        "label_counts": label_count_dict,
        "planned_samples": len(sample_plan),
        "records": records,
        "classification_report": report,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(args.repo_root)
    import sys

    sys.path.insert(0, str(args.repo_root))
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    with args.activity_label_json.open("r", encoding="utf-8") as f:
        dataset_activity_label_dict = json.load(f)
    unique_classes, unique_classes_dict = build_class_maps(dataset_activity_label_dict)
    prompt = build_prompt(unique_classes)

    start = time.time()
    model_name = get_model_name_from_path(args.model_path)
    conv_mode = infer_conv_mode(model_name)
    model_objects = load_pretrained_model(args.model_path, None, model_name)
    embedded_vision_tower = None
    if args.load_embedded_vision_tower:
        embedded_vision_tower = load_embedded_vision_tower(args.model_path, model_objects[1])

    results = []
    for dataset in args.datasets:
        data = np.load(args.root_data_dir / dataset / "data_20_120.npy", mmap_mode="r")
        labels = np.load(args.root_data_dir / dataset / "label_20_120.npy", mmap_mode="r")
        result = eval_one(
            dataset=dataset,
            data=data,
            labels=labels,
            label_dict=dataset_activity_label_dict[dataset],
            unique_classes_dict=unique_classes_dict,
            prompt=prompt,
            conv_mode=conv_mode,
            image_root=args.image_root,
            model_objects=model_objects,
            args=args,
        )
        results.append(result)
        (args.output_dir / f"{dataset}_records.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    artifact = {
        "method": "LLaSA",
        "official_repo": "https://github.com/BASHLab/LLaSA",
        "official_repo_commit": os.popen("git rev-parse HEAD").read().strip(),
        "model_path": args.model_path,
        "seed": args.seed,
        "per_class_samples": args.per_class_samples,
        "conv_mode": conv_mode,
        "embedded_vision_tower": embedded_vision_tower,
        "prompt": prompt,
        "runtime_seconds": round(time.time() - start, 3),
        "results": results,
    }
    (args.output_dir / "results.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({r["dataset"]: r["metrics"] for r in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
