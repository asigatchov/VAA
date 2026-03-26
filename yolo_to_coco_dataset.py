#!/usr/bin/env python3
"""Convert a directory of YOLO datasets into train/valid/test COCO datasets."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert multiple YOLO dataset folders into a single COCO dataset with "
            "train/valid/test splits."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Root directory containing YOLO datasets")
    parser.add_argument("--output_dir", required=True, help="Output COCO dataset directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def discover_samples(input_dir: Path) -> tuple[list[dict], dict[int, str]]:
    samples: list[dict] = []
    category_names: set[int] = set()

    for dataset_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            annotations = []
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as file_obj:
                    for line in file_obj:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        class_id_str, x_str, y_str, w_str, h_str = stripped.split()
                        class_id = int(class_id_str)
                        category_names.add(class_id)
                        annotations.append(
                            {
                                "category_id": class_id,
                                "x_center": float(x_str),
                                "y_center": float(y_str),
                                "width": float(w_str),
                                "height": float(h_str),
                            }
                        )

            samples.append(
                {
                    "dataset_name": dataset_dir.name,
                    "image_path": image_path,
                    "label_path": label_path,
                    "annotations": annotations,
                }
            )

    categories = {class_id: f"class_{class_id}" for class_id in sorted(category_names)}
    return samples, categories


def split_samples(samples: list[dict], train_ratio: float, valid_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict]]:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)

    train_end = int(count * train_ratio)
    valid_end = train_end + int(count * valid_ratio)

    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def yolo_to_coco_bbox(annotation: dict, image_width: int, image_height: int) -> tuple[list[float], float]:
    bbox_width = annotation["width"] * image_width
    bbox_height = annotation["height"] * image_height
    center_x = annotation["x_center"] * image_width
    center_y = annotation["y_center"] * image_height
    x_min = center_x - bbox_width / 2
    y_min = center_y - bbox_height / 2
    area = bbox_width * bbox_height
    return [x_min, y_min, bbox_width, bbox_height], area


def build_coco_split(split_name: str, samples: list[dict], output_dir: Path, categories: dict[int, str]) -> dict:
    split_dir = output_dir / split_name
    ensure_dir(split_dir)

    coco = {
        "info": {
            "description": f"COCO dataset split: {split_name}",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id, "name": name, "supercategory": "object"}
            for class_id, name in categories.items()
        ],
    }

    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        image_path = sample["image_path"]
        unique_name = f"{sample['dataset_name']}_{image_path.name}"
        target_image = split_dir / unique_name
        shutil.copy2(image_path, target_image)

        width, height = load_image_size(target_image)
        coco["images"].append(
            {
                "id": image_id,
                "license": 0,
                "file_name": unique_name,
                "width": width,
                "height": height,
            }
        )

        for annotation in sample["annotations"]:
            bbox, area = yolo_to_coco_bbox(annotation, width, height)
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation["category_id"],
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with open(split_dir / "_annotations.coco.json", "w", encoding="utf-8") as file_obj:
        json.dump(coco, file_obj, indent=2, ensure_ascii=False)

    return {
        "split": split_name,
        "num_images": len(coco["images"]),
        "num_annotations": len(coco["annotations"]),
        "output_dir": str(split_dir),
    }


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    samples, categories = discover_samples(input_dir)
    if not samples:
        raise ValueError(f"No YOLO samples found in: {input_dir}")

    ensure_dir(output_dir)
    splits = split_samples(samples, args.train_ratio, args.valid_ratio, args.test_ratio, args.seed)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_categories": len(categories),
        "categories": categories,
        "splits": [],
    }

    for split_name, split_samples_list in splits.items():
        split_summary = build_coco_split(split_name, split_samples_list, output_dir, categories)
        summary["splits"].append(split_summary)

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
