"""Train the lightweight volleyball action detector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .action_detector import (
    HELPER_CLASS_IDS,
    VballActionDetector,
    VballActionDetectorV2,
    detection_loss,
    detection_loss_v2,
)
from .constants import EXPORT_CLASS_NAMES
from .data import ActionDetectionDataset, collate_detection_batch, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VballActionDetector.")
    parser.add_argument("--dataset-dir", default="data/action_detector")
    parser.add_argument("--output-dir", default="model_vball_action_detector")
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--in-dim", type=int, default=None, help="Defaults to dataset_manifest.json in_dim or 9.")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--model-variant", default="v1", choices=("v1", "v2"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--obj-pos-weight", type=float, default=1.0)
    parser.add_argument("--action-cls-weight", type=float, default=1.0)
    parser.add_argument("--helper-cls-weight", type=float, default=1.0)
    parser.add_argument("--action-focal-gamma", type=float, default=0.0)
    parser.add_argument("--helper-focal-gamma", type=float, default=0.0)
    parser.add_argument(
        "--class-weights",
        default="none",
        help="Class weights: 'none', 'auto', or comma-separated weights for classes 0..N-1.",
    )
    parser.add_argument("--resume", help="Checkpoint path to resume from.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--hflip-p", type=float, default=0.0, help="Training horizontal flip probability.")
    parser.add_argument("--rotate-degrees", type=float, default=0.0, help="Max absolute random rotation angle.")
    parser.add_argument("--rotate-p", type=float, default=0.0, help="Training rotation probability.")
    return parser.parse_args()


def resolve_in_dim(dataset_dir: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if "in_dim" in manifest:
            return int(manifest["in_dim"])
    return 9


def resolve_num_classes(dataset_dir: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if "num_classes" in manifest:
            return int(manifest["num_classes"])
    return 6


def load_dataset_manifest(dataset_dir: Path) -> dict:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    return {}


def resolve_class_weights(
    spec: str,
    dataset_manifest: dict,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if spec.lower() in {"", "none", "off", "false"}:
        return None
    if spec.lower() == "auto":
        counts_raw = dataset_manifest.get("class_counts", {})
        counts = torch.tensor(
            [max(1.0, float(counts_raw.get(str(class_id), 1.0))) for class_id in range(num_classes)],
            dtype=torch.float32,
        )
        weights = counts.sum() / (num_classes * counts)
        return (weights / weights.mean()).to(device)

    values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if len(values) != num_classes:
        raise ValueError(f"--class-weights must contain {num_classes} values, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def make_loader(
    dataset_dir: Path,
    split: str,
    image_size: tuple[int, int],
    in_dim: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment: bool = False,
    hflip_p: float = 0.0,
    rotate_degrees: float = 0.0,
    rotate_p: float = 0.0,
) -> DataLoader:
    dataset = ActionDetectionDataset(
        dataset_dir,
        split=split,
        image_size=image_size,
        in_dim=in_dim,
        augment=augment,
        hflip_p=hflip_p,
        rotate_degrees=rotate_degrees,
        rotate_p=rotate_p,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )


def save_checkpoint(
    path: Path,
    model: VballActionDetector | VballActionDetectorV2,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
            "class_names": config.get("class_names", EXPORT_CLASS_NAMES),
        },
        path,
    )


def append_metrics(metrics_path: Path, row: dict[str, float | int | str]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model: VballActionDetector | VballActionDetectorV2,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
    model_variant: str,
    grad_clip: float,
    grad_accum_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    obj_pos_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
    action_cls_weight: float = 1.0,
    helper_cls_weight: float = 1.0,
    action_focal_gamma: float = 0.0,
    helper_focal_gamma: float = 0.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "obj": 0.0, "box": 0.0, "cls": 0.0, "action_cls": 0.0, "helper_cls": 0.0}
    samples = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, targets, _metas) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = [target.to(device, non_blocking=True) for target in targets]
        with torch.set_grad_enabled(training):
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(images)
                if model_variant == "v2":
                    action_class_weights = class_weights[:4] if class_weights is not None else None
                    helper_class_weights = (
                        class_weights[-len(HELPER_CLASS_IDS) :] if class_weights is not None else None
                    )
                    losses = detection_loss_v2(
                        output,
                        targets,
                        obj_pos_weight=obj_pos_weight,
                        action_cls_weight=action_cls_weight,
                        helper_cls_weight=helper_cls_weight,
                        action_class_weights=action_class_weights,
                        helper_class_weights=helper_class_weights,
                        action_focal_gamma=action_focal_gamma,
                        helper_focal_gamma=helper_focal_gamma,
                    )
                else:
                    losses = detection_loss(
                        output["raw"],
                        targets,
                        num_classes=num_classes,
                        obj_pos_weight=obj_pos_weight,
                        class_weights=class_weights,
                    )
            if training:
                backward_loss = losses["total"] / grad_accum_steps
                if scaler is not None and device.type == "cuda":
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()

                is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_step = (batch_idx + 1) == len(loader)
                if is_accum_step or is_last_step:
                    if scaler is not None and device.type == "cuda":
                        if grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        batch_size = images.shape[0]
        samples += batch_size
        for key in totals:
            if key in losses:
                totals[key] += float(losses[key].detach().cpu()) * batch_size

    metrics = {key: value / max(1, samples) for key, value in totals.items()}
    if model_variant == "v2":
        metrics["cls"] = metrics["action_cls"] + metrics["helper_cls"]
    return metrics


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = load_dataset_manifest(dataset_dir)
    in_dim = resolve_in_dim(dataset_dir, args.in_dim)
    num_classes = resolve_num_classes(dataset_dir, args.num_classes)
    class_names = dataset_manifest.get("class_names") or EXPORT_CLASS_NAMES
    image_size = (args.height, args.width)
    config = {
        "in_dim": in_dim,
        "num_classes": num_classes,
        "width_mult": args.width_mult,
        "model_variant": args.model_variant,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "obj_pos_weight": args.obj_pos_weight,
        "action_cls_weight": args.action_cls_weight,
        "helper_cls_weight": args.helper_cls_weight,
        "action_focal_gamma": args.action_focal_gamma,
        "helper_focal_gamma": args.helper_focal_gamma,
        "class_weights": args.class_weights,
        "hflip_p": args.hflip_p,
        "rotate_degrees": args.rotate_degrees,
        "rotate_p": args.rotate_p,
        "image_size": {"height": args.height, "width": args.width},
        "frame_offsets": dataset_manifest.get("frame_offsets"),
        "class_names": class_names,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as file_obj:
        json.dump(config, file_obj, indent=2, ensure_ascii=False)

    train_loader = make_loader(
        dataset_dir,
        "train",
        image_size,
        in_dim,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        augment=True,
        hflip_p=args.hflip_p,
        rotate_degrees=args.rotate_degrees,
        rotate_p=args.rotate_p,
    )
    try:
        valid_loader = make_loader(
            dataset_dir,
            "valid",
            image_size,
            in_dim,
            args.batch_size,
            args.num_workers,
            shuffle=False,
        )
    except ValueError:
        valid_loader = None

    device = torch.device(args.device)
    if args.model_variant == "v2":
        model = VballActionDetectorV2(
            in_dim=in_dim,
            num_classes=num_classes,
            width_mult=args.width_mult,
        ).to(device)
    else:
        model = VballActionDetector(
            in_dim=in_dim,
            num_classes=num_classes,
            width_mult=args.width_mult,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    class_weights = resolve_class_weights(args.class_weights, dataset_manifest, num_classes, device)
    if class_weights is not None:
        config["resolved_class_weights"] = [float(value) for value in class_weights.detach().cpu().tolist()]
        with open(output_dir / "config.json", "w", encoding="utf-8") as file_obj:
            json.dump(config, file_obj, indent=2, ensure_ascii=False)
    writer = SummaryWriter(log_dir=str(output_dir))

    start_epoch = 1
    best_valid = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid = float(checkpoint.get("metrics", {}).get("valid_total", best_valid))

    metrics_path = output_dir / "metrics.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            num_classes=num_classes,
            model_variant=args.model_variant,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            obj_pos_weight=args.obj_pos_weight,
            class_weights=class_weights,
            action_cls_weight=args.action_cls_weight,
            helper_cls_weight=args.helper_cls_weight,
            action_focal_gamma=args.action_focal_gamma,
            helper_focal_gamma=args.helper_focal_gamma,
        )
        valid_metrics = None
        if valid_loader is not None:
            with torch.no_grad():
                valid_metrics = run_epoch(
                    model,
                    valid_loader,
                    device,
                    optimizer=None,
                    num_classes=num_classes,
                    model_variant=args.model_variant,
                    grad_clip=0.0,
                    grad_accum_steps=1,
                    scaler=None,
                    obj_pos_weight=args.obj_pos_weight,
                    class_weights=class_weights,
                    action_cls_weight=args.action_cls_weight,
                    helper_cls_weight=args.helper_cls_weight,
                    action_focal_gamma=args.action_focal_gamma,
                    helper_focal_gamma=args.helper_focal_gamma,
                )

        row: dict[str, float | int | str] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        if valid_metrics is not None:
            row.update({f"valid_{key}": value for key, value in valid_metrics.items()})
        append_metrics(metrics_path, row)
        for key, value in train_metrics.items():
            writer.add_scalar(f"loss/train_{key}", value, epoch)
        if valid_metrics is not None:
            for key, value in valid_metrics.items():
                writer.add_scalar(f"loss/valid_{key}", value, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.flush()

        metrics = {key: float(value) for key, value in row.items() if isinstance(value, (int, float))}
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, metrics)
        valid_total = float(row.get("valid_total", row["train_total"]))
        if valid_total < best_valid:
            best_valid = valid_total
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, metrics)

        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"valid_total={valid_total:.4f}"
        )

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
