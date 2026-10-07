# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ
#
# SPDX-License-Identifier: MIT

import argparse
import random
from pathlib import Path
from typing import cast

import albumentations as A
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sixdrepnet360 import datasets
from sixdrepnet360.datasets import Sample, Transform
from sixdrepnet360.evaluate import evaluate, make_eval_transform
from sixdrepnet360.model import SixDRepNet360


def parse_args() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Train the 6DRepNet360 head pose model."
    )
    parser.add_argument(
        "--gpu", dest="gpu_id", help="GPU device id to use [0]", default=0, type=int
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Directory path for training data.",
        default="datasets/300W_LP",
        type=str,
    )
    parser.add_argument(
        "--filename_list",
        dest="filename_list",
        help="Path to text file containing relative paths for training examples.",
        default="datasets/300W_LP/files.txt",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="Training dataset type.",
        default="Pose_300W_LP",
        type=str,
    )
    parser.add_argument(
        "--eval_dataset",
        dest="eval_dataset",
        help="Evaluation dataset type, run after every epoch.",
        default="AFLW2000",
        type=str,
    )
    parser.add_argument(
        "--eval_data_dir",
        dest="eval_data_dir",
        help="Directory path for evaluation data.",
        default="datasets/AFLW2000",
        type=str,
    )
    parser.add_argument(
        "--eval_filename_list",
        dest="eval_filename_list",
        help="Evaluation file list. Evaluation is skipped when empty.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=80, type=int
    )
    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        help="Evaluation batch size. Does not affect the metrics, only memory use.",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--epochs", dest="epochs", help="Number of epochs.", default=80, type=int
    )
    parser.add_argument(
        "--lr", dest="lr", help="Adam learning rate.", default=1e-4, type=float
    )
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        help="Data loader workers.",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--precision",
        dest="precision",
        help="Training precision. fp16 and bf16 use autocast on CUDA/ROCm only.",
        choices=["fp32", "bf16", "fp16"],
        default="fp16",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Initialize the backbone from torchvision's ImageNet ResNet-50 weights.",
        action="store_true",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        help="Checkpoint to resume training from.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Directory where checkpoints and TensorBoard logs are written.",
        default="output",
        type=str,
    )
    parser.add_argument(
        "--save_every",
        dest="save_every",
        help="Save a numbered checkpoint every N epochs.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--log_every",
        dest="log_every",
        help="Log the training loss to TensorBoard every N steps.",
        default=10,
        type=int,
    )

    return parser.parse_args()


def make_train_transform() -> Transform:
    """Training augmentation: PIL image in, normalized tensor out.

    No horizontal flip here: the dataset already flips the image and negates
    yaw and roll, which keeps the rotation label consistent.
    """
    augment = A.Compose(
        [
            A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.1, 0.25),
                hole_width_range=(0.1, 0.25),
                fill=0,
                p=0.5,
            ),
            A.Blur(blur_range=(3, 5), p=0.1),
            A.RandomBrightnessContrast(p=0.5),
            A.RGBShift(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    def transform(img: Image.Image) -> torch.Tensor:
        out = augment(image=np.asarray(img))["image"]
        return cast(torch.Tensor, out)

    return transform


def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """Mean angle (radians) between predicted and ground truth rotation matrices."""
    m = torch.bmm(R_pred, R_gt.transpose(1, 2))
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Clamp away from +-1 so acos has a finite gradient.
    cos = torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos).mean()


def seed_worker(worker_id: int) -> None:
    # Forked workers inherit the parent's RNG state, so each one would produce the
    # same augmentations. Seed every worker from the torch seed it was given.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    # Global NumPy state, which albumentations may draw from.
    np.random.seed(seed)  # noqa: NPY002
    datasets.rng = np.random.default_rng(seed)


def load_imagenet_backbone(model: SixDRepNet360) -> None:
    """Copy torchvision's ImageNet ResNet-50 weights into the backbone.

    The classifier is skipped (the model has its own 6-output head), and any
    tensor whose name or shape differs is left at its random initialization.
    """
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    source = torchvision.models.resnet50(weights=weights).state_dict()
    own = model.state_dict()
    matching = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    result = model.load_state_dict(matching, strict=False)
    print(
        f"Loaded {len(matching)} ImageNet tensors. "
        f"Still random: {', '.join(result.missing_keys)}"
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def autocast_dtype(precision: str) -> torch.dtype | None:
    """Map a precision name to an autocast dtype. None means full fp32."""
    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def main() -> None:
    args = parse_args()
    # Input shapes are fixed, so let cuDNN/MIOpen pick the fastest conv algorithms.
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    precision = args.precision
    if device.type != "cuda" and precision != "fp32":
        print(f"No GPU found, using fp32 instead of {precision}.")
        precision = "fp32"
    amp_dtype = autocast_dtype(precision)
    # Loss scaling is only needed for fp16; bf16 has fp32's exponent range.
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "fp16")

    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
    elif args.pretrained:
        load_imagenet_backbone(model)

    print("Loading data.")
    pose_dataset = datasets.getDataset(
        args.dataset,
        args.data_dir,
        args.filename_list,
        make_train_transform(),
        train_mode=True,
    )
    loader = DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    eval_loader: DataLoader[Sample] | None = None
    if args.eval_filename_list:
        eval_dataset = datasets.getDataset(
            args.eval_dataset,
            args.eval_data_dir,
            args.eval_filename_list,
            make_eval_transform(),
            train_mode=False,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        print("No --eval_filename_list given, skipping per-epoch evaluation.")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for step, (images, R_gt, _cont_labels, _names) in enumerate(loader):
            images = images.to(device)
            R_gt = R_gt.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=amp_dtype is not None,
            ):
                R_pred = model(images)
            # The loss stays in fp32 even when the forward pass is autocast.
            loss = geodesic_loss(R_pred.float(), R_gt)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            steps += 1

            global_step = epoch * len(loader) + step
            if global_step % args.log_every == 0:
                writer.add_scalar("train/geodesic_loss", loss.item(), global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )

        mean_loss = running_loss / max(steps, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}, geodesic loss: {mean_loss:.4f}")
        writer.add_scalar("train/epoch_geodesic_loss", mean_loss, epoch + 1)

        if eval_loader is not None:
            metrics = evaluate(model, eval_loader, device)
            print(
                f"  Eval {args.eval_dataset}: Yaw {metrics.yaw:.4f}, "
                f"Pitch {metrics.pitch:.4f}, Roll {metrics.roll:.4f}, "
                f"MAE {metrics.mae:.4f}, VMAE {metrics.vmae:.4f}"
            )
            writer.add_scalar("eval/mae", metrics.mae, epoch + 1)
            writer.add_scalar("eval/vmae", metrics.vmae, epoch + 1)
            writer.add_scalar("eval/yaw", metrics.yaw, epoch + 1)
            writer.add_scalar("eval/pitch", metrics.pitch, epoch + 1)
            writer.add_scalar("eval/roll", metrics.roll, epoch + 1)

        save_checkpoint(checkpoint_dir / "last.pth", model, optimizer, scaler, epoch)
        is_last = epoch + 1 == args.epochs
        if (epoch + 1) % args.save_every == 0 or is_last:
            save_checkpoint(
                checkpoint_dir / f"snapshot_{epoch + 1:03d}.pth",
                model,
                optimizer,
                scaler,
                epoch,
            )

    writer.close()


if __name__ == "__main__":
    main()
