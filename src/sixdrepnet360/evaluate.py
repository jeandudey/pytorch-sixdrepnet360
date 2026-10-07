# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ
#
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from sixdrepnet360 import utils
from sixdrepnet360.datasets import Sample, Transform
from sixdrepnet360.model import SixDRepNet360

# Called once per batch with (names, yaw, pitch, roll) predictions in degrees.
BatchCallback = Callable[
    [Sequence[str | NDArray[np.uint8]], torch.Tensor, torch.Tensor, torch.Tensor],
    None,
]


@dataclass(frozen=True)
class Metrics:
    """Mean absolute errors in degrees, plus rotation-vector errors."""

    yaw: float
    pitch: float
    roll: float
    mae: float
    vec1: float
    vec2: float
    vec3: float
    vmae: float


def make_eval_transform() -> Transform:
    """Deterministic test-time transform: resize, center crop, normalize."""
    compose = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def transform(img: Image.Image) -> torch.Tensor:
        return cast(torch.Tensor, compose(img))

    return transform


def _min_angle_error(gt_deg: torch.Tensor, pred_deg: torch.Tensor) -> torch.Tensor:
    # Euler angles wrap around 360 degrees, so take the smallest of the three offsets.
    return torch.sum(
        torch.min(
            torch.stack(
                (
                    torch.abs(gt_deg - pred_deg),
                    torch.abs(pred_deg + 360 - gt_deg),
                    torch.abs(pred_deg - 360 - gt_deg),
                )
            ),
            0,
        )[0]
    )


def _vector_error(R_gt: torch.Tensor, R_pred: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.sum(
        torch.acos(torch.clamp(torch.sum(R_gt[:, axis] * R_pred[:, axis], 1), -1, 1))
        * 180
        / np.pi
    )


def evaluate(
    model: SixDRepNet360,
    loader: DataLoader[Sample],
    device: torch.device,
    on_batch: BatchCallback | None = None,
) -> Metrics:
    """Run the model over a loader and return mean errors.

    Leaves the model in eval mode.
    """
    model.eval()
    total = 0
    yaw_error = pitch_error = roll_error = 0.0
    v1_err = v2_err = v3_err = 0.0

    with torch.no_grad():
        for images, r_label, cont_labels, names in loader:
            images = images.to(device)
            total += cont_labels.size(0)

            # gt matrix and gt euler (labels are in radians)
            R_gt = r_label
            y_gt_deg = cont_labels[:, 0].float() * 180 / np.pi
            p_gt_deg = cont_labels[:, 1].float() * 180 / np.pi
            r_gt_deg = cont_labels[:, 2].float() * 180 / np.pi

            R_pred = model(images)
            euler = (
                utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            )
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            R_pred = R_pred.cpu()
            v1_err += _vector_error(R_gt, R_pred, 0).item()
            v2_err += _vector_error(R_gt, R_pred, 1).item()
            v3_err += _vector_error(R_gt, R_pred, 2).item()

            pitch_error += _min_angle_error(p_gt_deg, p_pred_deg).item()
            yaw_error += _min_angle_error(y_gt_deg, y_pred_deg).item()
            roll_error += _min_angle_error(r_gt_deg, r_pred_deg).item()

            if on_batch is not None:
                on_batch(names, y_pred_deg, p_pred_deg, r_pred_deg)

    if total == 0:
        raise ValueError("No samples found in the dataset.")

    yaw = yaw_error / total
    pitch = pitch_error / total
    roll = roll_error / total
    vec1 = v1_err / total
    vec2 = v2_err / total
    vec3 = v3_err / total
    return Metrics(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        mae=(yaw + pitch + roll) / 3,
        vec1=vec1,
        vec2=vec2,
        vec3=vec3,
        vmae=(vec1 + vec2 + vec3) / 3,
    )
