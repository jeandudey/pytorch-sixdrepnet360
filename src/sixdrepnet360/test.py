# SPDX-FileCopyrightText: 2023 Thorsten Hempel
#
# SPDX-License-Identifier: MIT

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
import torchvision
from numpy.typing import NDArray
from torch.backends import cudnn
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader

from sixdrepnet360 import datasets, utils
from sixdrepnet360.evaluate import evaluate, make_eval_transform
from sixdrepnet360.model import SixDRepNet360

# matplotlib.use("gtk")


def parse_args() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Head pose estimation using the Hopenet network."
    )
    parser.add_argument(
        "--gpu", dest="gpu_id", help="GPU device id to use [0]", default=0, type=int
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Directory path for data.",
        default="datasets/AFLW2000",
        type=str,
    )
    parser.add_argument(
        "--filename_list",
        dest="filename_list",
        help="Path to text file containing relative paths for every example.",
        default="datasets/AFLW2000/files.txt",
        type=str,
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot",
        help="Name of model snapshot.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=80, type=int
    )
    parser.add_argument(
        "--show_viz",
        dest="show_viz",
        help="Save images with pose cube.",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--dataset", dest="dataset", help="Dataset type.", default="AFLW2000", type=str
    )  # Panoptic

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cudnn.enabled = True
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    snapshot_path = args.snapshot
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    print("Loading data.")

    pose_dataset = datasets.getDataset(
        args.dataset,
        args.data_dir,
        args.filename_list,
        make_eval_transform(),
        train_mode=False,
    )
    test_loader = DataLoader(
        dataset=pose_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
    )

    # Load snapshot
    if snapshot_path == "":
        saved_state_dict = load_state_dict_from_url(
            "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
        )
    else:
        saved_state_dict = torch.load(snapshot_path)

    if "model_state_dict" in saved_state_dict:
        model.load_state_dict(saved_state_dict["model_state_dict"])
    else:
        model.load_state_dict(saved_state_dict)

    model.to(device)

    def draw(
        names: Sequence[str | NDArray[np.uint8]],
        y_pred_deg: torch.Tensor,
        p_pred_deg: torch.Tensor,
        r_pred_deg: torch.Tensor,
    ) -> None:
        name = names[0]
        if args.dataset == "Panoptic":
            cv2_img = cv2.imread(
                str(Path(args.data_dir) / cast(str, name).split(",")[0])
            )

        elif args.dataset == "AFLW2000":
            cv2_img = cv2.imread(str(Path(args.data_dir) / (cast(str, name) + ".jpg")))

        elif args.dataset == "BIWI":
            vis = np.asarray(name, dtype=np.uint8)
            cv2_img = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        else:
            raise ValueError(f"Visualization not supported for {args.dataset}")

        if cv2_img is None:
            raise ValueError("Failed to load image.")
        cv2_img = cv2_img.astype(np.uint8)
        utils.draw_axis(
            cv2_img,
            y_pred_deg[0],
            p_pred_deg[0],
            r_pred_deg[0],
            tdx=cv2_img.shape[1] / 2,
            tdy=cv2_img.shape[0] / 2,
            size=100,
        )
        # utils.plot_pose_cube(
        #     cv2_img, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], size=200
        # )
        cv2.imshow("Test", cv2_img)
        cv2.waitKey(0)
        cv2.imwrite(str(Path("output/img") / (cast(str, name) + ".png")), cv2_img)

    metrics = evaluate(
        model, test_loader, device, on_batch=draw if args.show_viz else None
    )
    print(
        f"Yaw: {metrics.yaw:.4f}, Pitch: {metrics.pitch:.4f}, "
        f"Roll: {metrics.roll:.4f}, MAE: {metrics.mae:.4f}"
    )
    print(
        f"Vec1: {metrics.vec1:.4f}, Vec2: {metrics.vec2:.4f}, "
        f"Vec3: {metrics.vec3:.4f}, VMAE: {metrics.vmae:.4f}"
    )


if __name__ == "__main__":
    main()
