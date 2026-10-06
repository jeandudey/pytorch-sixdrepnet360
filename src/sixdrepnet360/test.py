# SPDX-FileCopyrightText: 2023 Thorsten Hempel
#
# SPDX-License-Identifier: MIT

import argparse
import os

import cv2
import numpy as np
import torch
import torchvision
from torch.backends import cudnn
from torch.hub import load_state_dict_from_url
from torchvision import transforms

from sixdrepnet360 import datasets, utils
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
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    print("Loading data.")

    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pose_dataset = datasets.getDataset(
        args.dataset,
        args.data_dir,
        args.filename_list,
        transformations,
        train_mode=False,
    )
    test_loader = torch.utils.data.DataLoader(
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

    model.cuda(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    total = 0
    yaw_error = pitch_error = roll_error = 0.0
    v1_err = v2_err = v3_err = 0.0

    with torch.no_grad():
        for _i, (images, r_label, cont_labels, name) in enumerate(test_loader):
            images = torch.Tensor(images).cuda(gpu)
            total += cont_labels.size(0)

            # gt matrix
            R_gt = r_label

            # gt euler
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
            v1_err += torch.sum(
                torch.acos(torch.clamp(torch.sum(R_gt[:, 0] * R_pred[:, 0], 1), -1, 1))
                * 180
                / np.pi
            )
            v2_err += torch.sum(
                torch.acos(torch.clamp(torch.sum(R_gt[:, 1] * R_pred[:, 1], 1), -1, 1))
                * 180
                / np.pi
            )
            v3_err += torch.sum(
                torch.acos(torch.clamp(torch.sum(R_gt[:, 2] * R_pred[:, 2], 1), -1, 1))
                * 180
                / np.pi
            )

            pitch_error += torch.sum(
                torch.min(
                    torch.stack(
                        (
                            torch.abs(p_gt_deg - p_pred_deg),
                            torch.abs(p_pred_deg + 360 - p_gt_deg),
                            torch.abs(p_pred_deg - 360 - p_gt_deg),
                        )
                    ),
                    0,
                )[0]
            )
            yaw_error += torch.sum(
                torch.min(
                    torch.stack(
                        (
                            torch.abs(y_gt_deg - y_pred_deg),
                            torch.abs(y_pred_deg + 360 - y_gt_deg),
                            torch.abs(y_pred_deg - 360 - y_gt_deg),
                        )
                    ),
                    0,
                )[0]
            )
            roll_error += torch.sum(
                torch.min(
                    torch.stack(
                        (
                            torch.abs(r_gt_deg - r_pred_deg),
                            torch.abs(r_pred_deg + 360 - r_gt_deg),
                            torch.abs(r_pred_deg - 360 - r_gt_deg),
                        )
                    ),
                    0,
                )[0]
            )

            if args.show_viz:
                name = name[0]
                if args.dataset == "Panoptic":
                    cv2_img = cv2.imread(
                        os.path.join(args.data_dir, name.split(",")[0])
                    )

                elif args.dataset == "AFLW2000":
                    cv2_img = cv2.imread(os.path.join(args.data_dir, name + ".jpg"))

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
                cv2.imwrite(os.path.join("output/img/", name + ".png"), cv2_img)

        mae = (yaw_error + pitch_error + roll_error) / (total * 3)
        print(
            f"Yaw: {yaw_error / total:.4f}, Pitch: {pitch_error / total:.4f}, "
            f"Roll: {roll_error / total:.4f}, MAE: {mae:.4f}"
        )

        vmae = (v1_err + v2_err + v3_err) / (total * 3)
        print(
            f"Vec1: {v1_err / total:.4f}, Vec2: {v2_err / total:.4f}, "
            f"Vec3: {v3_err / total:.4f}, VMAE: {vmae:.4f}"
        )


if __name__ == "__main__":
    main()
