# SPDX-FileCopyrightText: 2026 Jean-Pierre De Jesus DIAZ
#
# SPDX-License-Identifier: MIT

import argparse

import onnx
import torch
import torchvision
from torch.hub import load_state_dict_from_url

from sixdrepnet360.model import SixDRepNet360

DEFAULT_SNAPSHOT_URL = (
    "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/"
    "6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
)


def parse_args() -> argparse.Namespace:
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Export a 6DRepNet360 snapshot to ONNX."
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot",
        help="Path to model snapshot. Empty downloads the default pretrained model.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output",
        dest="output",
        help="Path of the ONNX file to write.",
        default="sixdrepnet360.onnx",
        type=str,
    )
    parser.add_argument(
        "--opset", dest="opset", help="ONNX opset version.", default=18, type=int
    )

    return parser.parse_args()


def load_model(snapshot_path: str) -> SixDRepNet360:
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
    if snapshot_path == "":
        state_dict = load_state_dict_from_url(DEFAULT_SNAPSHOT_URL, map_location="cpu")
    else:
        state_dict = torch.load(snapshot_path, map_location="cpu")

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    model = load_model(args.snapshot)

    # 224x224 RGB, the same input size used by the test transforms.
    dummy_input = torch.randn(1, 3, 224, 224)
    batch = torch.export.Dim("batch")
    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        input_names=["image"],
        output_names=["rotation_matrix"],
        dynamic_shapes=({0: batch},),
        opset_version=args.opset,
        dynamo=True,
    )
    if onnx_program is None:
        raise RuntimeError("ONNX export did not return a program.")
    # Keep the weights inside the .onnx file instead of a separate .data file.
    onnx_program.save(args.output, external_data=False)

    onnx.checker.check_model(args.output)
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
