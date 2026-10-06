# SPDX-FileCopyrightText: 2023 Thorsten Hempel
#
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageFilter
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from sixdrepnet360 import utils

Transform = Callable[[Image.Image], torch.Tensor]
Sample = tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | NDArray[np.uint8]]

rng = np.random.default_rng()


def get_list_from_filenames(file_path: str) -> list[str]:
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    return Path(file_path).read_text().splitlines()


class AFLW2000(Dataset[Sample]):
    data_dir: str
    transform: Transform
    img_ext: str
    annot_ext: str
    image_mode: str
    X_train: list[str]
    y_train: list[str]
    length: int

    def __init__(
        self,
        data_dir: str,
        filename_path: str,
        transform: Transform,
        img_ext: str = ".jpg",
        annot_ext: str = ".mat",
        image_mode: str = "RGB",
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    @override
    def __getitem__(self, index: int) -> Sample:
        img = Image.open(Path(self.data_dir) / (self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = str(Path(self.data_dir) / (self.y_train[index] + self.annot_ext))

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]  # * 180 / np.pi
        yaw = pose[1]  # * 180 / np.pi
        roll = pose[2]  # * 180 / np.pi

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        img = self.transform(img)

        return img, torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self) -> int:
        # 2,000
        return self.length


class AFLW(Dataset[Sample]):
    data_dir: str
    transform: Transform
    img_ext: str
    annot_ext: str
    image_mode: str
    X_train: list[str]
    y_train: list[str]
    length: int

    def __init__(
        self,
        data_dir: str,
        filename_path: str,
        transform: Transform,
        img_ext: str = ".jpg",
        annot_ext: str = ".txt",
        image_mode: str = "RGB",
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    @override
    def __getitem__(self, index: int) -> Sample:
        img = Image.open(Path(self.data_dir) / (self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = Path(self.data_dir) / (self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        with txt_path.open() as annot:
            line = annot.readline().split(" ")
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self) -> int:
        # train: 18,863
        # test: 1,966
        return self.length


class AFW(Dataset[Sample]):
    data_dir: str
    transform: Transform
    img_ext: str
    annot_ext: str
    image_mode: str
    X_train: list[str]
    y_train: list[str]
    length: int

    def __init__(
        self,
        data_dir: str,
        filename_path: str,
        transform: Transform,
        img_ext: str = ".jpg",
        annot_ext: str = ".txt",
        image_mode: str = "RGB",
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    @override
    def __getitem__(self, index: int) -> Sample:
        txt_path = Path(self.data_dir) / (self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split("_")[0]

        img = Image.open(Path(self.data_dir) / (img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = Path(self.data_dir) / (self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        with txt_path.open() as annot:
            line = annot.readline().split(" ")
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self) -> int:
        # Around 200
        return self.length


class BIWI(Dataset[Sample]):
    data_dir: str
    transform: Transform
    X_train: NDArray[Any]
    y_train: NDArray[Any]
    image_mode: str
    train_mode: bool
    length: int

    def __init__(
        self,
        data_dir: str,
        filename_path: str,
        transform: Transform,
        image_mode: str = "RGB",
        train_mode: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d["image"]
        y_data = d["pose"]
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    @override
    def __getitem__(self, index: int) -> Sample:
        img = Image.fromarray(self.X_train[index].astype(np.uint8))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2] / 180 * np.pi
        yaw = self.y_train[index][0] / 180 * np.pi
        pitch = self.y_train[index][1] / 180 * np.pi
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = rng.random()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = rng.random()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        img = self.transform(img)

        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self) -> int:
        # 15,667
        return self.length


class Pose_300W_LP(Dataset[Sample]):
    # Head pose from 300W-LP dataset
    data_dir: str
    transform: Transform
    img_ext: str
    annot_ext: str
    image_mode: str
    X_train: list[str]
    y_train: list[str]
    length: int

    def __init__(
        self,
        data_dir: str,
        filename_path: str,
        transform: Transform,
        img_ext: str = ".jpg",
        annot_ext: str = ".mat",
        image_mode: str = "RGB",
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    @override
    def __getitem__(self, index: int) -> Sample:
        img = Image.open(Path(self.data_dir) / (self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = str(Path(self.data_dir) / (self.y_train[index] + self.annot_ext))

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = rng.random() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]  # * 180 / np.pi
        yaw = pose[1]  # * 180 / np.pi
        roll = pose[2]  # * 180 / np.pi

        # Gray images

        # Flip?
        rnd = rng.random()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = rng.random()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        # mu, sigma = 0, 0.01
        # noise = np.random.normal(mu, sigma, [3,3])
        # print(noise)

        # Get target tensors
        R = utils.get_R(pitch, yaw, roll)  # + noise

        # labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        img = self.transform(img)

        return img, torch.FloatTensor(R), torch.empty(0), self.X_train[index]

    def __len__(self) -> int:
        # 122,450
        return self.length


def getDataset(
    dataset: str,
    data_dir: str,
    filename_path: str,
    transform: Transform,
    train_mode: bool = True,
) -> Dataset[Sample]:
    if dataset == "Pose_300W_LP":
        pose_dataset = Pose_300W_LP(data_dir, filename_path, transform)
    elif dataset == "AFLW2000":
        pose_dataset = AFLW2000(data_dir, filename_path, transform)
    elif dataset == "BIWI":
        pose_dataset = BIWI(data_dir, filename_path, transform, train_mode=train_mode)
    elif dataset == "AFLW":
        pose_dataset = AFLW(data_dir, filename_path, transform)
    elif dataset == "AFW":
        pose_dataset = AFW(data_dir, filename_path, transform)
    else:
        raise NameError("Error: not a valid dataset name")

    return pose_dataset
