import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data as data
import torchvision
from absl import logging
from PIL import Image

from dataset.image_dataset.utils import (
    MEAN_STD_IMAGE_DATASETS,
    center_crop,
    fast_normalize,
    image_to_numpy,
)


class CelebA(data.Dataset):
    def __init__(self, data_path, crop_h=128, crop_w=None, resize_w=128):
        # https://github.com/podgorskiy/ALAE/blob/master/dataset_preparation/prepare_celeba.py
        self.corrupted = [
            "195995.jpg",
            "131065.jpg",
            "118355.jpg",
            "080480.jpg",
            "039459.jpg",
            "153323.jpg",
            "011793.jpg",
            "156817.jpg",
            "121050.jpg",
            "198603.jpg",
            "041897.jpg",
            "131899.jpg",
            "048286.jpg",
            "179577.jpg",
            "024184.jpg",
            "016530.jpg",
        ]

        mean, std = MEAN_STD_IMAGE_DATASETS["CelebA"]

        self.normalize_fn = lambda x: fast_normalize(x, mean, std)

        archive = zipfile.ZipFile(data_path / Path("Img/img_align_celeba.zip"), "r")
        annotations_folder = data_path / Path("Anno")
        self.modes = self._load_csv(data_path / Path("Eval/list_eval_partition.txt"))
        self.identity = self._load_csv(annotations_folder / Path("identity_CelebA.txt"))
        self.bbox = self._load_csv(annotations_folder / Path("list_bbox_celeba.txt"), header=1)
        self.landmarks_align = self._load_csv(
            (annotations_folder / "list_landmarks_align_celeba.txt"), header=1
        )
        self.attr = self._load_csv(annotations_folder / Path("list_attr_celeba.txt"), header=1)

        names = archive.namelist()

        names = [x for x in names if x[-4:] == ".jpg"]

        names = [x for x in names if x[-10:] not in self.corrupted]

        self.archive = archive
        self.names = names
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.resize_w = resize_w

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ):
        with open(filename) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        attributes = {row[0]: row[1:] for row in data if row[0][-10:] not in self.corrupted}

        return headers, attributes

    def __getitem__(self, index):
        name = self.names[index]
        img = self.archive.open(name)
        img = self.normalize_fn(
            center_crop(
                image_to_numpy(Image.open(img)), self.crop_h, self.crop_w, self.resize_w, offset=15
            )
        )
        return img[None, ...], None

    def __len__(self):
        return len(self.names)
