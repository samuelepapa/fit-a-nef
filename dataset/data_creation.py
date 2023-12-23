import os
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Sequence, Tuple, Union

import torchvision
from absl import logging
from ml_collections import ConfigDict
from torch.utils import data

# image datasets
from dataset.image_dataset.CelebA import CelebA
from dataset.image_dataset.image_data import load_images
from dataset.image_dataset.ImageNet import ImageNetKaggle
from dataset.image_dataset.MicroImageNet import MicroImageNet
from dataset.image_dataset.TinyImageNet import TinyImageNet
from dataset.image_dataset.utils import (
    MEAN_STD_IMAGE_DATASETS,
    fast_normalize,
    image_to_numpy,
)

# shape datasets
from dataset.shape_dataset.shape_data import load_shapes
from dataset.shape_dataset.shapenet import ShapeNet
from dataset.shape_dataset.shapenet_val import ShapeNetVal


def get_dataset(cfg_dataset: Dict[str, Any]) -> data.Dataset:
    """Create a dataset from the configuration. The default root path for the dataset is obtained
    from the DATA_PATH environment variable. If the variable is not set, the default value is
    "data".

    Args:
        cfg_dataset (Dict[str, Any]): The dataset configuration.

    Returns:
        The dataset.
    """

    available_datasets = [
        "CIFAR10",
        "MNIST",
        "CelebA",
        "ImageNet",
        "TinyImageNet",
        "STL10",
    ]

    if "DATA_PATH" not in os.environ:
        data_path = Path("data").absolute()
        data_path.mkdir(parents=True, exist_ok=True)
        logging.warning(f"DATA_PATH environment variable not set, using default value {data_path}")
        DATA = data_path
    else:
        DATA = Path(os.environ["DATA_PATH"])

    if cfg_dataset.name == "CIFAR10":
        mean, std = MEAN_STD_IMAGE_DATASETS["CIFAR10"]

        normalize_fn = lambda x: fast_normalize(x, mean, std)

        train_dataset = torchvision.datasets.CIFAR10(
            root=DATA / Path(cfg_dataset.path),
            train=True,
            transform=torchvision.transforms.Compose([image_to_numpy, normalize_fn]),
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=DATA / Path(cfg_dataset.path),
            train=False,
            transform=torchvision.transforms.Compose([image_to_numpy, normalize_fn]),
            download=True,
        )
        source_dataset = data.ConcatDataset([train_dataset, test_dataset])

    elif cfg_dataset.name == "MNIST":
        mean, std = MEAN_STD_IMAGE_DATASETS["MNIST"]

        normalize_fn = lambda x: fast_normalize(x, mean, std)

        train_dataset = torchvision.datasets.MNIST(
            root=DATA / Path(cfg_dataset.path),
            train=True,
            transform=torchvision.transforms.Compose(
                [image_to_numpy, lambda x: x.reshape(28, 28, 1), normalize_fn]
            ),
            download=True,
        )
        test_dataset = torchvision.datasets.MNIST(
            root=DATA / Path(cfg_dataset.path),
            train=False,
            transform=torchvision.transforms.Compose(
                [image_to_numpy, lambda x: x.reshape(28, 28, 1), normalize_fn]
            ),
            download=True,
        )
        source_dataset = data.ConcatDataset([train_dataset, test_dataset])

    elif cfg_dataset.name == "STL10":
        mean, std = MEAN_STD_IMAGE_DATASETS["STL10"]

        normalize_fn = lambda x: fast_normalize(x, mean, std)

        train_dataset = torchvision.datasets.STL10(
            root=DATA / Path(cfg_dataset.path),
            split="train",
            transform=torchvision.transforms.Compose([image_to_numpy, normalize_fn]),
            download=True,
        )

        test_dataset = torchvision.datasets.STL10(
            root=DATA / Path(cfg_dataset.path),
            split="test",
            transform=torchvision.transforms.Compose([image_to_numpy, normalize_fn]),
            download=True,
        )

        source_dataset = data.ConcatDataset([train_dataset, test_dataset])

    elif cfg_dataset.name == "CelebA":
        crop_h = cfg_dataset.get("crop_h", 128)
        crop_w = cfg_dataset.get("crop_w", None)
        resize_w = cfg_dataset.get("resize_w", 128)
        source_dataset = CelebA(
            DATA / Path(cfg_dataset.path), crop_h=crop_h, crop_w=crop_w, resize_w=resize_w
        )

    elif cfg_dataset.name == "ImageNet":
        crop_h = cfg_dataset.get("crop_h", 256)
        crop_w = cfg_dataset.get("crop_w", None)
        resize_w = cfg_dataset.get("resize_w", 256)
        source_dataset = ImageNetKaggle(
            DATA / Path(cfg_dataset.path), crop_h=crop_h, crop_w=crop_w, resize_w=resize_w
        )
    elif cfg_dataset.name == "MicroImageNet":
        crop_h = cfg_dataset.get("crop_h", 256)
        crop_w = cfg_dataset.get("crop_w", None)
        resize_w = cfg_dataset.get("resize_w", 64)
        source_dataset = MicroImageNet(
            DATA / Path(cfg_dataset.path),
            crop_h=crop_h,
            crop_w=crop_w,
            resize_w=resize_w,
            num_selected=6000,
            seed=42,
        )
    elif cfg_dataset.name == "TinyImageNet":
        train_dataset = TinyImageNet(
            root=str(DATA / cfg_dataset.path), split="train", download=True, seed=42
        )
        val_dataset = TinyImageNet(
            root=str(DATA / cfg_dataset.path), split="val", download=True, seed=42
        )

        source_dataset = data.ConcatDataset([train_dataset, val_dataset])
    elif cfg_dataset.name == "ShapeNet":
        train_dataset = ShapeNet(root=str(DATA / cfg_dataset.path), num_points=(1000, 1000))

        source_dataset = train_dataset
    elif cfg_dataset.name == "ShapeNetVal":
        train_dataset = ShapeNetVal(root=str(DATA / cfg_dataset.path), num_points=(1000, 1000))

        source_dataset = train_dataset
    else:
        raise ValueError(
            f"Dataset name must be one of {available_datasets} and is now {cfg_dataset.name}"
        )

    return source_dataset


def load_data(
    source_dataset: data.Dataset, cfg: ConfigDict, start_idx: int, end_idx: int, **kwargs
):
    if cfg.task == "image":
        return load_images(
            source_dataset=source_dataset, start_idx=start_idx, end_idx=end_idx, **kwargs
        )
    elif cfg.task == "shape":
        return load_shapes(
            source_dataset=source_dataset,
            start_idx=start_idx,
            end_idx=end_idx,
            **kwargs,
        )
