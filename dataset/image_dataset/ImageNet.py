import json
import os

import numpy as np
import torch.utils.data as data
from PIL import Image

from dataset.image_dataset.utils import (
    MEAN_STD_IMAGE_DATASETS,
    center_crop,
    fast_normalize,
    image_to_numpy,
)


class ImageNetKaggle(data.Dataset):
    """
    To download ImageNet do the following:
    1. Register at https://www.kaggle.com/
    2. Install the kaggle API:
        pip install kaggle
    3. Go to your account page (the little face in the top right)
    4. Select 'Create API Token'. This will download kaggle.json
    5. Move kaggle.json to ~/.kaggle/
    6. Run the following command:
        kaggle competitions download -c imagenet-object-localization-challenge
    7. Unzip the downloaded files and move them to the desired location
    8. Run the following commands:
        wget https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json
        wget https://gist.githubusercontent.com/paulgavrikov/3af1efe6f3dff63f47d48b91bb1bca6b/raw/00bad6903b5e4f84c7796b982b72e2e617e5fde1/ILSVRC2012_val_labels.json
    9. Move the downloaded json files to the same directory as the downloaded images
    """

    def __init__(self, root, crop_h=256, crop_w=None, resize_w=256):
        # from: https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.resize_w = resize_w
        self.samples = []
        self.targets = []
        self.train_size = 0
        self.val_size = 0
        self.syn_to_class = {}
        self.mean, self.std = MEAN_STD_IMAGE_DATASETS["ImageNet"]
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)

        for split in ["train", "val"]:
            samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
            for entry in os.listdir(samples_dir):
                if split == "train":
                    syn_id = entry
                    target = self.syn_to_class[syn_id]
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
                        self.train_size += 1
                elif split == "val":
                    syn_id = self.val_to_syn[entry]
                    target = self.syn_to_class[syn_id]
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)
                    self.targets.append(target)
                    self.val_size += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = fast_normalize(
            center_crop(
                image_to_numpy(Image.open(self.samples[idx]).convert("RGB")),
                self.crop_h,
                self.crop_w,
                self.resize_w,
                offset=0,
            ),
            self.mean,
            self.std,
        )
        return x[None, ...], self.targets[idx]
