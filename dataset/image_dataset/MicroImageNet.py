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


def collect_ids_subtree(subtree, ids=None):
    if ids is None:
        ids = []

    if "children" in subtree:
        for child in subtree["children"]:
            ids.append(child["id"])
            collect_ids_subtree(child, ids)

    if "id" in subtree:
        ids.append(subtree["id"])

    return ids


class MicroImageNet(data.Dataset):
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

    def __init__(self, root, crop_h=256, crop_w=None, resize_w=256, seed=42, num_selected=6000):
        # from: https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.resize_w = resize_w
        self.samples = []
        self.targets = []
        self.train_size = 0
        self.val_size = 0
        self.syn_to_class = {}

        with open(os.path.join(root, "imagenet.json")) as f:
            metadata = json.load(f)

        self.mean, self.std = MEAN_STD_IMAGE_DATASETS["MicroImageNet"]
        organism = metadata["children"][20]
        animal = organism["children"][3]
        artifact = metadata["children"][19]
        structure = artifact["children"][1]
        wheeled_vehicle = artifact["children"][0]["children"][5]["children"][0]["children"][4]

        amphibian = organism["children"][3]["children"][3]["children"][0]["children"][3]
        electronic_equipment = artifact["children"][0]["children"][3]["children"][4]
        birds = animal["children"][3]["children"][0]["children"][1]
        primates = animal["children"][3]["children"][0]["children"][0]["children"][0]["children"][
            1
        ]
        dogs = animal["children"][1]["children"][1]
        cats = animal["children"][1]["children"][0]
        houses = structure["children"][10]
        fungus = organism["children"][1]
        trucks = wheeled_vehicle["children"][4]["children"][2]["children"][6]
        cars = wheeled_vehicle["children"][4]["children"][2]["children"][1]

        selected_classes = {
            "amphibian": collect_ids_subtree(amphibian),
            "electronic_equipment": collect_ids_subtree(electronic_equipment),
            "birds": collect_ids_subtree(birds),
            "primates": collect_ids_subtree(primates),
            "dogs": collect_ids_subtree(dogs),
            "cats": collect_ids_subtree(cats),
            "houses": collect_ids_subtree(houses),
            "fungus": collect_ids_subtree(fungus),
            "trucks": collect_ids_subtree(trucks),
            "cars": collect_ids_subtree(cars),
        }

        samples = []
        targets = []
        train_size = 0
        val_size = 0
        syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            val_to_syn = json.load(f)

        concat_classes = []
        for classes in selected_classes.values():
            for syn in classes:
                if syn in syn_to_class:
                    concat_classes.append(syn)

        class_syn_to_id = {}
        for i, (class_name, syn_list) in enumerate(selected_classes.items()):
            for syn in syn_list:
                if syn in syn_to_class:
                    class_syn_to_id[syn] = i
        class_samples = {}
        for i in range(len(selected_classes)):
            class_samples[i] = []

        for split in ["train", "val"]:
            samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
            for entry in os.listdir(samples_dir):
                if entry in concat_classes:
                    if split == "train":
                        syn_id = entry
                        target = class_syn_to_id[entry]  # syn_to_class[syn_id]
                        syn_folder = os.path.join(samples_dir, syn_id)
                        for sample in os.listdir(syn_folder):
                            sample_path = os.path.join(syn_folder, sample)
                            samples.append(sample_path)
                            targets.append(target)
                            train_size += 1
                            class_samples[target].append(sample_path)
                    elif split == "val":
                        syn_id = val_to_syn[entry]
                        target = class_syn_to_id[entry]  # syn_to_class[syn_id]
                        sample_path = os.path.join(samples_dir, entry)
                        samples.append(sample_path)
                        targets.append(target)
                        class_samples[target].append(sample_path)
                        val_size += 1

        # select num_selected samples from each class uniformly at random
        rng = np.random.default_rng(seed)
        selected_samples = []
        selected_targets = []
        for target, samples in class_samples.items():
            cur_selected_samples = rng.choice(samples, size=num_selected, replace=False)
            selected_samples.extend(cur_selected_samples)
            selected_targets.extend([target] * num_selected)

        train_size = len(selected_samples)
        val_size = 0
        permuted_idxs = rng.permutation(train_size)
        selected_samples = [selected_samples[idx] for idx in permuted_idxs]
        selected_targets = [selected_targets[idx] for idx in permuted_idxs]
        self.samples = selected_samples
        self.targets = selected_targets

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
