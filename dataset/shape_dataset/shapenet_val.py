import logging
import os
from typing import Tuple

import numpy as np
import yaml
from torch.utils import data

logger = logging.getLogger(__name__)

category_list = [
    ("04379243", "table"),
    ("03593526", "jar"),
    ("04225987", "skateboard"),
    ("02958343", "car"),
    ("02876657", "bottle"),
    ("04460130", "tower"),
    ("03001627", "chair"),
    ("02871439", "bookshelf"),
    ("02942699", "camera"),
    ("02691156", "airplane"),
    ("03642806", "laptop"),
    ("02801938", "basket"),
    ("04256520", "sofa"),
    ("03624134", "knife"),
    ("02946921", "can"),
    ("04090263", "rifle"),
    ("04468005", "train"),
    ("03938244", "pillow"),
    ("03636649", "lamp"),
    ("02747177", "trash bin"),
    ("03710193", "mailbox"),
    ("04530566", "watercraft"),
    ("03790512", "motorbike"),
    ("03207941", "dishwasher"),
    ("02828884", "bench"),
    ("03948459", "pistol"),
    ("04099429", "rocket"),
    ("03691459", "loudspeaker"),
    ("03337140", "file cabinet"),
    ("02773838", "bag"),
    ("02933112", "cabinet"),
    ("02818832", "bed"),
    ("02843684", "birdhouse"),
    ("03211117", "display"),
    ("03928116", "piano"),
    ("03261776", "earphone"),
    ("04401088", "telephone"),
    ("04330267", "stove"),
    ("03759954", "microphone"),
    ("02924116", "bus"),
    ("03797390", "mug"),
    ("04074963", "remote"),
    ("02808440", "bathtub"),
    ("02880940", "bowl"),
    ("03085013", "keyboard"),
    ("03467517", "guitar"),
    ("04554684", "washer"),
    ("02834778", "bicycle"),
    ("03325088", "faucet"),
    ("04004475", "printer"),
    ("02954340", "cap"),
    # We added these manually
    ("02992529", "cellphone"),
    ("03046257", "clock"),
    ("03513137", "helmet"),
    ("03761084", "microwave"),
    ("03991062", "flowerpot"),
]


class ShapeNetVal(data.Dataset):
    def __init__(
        self,
        root: str,
        num_points: Tuple[int, int] = (1024, 1024),
        download: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            root (str): dataset root, place the 'ShapeNet' folder in this directory
            split (str): which split is used, ['train', 'test', 'val']
            num_points, tuple of ints: how many points are subsampled inside and outside each object
        """
        # Attributes
        self.root = root + "/ShapeNetCoreResampledValidation"
        self.num_points = num_points

        # If ShapeNet dir doesn't exist or is empty, download dataset
        if (not os.path.exists(self.root) or not len(os.listdir(self.root))) and download:
            raise NotImplementedError("Downloading ShapeNet is not implemented yet.")

        # Get ShapeNet categories
        categories = os.listdir(self.root)
        categories = [c for c in categories if os.path.isdir(os.path.join(self.root, c))]

        # selected_categories = [
        #     "04379243",
        #     "03593526",
        #     "04225987",
        #     "02958343",
        #     "02876657",
        #     "04460130",
        #     "03001627",
        #     "02871439",
        #     "02942699",
        #     "02691156",
        # ]

        selected_categories = [
            "03691459",  # 8436
            "02828884",  # 6778
            "04530566",  # 4045
            "03636649",  # 3514
            "04090263",  # 3173
            "04256520",  # 2373
            "02958343",  # 2318
            "02691156",  # 1939
            "03001627",  # 1813
            "04379243",  # 1597
        ]

        # Counts for each class
        selected_counts = [8436, 6778, 4045, 3514, 3173, 2373, 2318, 1939, 1813, 1597]

        categories = selected_categories

        # Construct metadata
        self.metadata = {c: {"id": c, "name": "n/a"} for c in categories}

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]["idx"] = c_idx

        # Get all models
        self.models = []
        idx = 0
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(self.root, c)
            if not os.path.isdir(subpath):
                logger.warning("Category %s does not exist in dataset." % c)

            models_c = sorted(os.listdir(subpath))
            for m in models_c:
                if m == "a76f63c6b3702a4981c9b20aad15512.npz" and c == "03691459":
                    continue
                if m == "a5d21835219c8fed19fb4103277a6b93.npz" and c == "03001627":
                    continue
                # if idx == 24934:
                #     idx += 1
                #     continue
                self.models.append(
                    {
                        "category": c,
                        "model": m,
                    }
                )
                idx += 1

        # shuffle the samples
        rng = np.random.default_rng(seed)
        rng.shuffle(self.models)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        file_path = os.path.join(
            self.root, self.models[idx]["category"], self.models[idx]["model"]
        )

        # Get label
        label = self.metadata[self.models[idx]["category"]]["idx"]

        # Load points and occupancy values
        points_dict = np.load(file_path)

        random_points = points_dict["random_points"]
        random_points_near_surface = points_dict["near_surf_points"]

        points = np.concatenate(
            [random_points, random_points_near_surface], axis=0, dtype=np.float32
        )

        random_points_occupancy = points_dict["random_points_occupancy"]
        random_points_near_surface_occupancy = points_dict["near_surf_points_occupancy"]

        occupancies = np.concatenate(
            [random_points_occupancy, random_points_near_surface_occupancy],
            axis=0,
            dtype=np.float32,
        )

        if np.isnan(points).any():
            print("nan: ", self.root, self.models[idx]["category"], self.models[idx]["model"])

        return points, occupancies, label


if __name__ == "__main__":
    train_dataset = ShapeNetVal(
        root="/media/davidknigge/hard-disk1/storage/", num_points=(1024, 1024)
    )
