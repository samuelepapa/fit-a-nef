import json
import os

import numpy as np
from absl import app, flags, logging

from dataset.image_dataset.ImageNet import ImageNetKaggle

FLAGS = flags.FLAGS

flags.DEFINE_string("path", default=None, help="Path to the dataset.")
flags.DEFINE_string("storage", default=None, help="Path to the storage folder.")


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


def main(_):
    root = FLAGS.path

    imagenet_dataset = ImageNetKaggle(root=root)

    with open("dataset/image_dataset/imagenet.json") as f:
        metadata = json.load(f)

    organism = metadata["children"][20]
    animal = organism["children"][3]
    artifact = metadata["children"][19]
    structure = artifact["children"][1]
    wheeled_vehicle = artifact["children"][0]["children"][5]["children"][0]["children"][4]

    amphibian = organism["children"][3]["children"][3]["children"][0]["children"][3]
    electronic_equipment = artifact["children"][0]["children"][3]["children"][4]
    birds = animal["children"][3]["children"][0]["children"][1]
    primates = animal["children"][3]["children"][0]["children"][0]["children"][0]["children"][1]
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

    class_samples = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
    }

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

    # select 1000 samples from each class uniformly at random
    num_selected = 1000
    seed = 42
    rng = np.random.default_rng(seed)
    selected_samples = []
    selected_targets = []
    for target, samples in class_samples.items():
        print(target, len(samples))
        cur_selected_samples = rng.choice(samples, size=num_selected, replace=False)
        selected_samples.extend(cur_selected_samples)
        selected_targets.extend([target] * num_selected)

    train_size = len(selected_samples)
    val_size = 0

    print(imagenet_dataset[0])


if __name__ == "__main__":
    app.run(main)
