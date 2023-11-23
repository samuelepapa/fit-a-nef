import argparse

import numpy as np
from absl import app, flags, logging
from ml_collections import ConfigDict

from dataset.data_creation import get_dataset
from dataset.image_dataset import load_images

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "ImageNet", "Dataset to use")
flags.DEFINE_string("path", ".", "Path to dataset")
flags.DEFINE_integer("out_channels", 3, "Number of output channels")
flags.DEFINE_integer("start_idx", 0, "Start index")
flags.DEFINE_integer("end_idx", 60000, "End index")
flags.DEFINE_integer("split_size", 10000, "Split size")


def main(_):
    dataset_cfg = {
        "name": FLAGS.dataset,
        "path": FLAGS.path,
        "out_channels": FLAGS.out_channels,
    }
    start_idx = FLAGS.start_idx
    end_idx = FLAGS.end_idx

    cfg_dataset = ConfigDict(dataset_cfg)
    source_dataset = get_dataset(cfg_dataset)

    if end_idx == -1:
        splits = np.arange(0, len(source_dataset), FLAGS.split_size)
        if splits[-1] != len(source_dataset):
            splits = np.append(splits, len(source_dataset))

        logging.info(f"Splits: {splits}")
        end_idxs = splits[1:]
        start_idxs = splits[:-1]

        means = []
        mean_squares = []
        sizes = []

        for start_idx, end_idx in zip(start_idxs, end_idxs):
            logging.info(f"Loading images from {start_idx} to {end_idx}")
            coords, images, images_shape, rng = load_images(source_dataset, start_idx, end_idx)
            means.append(np.mean(images, axis=(0, 1)))
            mean_squares.append(np.mean(images**2, axis=(0, 1)))
            sizes.append(images.shape[0])
            del images
            del coords
        logging.info("Computing mean and std")
        mean = np.average(means, axis=0, weights=sizes)
        mean_square = np.average(mean_squares, axis=0, weights=sizes)
        std = np.sqrt(mean_square - mean**2)

    else:
        # load at maximum 10k samples at a time
        cur_start_idx = start_idx
        cur_end_idx = min(start_idx + 10000, end_idx)
        means = []
        squared_means = []
        sizes = []
        while cur_end_idx < end_idx:
            print(f"Loading images from {cur_start_idx} to {cur_end_idx}")
            coords, images, images_shape, rng, index_perm = load_images(
                source_dataset, cur_start_idx, cur_end_idx
            )
            mean = np.mean(images, axis=(0, 1))
            means.append(mean)
            squared_means.append(np.mean(np.power(images, 2), axis=(0, 1)))
            sizes.append(images.shape[0])
            cur_start_idx = cur_end_idx
            cur_end_idx = min(cur_end_idx + 10000, end_idx)
            del images
            del coords

        mean = np.average(means, axis=0, weights=sizes)
        std = np.sqrt(np.average(squared_means, axis=0, weights=sizes) - mean**2)

    def to_array_str(x):
        return "[" + ",".join([str(y) for y in x]) + "]"

    logging.info(f"Mean: {to_array_str(mean)}")
    logging.info(f"Standard deviation: {to_array_str(std)}")


if __name__ == "__main__":
    app.run(main)
