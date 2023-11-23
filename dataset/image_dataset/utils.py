import jax
import jax.numpy as jnp
import numpy as np


def image_to_numpy(image):
    return np.array(image) / 255


def fast_normalize(image, mean, std):
    image = (image - mean) / std
    return image.astype(np.float32)


def center_crop(x, crop_h=128, crop_w=None, resize_w=128, offset=15, inter_mode="bilinear"):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h  # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.0)) + offset
    i = int(round((w - crop_w) / 2.0))
    return jax.image.resize(x[j : j + crop_h, i : i + crop_w], [resize_w, resize_w, 3], inter_mode)


MEAN_STD_IMAGE_DATASETS = {
    "MNIST": [np.array([0.13066047]), np.array([0.30810782])],
    "CIFAR10": [
        np.array(
            [
                0.49186882,
                0.48265392,
                0.44717726,
            ]
        ),
        np.array(
            [
                0.24697122,
                0.24338895,
                0.2615926,
            ]
        ),
    ],
    "CelebA": [
        np.array([0.5478978610762312, 0.42231354296134166, 0.3591953807599443]),
        np.array([0.2892034399873768, 0.2524239196548999, 0.24213944150717998]),
    ],
    "ImageNet": [
        np.array([0.4825423682551735, 0.43886259889903234, 0.3864390052222265]),
        np.array([0.2763444786550072, 0.26488767608904246, 0.2720352727653687]),
    ],
    "TinyImageNet": [
        np.array([0.4804297956553372, 0.44819699905135413, 0.39755623178048566]),
        np.array([0.27643974569425867, 0.26888658533363985, 0.2816685219463138]),
    ],
    "STL10": [
        np.array([0.4470306, 0.43970686, 0.40560696]),
        np.array([0.26047868, 0.25662908, 0.27047333]),
    ],
    "MicroImageNet": [
        np.array([0.4643375754356384, 0.4339415729045868, 0.38213176727294923]),
        np.array([0.18949764454628706, 0.18707174788855566, 0.21445986703967326]),
    ],
}


def normalize_image(image: jnp.ndarray, dataset_name: str) -> jnp.ndarray:
    mean, std = MEAN_STD_IMAGE_DATASETS[dataset_name]

    return (image - mean) / std


def unnormalize_image(image: jnp.ndarray, dataset_name: str) -> jnp.ndarray:
    mean, std = MEAN_STD_IMAGE_DATASETS[dataset_name]

    return image * std + mean
