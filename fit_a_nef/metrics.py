from functools import partial
from typing import Callable, Optional

import faiss
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from absl import logging
from sklearn.metrics import normalized_mutual_info_score

# metrics inspired by https://github.com/google-deepmind/dm_pix/blob/master/dm_pix/_src/metrics.py


def kmeans(feats, k, niter=100, gpu=False):
    """Compute K-means clustering on the given features.

    Args:
        feats (np.ndarray): Features to cluster. Shape (N, D).
        K (int): Number of clusters.
        niter (int): Number of iterations.
    Returns:
        (np.ndarray, np.ndarray): Cluster centers and cluster assignments.
    """
    kmeans = faiss.Kmeans(d=feats.shape[1], k=k, niter=niter, gpu=gpu)
    kmeans.train(feats)
    _, assignments = kmeans.index.search(feats, 1)
    return kmeans.centroids, assignments


def nmi(feats, labels, k, niter=100, gpu=False):
    centroids, assignments = kmeans(feats, k=k, niter=niter)
    return normalized_mutual_info_score(labels, assignments[:, 0])


# pdist from https://github.com/google/jax/blob/82b64ede77e50f62211d37ce1ecf468d2ae5962a/jax/scipy/spatial/distance.py


@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def simse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Returns the Scale-Invariant Mean Squared Error between `a` and `b`.

    For each image pair, a scaling factor for `b` is computed as the solution to
    the following problem:

      min_alpha || vec(a) - alpha * vec(b) ||_2^2

    where `a` and `b` are flattened, i.e., vec(x) = np.flatten(x). The MSE between
    the optimally scaled `b` and `a` is returned: mse(a, alpha*b).

    This is a scale-invariant metric, so for example: simse(x, y) == sims(x, y*5).

    This metric was used in "Shape, Illumination, and Reflectance from Shading" by
    Barron and Malik, TPAMI, '15.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      SIMSE between `a` and `b`.
    """

    a_dot_b = (a * b).sum()
    b_dot_b = (b * b).sum()
    alpha = a_dot_b / b_dot_b
    return jnp.square(a - alpha * b).mean()


@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def rmse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Returns the Mean Squared Error between `a` and `b`.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      MSE between `a` and `b`.
    """
    return jnp.sqrt(mse(a, b))


@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def mse(a: jax.Array, b: jax.Array) -> jax.Array:
    """Returns the Mean Squared Error between `a` and `b`.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      MSE between `a` and `b`.
    """
    return jnp.square(a - b).mean()


@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def mae(a: jax.Array, b: jax.Array) -> jax.Array:
    """Returns the Mean Absolute Error between `a` and `b`.

    Args:
      a: First image (or set of images).
      b: Second image (or set of images).

    Returns:
      MAE between `a` and `b`.
    """
    return jnp.abs(a - b).mean()


def iou(occ1: jax.Array, occ2: jax.Array) -> np.ndarray:
    """Computes the Intersection over Union (IoU) value for two sets of occupancy values.

    Args:
        occ1 (jax.Array): first set of occupancy values
        occ2 (jax.Array): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.0

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = area_intersect / area_union

    return iou


@partial(jax.vmap, in_axes=(0, 0, None, None), out_axes=0)
def psnr(
    image: jnp.ndarray, ground_truth: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray
) -> jnp.ndarray:
    """Computes the Peak Signal to Noise Ration (PSNR). Peak signal found from ground_truth and
    noise is given as the mean square error (MSE) between image and ground_truth. The value is
    returned in decibels.

    https://github.com/photosynthesis-team/piq/blob/master/piq/psnr.py

    Args:
        - image (jnp.ndarray): the first tensor used in the calculation.
        - ground_truth (jnp.ndarray): the second tensor used in the calculation.
    Returns:
        (jnp.ndarray) with the mean of the PNSR of the image according to the peak
        signal that can be obtained in ground_truth:
        mean(log10(peak_signal**2/MSE(image-ground_truth))).
    """
    # change the view to compute the metric in a batched way correctly
    w_image = image * std + mean
    w_ground_truth = ground_truth * std + mean

    maxval = jnp.max(w_ground_truth)

    w_image = w_image / maxval
    w_ground_truth = w_ground_truth / maxval

    EPS = 1e-8

    mse = jnp.maximum(0, jnp.mean((w_image - w_ground_truth) ** 2))

    return jnp.mean(-10 * jnp.log10(mse + EPS))


@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def ssim(
    a: jax.Array,
    b: jax.Array,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    precision=jax.lax.Precision.HIGHEST,
    filter_fn: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> jax.Array:
    """Computes the structural similarity index (SSIM) between image pairs.

    This function is based on the standard SSIM implementation from:
    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity",
    in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Note: the true SSIM is only defined on grayscale. This function does not
    perform any colorspace transform. If the input is in a color space, then it
    will compute the average SSIM.

    Args:
        a: First image (or set of images).
        b: Second image (or set of images).
        max_val: The maximum magnitude that `a` or `b` can have.
        filter_size: Window size (>= 1). Image dims must be at least this small.
        filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
        k1: One of the SSIM dampening parameters (> 0.).
        k2: One of the SSIM dampening parameters (> 0.).
        return_map: If True, will cause the per-pixel SSIM "map" to be returned.
        precision: The numerical precision to use when performing convolution.
        filter_fn: An optional argument for overriding the filter function used by
            SSIM, which would otherwise be a 2D Gaussian blur specified by filter_size
            and filter_sigma.

    Returns:
        Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """

    if filter_fn is None:
        # Construct a 1D Gaussian blur filter.
        hw = filter_size // 2
        shift = (2 * hw - filter_size + 1) / 2
        f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
        filt = jnp.exp(-0.5 * f_i)
        filt /= jnp.sum(filt)

        # Construct a 1D convolution.
        def filter_fn_1(z):
            return jnp.convolve(z, filt, mode="valid", precision=precision)

        filter_fn_vmap = jax.vmap(filter_fn_1)

        # Apply the vectorized filter along the y axis.
        def filter_fn_y(z):
            z_flat = jnp.moveaxis(z, -3, -1).reshape((-1, z.shape[-3]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-2],
                z.shape[-1],
                -1,
            )
            z_filtered = jnp.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -3)
            return z_filtered

        # Apply the vectorized filter along the x axis.
        def filter_fn_x(z):
            z_flat = jnp.moveaxis(z, -2, -1).reshape((-1, z.shape[-2]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-3],
                z.shape[-1],
                -1,
            )
            z_filtered = jnp.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -2)
            return z_filtered

    # Apply the blur in both x and y.
    filter_fn = lambda z: filter_fn_y(filter_fn_x(z))

    mu0 = filter_fn(a)
    mu1 = filter_fn(b)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filter_fn(a**2) - mu00
    sigma11 = filter_fn(b**2) - mu11
    sigma01 = filter_fn(a * b) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = jnp.finfo(jnp.float32).eps ** 2
    sigma00 = jnp.maximum(epsilon, sigma00)
    sigma11 = jnp.maximum(epsilon, sigma11)
    sigma01 = jnp.sign(sigma01) * jnp.minimum(jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim_value = jnp.mean(ssim_map)
    return ssim_map if return_map else ssim_value
