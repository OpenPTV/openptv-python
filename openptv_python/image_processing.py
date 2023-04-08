import copy

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates, uniform_filter

from openptv_python.parameters import ControlPar

filter_t = np.zeros((3, 3), dtype=float)


def filter_3(img, kernel=None) -> np.ndarray:
    """Apply a 3x3 filter to an image."""
    if kernel is None:  # default is a low pass
        kernel = np.ones((3, 3)) / 9
    filtered_img = ndimage.convolve(img, kernel)
    return filtered_img


def lowpass_3(img):
    # Define the 3x3 lowpass filter kernel
    kernel = np.ones((3, 3)) / 9

    # Apply the filter to the image using scipy.ndimage.convolve()
    img_lp = ndimage.convolve(img, kernel, mode="constant", cval=0.0)

    return img_lp


def fast_box_blur(filt_span, src, dest, cpar):
    n = 2 * filt_span + 1
    weights = [1] * n
    row_accum = uniform_filter(
        src.reshape((cpar.imy, cpar.imx)),
        size=n,
        mode="constant",
        cval=0,
        weights=weights,
    ).reshape(-1)
    dest[:] = row_accum[:]


def split(img: np.ndarray, half_selector: int, cpar: ControlPar) -> None:
    cond_offs = cpar.imx if half_selector % 2 else 0

    if half_selector == 0:
        return

    coords_x, coords_y = np.meshgrid(np.arange(cpar.imx), np.arange(cpar.imy // 2))

    coords_x = coords_x.flatten()
    coords_y = coords_y.flatten() * 2 + cond_offs

    new_img = map_coordinates(img, [coords_y, coords_x], mode="constant", cval=0)
    img[:] = new_img


def subtract_img(img1: np.ndarray, img2: np.ndarray, img_new: np.ndarray) -> None:
    """
    Subtract img2 from img1 and store the result in img_new.

    Args:
    ----
    img1, img2: numpy arrays containing the original images.
    img_new: numpy array to store the result.
    """
    img_new[:] = ndimage.maximum(img1 - img2, 0)


def subtract_mask(img, img_mask, img_new, cpar):
    img_new[:] = np.where(img_mask == 0, 0, img)


def copy_images(src: np.ndarray) -> np.ndarray:
    """Copy src image to dest."""
    dest = copy.deepcopy(src)
    return dest


def prepare_image(img, img_hp, dim_lp, filter_hp, filter_file, cpar):
    image_size = cpar.imx * cpar.imy
    img_lp = np.zeros(image_size, dtype=np.uint8)

    # Apply low-pass filter
    img = img.reshape((cpar.imy, cpar.imx))  # Reshape to 2D image
    img_lp = ndimage.uniform_filter(
        img,
        size=(dim_lp * 2 + 1, dim_lp * 2 + 1),
        mode="constant",
        cval=0.0,
    )

    # Subtract low-pass filtered image from original image
    img_hp = np.subtract(img, img_lp, dtype=np.int16)

    # Filter highpass image, if wanted
    if filter_hp == 1:
        img_hp = ndimage.uniform_filter(
            img_hp.reshape((cpar.imy, cpar.imx)),
            size=(3, 3),
            mode="constant",
            cval=0.0,
        ).flatten()
    elif filter_hp == 2:
        try:
            with open(filter_file, "r") as fp:
                filt = np.array(fp.read().split(), dtype=np.float64).reshape((3, 3))
        except Exception:
            return 0

        img_hp = ndimage.convolve(
            img_hp.reshape((cpar.imy, cpar.imx)),
            weights=filt,
            mode="constant",
            cval=0.0,
        ).flatten()

    return 1
