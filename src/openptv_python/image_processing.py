"""Image processing functions."""

import copy

import numpy as np
from numba import njit

from ._native_compat import native_preprocess_image, should_use_native
from ._native_convert import to_native_control_par
from .parameters import ControlPar

filter_t = np.zeros((3, 3), dtype=float)


@njit
def filter_3(img, kernel=None) -> np.ndarray:
    """Apply a 3x3 filter to an image."""
    if img.dtype != np.uint8:
        raise TypeError("Image must be of type uint8")

    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.float64)

    kernel_sum = float(np.sum(kernel))
    if kernel_sum == 0:
        raise ValueError("Filter kernel sum must not be zero")

    imx = img.shape[1]
    imy = img.shape[0]
    image_size = imx * imy
    flat_in = img.reshape(-1)
    flat_out = flat_in.copy()

    for index in range(imx + 1, image_size - imx - 1):
        buf = (
            kernel[0, 0] * flat_in[index - imx - 1]
            + kernel[0, 1] * flat_in[index - imx]
            + kernel[0, 2] * flat_in[index - imx + 1]
            + kernel[1, 0] * flat_in[index - 1]
            + kernel[1, 1] * flat_in[index]
            + kernel[1, 2] * flat_in[index + 1]
            + kernel[2, 0] * flat_in[index + imx - 1]
            + kernel[2, 1] * flat_in[index + imx]
            + kernel[2, 2] * flat_in[index + imx + 1]
        )
        buf = int(buf / kernel_sum)
        if buf > 255:
            buf = 255
        if buf < 8:
            buf = 8
        flat_out[index] = np.uint8(buf)

    return flat_out.reshape(img.shape)


@njit
def lowpass_3(img: np.ndarray) -> np.ndarray:
    """Lowpass filter matching the native 3x3 implementation."""
    if img.dtype != np.uint8:
        raise TypeError("Image must be of type uint8")

    imx = img.shape[1]
    imy = img.shape[0]
    image_size = imx * imy
    flat_in = img.reshape(-1)
    flat_out = flat_in.copy()

    for index in range(imx + 1, image_size - imx - 1):
        buf = (
            int(flat_in[index - imx - 1])
            + int(flat_in[index - imx])
            + int(flat_in[index - imx + 1])
            + int(flat_in[index - 1])
            + int(flat_in[index])
            + int(flat_in[index + 1])
            + int(flat_in[index + imx - 1])
            + int(flat_in[index + imx])
            + int(flat_in[index + imx + 1])
        )
        flat_out[index] = np.uint8(buf // 9)

    return flat_out.reshape(img.shape)


def fast_box_blur(
    filt_span: int,
    src: np.ndarray,
    cpar: ControlPar | None = None,
) -> np.ndarray:
    """Fast box blur matching liboptv border handling."""
    if src.dtype != np.uint8:
        raise TypeError("Image must be of type uint8")

    if src.ndim != 2:
        raise TypeError("Input array must be two-dimensional")

    if cpar is not None:
        imx = int(cpar.imx)
        imy = int(cpar.imy)
        if src.shape != (imy, imx):
            raise ValueError("Image shape does not match control parameters")
    else:
        imy, imx = src.shape

    image_size = imx * imy
    n = 2 * filt_span + 1
    nq = n * n
    src_flat = src.reshape(-1)
    dest_flat = np.zeros(image_size, dtype=np.uint8)
    row_accum = np.zeros(image_size, dtype=np.int64)
    col_accum = np.zeros(imx, dtype=np.int64)

    for row in range(imy):
        row_start = row * imx
        accum = int(src_flat[row_start])
        row_accum[row_start] = accum * n

        for col in range(1, min(filt_span + 1, imx)):
            right = row_start + 2 * col
            left = right - 1
            if right >= row_start + imx:
                break
            accum += int(src_flat[left]) + int(src_flat[right])
            row_accum[row_start + col] = accum * n // (2 * col + 1)

        for col in range(filt_span + 1, max(imx - filt_span, filt_span + 1)):
            accum += int(src_flat[row_start + col + filt_span])
            accum -= int(src_flat[row_start + col - filt_span - 1])
            row_accum[row_start + col] = accum

        m = n - 2
        col = imx - filt_span
        left = row_start + imx - n
        right = left + 1
        while col < imx and m > 0 and right < row_start + imx:
            accum -= int(src_flat[left]) + int(src_flat[right])
            row_accum[row_start + col] = accum * n // m
            left += 2
            right += 2
            col += 1
            m -= 2

    for col in range(imx):
        col_accum[col] = row_accum[col]
        dest_flat[col] = np.uint8(col_accum[col] // n)

    max_top = min(filt_span, (imy - 1) // 2)
    for row in range(1, max_top + 1):
        base1 = (2 * row - 1) * imx
        base2 = base1 + imx
        out_base = row * imx
        for col in range(imx):
            col_accum[col] += row_accum[base1 + col] + row_accum[base2 + col]
            dest_flat[out_base + col] = np.uint8(
                n * col_accum[col] // nq // (2 * row + 1)
            )

    for row in range(filt_span + 1, max(imy - filt_span, filt_span + 1)):
        remove_base = (row - filt_span - 1) * imx
        add_base = (row + filt_span) * imx
        out_base = row * imx
        for col in range(imx):
            col_accum[col] += row_accum[add_base + col] - row_accum[remove_base + col]
            dest_flat[out_base + col] = np.uint8(col_accum[col] // nq)

    for remaining in range(min(filt_span, imy - 1), 0, -1):
        remove_base = (imy - 2 * remaining - 1) * imx
        remove_base2 = remove_base + imx
        out_base = (imy - remaining) * imx
        if remove_base < 0:
            continue
        for col in range(imx):
            col_accum[col] -= (
                row_accum[remove_base + col] + row_accum[remove_base2 + col]
            )
            dest_flat[out_base + col] = np.uint8(
                n * col_accum[col] // nq // (2 * remaining + 1)
            )

    return dest_flat.reshape(src.shape)


# def split(img: np.ndarray, half_selector: int, cpar: ControlPar) -> np.ndarray:
#     """Split image into two halves."""
#     cond_offs = cpar.imx if half_selector % 2 else 0

#     if half_selector == 0:
#         return

#     coords_x, coords_y = np.meshgrid(np.arange(cpar.imx), np.arange(cpar.imy // 2))

#     coords_x = coords_x.flatten()
#     coords_y = coords_y.flatten() * 2 + cond_offs

#     new_img = map_coordinates(img, [coords_y, coords_x], mode="constant", cval=0)
#     return new_img


@njit
def subtract_img(img1: np.ndarray, img2: np.ndarray, img_new: np.ndarray) -> None:
    """
    Subtract img2 from img1 and store the result in img_new.

    Args:
    ----
    img1, img2: numpy arrays containing the original images.
    img_new: numpy array to store the result.
    """
    img_new[:] = np.maximum(img1.astype(np.int16) - img2.astype(np.int16), 0).astype(
        np.uint8
    )


@njit
def subtract_mask(img: np.ndarray, img_mask: np.ndarray):
    """Subtract mask from image."""
    img_new = np.where(img_mask == 0, 0, img)
    return img_new


def copy_images(src: np.ndarray) -> np.ndarray:
    """Copy src image to dest."""
    dest = copy.deepcopy(src)
    return dest


def prepare_image(
    img: np.ndarray,
    dim_lp: int = 1,
    filter_hp: int = 0,  # or 1,2
    filter_file: str = "",
) -> np.ndarray:
    """Prepare an image for particle detection: an averaging (smoothing).

    filter on an image, optionally followed by additional user-defined filter.
    """
    if img.dtype != np.uint8:
        raise TypeError("Image must be of type uint8")

    if img.ndim != 2:
        raise TypeError("Input array must be two-dimensional")

    img_lp = fast_box_blur(dim_lp, img)
    img_hp = np.empty_like(img)
    subtract_img(img, img_lp, img_hp)

    # Filter highpass image, if wanted, if filter_hp == 0, no highpass filtering
    if filter_hp == 1:
        img_hp = lowpass_3(img_hp)
    elif filter_hp == 2 and filter_file != "":
        try:
            with open(filter_file, "r", encoding="utf-8") as fp:
                filt = np.array(fp.read().split(), dtype=np.float64).reshape((3, 3))
        except Exception as exc:
            raise IOError(f"Could not open filter file: {filter_file}") from exc

        img_hp = filter_3(img_hp, filt)

    return img_hp


def preprocess_image(img, filter_hp, cpar, dim_lp) -> np.ndarray:
    """Decorate prepare_image with default parameters."""
    if should_use_native("preprocess_image"):
        native_cpar = to_native_control_par(cpar)
        return native_preprocess_image(
            img,
            filter_hp,
            native_cpar,
            lowpass_dim=dim_lp,
            filter_file=None,
            output_img=None,
        )
    return prepare_image(img=img, dim_lp=dim_lp, filter_hp=filter_hp, filter_file="")


# def preprocess_image(
#     input_img: np.ndarray,
#     filter_hp: int,
#     control: ControlPar,
#     lowpass_dim=1,
#     filter_file=None,
#     output_img=None,
# ):
#     """
#     Perform the steps necessary for preparing an image

#     for particle detection: an averaging (smoothing) filter on an image, optionally
#     followed by additional user-defined filter.

#     Arguments:
#     numpy.ndarray input_img - numpy 2d array representing the source image to filter.
#     int filter_hp - flag for additional filtering of _hp. 1 for lowpass, 2 for
#         general 3x3 filter given in parameter ``filter_file``.
#     ControlParams control - image details such as size and image half for
#     interlaced cases.
#     int lowpass_dim - half-width of lowpass filter, see fast_box_blur()'s filt_span
#       parameter.
#     filter_file - path to a text file containing the filter matrix to be
#         used in case ```filter_hp == 2```. One line per row, white-space
#         separated columns.
#     numpy.ndarray output_img - result numpy 2d array representing the source
#         image to filter. Same size as img.

#     Returns:
#     numpy.ndarray representing the result image.
#     """

#     # check arrays dimensions
#     if input_img.ndim != 2:
#         raise TypeError("Input array must be two-dimensional")
#     if (output_img is not None) and (
#         input_img.shape[0] != output_img.shape[0]
#         or input_img.shape[1] != output_img.shape[1]
#     ):
#         raise ValueError("Different shapes of input and output images.")
#     else:
#         output_img = np.empty_like(input_img)

#     if filter_hp == 2:
#         if filter_file is None or not isinstance(filter_file, str):
#             raise ValueError(
#                 "Expecting a filter file name, received None or non-string."
#             )
#     else:
#         filter_file = b""

#     for arr in (input_img, output_img):
#         if not arr.flags["C_CONTIGUOUS"]:
#             np.ascontiguousarray(arr)

#     output_img = prepare_image(input_img, lowpass_dim, filter_hp, filter_file, control):
#     if output_img is None:
#         raise Exception(
#             "prepare_image C function failed: failure of memory allocation or filter file reading"
#         )

#     return output_img


# def prepare_image(input_img, output_img, lowpass_dim, filter_hp, filter_file, control_par):
#     '''
#     prepare_image() - C implementation of image preprocessing

#     Arguments:
#     input_img - numpy 2d array representing the source image to filter.
#     output_img - result numpy 2d array representing the source image to filter.
#       Same size as input_img.
#     lowpass_dim - half-width of lowpass filter, see fast_box_blur()'s filt_span parameter.
#     filter_hp - flag for additional filtering of _hp. 1 for lowpass, 2 for general
#       3x3 filter given in parameter filter_file.
#     filter_file - path to a text file containing the filter matrix to be used in case
#       filter_hp == 2. One line per row, white-space separated columns.
#     control_par - control parameters

#     Returns:
#     int representing the success status of the function
#     '''

#     # implementation of the function here
#     pass # replace this with the actual implementation of the function


# result = fast_box_blur_numba(filt_span, src, cpar)
