"""Image processing functions."""
import copy

import numpy as np
from numba import njit, prange
from scipy import ndimage
from scipy.ndimage import uniform_filter

from .parameters import ControlPar

filter_t = np.zeros((3, 3), dtype=float)

@njit
def filter_3(img, kernel=None) -> np.ndarray:
    """Apply a 3x3 filter to an image."""
    if kernel is None:  # default is a low pass
        kernel = np.ones((3, 3)) / 9
    filtered_img = ndimage.convolve(img, kernel)
    return filtered_img

@njit
def lowpass_3(img: np.ndarray) -> np.ndarray:
    """Lowpass filter of 3x3."""
    # Define the 3x3 lowpass filter kernel
    kernel = np.ones((3, 3)) / 9

    # Apply the filter to the image using scipy.ndimage.convolve()
    img_lp = ndimage.convolve(img, kernel, mode="constant", cval=0.0)

    return img_lp

@njit
def fast_box_blur(filt_span: int, src: np.ndarray, cpar: ControlPar) -> np.ndarray:
    """Fast box blur."""
    n = 2 * filt_span + 1
    row_accum = uniform_filter(
        src.reshape((cpar.imy, cpar.imx)),
        size=n,
        mode="constant",
        cval=0,
    ).reshape(-1)
    return row_accum


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
    img_new[:] = ndimage.maximum(img1 - img2, 0)

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
    filter_hp: int = 0, # or 1,2
    filter_file: str = "",
) -> np.ndarray:
    """Prepare an image for particle detection: an averaging (smoothing).

    filter on an image, optionally followed by additional user-defined filter.
    """
    # image_size = cpar.imx * cpar.imy
    # img_lp = np.zeros_like(img, dtype=np.uint8)

    # Apply low-pass filter
    # img = img.reshape((cpar.imy, cpar.imx))  # Reshape to 2D image
    if img.dtype != np.uint8:
        raise TypeError("Image must be of type uint8")

    img_lp = ndimage.uniform_filter(
        img,
        size=dim_lp * 2 + 1,
        mode="constant",
        cval=0,
    )

    # Subtract low-pass filtered image from original image
    img_hp = img | img_lp

    # Filter highpass image, if wanted, if filter_hp == 0, no highpass filtering
    if filter_hp == 1:
        img_hp = ndimage.uniform_filter(
            # img_hp.reshape((cpar.imy, cpar.imx)),
            img_hp,
            size=3,
            mode="constant",
            cval=0.0,
        )
    elif filter_hp == 2 and filter_file != "":
        try:
            with open(filter_file, "r", encoding="utf-8") as fp:
                filt = np.array(fp.read().split(), dtype=np.float64).reshape((3, 3))
        except Exception as exc:
            raise IOError(f"Could not open filter file: {filter_file}") from exc

        img_hp = ndimage.convolve(
            # img_hp.reshape((cpar.imy, cpar.imx)),
            img_hp,
            weights=filt,
            mode="constant",
            cval=0.0,
        )

    return img_hp

def preprocess_image(img, filter_hp, cpar, dim_lp)-> np.ndarray:
    """Decorate prepare_image with default parameters."""
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

@njit(parallel=True)
def fast_box_blur_numba(filt_span, src, cpar):
    imx, imy = cpar['imx'], cpar['imy']
    n = 2 * filt_span + 1
    nq = n * n

    row_accum = np.zeros(imx * imy, dtype=np.int32)
    col_accum = np.zeros(imx, dtype=np.int32)
    dest = np.zeros_like(src, dtype=np.int32)

    # Sum over lines first [1]:
    for i in prange(imy):
        row_start = i * imx
        accum = src[row_start]
        row_accum[row_start] = accum * n

        for j in range(1, filt_span + 1):
            accum += src[row_start + j - 1] + src[row_start + j + filt_span]
            row_accum[row_start + j] = accum * n // (2 * j + 1)

        for j in range(filt_span + 1, imx - filt_span):
            accum += src[row_start + j + filt_span] - src[row_start + j - filt_span - 1]
            row_accum[row_start + j] = accum

        for j in range(imx - filt_span, imx):
            accum -= src[row_start + j - filt_span - 1] + src[row_start + j + filt_span]
            row_accum[row_start + j] = accum * n // (2 * (imx - j - 1) + 1)

    # Sum over columns:
    col_accum[:imx] = row_accum[:imx]
    dest[:imx] = col_accum[:imx] // n

    for i in range(1, filt_span + 1):
        ptr1 = row_accum[(2 * i - 1) * imx:(2 * i + 1) * imx]
        ptr2 = ptr1[imx:]
        col_accum += ptr1 + ptr2
        dest[i * imx:(i + 1) * imx] = n * col_accum // nq // (2 * i + 1)

    for i in range(filt_span + 1, imy - filt_span):
        ptr1 = row_accum[(i - filt_span - 1) * imx:i * imx]
        ptr2 = row_accum[(i + filt_span) * imx:(i + filt_span + 1) * imx]
        col_accum += ptr2 - ptr1
        dest[i * imx:(i + 1) * imx] = col_accum // nq

    for i in range(filt_span, 0, -1):
        ptr1 = row_accum[(imy - 2 * i - 1) * imx:(imy - 2 * i + 1) * imx]
        ptr2 = ptr1[imx:]
        col_accum -= ptr1 + ptr2
        dest[(imy - i) * imx:] = n * col_accum // nq // (2 * i + 1)

    return dest

# # Example usage:
# filt_span = 3
# src = np.random.randint(0, 256, size=(1000, 1000), dtype=np.uint8)
# cpar = {'imx': src.shape[1], 'imy': src.shape[0]}

# result = fast_box_blur_numba(filt_span, src, cpar)




def filter_3_numpy(img: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """
    Performs a 3x3 filtering over an image.

    Args:
        img: Original image (NumPy array).
        filt: 3x3 filter matrix (NumPy array).

    Returns:
        Filtered image (NumPy array).
    """
    sum_filt = np.sum(filt)
    if sum_filt == 0:
        return img  # Or raise an exception, depending on desired behavior

    img_pad = np.pad(img, pad_width=1, mode='wrap')  # Wrap-around edges
    img_lp = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Use array slicing for efficiency
            region = img_pad[i:i + 3, j:j + 3]
            buf = np.sum(filt * region)
            buf /= sum_filt

            buf = np.clip(buf, 8, 255)  # Enforce minimal brightness and max value
            img_lp[i, j] = int(buf)

    return img_lp


def lowpass_3_numpy(img: np.ndarray) -> np.ndarray:
    """
    Applies a 3x3 lowpass filter (average of all 9 pixels).

    Args:
        img: Original image (NumPy array).

    Returns:
        Filtered image (NumPy array).
    """
    img_pad = np.pad(img, pad_width=1, mode='wrap')
    img_lp = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = img_pad[i:i + 3, j:j + 3]
            buf = np.sum(region)
            img_lp[i, j] = buf // 9  # Integer division

    return img_lp


def fast_box_blur_numpy(img: np.ndarray, filt_span: int) -> np.ndarray:
    """
    Performs a fast box blur on an image.

    Args:
        img: Original image (NumPy array).
        filt_span: Half-width of the box blur filter.

    Returns:
        Blurred image (NumPy array).
    """
    n = 2 * filt_span + 1
    nq = n * n
    imy, imx = img.shape

    row_accum = np.zeros(img.size, dtype=np.int32).reshape(imy, imx)
    col_accum = np.zeros(imx, dtype=np.int32)

    # Sum over lines
    for i in range(imy):
        row_start = i * imx
        accum = img[i, 0]
        row_accum[i, 0] = accum * n

        for m in range(3, 2 * filt_span + 2, 2):
            accum += (img[i, m // 2] + img[i, m // 2 + 1])
            row_accum[i, m // 2] = accum * n // m
        
        for j in range(filt_span + 1, imx):
            accum += (img[i, j] - img[i, j - n])
            row_accum[i, j] = accum

        for m in range(n - 2, 1, -2):
            accum -= (img[i, imx - 1 - m // 2] + img[i, imx - 2 - m // 2])
            row_accum[i, imx - 1 - m // 2] = accum * n // m

    # Sum over columns
    dest = np.zeros_like(img, dtype=np.uint8)

    for j in range(imx):
        col_accum[j] = row_accum[0, j]
        dest[0, j] = col_accum[j] // n

    for i in range(1, filt_span + 1):
        for j in range(imx):
            col_accum[j] += (row_accum[2 * i - 1, j] + row_accum[2 * i, j])
            dest[i, j] = n * col_accum[j] // nq // (2 * i + 1)

    for i in range(filt_span + 1, imy - filt_span):
        for j in range(imx):
            col_accum[j] += (row_accum[i + filt_span, j] - row_accum[i - filt_span - 1, j])
            dest[i, j] = col_accum[j] // nq

    for i in range(filt_span, 0, -1):
        for j in range(imx):
            col_accum[j] -= (row_accum[imy - 2 * i - 1, j] + row_accum[imy - 2 * i, j])
            dest[imy - i, j] = n * col_accum[j] // nq // (2 * i + 1)

    return dest.astype(np.uint8)


def split_numpy(img: np.ndarray, half_selector: int) -> np.ndarray:
    """
    Crams either even or odd lines into the first half of the image.
    The lower half of the image is set to 2.

    Args:
        img: Image to modify (NumPy array).
        half_selector: 1 for odd rows, 2 for even rows, 0 to do nothing.

    Returns:
        Modified image (NumPy array).
    """
    if half_selector == 0:
        return img

    imy, imx = img.shape
    cond_offs = imx if (half_selector % 2) else 0

    for row in range(imy // 2):
        img[row] = img[2 * row][cond_offs:cond_offs + imx]

    img[imy // 2:] = 2  # Set lower half to 2

    return img


def subtract_img_numpy(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Subtracts img2 from img1.

    Args:
        img1: First image (NumPy array).
        img2: Second image (NumPy array).

    Returns:
        Resulting image (NumPy array).
    """
    img_new = np.maximum(0, img1.astype(np.int16) - img2.astype(np.int16)).astype(np.uint8)
    return img_new


def subtract_mask_numpy(img: np.ndarray, img_mask: np.ndarray) -> np.ndarray:
    """
    Creates a masked image where pixels equal to zero in img_mask are set to 0.

    Args:
        img: Original image (NumPy array).
        img_mask: Mask image (NumPy array).

    Returns:
        Resulting image (NumPy array).
    """
    img_new = np.where(img_mask == 0, 0, img)
    return img_new.astype(np.uint8)


def copy_images_numpy(src: np.ndarray) -> np.ndarray:
    """
    Copies one image into another.

    Args:
        src: Source image (NumPy array).

    Returns:
        Destination image (NumPy array).
    """
    dest = src.copy()
    return dest


def prepare_image_numpy(img: np.ndarray, dim_lp: int, filter_hp: int, filter_file: str, cpar: Tuple[int, int, int]) -> np.ndarray:
    """
    Prepares an image for particle detection.

    Args:
        img: Source image (NumPy array).
        dim_lp: Half-width of the lowpass filter.
        filter_hp: Flag for additional filtering (0: none, 1: lowpass, 2: 3x3 filter).
        filter_file: Path to the filter matrix file.
	    cpar: image details such as size and image half for interlaced cases.

    Returns:
        Filtered image (NumPy array).
    """
    imx, imy, chfield = cpar
    img_lp = fast_box_blur(img, dim_lp)
    img_hp = subtract_img(img, img_lp)

    # consider field mode
    if chfield == 1 or chfield == 2:
        img_hp = split(img_hp, chfield)

    # filter highpass image, if wanted
    if filter_hp == 1:
        img_hp = lowpass_3(img_hp)
    elif filter_hp == 2:
        try:
            filt = np.loadtxt(filter_file)
            if filt.shape != (3, 3):
                raise ValueError("Filter file must contain a 3x3 matrix.")
            img_hp = filter_3(img_hp, filt)
        except Exception as e:
            print(f"Error reading or applying filter: {e}")
            return None

    return img_hp

