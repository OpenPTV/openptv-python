from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label, maximum_filter

from .constants import CORRES_NONE
from .parameters import ControlPar, TargetPar
from .tracking_frame_buf import Target


@dataclass
class Peak:
    """Peak dataclass."""

    pos: int = 0
    status: int = 0
    xmin: int = 0
    xmax: int = 0
    ymin: int = 0
    ymax: int = 0
    n: int = 0
    sumg: int = 0
    x: float = 0.0
    y: float = 0.0
    unr: int = 0
    touch: list[int] = field(default_factory=list, repr=False)
    n_touch: int = 0


def targ_rec(
    img: np.ndarray,
    targ_par: TargetPar,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    cpar: ControlPar,
    num_cam,
) -> List[Target]:
    """Target recognition function."""
    n = 0
    n_wait = 0
    n_targets = 0
    sumg = 0
    numpix = 0
    thres = targ_par.gvthresh[num_cam]
    disco = targ_par.discont

    imx = cpar.imx
    imy = cpar.imy

    # img0 = [0] * (imx * imy)  # create temporary mask
    img0 = img.copy()  # copy the original image

    # Make sure the min/max coordinates don't cause us to access memory
    # outside the image memory.
    if xmin <= 0:
        xmin = 1
    if ymin <= 0:
        ymin = 1
    if xmax >= imx:
        xmax = imx - 1
    if ymax >= imy:
        ymax = imy - 1

    waitlist = [[0] * 2 for _ in range(2048)]

    xa = 0
    ya = 0
    xb = 0
    yb = 0
    x4 = [0] * 4
    y4 = [0] * 4

    pix = []

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            # note i=y = rows = top to bottom
            # j=x = columns - left to right
            gv = img0[i, j]
            if gv > thres:
                if (
                    (gv >= img0[i, j - 1])
                    and (gv >= img0[i, j + 1])
                    and (gv >= img0[(i - 1), j])
                    and (gv >= img0[(i + 1), j])
                    and (gv >= img0[(i - 1), j - 1])
                    and (gv >= img0[(i + 1), j - 1])
                    and (gv >= img0[(i - 1), j + 1])
                    and (gv >= img0[(i + 1), j + 1])
                ):
                    yn = i
                    xn = j

                    sumg = gv
                    img0[i, j] = 0

                    xa = xn
                    xb = xn
                    ya = yn
                    yb = yn

                    gv -= thres  # intensity above the threshold
                    x = (xn) * gv
                    y = yn * gv
                    numpix = 1
                    waitlist[0][0] = j
                    waitlist[0][1] = i
                    n_wait = 1

                    while n_wait > 0:
                        gvref = img[waitlist[0][1], waitlist[0][0]]

                        x4[0] = waitlist[0][0] - 1
                        y4[0] = waitlist[0][1]
                        x4[1] = waitlist[0][0] + 1
                        y4[1] = waitlist[0][1]
                        x4[2] = waitlist[0][0]
                        y4[2] = waitlist[0][1] - 1
                        x4[3] = waitlist[0][0]
                        y4[3] = waitlist[0][1] + 1

                        for n in range(4):
                            xn = x4[n]
                            yn = y4[n]
                            if xn >= xmax or yn >= ymax or xn < 0 or yn < 0:
                                continue

                            gv = img0[yn, xn]

                            if (
                                (gv > thres)
                                and (xn > xmin - 1)
                                and (xn < xmax + 1)
                                and (yn > ymin - 1)
                                and (yn < ymax + 1)
                                and (gv <= gvref + disco)
                                and (gvref + disco >= img[yn - 1, xn])
                                and (gvref + disco >= img[yn + 1, xn])
                                and (gvref + disco >= img[yn, xn - 1])
                                and (gvref + disco >= img[yn, xn + 1])
                            ):
                                print(f"gv = {gv} sumg = {sumg}")
                                sumg += gv
                                img0[yn, xn] = 0
                                if xn < xa:
                                    xa = xn
                                if xn > xb:
                                    xb = xn
                                if yn < ya:
                                    ya = yn
                                if yn > yb:
                                    yb = yn
                                waitlist[n_wait][0] = xn
                                waitlist[n_wait][1] = yn

                                # Coordinates are weighted by grey value, normed later.
                                x += (xn) * (gv - thres)
                                y += yn * (gv - thres)

                                numpix += 1
                                n_wait += 1

                        n_wait -= 1
                        for m in range(n_wait):
                            waitlist[m][0] = waitlist[m + 1][0]
                            waitlist[m][1] = waitlist[m + 1][1]
                        waitlist[n_wait][0] = 0
                        waitlist[n_wait][1] = 0

                    if (
                        xa == (xmin - 1)
                        or ya == (ymin - 1)
                        or xb == (xmax + 1)
                        or yb == (ymax + 1)
                    ):
                        continue

                    nx = xb - xa + 1
                    ny = yb - ya + 1

                    if (
                        numpix >= targ_par.nnmin
                        and numpix <= targ_par.nnmax
                        and nx >= targ_par.nxmin
                        and nx <= targ_par.nxmax
                        and ny >= targ_par.nymin
                        and ny <= targ_par.nymax
                        and sumg > targ_par.sumg_min
                    ):
                        pix.append(Target(n=numpix, nx=nx, ny=ny, sumg=sumg))
                        sumg -= numpix * thres
                        # finish the grey-value weighting:
                        x /= sumg
                        x += 0.5
                        y /= sumg
                        y += 0.5
                        pix[n_targets].x = x
                        pix[n_targets].y = y
                        pix[n_targets].tnr = CORRES_NONE
                        pix[n_targets].pnr = n_targets
                        n_targets += 1
                        xn = x
                        yn = y

    # t = TargetArray(num_targs=n_targets)
    # t.num_targs = n_targets
    # t.targs = pix
    return pix


def peak_fit(
    img: np.ndarray,
    targ_par: TargetPar,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    cpar: ControlPar,
    num_cam: int,
) -> List[Target]:
    """Fit the peaks in the image to a gaussian."""
    imx, imy = cpar.imx, cpar.imy
    n_peaks = 0
    n_wait = 0
    x8, y8 = [0, 1, 0, -1], [1, 0, -1, 0]
    p2 = 0
    thres = targ_par.gvthresh[num_cam]
    disco = targ_par.discont
    intx1, inty1 = 0, 0
    unify = 0
    unified = 0
    non_unified = 0
    gv, gvref = 0, 0
    gv1, gv2 = 0, 0
    x1, x2, y1, y2, s12 = 0.0, 0.0, 0.0, 0.0, 0.0
    label_img = [0] * (imx * imy)
    peaks = []
    waitlist = [[]]
    pix = []
    n_target = 0

    for i in range(ymin, ymax - 1):
        for j in range(xmin, xmax):
            n = i * imx + j

            # compare with threshold
            gv = img[n]
            if gv <= thres:
                continue

            # skip already labeled pixel
            if label_img[n] != 0:
                continue

            # check whether pixel is a local maximum
            if (
                gv >= img[n - 1]
                and gv >= img[n + 1]
                and gv >= img[n - imx]
                and gv >= img[n + imx]
                and gv >= img[n - imx - 1]
                and gv >= img[n + imx - 1]
                and gv >= img[n - imx + 1]
                and gv >= img[n + imx + 1]
            ):
                # label peak in label_img, initialize peak
                n_peaks += 1
                label_img[n] = n_peaks
                peaks.append(Peak(pos=n, status=1, xmin=j, xmax=i, ymin=i, ymax=i))

                waitlist[0][0] = j
                waitlist[0][1] = i
                n_wait = 1

                while n_wait > 0:
                    gvref = img[imx * waitlist[0][1] + waitlist[0][0]]
                    x8 = [
                        waitlist[0][0] - 1,
                        waitlist[0][0] + 1,
                        waitlist[0][0],
                        waitlist[0][0],
                    ]
                    y8 = [
                        waitlist[0][1],
                        waitlist[0][1],
                        waitlist[0][1] - 1,
                        waitlist[0][1] + 1,
                    ]

                    for k in range(4):
                        yn = y8[k]
                        xn = x8[k]

                        if xn < 0 or xn >= imx or yn < 0 or yn >= imy:
                            continue

                        n = imx * yn + xn
                        if label_img[n] != 0:
                            continue

                        gv = img[n]

                        # conditions for threshold, discontinuity, image borders and peak fitting
                        if (
                            (gv > thres)
                            and (xn >= xmin)
                            and (xn < xmax)
                            and (yn >= ymin)
                            and (yn < ymax - 1)
                            and (gv <= gvref + disco)
                            and (gvref + disco >= img[imx * (yn - 1) + xn])
                            and (gvref + disco >= img[imx * (yn + 1) + xn])
                            and (gvref + disco >= img[imx * yn + (xn - 1)])
                            and (gvref + disco >= img[imx * yn + (xn + 1)])
                        ):
                            label_img[imx * yn + xn] = n_peaks
                            waitlist[n_wait][0] = xn
                            waitlist[n_wait][1] = yn
                            n_wait += 1

                    n_wait -= 1
                    for m in range(n_wait):
                        waitlist[m][0] = waitlist[m + 1][0]
                        waitlist[m][1] = waitlist[m + 1][1]
                    waitlist[n_wait][0] = 0
                    waitlist[n_wait][1] = 0

    # 2.:    process label image
    #        (collect data for center of gravity, shape and brightness parameters)
    #        get touch events

    for i in range(ymin, ymax):
        for j in range(xmin, xmax):
            n = i * imx + j

            if label_img[n] > 0:
                # process pixel
                pnr = label_img[n]
                gv = img[n]
                ptr_peak = peaks[pnr - 1]
                ptr_peak.n += 1
                ptr_peak.sumg += gv
                ptr_peak.x += j * gv
                ptr_peak.y += i * gv

                if j < ptr_peak.xmin:
                    ptr_peak.xmin = j
                if j > ptr_peak.xmax:
                    ptr_peak.xmax = j
                if i < ptr_peak.ymin:
                    ptr_peak.ymin = i
                if i > ptr_peak.ymax:
                    ptr_peak.ymax = i

                # get touch events
                if i > 0 and j > 1:
                    check_touch(ptr_peak, pnr, label_img[n - imx - 1])
                if i > 0:
                    check_touch(ptr_peak, pnr, label_img[n - imx])
                if i > 0 and j < imy - 1:
                    check_touch(ptr_peak, pnr, label_img[n - imx + 1])
                if j > 0:
                    check_touch(ptr_peak, pnr, label_img[n - 1])
                if j < imy - 1:
                    check_touch(ptr_peak, pnr, label_img[n + 1])
                if i < imx - 1 and j > 0:
                    check_touch(ptr_peak, pnr, label_img[n + imx - 1])
                if i < imx - 1:
                    check_touch(ptr_peak, pnr, label_img[n + imx])
                if i < imx - 1 and j < imy - 1:
                    check_touch(ptr_peak, pnr, label_img[n + imx + 1])

    # 3.: reunification test: profile and distance

    for i in range(n_peaks):
        if peaks[i].n_touch == 0:
            continue  # no touching targets
        if peaks[i].unr != 0:
            continue  # target already unified

        # profile criterion
        # point 1
        x1 = peaks[i].x / peaks[i].sumg
        y1 = peaks[i].y / peaks[i].sumg
        gv1 = img[peaks[i].pos]

        # consider all touching points
        for j in range(peaks[i].n_touch):
            p2 = peaks[i].touch[j] - 1

            if p2 >= n_peaks:
                continue  # workaround memory overwrite problem
            if p2 < 0:
                continue  # workaround memory overwrite problem
            if peaks[p2].unr != 0:
                continue  # target already unified

            # point 2
            x2 = peaks[p2].x / peaks[p2].sumg
            y2 = peaks[p2].y / peaks[p2].sumg

            gv2 = img[peaks[p2].pos]

            s12 = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # consider profile dot for dot
            # if any points is by more than disco below profile, do not unify
            if s12 < 2.0:
                unify = 1
            else:
                unify = 1
                for ll in range(1, int(s12)):
                    intx1 = int(x1 + ll * (x2 - x1) / s12)
                    inty1 = int(y1 + ll * (y2 - y1) / s12)
                    gv = img[inty1 * imx + intx1] + disco
                    if gv < gv1 + ll * (gv2 - gv1) / s12 or gv < gv1 or gv < gv2:
                        unify = 0
                    if unify == 0:
                        break
            if unify == 0:
                non_unified += 1
                continue

            # otherwise unify targets
            unified += 1
            peaks[i].unr = p2
            peaks[p2].x += peaks[i].x
            peaks[p2].y += peaks[i].y
            peaks[p2].sumg += peaks[i].sumg
            peaks[p2].n += peaks[i].n
            if peaks[i].xmin < peaks[p2].xmin:
                peaks[p2].xmin = peaks[i].xmin
            if peaks[i].ymin < peaks[p2].ymin:
                peaks[p2].ymin = peaks[i].ymin
            if peaks[i].xmax > peaks[p2].xmax:
                peaks[p2].xmax = peaks[i].xmax
            if peaks[i].ymax > peaks[p2].ymax:
                peaks[p2].ymax = peaks[i].ymax

    # 4.: process targets
    for i in range(n_peaks):
        # check whether target touches image borders
        if peaks[i].xmin == xmin and (xmax - xmin) > 32:
            continue
        if peaks[i].ymin == ymin and (xmax - xmin) > 32:
            continue
        if peaks[i].xmax == xmax - 1 and (xmax - xmin) > 32:
            continue
        if peaks[i].ymax == ymax - 1 and (xmax - xmin) > 32:
            continue

        if (
            peaks[i].unr == 0
            and peaks[i].sumg > targ_par.sumg_min
            and (peaks[i].xmax - peaks[i].xmin + 1) >= targ_par.nxmin
            and (peaks[i].ymax - peaks[i].ymin + 1) >= targ_par.nymin
            and (peaks[i].xmax - peaks[i].xmin) < targ_par.nxmax
            and (peaks[i].ymax - peaks[i].ymin) < targ_par.nymax
            and peaks[i].n >= targ_par.nnmin
            and peaks[i].n <= targ_par.nnmax
        ):
            sumg = peaks[i].sumg

            # target coordinates
            pix.append(Target())
            pix[n_target].x = 0.5 + peaks[i].x / sumg
            pix[n_target].y = 0.5 + peaks[i].y / sumg

            # target shape parameters
            pix[n_target].sumg = sumg
            pix[n_target].n = peaks[i].n
            pix[n_target].nx = peaks[i].xmax - peaks[i].xmin + 1
            pix[n_target].ny = peaks[i].ymax - peaks[i].ymin + 1
            pix[n_target].tnr = CORRES_NONE
            pix[n_target].pnr = n_target
            n_target += 1

    # t = TargetArray(num_targets=n_target)
    # t.num_targs = n_target
    # t.append(pix)
    return pix


def check_touch(tpeak, p1, p2):
    done = False

    if p2 == 0:
        return
    if p2 == p1:
        return

    # check whether p1, p2 are already marked as touching
    for m in range(tpeak.n_touch):
        if tpeak.touch[m] == p2:
            done = True

    # mark touch event
    if not done:
        tpeak.touch[tpeak.n_touch] = p2
        tpeak.n_touch += 1
        # don't allow for more than 4 touches
        if tpeak.n_touch > 3:
            tpeak.n_touch = 3


def peak_fit_new(
    image: np.ndarray, threshold: float = 0.5, sigma: float = 1.0
) -> List[Peak]:
    """
    Find local maxima in an image using a Gaussian filter and a maximum filter,.

    and returns their positions and intensities.

    Args:
    ----
        image (np.ndarray): The image to analyze.
        threshold (float): The minimum intensity value for a peak to be considered.
        sigma (float): The standard deviation of the Gaussian filter.

    Returns:
    -------
        List[Peak]: A list of Peak objects representing the detected peaks.
    """
    smoothed = gaussian_filter(image, sigma)
    mask = smoothed > threshold * np.max(smoothed)  # type: ignore
    maxima = maximum_filter(smoothed, footprint=np.ones((3, 3))) == smoothed
    labeled, num_objects = label(maxima)  # type: ignore
    peaks = []
    for i in range(num_objects):
        indices = np.argwhere(labeled == i + 1)
        coordinates = [center_of_mass(mask[indices[:, 0], indices[:, 1]])[::-1]]
        intensity = smoothed[indices[:, 0], indices[:, 1]].max()
        x, y = np.mean(coordinates, axis=0)
        peaks.append(Peak(int(round(x)), int(round(y)), intensity))
    return peaks


def target_recognition(
    img: np.ndarray,
    tpar: TargetPar,
    cam: int,
    cparam: ControlPar,
    subrange_x=None,
    subrange_y=None,
) -> List[Target]:
    """
    Detect targets (contiguous bright blobs) in an image.

    Limited to ~20,000 targets per image for now. This limitation comes from
    the structure of underlying C code.

    Arguments:
    ---------
    img - a numpy array holding the 8-bit gray image.
    tpar - target recognition parameters s.a. size bounds etc.
    cam - number of camera that took the picture, needed for getting
        correct parameters for this image.
    cparam - an object holding general control parameters.
    subrange_x - optional, tuple of min and max pixel coordinates to search
        between. Default is to search entire image width.
    subrange_y - optional, tuple of min and max pixel coordinates to search
        between. Default is to search entire image height.

    Returns:
    -------
    Number of  object holding the targets found.
    """
    # Set the subrange (to default if not given):
    if subrange_x is None:
        xmin, xmax = 0, cparam.imx
    else:
        xmin, xmax = subrange_x

    if subrange_y is None:
        ymin, ymax = 0, cparam.imy
    else:
        ymin, ymax = subrange_y

    if img.shape[0] < ymax or img.shape[1] < xmax:
        raise ValueError("region of detection is larger than image dimensions")

    # The core Python implementation of targ_rec:
    target_array = targ_rec(img, tpar, xmin, xmax, ymin, ymax, cparam, cam)

    return target_array


def blob_detection(image, threshold):
    # Initialize the output list.
    blobs = []

    # Convert the image to grayscale.
    grayscale_image = np.array(image).mean(axis=2)

    # Find all of the pixels in the image that are above the threshold.
    above_threshold_pixels = np.where(grayscale_image > threshold)[0]

    # Create a queue to store the pixels that are currently being processed.
    waitlist = []

    # Add the first pixel to the queue.
    waitlist.append(above_threshold_pixels[0])

    # While the queue is not empty:
    while waitlist:
        # Pop the first pixel from the queue.
        pixel = waitlist.pop(0)

        # Get the x- and y-coordinates of the pixel.
        x = pixel % image.shape[1]
        y = pixel // image.shape[1]

        # Check to see if the pixel is within the bounds of the image.
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue

        # Check to see if the pixel is already part of a blob.
        if pixel in blobs:
            continue

        # Add the pixel to the current blob.
        blobs.append(pixel)

        # Add all of the neighboring pixels that are above the threshold to the queue.
        for neighbor in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if (
                neighbor[0] >= 0
                and neighbor[0] < image.shape[1]
                and neighbor[1] >= 0
                and neighbor[1] < image.shape[0]
                and grayscale_image[neighbor] > threshold
            ):
                waitlist.append(neighbor)

    return blobs
