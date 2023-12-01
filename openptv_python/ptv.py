from pathlib import Path
from typing import List, Tuple

import numpy as np
from skimage.io import imread
from skimage.util import img_as_ubyte

from openptv_python.calibration import Calibration
from openptv_python.constants import MAX_TARGETS, TR_BUFSPACE
from openptv_python.correspondences import MatchedCoords, correspondences
from openptv_python.image_processing import prepare_image
from openptv_python.orientation import (
    full_calibration,
    point_positions,
)
from openptv_python.parameters import (
    ControlPar,
    ExaminePar,
    OrientPar,
    PftVersionPar,
    SequencePar,
    TargetPar,
    TrackPar,
    VolumePar,
)
from openptv_python.segmentation import target_recognition
from openptv_python.track import (
    track_forward_start,
    trackback_c,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv_python.tracking_frame_buf import TargetArray, read_targets
from openptv_python.tracking_run import tr_new

default_naming = {
    'corres': 'res/rt_is',
    'linkage': 'res/ptv_is',
    'prio': 'res/added'
}


DEFAULT_NAMING = {
    'corres': 'corres',
    'linkage': 'linkage',
    'prio': 'prio'
}  # Assuming default_naming is defined somewhere in your code


class Tracker:
    """A class to run tracking.

    Workflow: instantiate, call restart() to initialize the frame buffer, then
    call either ``step_forward()`` while it still return True, then call
    ``finalize()`` to finish the run. Alternatively, ``full_forward()`` will
    do all this for you.
    """

    def __init__(self, cpar, vpar, tpar, spar, cals, naming=DEFAULT_NAMING, flatten_tol=0.0001):
        """Initialize.

        Arguments:
        ControlParams cpar, VolumeParams vpar, TrackingParams tpar,
        SequenceParams spar - the usual parameter objects, as read from
            anywhere.
        cals - a list of Calibration objects.
        dict naming - a dictionary with naming rules for the frame buffer
            files. See the ``DEFAULT_NAMING`` member (which is the default).
        """
        # We need to keep a reference to the Python objects so that their
        # allocations are not freed.
        self._keepalive = (cpar, vpar, tpar, spar, cals)

        self.run_info = tr_new(spar._sequence_par, tpar._track_par,
            vpar._volume_par, cpar._control_par, TR_BUFSPACE, MAX_TARGETS,
            naming['corres'], naming['linkage'], naming['prio'],
            cals, flatten_tol)

    def restart(self):
        """Restart and initialize a tracking run.

        Prepare a tracking run. Sets up initial buffers and performs the
        one-time calculations used throughout the loop.
        """
        self.step = self.run_info.seq_par.first
        track_forward_start(self.run_info)

    def step_forward(self):
        """Perform one tracking step for the current frame of iteration."""
        if self.step >= self.run_info.seq_par.last:
            return False

        trackcorr_c_loop(self.run_info, self.step)
        self.step += 1
        return True

    def finalize(self):
        """Finish a tracking run."""
        trackcorr_c_finish(self.run_info, self.step)

    def full_forward(self):
        """Do a full tracking run from restart to finalize."""
        track_forward_start(self.run_info)
        for step in range(
                self.run_info.seq_par.first, self.run_info.seq_par.last):
            trackcorr_c_loop(self.run_info, step)
        trackcorr_c_finish(self.run_info, self.run_info.seq_par.last)

    def full_backward(self):
        """Do a full backward run on existing tracking results.

        so make sure results exist or it will explode in your face.
        """
        trackback_c(self.run_info)

    def current_step(self):
        return self.step

    def __del__(self):
        # Don't call tr_free, just free the memory that belongs to us.
        del self.run_info.fb
        del self.run_info.cal  # allocated by cal_list2arr, leaves belong to
                               # owner of the Tracker.
        del self.run_info  # not using tr_free() which assumes ownership of
                           # parameter structs.



def negative(img: np.ndarray) -> np.ndarray:
    """Negative 8-bit image."""
    if img.dtype != np.uint8:
        raise ValueError("Image must be 8-bit")
    return 255 - img

def simple_highpass(img: np.ndarray) -> np.ndarray:
    """Apply highpass with default values."""
    return prepare_image(img, 1, 0)


def py_start_proc_c(n_cams: int) -> Tuple[ControlPar, SequencePar, VolumePar,
                                          TrackPar, TargetPar, List[Calibration],
                                          ExaminePar]:
    """Read parameters."""
    # Control parameters
    cpar = ControlPar(n_cams).from_file("parameters/ptv.par")

    # Sequence parameters
    spar = SequencePar().from_file("parameters/sequence.par", num_cams=n_cams)

    # Volume parameters
    vpar = VolumePar().from_file("parameters/criteria.par")

    # Tracking parameters
    track_par = TrackPar().from_file("parameters/track.par")

    # Target parameters
    tpar = TargetPar().from_file("parameters/targ_rec.par")

    # Examine parameters, multiplane (single plane vs combined calibration)
    epar = ExaminePar().from_file("paramters/examine.par")

    # Calibration parameters
    cals = []
    for i_cam in range(n_cams):
        cal = Calibration()
        tmp = cpar.cal_img_base_name[i_cam]
        cal.from_file(tmp + ".ori", tmp + ".addpar")
        cals.append(cal)

    return cpar, spar, vpar, track_par, tpar, cals, epar


def py_pre_processing_c(list_of_images: List[np.ndarray]) -> List[np.ndarray]:
    """Preprocess images.

    Image pre-processing, mostly highpass filter, could be extended in
    the future.

    Inputs:
        list of images
        cpar ControlPar()
    """
    newlist = []
    for img in list_of_images:
        newlist.append(simple_highpass(img))
    return newlist


def py_detection_proc_c(list_of_images, cpar, tpar, cals):
    """Detect targets."""
    pftVersionPar = PftVersionPar().from_file("parameters/pft_version.par")
    Existing_Target = bool(pftVersionPar.existing_target_flag)

    detections, corrected = [], []
    for i_cam, img in enumerate(list_of_images):
        if Existing_Target:
            targs = read_targets(cpar.get_img_base_name(i_cam), 0)
        else:
            targs = target_recognition(img, tpar, i_cam, cpar)

        targs.sort(key=lambda t: t.y)
        detections.append(targs)
        mc = MatchedCoords(targs, cpar, cals[i_cam])
        corrected.append(mc)

    return detections, corrected


def py_correspondences_proc_c(exp):
    """Provide correspondences.

    Inputs:
        exp = info.object from the pyptv_gui
    Outputs:
        quadruplets, ... : four empty lists filled later with the
    correspondences of quadruplets, triplets, pairs, and so on.
    """
    frame = 123456789  # just a temporary workaround. todo: think how to write

    #        if any([len(det) == 0 for det in detections]):
    #            return False

    # Corresp. + positions.
    sorted_pos, sorted_corresp, num_targs = correspondences(
        exp.detections, exp.corrected, exp.cals, exp.vpar, exp.cpar)

    # Save targets only after they've been modified:
    for i_cam in range(exp.n_cams):
        exp.detections[i_cam].write(exp.spar.get_img_base_name(i_cam), frame)

    print("Frame " + str(frame) + " had " +
          repr([s.shape[1] for s in sorted_pos]) + " correspondences.")

    return sorted_pos, sorted_corresp, num_targs


def py_determination_proc_c(n_cams, sorted_pos, sorted_corresp, corrected):
    """Return 3d positions."""
    # Control parameters
    cpar = ControlPar(n_cams).from_file("parameters/ptv.par")

    # Volume parameters
    vpar = VolumePar().from_file("parameters/criteria.par")

    cals = []
    for i_cam in range(n_cams):
        tmp = cpar.cal_img_base_name[i_cam]
        cal = Calibration().from_file(tmp + ".ori", tmp + ".addpar")
        cals.append(cal)

    # Distinction between quad/trip irrelevant here.
    sorted_pos = np.concatenate(sorted_pos, axis=1)
    sorted_corresp = np.concatenate(sorted_corresp, axis=1)

    flat = np.array([
        corrected[i].get_by_pnrs(sorted_corresp[i]) for i in range(len(cals))
    ])
    pos, rcm = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

    if len(cals) < 4:
        print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]))
        print_corresp[:len(cals), :] = sorted_corresp
    else:
        print_corresp = sorted_corresp

    # Save rt_is in a temporary file
    fname = default_naming["corres"] + ".123456789"  # hard-coded frame number
    print(f'Prepared {fname} to write positions\n')

    try:
        with open(fname, "w", encoding='utf-8') as rt_is:
            print(f'Opened {fname} \n')
            rt_is.write(str(pos.shape[0]) + "\n")
            for pix, pt in enumerate(pos):
                pt_args = (pix + 1, ) + tuple(pt) + tuple(print_corresp[:, pix])
                rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
    except FileNotFoundError:
        msg = "Sorry, the file "+ fname + "does not exist."
        print(msg) # Sorry, the file John.txt does not exist.

    # rt_is.close()



def py_sequence_loop(exp):
    """Run a sequence of detection, stereo-correspondence, determination.

    and store the data in the cam#.XXX_targets (rewritten) and rt_is.XXX files.
    Basically it is to run the batch as in pyptv_batch.py without tracking.
    """
    n_cams, cpar, spar, vpar, tpar, cals = (
        exp.n_cams,
        exp.cpar,
        exp.spar,
        exp.vpar,
        exp.tpar,
        exp.cals,
    )

    pftVersionPar = PftVersionPar().from_file("parameters/pft_version.par")

    # sequence loop for all frames
    for frame in range(spar.get_first(), spar.get_last() + 1):
        # print(f"processing {frame} frame")

        detections = []
        corrected = []
        for i_cam in range(n_cams):
            if pftVersionPar.existing_target_flag:
                targs = read_targets(spar.get_img_base_name(i_cam), frame)
            else:
                # imname = spar.get_img_base_name(i_cam) + str(frame).encode()
                imname = spar.get_img_base_name(i_cam).decode()
                imname = Path(imname % frame)
                # imname = Path(imname.replace('#',f'{frame}'))
                # print(f'Image name {imname}')

                if not imname.exists():
                    print(f"{imname} does not exist")

                img = img_as_ubyte(imread(imname))
                # time.sleep(.1) # I'm not sure we need it here

                if 'exp1' in exp.__dict__:
                    if exp.exp1.active_Par.m_Par.Inverse:
                        print("Invert image")
                        img = 255 - img

                    if exp.exp1.active_Par.m_Par.Subtr_Mask:
                        # print("Subtracting mask")
                        try:
                            mask_name = exp.exp1.active_Par.m_Par.Base_Name_Mask.replace('#',str(i_cam+1))
                            mask = imread(mask_name)
                            img[mask] = 0

                        except ValueError:
                            print("failed to read the mask")


                high_pass = simple_highpass(img)
                targs = target_recognition(high_pass, tpar, i_cam, cpar)

            targs.sort_y()
            detections.append(targs)
            masked_coords = MatchedCoords(targs, cpar, cals[i_cam])
            pos, _ = masked_coords.as_arrays()
            corrected.append(masked_coords)

        #        if any([len(det) == 0 for det in detections]):
        #            return False

        # Corresp. + positions.
        sorted_pos, sorted_corresp, _ = correspondences(
            detections, corrected, cals, vpar, cpar)

        # Save targets only after they've been modified:
        # this is a workaround of the proper way to construct _targets name
        for i_cam in range(n_cams):
            detections[i_cam].write(
                spar.get_img_base_name(i_cam),
                frame
                )

        print("Frame " + str(frame) + " had " +
              repr([s.shape[1] for s in sorted_pos]) + " correspondences.")

        # Distinction between quad/trip irrelevant here.
        sorted_pos = np.concatenate(sorted_pos, axis=1)
        sorted_corresp = np.concatenate(sorted_corresp, axis=1)

        flat = np.array([
            corrected[i].get_by_pnrs(sorted_corresp[i])
            for i in range(len(cals))
        ])
        pos, _ = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

        # if len(cals) == 1: # single camera case
        #     sorted_corresp = np.tile(sorted_corresp,(4,1))
        #     sorted_corresp[1:,:] = -1

        if len(cals) < 4:
            print_corresp = -1 * np.ones((4, sorted_corresp.shape[1]))
            print_corresp[:len(cals), :] = sorted_corresp
        else:
            print_corresp = sorted_corresp

        # Save rt_is
        rt_is_filename = default_naming["corres"].decode()
        rt_is_filename = rt_is_filename + f'.{frame}'
        with open(rt_is_filename, "w", encoding="utf8") as rt_is:
            rt_is.write(str(pos.shape[0]) + "\n")
            for pix, pt in enumerate(pos):
                pt_args = (pix + 1, ) + tuple(pt) + tuple(print_corresp[:, pix])
                rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
        # rt_is.close()
    # end of a sequence loop


def py_trackcorr_init(exp):
    """Read all the necessary stuff into Tracker."""
    tracker = Tracker(exp.cpar, exp.vpar, exp.track_par, exp.spar, exp.cals,
                      default_naming)
    return tracker



def py_calibration(selection):
    """Calibration call."""
    if selection == 1:  # read calibration parameters into liboptv
        pass

    if selection == 2:  # run detection of targets
        pass

    if selection == 9:  # initial guess
        """Reads from a target file the 3D points and projects them on
        the calibration images
        It is the same function as show trajectories, just read from a different
        file
        """


def py_multiplanecalibration(exp):
    """Perform multiplane calibration.

    in which for all cameras the pre-processed plane in multiplane.par
    all combined.
    Overwrites the ori and addpar files of the cameras specified
    in cal_ori.par of the multiplane parameter folder.
    """
    for i_cam in range(exp.n_cams):  # iterate over all cameras
        all_known = []
        all_detected = []
        for i in range(exp.MultiPar.n_planes):  # combine all single planes

            c = exp.calPar.img_ori[i_cam][-9]  # Get camera id

            file_known = exp.MultiPar.plane_name[i] + str(c) + ".tif.fix"
            file_detected = exp.MultiPar.plane_name[i] + str(c) + ".tif.crd"

            # Load calibration point information from plane i
            known = np.loadtxt(file_known)
            detected = np.loadtxt(file_detected)

            if np.any(detected == -999):
                raise ValueError(
                    ("Using undetected points in {} will cause " +
                     "silliness. Quitting.").format(file_detected))

            num_known = len(known)
            num_detect = len(detected)

            if num_known != num_detect:
                raise ValueError(
                    f"Number of detected points {num_known} does not match" + \
                    " number of known points {num_detect} for {file_known}, {file_detected}")

            if len(all_known) > 0:
                detected[:, 0] = (all_detected[-1][-1, 0] + 1 +
                                  np.arange(len(detected)))

            # Append to list of total known and detected points
            all_known.append(known)
            all_detected.append(detected)

        # Make into the format needed for full_calibration.
        all_known = np.vstack(all_known)[:, 1:]
        all_detected = np.vstack(all_detected)

        targs = TargetArray(len(all_detected))
        for tix in range(len(all_detected)):
            targ = targs[tix]
            det = all_detected[tix]

            targ.set_pnr(tix)
            targ.set_pos(det[1:])

        # backup the ORI/ADDPAR files first
        exp.backup_ori_files()

        op = OrientPar().from_file("parameters/cal_ori.par")

        # recognized names for the flags:
        names = [
            "cc",
            "xh",
            "yh",
            "k1",
            "k2",
            "k3",
            "p1",
            "p2",
            "scale",
            "shear",
        ]
        op_names = [
            op.cc,
            op.xh,
            op.yh,
            op.k1,
            op.k2,
            op.k3,
            op.p1,
            op.p2,
            op.scale,
            op.shear,
        ]

        flags = []
        for name, op_name in zip(names, op_names):
            if op_name == 1:
                flags.append(name)

        # Run the multiplane calibration
        residuals, targ_ix, err_est = full_calibration(exp.cals[0], all_known,
                                                       targs, exp.cpar, flags)

        # Save the results
        exp._write_ori(i_cam,
                       addpar_flag=True)  # addpar_flag to save addpar file
        print("End multiplane")
