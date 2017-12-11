from optv.correspondences import correspondences, MatchedCoords
from optv.segmentation import target_recognition
from optv.orientation import point_positions
from optv.image_processing import preprocess_image
#from optv.tracking_framebuf import CORRES_NONE
from optv.tracker import Tracker, default_naming
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, \
    SequenceParams, TargetParams


def py_init_proc_c():
    """ Was supposed to read/set globals that do not exist anymore """
    pass
    
def py_set_img(img,i):
    """ Not used anymore, was transferring images to the C """
    pass
    
def py_start_proc_c():
    """ Read parameters """
    with open('parameters/ptv.par','r') as f:
        n_cams = int(f.readline())

    # Control parameters
    cpar = ControlParams(n_cams)
    cpar.read_control_par('parameters/ptv.par')

    # Sequence parameters
    spar = SequenceParams(num_cams=n_cams)
    spar.read_sequence_par('parameters/sequence.par',n_cams)

    # Volume parameters
    vpar = VolumeParams()
    vpar.read_volume_par('parameters/criteria.par')

    # Tracking parameters
    track_par = TrackingParams()
    track_par.read_track_par('parameters/track.par')

    # Target parameters
    tpar = TargetParams()
    tpar.read('parameters/targ_rec.par')

    # 

    # Calibration parameters

    cals =[]
    for i_cam in xrange(n_cams):
        cal = Calibration()
        tmp = cpar.get_cal_img_base_name(i_cam)
        cal.from_file(tmp+'.ori', tmp+'.addpar')
        cals.append(cal)
        
def py_pre_processing_c(list_of_images):
    """ Image pre-processing, mostly highpass filter, could be extended in the future """
    cpar = ControlParams(len(list_of_images))
    cpar.read_control_par('parameters/ptv.par')
    newlist = []
    for img in list_of_images:
        newlist.append(preprocess_image(img, 0, cpar, 12))
    return newlist
    
def py_detection_proc_c(list_of_images):
    """ Detection of targets """
    cpar = ControlParams(len(list_of_images))
    cpar.read_control_par('parameters/ptv.par')
    tpar = TargetParams()
    tpar.read('parameters/targ_rec.par')

    detections = []
    for i_cam,img in enumerate(list_of_images):
        targs = target_recognition(img, tpar, i_cam, cpar)
        targs.sort_y()
        detections.append(targs)
        
    return detections
    

