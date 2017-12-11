from optv.correspondences import correspondences, MatchedCoords
from optv.segmentation import target_recognition
from optv.orientation import point_positions
from optv.image_processing import preprocess_image
#from optv.tracking_framebuf import CORRES_NONE
from optv.tracker import Tracker, default_naming
from optv.calibration import Calibration
from optv.parameters import ControlParams, VolumeParams, TrackingParams, \
    SequenceParams, TargetParams
    
    
import numpy as np


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
    
    # Calibration parameters

    cals =[]
    for i_cam in xrange(len(list_of_images)):
        cal = Calibration()
        tmp = cpar.get_cal_img_base_name(i_cam)
        cal.from_file(tmp+'.ori', tmp+'.addpar')
        cals.append(cal)

    detections, corrected = [],[]
    for i_cam, img in enumerate(list_of_images):
        targs = target_recognition(img, tpar, i_cam, cpar)
        targs.sort_y()
        detections.append(targs)
        mc = MatchedCoords(targs, cpar, cals[i_cam])
        corrected.append(mc)
        
    return detections, corrected
    
def py_correspondences_proc_c(n_cams, detections, corrected):
    """ Provides correspondences 
    Inputs: 
        detections, corrected: output of the py_detection_proc_c
    Outputs:
        quadruplets, ... : four empty lists filled later with the 
    correspondences of quadruplets, triplets, pairs, and so on
    """
    
    frame = 123456789 # just a temporary workaround. todo: think how to write
    
    # Control parameters
    cpar = ControlParams(n_cams)
    cpar.read_control_par('parameters/ptv.par')
        
    cals =[]
    for i_cam in xrange(n_cams):
        cal = Calibration()
        tmp = cpar.get_cal_img_base_name(i_cam)
        cal.from_file(tmp+'.ori', tmp+'.addpar')
        cals.append(cal)


    # Sequence parameters
    spar = SequenceParams(num_cams=n_cams)
    spar.read_sequence_par('parameters/sequence.par',n_cams)
    
    # Volume parameters
    vpar = VolumeParams()
    vpar.read_volume_par('parameters/criteria.par')


#        if any([len(det) == 0 for det in detections]):
#            return False

    # Corresp. + positions.
    sorted_pos, sorted_corresp, num_targs = correspondences(
        detections, corrected, cals, vpar, cpar)
        
        
    quadruplets = sorted_pos[0]
    triplets = sorted_pos[1]
    pairs = sorted_pos[2]
    unused = [] # temporary solution 
    

    # Save targets only after they've been modified:
    for i_cam in xrange(n_cams):
        detections[i_cam].write(spar.get_img_base_name(i_cam),frame)


    print("Frame " + str(frame) + " had " \
          + repr([s.shape[1] for s in sorted_pos]) + " correspondences.")
          
    # import pdb; pdb.set_trace()

    # Distinction between quad/trip irrelevant here.
    sorted_pos = np.concatenate(sorted_pos, axis=1)
    sorted_corresp = np.concatenate(sorted_corresp, axis=1)

    flat = np.array([corrected[i].get_by_pnrs(sorted_corresp[i]) \
                     for i in xrange(len(cals))])
    pos, rcm = point_positions(
        flat.transpose(1,0,2), cpar, cals)

    # Save rt_is
    rt_is = open(default_naming['corres']+'.'+str(frame), 'w')
    rt_is.write(str(pos.shape[0]) + '\n')
    for pix, pt in enumerate(pos):
        pt_args = (pix + 1,) + tuple(pt) + tuple(sorted_corresp[:,pix])
        rt_is.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % pt_args)
    rt_is.close()
    
    
#     import copy
#     quadruplets = [[] for i in xrange(n_cams)]
#     triplets  = copy.copy(quadruplets)
#     pairs = copy.copy(triplets)
#     unused = copy.copy(pairs)
# 
#     
# 
# 
#     for row in sorted_corresp.T:
#         if sum(row != -1) == 4:
#             print('quadruplet')
#             for i_cam, ix in enumerate(row):
#                 print(i_cam,ix,sorted_pos[i_cam][ix])
#                 quadruplets[i_cam].append(sorted_pos[i_cam][ix])
#     
#         elif sum(row != -1) == 3:
#             # triplets
#             for i_cam, ix in enumerate(row):
#                 if ix == -1:
#                     continue
#                 triplets[i_cam].append(sorted_pos[i_cam][ix])
#                 
#         elif sum(row != -1) == 2:
#             # pairs
#             for i_cam, ix in enumerate(row):
#                 if ix == -1:
#                     continue
#                 pairs[i_cam].append(sorted_pos[i_cam][ix])
#         else:
#             # unused
#                 unused[i_cam].append(sorted_pos[i_cam][ix])

    return quadruplets,triplets,pairs,unused
    
    
