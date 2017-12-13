#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:47:14 2017

@author: alex
"""

import sys, os
sys.path.append(os.path.abspath('../openptv-python/pyptv_gui/'))

from ptv1 import *
import parameters as par
import matplotlib.pyplot as plt


cpar = ControlParams(4)
cpar.read_control_par('parameters/ptv.par')
n_cams = cpar.get_num_cams()
print(n_cams)


im = imread('cal/cam1.tif')
im1 = simple_highpass(im,cpar)

plt.imshow(im); plt.show()
plt.imshow(im1); plt.show()


# Sequence parameters
spar = SequenceParams(num_cams = n_cams)
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

cals =[]
for i_cam in xrange(n_cams):
    cal = Calibration()
    tmp = cpar.get_cal_img_base_name(i_cam)
    print(tmp)
    cal.from_file(tmp+'.ori', tmp+'.addpar')
    cals.append(cal)




calParams = par.CalOriParams(n_cams,'parameters')
calParams.read()

calParams.img_cal_name

list_of_images = []
for imname in calParams.img_cal_name:
    list_of_images.append(imread(imname))
    
list_of_images = py_pre_processing_c(list_of_images,cpar)
d,c = py_detection_proc_c(list_of_images,cpar,tpar,cals)