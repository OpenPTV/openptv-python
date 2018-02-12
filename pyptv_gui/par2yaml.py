"""
This file converts all the parameter files that are recognizable by the parameters.py 
Classes to YAML files
Usage:
    par2yaml ~/test_cavity

"""
from __future__ import print_function
from __future__ import absolute_import

import sys, os
import parameters


def paramsFactory(filename, n_img, n_pts = 4):
    if filename == "ptv.par":
        return parameters.PtvParams()
    if filename == "cal_ori.par":
        return parameters.CalOriParams(n_img)
    if filename == "sequence.par":
        return parameters.SequenceParams(n_img)
    if filename == "sequence_db.par":
        return parameters.SequenceParams(n_img)
    if filename == "criteria.par":
        return parameters.CriteriaParams()
    if filename == "targ_rec.par":
        return parameters.TargRecParams(n_img)
    if filename == "man_ori.par":
        return parameters.ManOriParams(n_img, n_pts)
    if filename == "detect_plate.par":
        return parameters.DetectPlateParams()
    if filename == "orient.par":
        return parameters.OrientParams()
    if filename == "track.par":
        return parameters.TrackingParams()
    if filename == "pft_version.par":
        return parameters.PftVersionParams()
    if filename == "examine.par":
        return parameters.ExamineParams()
    if filename == "dumbbell.par":
        return parameters.DumbbellParams()
    if filename == "shaking.par":
        return parameters.ShakingParams()
    if filename == 'multi_planes.par':
        return parameters.MultiPlaneParams()
    if filename == 'sortgrid.par':
        return parameters.SortGridParams()
        
    return None

        
def main(referenceParamsDir):
    print('inside run')
    ptvParams = parameters.PtvParams()
    ptvParams.path = referenceParamsDir
    print(ptvParams.path)
    import pdb; pdb.set_trace()
    ptvParams.read()
    n_img = ptvParams.n_img
    print(n_img)
    n_pts = 4
    
    
    #loop all .par files in the parameters directory
    paramdirfiles = os.listdir(referenceParamsDir)
    print(paramdirfiles)
    for paramfile in paramdirfiles:
        if paramfile.endswith('.par'):
            yield SingleParamFile_to_yaml, paramfile, n_img, n_pts


def SingleParamFile_to_yaml(paramfile, n_img, n_pts):
    params = paramsFactory(paramfile, n_img, n_pts)
    if params is None:
        print("No parameters class found for file %s." % (paramfile))
        return
    
    params.path = referenceParamsDir
    referenceFile = params.filepath()
    try:
        params.read()
    except:
        print("Error reading %s from %s:" % (paramfile, params.path), sys.exc_info())
        assert False

    try:
        params._to_yaml()
    except:
        print("Error writing YAML %s " % paramfile.replace('.par','.yaml') )
        assert False    
    
    
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        referenceParamsDir  = os.path.join(os.path.abspath(sys.argv[1]),'parameters')
    else:
        referenceParamsDir = '/Users/alex/Documents/OpenPTV/test_cavity/parameters'
    ptvParams = parameters.PtvParams()
    ptvParams.path = referenceParamsDir
    print('Working in %s' % ptvParams.path)
    ptvParams.read()
    n_img = ptvParams.n_img
    print(n_img)
    n_pts = 4
    
    
    #loop all .par files in the parameters directory
    paramdirfiles = os.listdir(referenceParamsDir)
    print(paramdirfiles)
    for paramfile in paramdirfiles:
        if paramfile.endswith('.par'):
            SingleParamFile_to_yaml(paramfile, n_img, n_pts)
    
    
    
