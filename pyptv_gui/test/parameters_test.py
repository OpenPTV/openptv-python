"""
This file uses Python nose to test the correctness of writing and reading parameter files.
To run the test, 'parameters' directory should exists in the 'tests' directory with all .par files (with the correct names).
The parameters directory is iterated, and .par files are read (using the suitable param classes). Then they are written back, and the files are compared.
"""

import os
import sys

sys.path += ['..']
import parameters


referenceParamsDir = "testing_fodder/parameters"
testParamsDir = "parameters_test"


def paramsFactory(filename, n_img, n_pts = 4):
	if filename == "ptv.par":
		return parameters.PtvParams()
	if filename == "cal_ori.par":
		return parameters.CalOriParams(n_img)
	if filename == "sequence.par":
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
		
	return None

	
def setup_func():
	if not os.path.exists(testParamsDir):
		os.mkdir(testParamsDir)

def teardown_func():
	#print "Finished testing", __file__
	pass

	
def testReadWrite():
	#get n_img from ptv.par
	ptvParams = parameters.PtvParams()
	ptvParams.path = referenceParamsDir
	ptvParams.read()
	n_img = ptvParams.n_img
	n_pts = 4
	
	
	#loop all .par files in the parameters directory
	paramdirfiles = os.listdir(referenceParamsDir)
	for paramfile in paramdirfiles:
		if paramfile.strip()[-4:] == '.par':
			yield checkSingleParamFile, paramfile, n_img, n_pts
(testReadWrite.setup, testReadWrite.teardown) = (setup_func, teardown_func)


def checkSingleParamFile(paramfile, n_img, n_pts):
	params = paramsFactory(paramfile, n_img, n_pts)
	if params is None:
		print "No parameters class found for file %s." % (paramfile)
		assert False
		return
	
	params.path = referenceParamsDir
	referenceFile = params.filepath()
	try:
		params.read()
	except:
		print "Error reading %s from %s:" % (paramfile, params.path), sys.exc_info()
		assert False
	
	params.path = testParamsDir
	testFile = params.filepath()
	try:
		params.write()
	except:
		print "Error writing %s to %s:" % (paramfile, params.path), sys.exc_info()
		assert False

	if not compareFiles(referenceFile, testFile, True):
		assert False


def compareFiles(f1, f2, verbose = False):
	readlines = lambda f: file(f, "r").readlines()
	lns1 = readlines(f1)
	lns2 = readlines(f2)
	
	def getLen(lns):
		for i in reversed(range(len(lns))):
			if lns[i].strip() != '':
				return i
		return 0
	
	nlns1 = getLen(lns1)
	nlns2 = getLen(lns2)
	if nlns1 != nlns2:
		if verbose:
			print "Files %s and %s have different amount of lines (%d/%d)." % (f1, f2, nlns1, nlns2)
		return False
		
	for n in range(nlns1):
		l1 = lns1[n].strip()
		l2 = lns2[n].strip()
		if l1 != l2:
			try: #check for float values
				if abs(float(l1)-float(l2)) > 1e-10: #l1!=l2
					raise					
			except:
				if verbose:
					print "Files %s and %s differ at line %d (%s/%s)" % (f1, f2, n+1, l1, l2)
				return False
	return True
	
	
	
	
	
	
	
