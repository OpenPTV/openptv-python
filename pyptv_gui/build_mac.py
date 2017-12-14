"""
some trick to get it compiled on mac
"""
import os

cwd = os.getcwd()


for line in file('setup.py'):
	if line.strip().startswith('ext_modules'):
		lst = line


filenames = lst.partition('"ptv1.pyx",')[-1].lstrip().strip('],\n').split(',')
print filenames
# filenames = ["segmentation.c", "tools.c","image_processing.c", "trafo.c", "jw_ptv.c", "peakfitting.c", "rotation.c", "correspondences.c", "epi.c", "multimed.c", "ray_tracing.c","imgcoord.c","lsqadj.c", "orientation.c","sortgrid.c", "pointpos.c","intersect.c"]
newlines = []

# taking those which are C code:
# for filename in (f for f in filenames if f.endswith('.c')):

src_path = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0],'src_c')
os.chdir(src_path)


# or using only the given list
for filename in filenames:
	print filename.strip().strip('"')
	f = file(filename.strip().strip('"'))
	for line in f:
		if 'ptv.h' in line:
			pass # print line
		else:
			newlines.append(line)
	
	
	f.close()
	
	
	
	
outfile = file('tmp.c','w')
outfile.write('#include "ptv.h"\n')
outfile.writelines(newlines)
outfile.close()

print os.getcwd()

os.system('python setup_mac.py build_ext --inplace')
os.remove('tmp.c')
