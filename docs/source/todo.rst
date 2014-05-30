Todo:
1. Calibration
a)  calibration should run in a separate window - the images are different from those used in the main run, all the parameters are separate and the only interaction with the main program is through the *.ori and *.addpar files
b) Window similar to the main window, with 4 images

The calibration menu: 

"Show Calib. Image" -command "set sel 1;calib_cmd;bindings1;" 
"Detection" -command " set sel 2;bindings0;calib_cmd"
"Manual orientation" -command " set sel 3;calib_cmd"
"Orientation with file" -command " set sel 4;calib_cmd"
"Show initial guess" -command " set sel 9;calib_cmd"
"Sortgrid" -command " set sel 5;calib_cmd"
"Sortgrid = initial guess" -command " set sel 14;calib_cmd"
"Orientation" -command " set sel 6;calib_cmd"
"Orientation with particle positions (Sequence/Tracking/Shaking)" -command "sequence_cmd 2;trackcorr_cmd 2;  set sel 10;calib_cmd"
"Orientation with particle positions (Shaking)" -command " set sel 10;calib_cmd"
"Orientation from dumbbell (Sequence/Correction/Shaking)" -command "sequence_cmd 3; set sel 12;calib_cmd"
"Restore previous Orientation" -command "restore_cmd"
"Checkpoints" -command " set sel 7;calib_cmd"
"Ap figures" -command "set sel 8;calib_cmd"
   
The menu could appear on the left, instead of the tree, as buttons

1. Show Calib. Image is "load and show images" - in Python
2. Detection -> highpass + detection_proc
3. Manual orientation Here we must have some mouse interaction, the user should click on the images (4 times per image, according to prescribed order). the clicks are associated with the pre-defined points in the parameters - man_ori.par. the results of the clicks in pixels are written in the text file, man_ori.dat
4. Orientation with file - read the man_ori.par and read another text file which was created before manually: man_ori.dat and shown on the screen using a cross and a number next to it (use chaco.api.TextBoxOverlay)







Plus, we'll need to add the following buttons/actions:
View/Edit ORI files
View/Edit calibration block files

Add sliders for the x,y,z values in ori files and sliders for 
the angles. could use Enable demo: slider_example.py





Future:
1. in peakfitting GUI part we should be able to choose from GUI which 
peak_fitting_routine do we use. a) add peakfitting.py. b) think of colloids - 
breakable particles, c) think of two-phase experiments with small and larger
particles
2. use http://code.enthought.com/projects/mayavi/docs/development/html/mayavi/auto/examples.html to get 3D of trajectories


