""" OpenPTV-Python is the GUI for the OpenPTV (http://www.openptv.net) liboptv library 
based on Python/Enthought Traits GUI/Numpy/Chaco


Copyright (c) 2008-2013, Tel Aviv University

Copyright (c) 2013 - the OpenPTV team

The software is distributed under the terms of MIT-like license
http://opensource.org/licenses/MIT

.. moduleauthors:: OpenPTV group

"""

from traits.api \
	import HasTraits, Str, Int, List, Bool, Enum, Instance, Any
from traitsui.api \
	import TreeEditor, TreeNode, View, Item, \
		    Handler, Group		    
from enable.component_editor import ComponentEditor
from chaco.api import Plot, ArrayPlotData, gray
from traitsui.menu import MenuBar, Menu, Action
from chaco.tools.api import  ZoomTool,PanTool
from scipy.misc import imread
from threading import Thread
from pyface.api import GUI

import os
import sys
import numpy as np

# Parse inputs:

# Get the path to the software
if len(sys.argv) > 0:
	software_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
else:
	software_path = os.path.abspath(os.getcwd())
	
if not os.path.isdir(software_path):
	print ("Wrong experimental directory %s " % software_path)	

	
# Path to the experiment
if len(sys.argv) > 1:
	exp_path = os.path.abspath(sys.argv[1])
	if not os.path.isdir(exp_path):
		print ("Wrong experimental directory %s " % exp_path)

		
# change directory to the software path
try:
	os.chdir(software_path)
except:
	print("Wrong software path %s " % software_path)


src_path = os.path.join(os.path.split(software_path)[0],'src_c')
print 'src_path=', src_path
if not os.path.isdir(src_path):
	print("Wrong src_c path %s" % src_path)
sys.path.append(src_path) 

import ptv1 as ptv
from tracking_framebuf import read_targets

# pyPTV specific imports
import general
import parameters as par
from parameter_gui import *
from calibration_gui import *
from directory_editor import DirectoryEditorDialog
from quiverplot import QuiverPlot
from demo import *

if len(sys.argv) < 2:
	directory_dialog = DirectoryEditorDialog()
	directory_dialog.configure_traits()
	exp_path = directory_dialog.dir_name # default_path+os.sep+'exp1'

cwd = os.getcwd()

try:
	os.chdir(exp_path)
except:
	print('Wrong experimental directory %s' % exp_path)

# 
class Clicker(ImageInspectorTool):
	"""  Clicker class handles right mouse click actions from the tree and menubar actions
	"""
	left_changed=Int(1)
	right_changed=Int(1)
	x=0
	y=0
	def normal_left_down(self, event):
		""" Handles the left mouse button being clicked.
		Fires the **new_value** event with the data (if any) from the event's
		position.
		"""
		plot = self.component
		if plot is not None:
			ndx = plot.map_index((event.x, event.y))
			x_index, y_index = ndx
			image_data=plot.value
			self.x=(x_index)
			self.y=(y_index)
			self.data_value=image_data.data[y_index, x_index]
			self.left_changed=1-self.left_changed
			self.last_mouse_position = (event.x, event.y)
		return

	def normal_right_down(self, event):
		plot = self.component
		if plot is not None:
			ndx = plot.map_index((event.x, event.y))

			x_index, y_index = ndx
			# image_data=plot.value
			self.x=(x_index)
			self.y=(y_index)
			
			self.right_changed=1-self.right_changed
			print self.x
			print self.y
			
			self.last_mouse_position = (event.x, event.y)
		return

	def normal_mouse_move(self, event):
		pass

	def __init__(self, *args, **kwargs):
		super(Clicker, self).__init__(*args, **kwargs)

# --------------------------------------------------------------
class CameraWindow (HasTraits):
	""" CameraWindow class contains the relevant information and functions for a single camera window: image, zoom, pan
	important members:
		_plot_data  - contains image data to display (used by update_image)
		_plot - instance of Plot class to use with _plot_data
		_click_tool - instance of Clicker tool for the single camera window, to handle mouse processing
	"""
	_plot_data=Instance(ArrayPlotData)
	_plot=Instance(Plot)
	_click_tool=Instance(Clicker)
	rclicked=Int(0)
	
	cam_color = ''

	name=Str
	view = View( Item(name='_plot',editor=ComponentEditor(), show_label=False) )
	# view = View( Item(name='_plot',show_label=False) )
   
	def __init__(self, color):
		""" Initialization of plot system 
		"""
		padd=25
		self._plot_data=ArrayPlotData()
		self._plot=Plot(self._plot_data, default_origin="top left")
		self._plot.padding_left=padd
		self._plot.padding_right=padd
		self._plot.padding_top=padd
		self._plot.padding_bottom=padd
		self.right_p_x0,self.right_p_y0,self.right_p_x1,self.right_p_y1,self._quiverplots=[],[],[],[],[]
		self.cam_color = color

	def attach_tools(self):
		""" attach_tools(self) contains the relevant tools: clicker, pan, zoom
		"""
		self._click_tool=Clicker(self._img_plot)
		self._click_tool.on_trait_change(self.left_clicked_event, 'left_changed') #set processing events for Clicker
		self._click_tool.on_trait_change(self.right_clicked_event, 'right_changed')
		self._img_plot.tools.append(self._click_tool)
		pan = PanTool(self._plot, drag_button = 'middle')
		zoom_tool= ZoomTool(self._plot, tool_mode="box", always_on=False)
#		zoom_tool = BetterZoom(component=self._plot, tool_mode="box", always_on=False)
		zoom_tool.max_zoom_out_factor=1.0 # Disable "bird view" zoom out
		self._img_plot.overlays.append(zoom_tool)
		self._img_plot.tools.append(pan)

		
	def left_clicked_event(self): #TODO: why do we need the clicker_tool if we can handle mouse clicks here?
		""" left_clicked_event - processes left click mouse avents and displays coordinate and grey value information
		on the screen
		"""
		print "x = %d, y= %d, grey= %d " % (self._click_tool.x,self._click_tool.y,self._click_tool.data_value)
		#need to priny gray value

	def right_clicked_event(self):
		self.rclicked=1 #flag that is tracked by main_gui, for right_click_process function of main_gui
		#self._click_tool.y,self.name])
		#self.drawcross("coord_x","coord_y",self._click_tool.x,self._click_tool.y,"red",5)
		#print ("right clicked, "+self.name)
		#need to print cross and manage other camera's crosses


	def update_image(self,image,is_float):
		""" update_image - displays/updates image in the curren camera window
		parameters:
			image - image data
			is_float - if true, displays an image as float array,
			else displays as byte array (B&W or gray)
		example usage:
			update_image(image,is_float=False)
		"""
		if is_float:
			self._plot_data.set_data('imagedata',image.astype(np.float))
		else:
			self._plot_data.set_data('imagedata',image.astype(np.byte))
	   
		if not hasattr(self,'_img_plot'): #make a new plot if there is nothing to update
			self._img_plot=Instance(ImagePlot)
			self._img_plot=self._plot.img_plot('imagedata',colormap=gray)[0]
			self.attach_tools()
			
			
		   # self._plot.request_redraw()

	def drawcross(self, str_x,str_y,x,y,color1,mrk_size,marker1="plus"):
		""" drawcross draws crosses at a given location (x,y) using color and marker in the current camera window
		parameters:
			str_x - label for x coordinates
			str_y - label for y coordinates
			x - array of x coordinates
			y - array of y coordinates
			mrk_size - marker size
			makrer1 - type of marker, e.g "plus","circle"
		example usage:
			drawcross("coord_x","coord_y",[100,200,300],[100,200,300],2)
			draws plus markers of size 2 at points (100,100),(200,200),(200,300)
		"""
		self._plot_data.set_data(str_x,x)
		self._plot_data.set_data(str_y,y)
		self._plot.plot((str_x,str_y),type="scatter",color=color1,marker=marker1,marker_size=mrk_size)
		#self._plot.request_redraw()

	def drawquiver(self,x1c,y1c,x2c,y2c,color,linewidth=1.0):
		""" drawquiver draws multiple lines at once on the screen x1,y1->x2,y2 in the current camera window
		parameters:
			x1c - array of x1 coordinates
			y1c - array of y1 coordinates
			x2c - array of x2 coordinates
			y2c - array of y2 coordinates
			color - color of the line
			linewidth - linewidth of the line
		example usage:
			drawquiver ([100,200],[100,100],[400,400],[300,200],'red',linewidth=2.0)
			draws 2 red lines with thickness = 2 :  100,100->400,300 and 200,100->400,200
					
		"""
		x1,y1,x2,y2=self.remove_short_lines(x1c,y1c,x2c,y2c)
		if len(x1)>0:
			xs=ArrayDataSource(x1)
			ys=ArrayDataSource(y1)

			quiverplot=QuiverPlot(index=xs,value=ys,\
								  index_mapper=LinearMapper(range=self._plot.index_mapper.range),\
								  value_mapper=LinearMapper(range=self._plot.value_mapper.range),\
								  origin = self._plot.origin,arrow_size=0,\
								  line_color=color,line_width=linewidth,ep_index=np.array(x2),ep_value=np.array(y2)
								  )
			self._plot.add(quiverplot)
			self._quiverplots.append(quiverplot) #we need this to track how many quiverplots are in the current plot
			# import pdb; pdb.set_trace()
			
	def remove_short_lines(self,x1,y1,x2,y2):
		""" removes short lines from the array of lines 
		parameters:
			x1,y1,x2,y2 - start and end coordinates of the lines
		returns:
			x1f,y1f,x2f,y2f - start and end coordinates of the lines, with short lines removed
		example usage:
			x1,y1,x2,y2=remove_short_lines([100,200,300],[100,200,300],[100,200,300],[102,210,320])
			3 input lines, 1 short line will be removed (100,100->100,102)
			returned coordinates:
			x1=[200,300]; y1=[200,300]; x2=[200,300]; y2=[210,320]
		"""
		dx,dy=2,2 #minimum allowable dx,dy
		x1f,y1f,x2f,y2f=[],[],[],[]
		for i in range(len(x1)):
			if abs(x1[i]-x2[i])>dx or abs(y1[i]-y2[i])>dy:
				x1f.append(x1[i])
				y1f.append(y1[i])
				x2f.append(x2[i])
				y2f.append(y2[i])
		return x1f,y1f,x2f,y2f
				
	def drawline(self,str_x,str_y,x1,y1,x2,y2,color1):
		""" drawline draws 1 line on the screen by using lineplot x1,y1->x2,y2
		parameters:
			str_x - label of x coordinate
			str_y - label of y coordinate
			x1,y1,x2,y2 - start and end coordinates of the line
			color1 - color of the line
		example usage:
			drawline("x_coord","y_coord",100,100,200,200,red)
			draws a red line 100,100->200,200
		"""
		self._plot_data.set_data(str_x,[x1,x2])
		self._plot_data.set_data(str_y,[y1,y2])
		self._plot.plot((str_x,str_y),type="line",color=color1)
		#self._plot.request_redraw()	   


class TrackThread(Thread):
	""" TrackThread is used by tracking with display function - runs separate thread that updates the gui
	"""
	def run(self):
		print "tracking with display thread started"
		run_info = ptv.py_trackcorr_init() #init the relevant C function
		for step in range(*run_info.get_sequence_range()): #Loop over each step in sequence
			self.track_step=step
            
            #Call C function to process current step and store results for plotting
			self.intx0, self.intx1, self.intx2, self.inty0, self.inty1, \
                self.inty2, self.pnr1, self.pnr2, self.pnr3, self.m_tr = \
			    ptv.py_trackcorr_loop(run_info, step, display=1)

			self.can_continue=False
			GUI.invoke_later(setattr, main_gui, 'update_thread_plot', True) #invoke plotting when system is ready to process it
			while not self.can_continue: # wait while plotting of the current step is finished, then continue for the next step
				pass
			del self.intx0,self.intx1

		ptv.py_trackcorr_finish(run_info, step + 1) # call finishing C function (after all the steps in the loop are processed)
		for i in range(len(main_gui.camera_list)): # refresh cameras
			main_gui.camera_list[i]._plot.request_redraw()
		
		print "tracking with display thread finished"

class TreeMenuHandler (Handler):
	""" TreeMenuHanlder contains all the callback actions of menu bar, processing of tree editor, and reactions of the GUI to the user clicks
	possible function declarations:
		1) to process menubar actions:
			def function(self, info):
		parameters: self - needed for member function declaration,
			    info - contains pointer to calling parent class (e.g main_gui)
			    To access parent class objects use info.object, for example info.object.exp1 gives access to exp1 member of main_gui class
		2) to process tree editor actions:
			def function(self,editor,object) - see examples below
			    	
	"""
	def configure_main_par(self, editor, object):
		experiment = editor.get_parent(object)
		paramset = object
		print 'Total paramsets:', len(experiment.paramsets)
		if paramset.m_params==None:
			#TODO: is it possible that control reaches here? If not, probably the if should be removed.
			paramset.m_params=Main_Params()
		else:
			paramset.m_params._reload()
		paramset.m_params.edit_traits(kind='modal')
		
		
	def configure_cal_par(self, editor, object):
		experiment = editor.get_parent(object)
		paramset = object
		print len(experiment.paramsets)
		if paramset.c_params==None:
			#TODO: is it possible that control reaches here? If not, probably the if should be removed.
			paramset.c_params=Calib_Params()
		else:
			paramset.c_params._reload()
		paramset.c_params.edit_traits(kind='modal')
		
	def configure_track_par(self, editor, object):
		experiment = editor.get_parent(object)
		paramset = object
		print len(experiment.paramsets)
		if paramset.t_params==None:
			#TODO: is it possible that control reaches here? If not, probably the if should be removed.
			paramset.t_params=Tracking_Params()
		paramset.t_params.edit_traits(kind='modal')
		
	def set_active(self, editor, object):
		experiment = editor.get_parent(object)
		paramset = object
 		# experiment.active_params = paramset
 		experiment.setActive(paramset)
		experiment.changed_active_params = True
		# editor.object.__init__()
				
	def copy_set_params(self, editor, object):
		experiment = editor.get_parent(object)
		paramset = object
		i = 1
		new_name = None
		new_dir_path = None
		flag = False
		while not flag:
			new_name = "%s (%d)" % (paramset.name, i)
			new_dir_path = "%s%s" % (general.par_dir_prefix, new_name)
			if not os.path.isdir(new_dir_path):
				flag = True
			else:
				i=i+1
		
		os.mkdir(new_dir_path)
		par.copy_params_dir(paramset.par_path, new_dir_path)
		experiment.addParamset(new_name, new_dir_path)

	def rename_set_params(self,editor,object):
		""" rename_set_params renames the node name on the tree and also the folder of parameters
		"""
		experiment = editor.get_parent(object)
		paramset = object
		# rename
		editor._menu_rename_node()
		new_name = object.name
		new_dir_path = general.par_dir_prefix + new_name
		os.mkdir(new_dir_path)
		par.copy_params_dir(paramset.par_path, new_dir_path)
		[os.remove(os.path.join(paramset.par_path,f)) for f in os.listdir(paramset.par_path)]
		os.rmdir(paramset.par_path)
		experiment.removeParamset(paramset)
		experiment.addParamset(new_name, new_dir_path)
		
	def delete_set_params(self, editor, object):
		""" delete_set_params deletes the node and the folder of parameters 
		"""
		# experiment = editor.get_parent(object)
		paramset = object
		# delete node
		editor._menu_delete_node()
		# delete all the parameter files
		[os.remove(os.path.join(paramset.par_path,f)) for f in os.listdir(paramset.par_path)]
		# remove folder
		os.rmdir(paramset.par_path)


	#------------------------------------------
	# Menubar actions
	#------------------------------------------
	def new_action(self,info):
		print("Not implemented")
		
	def open_action(self,info):
		directory_dialog = DirectoryEditorDialog()
		directory_dialog.edit_traits()
		exp_path = directory_dialog.dir_name
		print "Changing experimental path to %s" % exp_path
		os.chdir(exp_path)
		info.object.exp1.populate_runs(exp_path)
		
	def exit_action(self,info):
		print ("not implemented yet")
		
	def saveas_action(self,info):
		print ("not implemented yet")
		
	def showimg_action(self,info):
		print "not implemented"
		
		info.object.update_plots(info.object.orig_image)
		 
	def highpass_action(self,info):
		""" highpass_action - calls ptv.py_pre_processing_c() binding which does highpass on working images (object.orig_image)
		that were set with init action
		"""
		print ("highpass started")
		ptv.py_pre_processing_c() # call the binding
		info.object.update_plots(info.object.orig_image)
		print ("highpass finished")

	def img_coord_action(self,info):
		""" img_coord_action - runs detection function by using ptv.py_detection_proc_c() binding. results are extracted
		with help of ptv.py_get_pix(x,y) binding and plotted on the screen
		"""
		print ("detection proc started")
		ptv.py_detection_proc_c()
		print ("detection proc finished")
		x=[]
		y=[]
		ptv.py_get_pix(x,y) # extract detected points from C with help of py_get_pix binding
		info.object.drawcross("x","y",x,y,"blue",3)

		   
	def corresp_action(self,info):
		""" corresp_action - calls ptv.py_correspondences_proc_c(quadriplets,triplets,pairs, unused) binding.
		Result of correspondence action is filled to uadriplets,triplets,pairs, unused arrays
		"""
		print ("correspondence proc started")
		quadriplets=[]
		triplets=[]
		pairs=[]
		unused=[]
		ptv.py_correspondences_proc_c(quadriplets,triplets,pairs, unused)
		info.object.clear_plots(remove_background=False)
		#info.object.update_plots(info.object.orig_image)
		info.object.drawcross("quad_x","quad_y",quadriplets[0],quadriplets[1],"red",3) #draw quadriplets, triplets, etc...
		info.object.drawcross("tripl_x","tripl_y",triplets[0],triplets[1],"green",3)
		info.object.drawcross("pair_x","pair_y",pairs[0],pairs[1],"yellow",3)
		info.object.drawcross("unused_x","unused_y",unused[0],unused[1],"blue",3)

	def init_action(self,info):
		""" init_action - clears existing plots from the camera windows,
		initializes C image arrays with mainGui.orig_image and 
		calls appropriate start_proc_c by using ptv.py_start_proc_c()
		"""
		mainGui = info.object
		mainGui.exp1.syncActiveDir() #synchronize the active run params dir with the temp params dir
		
		for i in range (0,len(mainGui.camera_list)):
			exec("mainGui.orig_image[%d]=imread(mainGui.exp1.active_params.m_params.Name_%d_Image).astype(np.ubyte)" %(i,i+1))
			if hasattr(mainGui.camera_list[i],'_img_plot'):
				del mainGui.camera_list[i]._img_plot
		mainGui.clear_plots()
		print("\nInit action\n")
		mainGui.update_plots(mainGui.orig_image,is_float=1)
		mainGui.set_images(mainGui.orig_image)
	   
		ptv.py_start_proc_c()
		mainGui.pass_init=True
		print ("done")
			 
		 
	 
	def calib_action(self,info):
		""" calib_action - initializes calib class with appropriate number of plot windows, 
		passes to calib class pointer to ptv module and to exp1 class,
		invokes the calibration GUI
		"""
		print ("Starting calibration dialog")
		mainGui = info.object
		mainGui.pass_init=False
		num_cams=len(mainGui.camera_list)
		#TODO: calibration_params should work with the actual parameters of the current run (not par.temp_path)
		# Right. I replace par_path with info.object.exp1.active_params.par_path, Alex, 20.01. 10:43
		mainGui.calib = calibration_gui(info.object.exp1.active_params.par_path)
		for i in range (0,num_cams):
				mainGui.calib.camera.append(plot_window())
				mainGui.calib.camera[i].name="Camera"+str(i+1)
				mainGui.calib.camera[i].cameraN=i
				mainGui.calib.camera[i].py_rclick_delete=ptv.py_rclick_delete
				mainGui.calib.camera[i].py_get_pix_N=ptv.py_get_pix_N
				
   
		mainGui.calib.ptv=ptv
		mainGui.calib.exp1=mainGui.exp1 #pass all the parameters to calib class
		mainGui.calib.configure_traits()  
   
	def sequence_action(self,info):
		"""sequence action - implements binding to C sequence function. Original function was split into 2 parts:
			1) initialization - binded by ptv.py_sequence_init(..) function
			2) main loop processing - binded by ptv.py_sequence_loop(..) function
		"""
		extern_sequence=info.object.plugins.sequence_alg
		if extern_sequence!='default':
			try:
				current_path=os.path.abspath(os.getcwd()) # save working path
				os.chdir(software_path) #change to software path, to load tracking module
				seq=__import__(extern_sequence) #import choosen tracker from software dir
			except:
				print "Error loading "+extern_sequence+". Falling back to default sequence algorithm"
				extern_sequence='default'
			os.chdir(current_path) # change back to working path
		if extern_sequence=='default':
			n_camera=len(info.object.camera_list)
			print ("Starting sequence action (default algorithm)")
			seq_first=info.object.exp1.active_params.m_params.Seq_First
			seq_last=info.object.exp1.active_params.m_params.Seq_Last
			print seq_first,seq_last
			base_name=[]
			for i in range (n_camera):
				exec("base_name.append(info.object.exp1.active_params.m_params.Basename_%d_Seq)" %(i+1))
				print base_name[i]
	   
			ptv.py_sequence_init(0) #init C sequence function
			stepshake=ptv.py_get_from_sequence_init() #get parameters and pass to main loop
			if not stepshake:
				stepshake=1
			
			print stepshake
			temp_img=np.array([],dtype=np.ubyte)
			# main loop - format image name, read it and call v.py_sequence_loop(..) for current step
			for i in range(seq_first,seq_last+1,stepshake):
				seq_ch="%04d" % i
					
				for j in range (n_camera):
					img_name=base_name[j]+seq_ch
					print ("Setting image: ",img_name)
					try:
						temp_img=imread(img_name).astype(np.ubyte)
					except:
						print "Error reading file"
						   
					ptv.py_set_img(temp_img,j)
					ptv.py_sequence_loop(0,i)
		else:
			print "Sequence by using "+extern_sequence
			sequence=seq.Sequence(ptv=ptv, exp1=info.object.exp1,camera_list=info.object.camera_list)
			sequence.do_sequence()
			#print "Sequence by using "+extern_sequence+" has failed."

	def track_no_disp_action(self, info):
		""" track_no_disp_action uses ptv.py_trackcorr_loop(..) binding to call tracking without display
		"""				   
		extern_tracker=info.object.plugins.track_alg
		if extern_tracker!='default':
			try:
				current_path=os.path.abspath(os.getcwd()) # save working path
				os.chdir(software_path) #change to software path, to load tracking module
				track=__import__(extern_tracker) #import choosen tracker from software dir
			except:
				print "Error loading "+extern_tracker+". Falling back to default tracker"
				extern_tracker='default'
			os.chdir(current_path) # change back to working path
		if extern_tracker=='default':
			print "Using default tracker"
			run_info = ptv.py_trackcorr_init()
			print run_info.get_sequence_range()
			for step in range(*run_info.get_sequence_range()):
				print step
				ptv.py_trackcorr_loop(run_info, step, display=0)
            
   			#finalize tracking
			ptv.py_trackcorr_finish(run_info, step + 1)
		else:
			print "Tracking by using "+extern_tracker
			tracker=track.Tracking(ptv=ptv, exp1=info.object.exp1)
			tracker.do_tracking()
			
			
		print "tracking without display finished"
	 
	def track_disp_action(self,info):
		""" tracking with display is handled by TrackThread which does processing step by step and
		waits for GUI to update before procceeding to the next step
		"""
		info.object.clear_plots(remove_background=False)
		info.object.tr_thread=TrackThread()
		info.object.tr_thread.start()

	def track_back_action(self,info):
		""" tracking back action is handled by ptv.py_trackback_c() binding
		"""
		print ("Starting back tracking")
		ptv.py_trackback_c()
		   
	def threed_positions(self,info):
		ptv.py_determination_proc_c(0)
   
        def multigrid_demo(self,info):
                demo_window=DemoGUI(ptv=ptv, exp1=info.object.exp1)
                demo_window.configure_traits()
                
   
	def detect_part_track(self, info):
		""" track detected particles is handled by 2 bindings:
			1) tracking_framebuf.read_targets(..)
			2) ptv.py_get_mark_track_c(..)
		"""
		info.object.clear_plots(remove_background=False) #clear everything
		info.object.update_plots(info.object.orig_image,is_float=1)
		
		prm = info.object.exp1.active_params.m_params
		seq_first = prm.Seq_First #get sequence parameters
		seq_last = prm.Seq_Last
		base_names = [prm.Basename_1_Seq, prm.Basename_2_Seq, 
			prm.Basename_3_Seq, prm.Basename_4_Seq]
		
		info.object.load_set_seq_image(seq_first) #load first seq image and set appropriate C array
		n_images=len(info.object.camera_list)
		print "Starting detect_part_track"
		x1_a,x2_a,y1_a,y2_a=[],[],[],[]
		for i in range (n_images): #initialize result arrays
			x1_a.append([])
			x2_a.append([])
			y1_a.append([])
			y2_a.append([])
        
		for i_seq in range(seq_first, seq_last+1): #loop over sequences
			for i_img in range(n_images):
				intx_green,inty_green,intx_blue,inty_blue=[],[],[],[]
				imx, imy, zoomx, zoomy, zoomf = ptv.py_get_mark_track_c(i_img)
				targets = read_targets(base_names[i_img], i_seq)
                
				for h in range(len(targets)):
					#get data from C
					tx, ty = targets[h].pos()
					
					if (targets[h].tnr() > -1):
						intx_green.append(int(imx/2 + zoomf*(tx - zoomx)))
						inty_green.append(int(imy/2 + zoomf*(ty - zoomy)))
					else:
						intx_blue.append(int(imx/2 + zoomf*(tx - zoomx)))
						inty_blue.append(int(imy/2 + zoomf*(ty - zoomy)))
			
				x1_a[i_img]=x1_a[i_img]+intx_green # add current step to result array
				x2_a[i_img]=x2_a[i_img]+intx_blue
				y1_a[i_img]=y1_a[i_img]+inty_green
				y2_a[i_img]=y2_a[i_img]+inty_blue
#				 info.object.camera_list[i_img].drawcross(str(i_seq)+"x_tr_gr",str(i_seq)+"y_tr_gr",intx_green,inty_green,"green",3)
#				 info.object.camera_list[i_img].drawcross(str(i_seq)+"x_tr_bl",str(i_seq)+"y_tr_bl",intx_blue,inty_blue,"blue",2)
		#plot result arrays
		for i_img in range(n_images):
			info.object.camera_list[i_img].drawcross("x_tr_gr","y_tr_gr",x1_a[i_img],y1_a[i_img],"green",3)
			info.object.camera_list[i_img].drawcross("x_tr_bl","y_tr_bl",x2_a[i_img],y2_a[i_img],"blue",2)
			info.object.camera_list[i_img]._plot.request_redraw()
									 
		print "Finished detect_part_track"

	def traject_action(self,info):
		""" show trajectories is handled by ptv.py_traject_loop(..) which returns data to be plotted.
		traject_action collects data to be plotted from all the steps and plots it at once.
		"""
		print "Starting show trajectories"
		info.object.clear_plots(remove_background=False)
		seq_first=info.object.exp1.active_params.m_params.Seq_First
		seq_last=info.object.exp1.active_params.m_params.Seq_Last
		info.object.load_set_seq_image(seq_first,display_only=True)
		n_camera=len(info.object.camera_list)
		x1_a,x2_a,y1_a,y2_a=[],[],[],[]
		for i in range (n_camera): #initialize result arrays 
			x1_a.append([])
			x2_a.append([])
			y1_a.append([])
			y2_a.append([])
		for i_seq in range(seq_first, seq_last):
			x1,y1,x2,y2,m1_tr=ptv.py_traject_loop(i_seq)
			for i in range(n_camera):
				x1_a[i]=x1_a[i]+x1[i]
				x2_a[i]=x2_a[i]+x2[i]
				y1_a[i]=y1_a[i]+y1[i]
				y2_a[i]=y2_a[i]+y2[i]
		print "Show trajectories finished"
		for i in range(n_camera):
			info.object.camera_list[i].drawcross("trajx1","trajy1",x1_a[i],y1_a[i],"blue",2)
			info.object.camera_list[i].drawcross("trajx2","trajy2",x2_a[i],y2_a[i],"red",2)
			info.object.camera_list[i].drawquiver(x1_a[i],y1_a[i],x2_a[i],y2_a[i],"green",linewidth=3.0)
			info.object.camera_list[i]._plot.request_redraw()


	def plugin_action(self,info):
		""" Configure plugins by using GUI
		"""
		info.object.plugins.read()
		info.object.plugins.configure_traits()
#----------------------------------------------------------------
# Actions associated with right mouse button clicks (treeeditor)
# ---------------------------------------------------------------
ConfigMainParams = Action(name="Main parameters",action='handler.configure_main_par(editor,object)')
ConfigCalibParams = Action(name="Calibration parameters",action='handler.configure_cal_par(editor,object)')
ConfigTrackParams = Action(name="Tracking parameters",action='handler.configure_track_par(editor,object)')
SetAsDefault = Action(name="Set as active",action='handler.set_active(editor,object)')
CopySetParams = Action(name="Copy set of parameters",action='handler.copy_set_params(editor,object)')
RenameSetParams = Action(name="Rename run",action='handler.rename_set_params(editor,object)')
DeleteSetParams = Action(name="Delete run",action='handler.delete_set_params(editor,object)')


# -----------------------------------------
# Defines the menubar
#------------------------------------------
menu_bar = MenuBar(
				Menu(
						Action(name='New',action='new_action'),
						Action(name='Open',action='open_action'),
						Action(name='Save As',action='saveas_action'),
						Action(name='Exit',action='exit_action'),
					name='File'
				 ),
				 Menu(
					Action(name='Init / Restart',action='init_action'),
					name='Start'
				),
				Menu(
					#Action(name='Show original image',action='showimg_action',enabled_when='pass_init'),
					Action(name='High pass filter',action='highpass_action',enabled_when='pass_init'),
					Action(name='Image coord',action='img_coord_action',enabled_when='pass_init'),
					Action(name='Correspondences',action='corresp_action',enabled_when='pass_init'),
					name='Preprocess'
				),
				Menu(
					Action(name='3D positions',action='threed_positions'),
					name='3D Positions'
				),
				Menu(
					Action(name='Create calibration',action='calib_action'), #,enabled_when='pass_init'),
					name='Calibration'
				),
				  Menu(
					Action(name='Sequence without display',action='sequence_action',enabled_when='pass_init'),
					name='Sequence'
				),
				Menu(
					Action(name='Detected Particles',action='detect_part_track',enabled_when='pass_init'),
					Action(name='Tracking without display',action='track_no_disp_action',enabled_when='pass_init'),
					Action(name='Tracking with display',action='track_disp_action',enabled_when='pass_init'),
					Action(name='Tracking backwards',action='track_back_action',enabled_when='pass_init'),
					Action(name='Show trajectories',action='traject_action',enabled_when='pass_init'),
					name='Tracking'
				),
				Menu(
					Action(name='Configure tracking/sequence',action='plugin_action'),
					name='Plugins'
				),
                                Menu(
                                        Action(name='Run multigrid demo',action='multigrid_demo'),
                                        name='Demo'
                                ),
				
			 )

#----------------------------------------
# tree editor for the Experiment() class
#
tree_editor_exp=TreeEditor(
		nodes=[
				TreeNode(
						node_for=[Experiment],
						auto_open=True,
						children = '',
						label = '=Experiment',
				),
				TreeNode(
						node_for=[Experiment],
						auto_open=True,
						children='paramsets',
						label= '=Parameters',
						add=[Paramset],
						menu = Menu(
									CopySetParams
									)
						),
				TreeNode(
						node_for=[Paramset],
						auto_open=True,
						children='',
						label= 'name',
						menu=Menu(
								NewAction,
								CopySetParams,
								RenameSetParams,
								DeleteSetParams,
								Separator(),
								ConfigMainParams,
								ConfigCalibParams,
								ConfigTrackParams,
								Separator(),
								SetAsDefault
								)
				)
			   
		],
		editable = False,
			   
)
# -------------------------------------------------------------------------
class Plugins (HasTraits):
	track_list=List
	seq_list=List
	track_alg=Enum(values='track_list')
	sequence_alg=Enum(values='seq_list')
	view = View(
			Group(
			Item(name='track_alg', label="Choose tracking algorithm:"),
			Item(name='sequence_alg', label="Choose sequence algorithm:")

			),
			buttons = [ 'OK'],
			title = 'External plugins configuration'
			)
	def __init__(self):
		self.read()
	def read(self):
		# reading external tracking
		try:
			f=open(os.path.join(software_path, "external_tracker_list.txt"), 'r')
			trackers=f.read().split('\n')
			trackers.insert(0,'default')
         		self.track_list=trackers
         		f.close()
		except:
			self.track_list=['default']
		# reading external sequence
		try:
			f=open(os.path.join(software_path, "external_sequence_list.txt"), 'r')
			seq=f.read().split('\n')
			seq.insert(0,'default')
			self.seq_list=seq
			f.close()
		except:
			self.seq_list=['default']
		
# ----------------------------------------------
class MainGUI (HasTraits):
	""" MainGUI is the main class under which the Model-View-Control (MVC) model is defined
	"""
	camera_list=List
	imgplt_flag=0
	pass_init=Bool(False)
	update_thread_plot=Bool(False)
	tr_thread=Instance(TrackThread)
	selected = Any
	
	# Defines GUI view --------------------------
	view = View(
				Group(
					Group(Item(name = 'exp1', editor = tree_editor_exp, show_label=False, width=-400, resizable=False),
						Item('camera_list',style  = 'custom',editor =
							ListEditor( use_notebook = True,deletable	 = False,
								dock_style	 = 'tab',
								page_name	 = '.name',
								selected='selected'),
							show_label = False),
						orientation='horizontal',
						show_left=False),
					orientation='vertical'),
				title = 'pyPTV',
				id='main_view',
				width = 1.,
				height = 1.,
				resizable = True,
				handler=TreeMenuHandler(), # <== Handler class is attached
				menubar=menu_bar)


   
	def _selected_changed(self):
		self.current_camera=int(self.selected.name.split(' ')[1])-1
	   
	   
	#---------------------------------------------------
	# Constructor and Chaco windows initialization
	#---------------------------------------------------
	def __init__(self):
		super(MainGUI, self).__init__()
		colors = ['yellow','green','red','blue']
		self.exp1=Experiment()
		self.exp1.populate_runs(exp_path)
		self.plugins=Plugins()
		self.n_camera=self.exp1.active_params.m_params.Num_Cam
		print self.n_camera
		self.orig_image=[]
		self.hp_image=[]
		self.current_camera=0
		self.camera_list = []
		for i in range(self.n_camera):
			self.camera_list.append(CameraWindow(colors[i]))
			self.camera_list[i].name="Camera "+str(i+1)
			self.camera_list[i].on_trait_change(self.right_click_process, 'rclicked')
			self.orig_image.append(np.array([],dtype=np.ubyte))
			self.hp_image.append(np.array([]))
		ptv.py_init_proc_c() #intialization of globals in ptv C module
		   
	#------------------------------------------------------
	def right_click_process(self):
		x_clicked,y_clicked,n_camera=0,0,0
		h_img=self.exp1.active_params.m_params.imx
		v_img=self.exp1.active_params.m_params.imy
		print h_img,v_img
		for i in range(len(self.camera_list)):
			
				n_camera=i
				x_clicked,y_clicked=self.camera_list[i]._click_tool.x,\
									self.camera_list[i]._click_tool.y
				x1,y1,x2,y2,x1_points,y1_points,intx1,inty1=ptv.py_right_click(x_clicked,y_clicked,n_camera)
				if (x1!=-1 and y1!=-1):
					self.camera_list[n_camera].right_p_x0.append(intx1)
					self.camera_list[n_camera].right_p_y0.append(inty1)
					self.camera_list[n_camera].drawcross("right_p_x0","right_p_y0",
						self.camera_list[n_camera].right_p_x0\
					,self.camera_list[n_camera].right_p_y0,"cyan",3,marker1="circle")
					self.camera_list[n_camera]._plot.request_redraw()
					print "right click process"
					print x1,y1,x2,y2,x1_points,y1_points
					color_camera=['yellow','red','blue','green']
					#print [x1[i]],[y1[i]],[x2[i]],[y2[i]]
					for j in range(len(self.camera_list)):
						if j is not n_camera:
							count=self.camera_list[i]._plot.plots.keys()
							self.camera_list[j].drawline("right_cl_x"+str(len(count)),"right_cl_y"+str(len(count)),x1[j],y1[j],x2[j],y2[j],color_camera[n_camera])
							self.camera_list[j]._plot.index_mapper.range.set_bounds(0,h_img)
							self.camera_list[j]._plot.value_mapper.range.set_bounds(0,v_img)
							self.camera_list[j].drawcross("right_p_x1","right_p_y1",x1_points[j],y1_points[j],\
							color_camera[n_camera],2)
							self.camera_list[j]._plot.request_redraw()
				else:
					print ("No nearby points for epipolar lines")
				self.camera_list[i].rclicked=0

   
	def update_plots(self,images,is_float=0):
		for i in range(len(images)):
			self.camera_list[i].update_image(images[i],is_float)
			self.camera_list[i]._plot.request_redraw()

	# set_images sets ptv's C module img[] array
	def set_images(self,images):
		for i in range(len(images)):
			ptv.py_set_img(images[i],i)

	def get_images(self,plot_index,images):
		for i in plot_index:
			ptv.py_get_img(images[i],i)

	def drawcross(self,str_x,str_y,x,y,color1,size1):
 		for i in range(len(self.camera_list)):
			self.camera_list[i].drawcross(str_x,str_y,x[i],y[i],color1,size1)
			self.camera_list[i]._plot.request_redraw()
				   
	def clear_plots(self,remove_background=True):
		# this function deletes all plotes except basic image plot
   
		if not remove_background:
			index='plot0'
		else:
			index=None
		   
		for i in range(len(self.camera_list)):
			plot_list=self.camera_list[i]._plot.plots.keys()
			#if not remove_background:
			#	index=None
			try:
				plot_list.remove(index)
			except:
				pass
			self.camera_list[i]._plot.delplot(*plot_list[0:])
			self.camera_list[i]._plot.tools=[]
			self.camera_list[i]._plot.request_redraw()
			for j in range(len(self.camera_list[i]._quiverplots)):
				self.camera_list[i]._plot.remove(self.camera_list[i]._quiverplots[j])
			self.camera_list[i]._quiverplots=[]
			self.camera_list[i].right_p_x0=[]
			self.camera_list[i].right_p_y0=[]
			self.camera_list[i].right_p_x1=[]
			self.camera_list[i].right_p_y1=[]
					
   
	def _update_thread_plot_changed(self):
		n_camera=len(self.camera_list)
	   
		if self.update_thread_plot and self.tr_thread:
			print "updating plots..\n"
			step=self.tr_thread.track_step
		   
			x0,x1,x2,y0,y1,y2,pnr1,pnr2,pnr3,m_tr=\
			self.tr_thread.intx0,self.tr_thread.intx1,self.tr_thread.intx2,\
			self.tr_thread.inty0,self.tr_thread.inty1,self.tr_thread.inty2,self.tr_thread.pnr1,\
			self.tr_thread.pnr2,self.tr_thread.pnr3,self.tr_thread.m_tr
			for i in range (n_camera):
				self.camera_list[i].drawcross(str(step)+"x0",str(step)+"y0",x0[i],y0[i],"green",2)
				self.camera_list[i].drawcross(str(step)+"x1",str(step)+"y1",x1[i],y1[i],"yellow",2)
				self.camera_list[i].drawcross(str(step)+"x2",str(step)+"y2",x2[i],y2[i],"white",2)
				self.camera_list[i].drawquiver(x0[i],y0[i],x1[i],y1[i],"orange")
				self.camera_list[i].drawquiver(x1[i],y1[i],x2[i],y2[i],"white")
##				  for j in range (m_tr):
##					  str_plt=str(step)+"_"+str(j)
##					  
##					  self.camera_list[i].drawline\
##					  (str_plt+"vec_x0",str_plt+"vec_y0",x0[i][j],y0[i][j],x1[i][j],y1[i][j],"orange")
##					  self.camera_list[i].drawline\
##					  (str_plt+"vec_x1",str_plt+"vec_y1",x1[i][j],y1[i][j],x2[i][j],y2[i][j],"white")
			self.load_set_seq_image(step,update_all=False,display_only=True)
			self.camera_list[self.current_camera]._plot.request_redraw()
			self.tr_thread.can_continue=True
			self.update_thread_plot=False


	def load_set_seq_image(self,seq, update_all=True,display_only=False):
		n_camera=len(self.camera_list)
		if not hasattr(self,'base_name'):
			self.base_name=[]
			for i in range (n_camera):
				exec("self.base_name.append(self.exp1.active_params.m_params.Basename_%d_Seq)" %(i+1))
				print self.base_name[i]

		i=seq
		seq_ch = "%04d" % i

		if not update_all:
			j=self.current_camera
			img_name=self.base_name[j]+seq_ch
			self.load_disp_image(img_name,j,display_only)
		else:
			for j in range (n_camera):
				img_name=self.base_name[j]+seq_ch
				self.load_disp_image(img_name,j,display_only)


	def load_disp_image(self, img_name,j,display_only=False):
		print ("Setting image: "+str(img_name))
		temp_img=np.array([],dtype=np.ubyte)
		try:
			temp_img=imread(img_name).astype(np.ubyte)
			if not display_only:
				ptv.py_set_img(temp_img,j)
			if len(temp_img)>0:
				self.camera_list[j].update_image(temp_img,is_float=1)
		except:
			print "Error reading file"


		   
# -------------------------------------------------------------	   
if __name__ == '__main__':
	try:
		main_gui = MainGUI()
    #gui1.exp1.populate_runs(exp_path)
		main_gui.configure_traits()
	except:
		print("something wrong with the software or folder")
		general.printException()
	
	os.chdir(cwd) #get back to the original workdir
