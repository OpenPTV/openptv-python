"""
Copyright (c) 2008-2013, Tel Aviv University
Copyright (c) 2013 - the OpenPTV team
The software is distributed under the terms of MIT-like license
http://opensource.org/licenses/MIT
"""
from traits.api \
    import HasTraits, Str, Float, Int, List, Bool, Enum, Instance, Button, File
from traitsui.api \
    import View, Item, VSplit, \
           HGroup, Handler, Group, VGroup, HGroup, Tabbed, ListEditor
from enable.component_editor import ComponentEditor
from chaco.api import Plot, ArrayPlotData, gray,    ImageData, \
                                ImagePlot,CMapImagePlot,ArrayDataSource, MultiArrayDataSource,\
                                LinearMapper
from traitsui.menu import MenuBar, ToolBar, Menu, Action
from chaco.tools.image_inspector_tool import ImageInspectorTool
from chaco.tools.simple_zoom import  SimpleZoom
from text_box_overlay import TextBoxOverlay
from scipy.misc import imread
import os, sys, shutil
from scipy.misc import imread
from code_editor import *
import numpy as np

from quiverplot import QuiverPlot

src_path = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0],'src_c')
# print src_path
sys.path.append(src_path)


import ptv1 as ptv
import parameter_gui as exp
import parameters as par



# -------------------------------------------
class clicker_tool(ImageInspectorTool):
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
            print self.x
            print self.y
            self.left_changed=1-self.left_changed
            self.last_mouse_position = (event.x, event.y)

    def normal_right_down(self, event):
        plot = self.component
        if plot is not None:
            ndx = plot.map_index((event.x, event.y))

            x_index, y_index = ndx
            image_data=plot.value
            self.x=(x_index)
            self.y=(y_index)

            self.right_changed=1-self.right_changed
            print self.x
            print self.y


            self.last_mouse_position = (event.x, event.y)

    def normal_mouse_move(self, event):
        pass
    def __init__(self, *args, **kwargs):
        super(clicker_tool, self).__init__(*args, **kwargs)


# ----------------------------------------------------------

class plot_window (HasTraits):
    _plot_data=Instance(ArrayPlotData)
    _plot=Instance(Plot)
    _click_tool=Instance(clicker_tool)
    _img_plot=Instance(ImagePlot)
    _right_click_avail=0
    name=Str
    view = View(
             Item(name='_plot',editor=ComponentEditor(), show_label=False),

             )

    def __init__(self):
        # -------------- Initialization of plot system ----------------
        padd=25
        self._plot_data=ArrayPlotData()
        self._x=[]
        self._y=[]
        self.man_ori=[1,2,3,4]
        self._plot=Plot(self._plot_data, default_origin="top left")
        self._plot.padding_left=padd
        self._plot.padding_right=padd
        self._plot.padding_top=padd
        self._plot.padding_bottom=padd
        self._quiverplots=[]

            # -------------------------------------------------------------
    def left_clicked_event(self):
        print ("left clicked")
        if len(self._x)<4:
            self._x.append( self._click_tool.x)
            self._y.append( self._click_tool.y)
        print self._x
        print self._y
        self.drawcross("coord_x","coord_y",self._x,self._y,"red",5)
        self._plot.overlays=[]
        self.plot_num_overlay(self._x,self._y,self.man_ori)

    def right_clicked_event(self):
        print ("right clicked")
        if len(self._x)>0:
            self._x.pop()
            self._y.pop()
            print self._x
            print self._y
            self.drawcross("coord_x","coord_y",self._x,self._y,"red",5)
            self._plot.overlays=[]
            self.plot_num_overlay(self._x,self._y,self.man_ori)
        else:
            if (self._right_click_avail):
                print "deleting point"
                self.py_rclick_delete(self._click_tool.x,self._click_tool.y,self.cameraN)
                x=[]
                y=[]
                self.py_get_pix_N(x,y,self.cameraN)
                self.drawcross("x","y",x[0],y[0],"blue",4)



    def attach_tools(self):
        self._click_tool=clicker_tool(self._img_plot)
        self._click_tool.on_trait_change(self.left_clicked_event, 'left_changed')
        self._click_tool.on_trait_change(self.right_clicked_event, 'right_changed')
        self._img_plot.tools.append(self._click_tool)
        self._zoom_tool = SimpleZoom(component=self._plot, tool_mode="box", always_on=False)
        self._zoom_tool.max_zoom_out_factor=1.0
        self._img_plot.tools.append(self._zoom_tool)
        if self._plot.index_mapper is not None:
            self._plot.index_mapper.on_trait_change(self.handle_mapper, 'updated', remove=False)
        if self._plot.value_mapper is not None:
            self._plot.value_mapper.on_trait_change(self.handle_mapper, 'updated', remove=False)


    def drawcross(self, str_x,str_y,x,y,color1,mrk_size):
        self._plot_data.set_data(str_x,x)
        self._plot_data.set_data(str_y,y)
        self._plot.plot((str_x,str_y),type="scatter",color=color1,marker="plus",marker_size=mrk_size)
        self._plot.request_redraw()

    def drawline(self,str_x,str_y,x1,y1,x2,y2,color1):
        self._plot_data.set_data(str_x,[x1,x2])
        self._plot_data.set_data(str_y,[y1,y2])
        self._plot.plot((str_x,str_y),type="line",color=color1)
        self._plot.request_redraw()


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

    def handle_mapper(self):
        for i in range (0,len(self._plot.overlays)):
            if hasattr(self._plot.overlays[i],'real_position'):
                coord_x1, coord_y1=self._plot.map_screen([self._plot.overlays[i].real_position])[0]
                self._plot.overlays[i].alternate_position=(coord_x1,coord_y1)

    def plot_num_overlay(self,x,y,txt):
        for i in range (0,len(x)):
            coord_x, coord_y=self._plot.map_screen([(x[i],y[i])])[0]
            ovlay=TextBoxOverlay(component=self._plot,
            text=str(txt[i]), alternate_position=(coord_x,coord_y),
                real_position=(x[i],y[i]),
                text_color = "white",
                border_color = "red"
                )
            self._plot.overlays.append(ovlay)

    def update_image(self,image,is_float):
        if is_float:
            self._plot_data.set_data('imagedata',image.astype(np.float))
        else:
            self._plot_data.set_data('imagedata',image.astype(np.byte))
        self._plot.request_redraw()

# ---------------------------------------------------------

class calibration_gui(HasTraits):

    camera=List
    status_text=Str("")
    ori_img_name=[]
    ori_img=[]
    pass_init=Bool(False)
    pass_init_disabled=Bool(False)
    # -------------------------------------------------------------
    button_edit_cal_parameters = Button()
    button_showimg=Button()
    button_detection=Button()
    button_manual=Button()
    button_file_orient=Button()
    button_init_guess=Button()
    button_sort_grid=Button()
    button_sort_grid_init=Button()
    button_orient=Button()
    button_orient_part=Button()
    button_orient_shaking=Button()
    button_orient_dumbbell=Button()
    button_restore_orient=Button()
    button_checkpoint=Button()
    button_ap_figures=Button()
    button_edit_ori_files = Button()
    button_test = Button()

        # Defines GUI view --------------------------

    view = View(
              HGroup(
                VGroup(
                    VGroup(
                            Item(name='button_showimg',label='Load/Show Images', show_label=False),
                            Item(name='button_detection',label='Detection', show_label=False,enabled_when='pass_init'),
                            Item(name='button_manual',label='Manual orient.', show_label=False,enabled_when='pass_init'),
                            Item(name='button_file_orient',label='Orient. with file', show_label=False,enabled_when='pass_init'),
                            Item(name='button_init_guess',label='Show initial guess', show_label=False,enabled_when='pass_init'),
                            Item(name='button_sort_grid',label='Sortgrid', show_label=False,enabled_when='pass_init'),
                            Item(name='button_sort_grid_init',label='Sortgrid = initial guess', show_label=False,enabled_when='pass_init'),
                            Item(name='button_orient',label='Orientation', show_label=False,enabled_when='pass_init'),
                            Item(name='button_orient_part',label='Orientation with particles', show_label=False,enabled_when='pass_init'),
                            Item(name='button_orient_dumbbell',label='Orientation from dumbbell', show_label=False,enabled_when='pass_init'),
                            Item(name='button_restore_orient',label='Restore ori files', show_label=False,enabled_when='pass_init'),
                            Item(name='button_checkpoint',label='Checkpoints', show_label=False,enabled_when='pass_init_disabled'),
                            Item(name='button_ap_figures',label='Ap figures', show_label=False,enabled_when='pass_init_disabled'),
                            show_left=False,

                            ),
                    VGroup(
                        Item(name='button_edit_cal_parameters',label='Edit calibration parameters', show_label=False),
                        Item(name='button_edit_ori_files',label='Edit ori files', show_label=False,),
                            show_left=False,
                            ),
                    ),

                            Item( 'camera',style  = 'custom',
                            editor = ListEditor( use_notebook = True,
                                       deletable    = False,
                                       dock_style   = 'tab',
                                       page_name    = '.name',
                                       ),
                                       show_label = False
                                       ),

                    orientation='horizontal'
                            ),
            title = 'Calibration',
            id='view1',
            width = 1.,
            height = 1.,
            resizable = True,
            statusbar='status_text'
    )
    #--------------------------------------------------
    def _button_edit_cal_parameters_fired(self):
        cp = exp.Calib_Params(par_path=self.par_path)
        cp.edit_traits(kind='modal')
        par.copy_params_dir(self.par_path, par.temp_path)

    def _button_showimg_fired(self):
        if os.path.isfile(os.path.join(self.exp1.active_params.par_path,'man_ori.dat')):
            shutil.copyfile(os.path.join(self.exp1.active_params.par_path,'man_ori.dat'),\
            os.path.join(os.getcwd(),'man_ori.dat'))

        print("Load Image fired")
        self.load_init_v1() # < - this should be united with the Calib_Params in experiment_01a.py
        print(len(self.ori_img))
        self.ptv.py_calibration(1)
        self.pass_init=True
        self.status_text="Initialization finished."


    def _button_detection_fired(self):
        if self.need_reset:
            self.reset_show_images()
            self.need_reset=0
        print("Detection procedure")
        self.ptv.py_calibration(2)
        x=[]
        y=[]
        self.ptv.py_get_pix(x,y)

        # self.update_plots(self.ori_img)
        self.drawcross("x","y",x,y,"blue",4)
        for i in range(len(self.camera)):
            self.camera[i]._right_click_avail=1

    def _button_manual_fired(self):
        points_set=True
        for i in range(len(self.camera)):
            if len(self.camera[i]._x)<4:
                print "inside manual click"
                print self.camera[i]._x
                points_set=False

        if points_set:
            man_ori_path = os.path.join(os.getcwd(),'man_ori.dat')
            f=open(man_ori_path, 'w')
            if f is None:
                self.status_text="Error saving man_ori.dat."
            else:
                for i in range(len(self.camera)):
                    for j in range(4):
                        f.write("%f %f\n" % (self.camera[i]._x[j],self.camera[i]._y[j]))

                self.status_text="man_ori.dat saved."
                f.close()
        else:
            self.status_text="Set 4 points on each calibration image for manual orientation"

    def _button_file_orient_fired(self):
        if self.need_reset:
            self.reset_show_images()
            self.need_reset=0

        man_ori_path = os.path.join(os.getcwd(),'man_ori.dat')
        try:
            f=open(man_ori_path, 'r')
        except:
            self.status_text="Error loading man_ori.dat."
        else:
            for i in range(len(self.camera)):
                self.camera[i]._x=[]
                self.camera[i]._y=[]
                for j in range(4):
                    line=f.readline().split()
                    self.camera[i]._x.append(float(line[0]))
                    self.camera[i]._y.append(float(line[1]))
            self.status_text="man_ori.dat loaded."
            f.close()
            shutil.copyfile(man_ori_path,os.path.join(self.exp1.active_params.par_path,'man_ori.dat'))

        #TODO: rewrite using Parameters subclass
        man_ori_par_path = os.path.join(os.getcwd(),'parameters','man_ori.par')
        f=open(man_ori_par_path, 'r')
        if f is None:
            self.status_text="Error loading man_ori.par."
        else:
            for i in range(len(self.camera)):
                for j in range(4):
                    self.camera[i].man_ori[j]=int(f.readline().split()[0])
                self.status_text="man_ori.par loded."
                self.camera[i].left_clicked_event()
            f.close()
            self.ptv.py_calibration(4)
            self.status_text="Loading orientation data from file finished."


    def _button_init_guess_fired(self):
        if self.need_reset:
            self.reset_show_images()
            self.need_reset=0
        self.ptv.py_calibration(9)
        x=[]
        y=[]
        self.ptv.py_get_from_calib(x,y)
        self.drawcross("init_x","init_y",x,y,"yellow",3)
        self.status_text="Initial guess finished."

    def _button_sort_grid_fired(self):
        if self.need_reset:
            self.reset_show_images()
            self.need_reset=0
        self.ptv.py_calibration(5)
        x=[]
        y=[]
        x1_cyan=[]
        y1_cyan=[]
        pnr=[]
        self.ptv.py_get_from_sortgrid(x,y,pnr)
        # filter out -999 which is returned for the missing points:
        while -999 in x[0]:
            id = x[0].index(-999)
            del x[0][id]
            del y[0][id]
            del pnr[0][id]

        self.drawcross("sort_x","sort_y",x,y,"white",4)
        self.ptv.py_get_from_calib(x1_cyan,y1_cyan)
        self.drawcross("init_x","init_y",x1_cyan,y1_cyan,"cyan",4)
        for i in range (len(self.camera)):
            self.camera[i]._plot.overlays=[]
            self.camera[i].plot_num_overlay(x[i],y[i],pnr[i])
        self.status_text="Sort grid finished."

    def _button_sort_grid_init_fired(self):
        if self.need_reset:
            self.reset_show_images()
            self.need_reset=0
        self.ptv.py_calibration(14)
        x=[]
        y=[]
        x1_cyan=[]
        y1_cyan=[]
        pnr=[]
        self.ptv.py_get_from_sortgrid(x,y,pnr)
        self.drawcross("sort_x_init","sort_y_init",x,y,"white",4)
        self.ptv.py_get_from_calib(x1_cyan,y1_cyan)
        self.drawcross("init_x","init_y",x1_cyan,y1_cyan,"cyan",4)
        for i in range (len(self.camera)):
            self.camera[i]._plot.overlays=[]
            self.camera[i].plot_num_overlay(x[i],y[i],pnr[i])
        self.status_text="Sort grid initial guess finished."

    def _button_orient_fired(self):
        # backup the ORI/ADDPAR files first
        self.backup_ori_files()
        self.ptv.py_calibration(6)
        self.protect_ori_files()
        self.need_reset=1
        x1=[]
        y1=[]
        x2=[]
        y2=[]
        self.ptv.py_get_from_orient(x1,y1,x2,y2)

        self.reset_plots()
        for i in range (len(self.camera)):
            self.camera[i]._plot_data.set_data('imagedata',self.ori_img[i].astype(np.float))
            self.camera[i]._img_plot=self.camera[i]._plot.img_plot('imagedata',colormap=gray)[0]
            self.camera[i].drawquiver(x1[i],y1[i],x2[i],y2[i],"red")
            self.camera[i]._plot.index_mapper.range.set_bounds(0,self.h_pixel)
            self.camera[i]._plot.value_mapper.range.set_bounds(0,self.v_pixel)

        self.drawcross("orient_x","orient_y",x1,y1,"orange",4)
        self.status_text="Orientation finished."

    def _button_orient_part_fired(self):
        self.backup_ori_files()
        self.ptv.py_calibration(10)
        x1,y1,x2,y2=[],[],[],[]
        self.ptv.py_get_from_orient(x1,y1,x2,y2)

        self.reset_plots()
        for i in range (len(self.camera)):
            self.camera[i]._plot_data.set_data('imagedata',self.ori_img[i].astype(np.float))
            self.camera[i]._img_plot=self.camera[i]._plot.img_plot('imagedata',colormap=gray)[0]
            self.camera[i].drawquiver(x1[i],y1[i],x2[i],y2[i],"red")
            self.camera[i]._plot.index_mapper.range.set_bounds(0,self.h_pixel)
            self.camera[i]._plot.value_mapper.range.set_bounds(0,self.v_pixel)
            self.drawcross("orient_x","orient_y",x1,y1,"orange",4)

        self.status_text="Orientation with particles finished."

    def _button_orient_dumbbell_fired(self):
        print "Starting orientation from dumbbell"
        self.backup_ori_files()
        self.ptv.py_ptv_set_dumbbell(1)
        n_camera=len(self.camera)
        print ("Starting sequence action")
        seq_first=self.exp1.active_params.m_params.Seq_First
        seq_last=self.exp1.active_params.m_params.Seq_Last
        print seq_first,seq_last
        base_name=[]
        for i in range (n_camera):
            exec("base_name.append(self.exp1.active_params.m_params.Basename_%d_Seq)" %(i+1))
            print base_name[i]
            self.ptv.py_sequence_init(1)
            stepshake=self.ptv.py_get_from_sequence_init()
            if not stepshake:
                stepshake=1

        temp_img=np.array([],dtype=np.ubyte)
        for i in range(seq_first,seq_last+1,stepshake):
            seq_ch="%04d" % i
            print seq_ch
            for j in range (n_camera):
                print ("j %d" % j)
                img_name=base_name[j]+seq_ch
                print ("Setting image: ",img_name)
                try:
                    temp_img=imread(img_name).astype(np.ubyte)
                except:
                    print "Error reading file"
                    self.ptv.py_set_img(temp_img,j)

            self.ptv.py_sequence_loop(1,i)

        print "Orientation from dumbbell - sequence finished"
        self.ptv.py_calibration(12)
        self.ptv.py_ptv_set_dumbbell(1)
        print "Orientation from dumbbell finished"

    def _button_restore_orient_fired(self):
        self.restore_ori_files()


    def load_init_v1(self):
        calOriParams = par.CalOriParams(len(self.camera), path = self.par_path)
        calOriParams.read()
        (fixp_name, img_cal_name, img_ori, tiff_flag, pair_flag, chfield) = \
            (calOriParams.fixp_name, calOriParams.img_cal_name, calOriParams.img_ori, calOriParams.tiff_flag, calOriParams.pair_flag, calOriParams.chfield)
        self.ori_img_name = img_cal_name

        ptvParams = par.PtvParams(path = self.par_path)
        ptvParams.read()
        (n_img, img_name, img_cal, hp_flag, allCam_flag, tiff_flag, imx, imy, pix_x, pix_y, chfield, mmp_n1, mmp_n2, mmp_n3, mmp_d) = \
            (ptvParams.n_img, ptvParams.img_name, ptvParams.img_cal, ptvParams.hp_flag, ptvParams.allCam_flag, ptvParams.tiff_flag, \
            ptvParams.imx, ptvParams.imy, ptvParams.pix_x, ptvParams.pix_y, ptvParams.chfield, ptvParams.mmp_n1, ptvParams.mmp_n2, ptvParams.mmp_n3, ptvParams.mmp_d)
        self.h_pixel = imx
        self.v_pixel = imy

        self.ori_img = []
        print("len(self.camera)")
        print(len(self.camera))
        for i in range (len(self.camera)):
            print ("reading "+self.ori_img_name[i])
            try:
                img1=imread(self.ori_img_name[i], flatten=1).astype(np.ubyte)
                print img1.shape
            except:
                print("Error reading image "+self.ori_img_name[i])
                break
            self.ori_img.append(img1)
            self.ptv.py_set_img(self.ori_img[i],i)

        self.reset_show_images()
        # Loading manual parameters here

        #TODO: rewrite using Parameters subclass
        man_ori_path = os.path.join(os.getcwd(),'parameters','man_ori.par')
        f = open(man_ori_path, 'r')
        if f is None:
            printf('\nError loading man_ori.par')
        else:
            for i in range (len(self.camera)):
                for j in range(4):
                    self.camera[i].man_ori[j]=int(f.readline().strip())
        f.close()



    def reset_plots(self):
        for i in range (len(self.camera)):
            self.camera[i]._plot.delplot(*self.camera[i]._plot.plots.keys()[0:])
            self.camera[i]._plot.overlays=[]
            for j in range(len(self.camera[i]._quiverplots)):
                self.camera[i]._plot.remove(self.camera[i]._quiverplots[j])
            self.camera[i]._quiverplots=[]

    def reset_show_images(self):
        for i in range (len(self.camera)):
            self.camera[i]._plot.delplot(*self.camera[i]._plot.plots.keys()[0:])
            self.camera[i]._plot.overlays=[]
            self.camera[i]._plot_data.set_data('imagedata',self.ori_img[i].astype(np.float))
            self.camera[i]._img_plot=self.camera[i]._plot.img_plot('imagedata',colormap=gray)[0]
            self.camera[i]._x=[]
            self.camera[i]._y=[]
            self.camera[i]._img_plot.tools=[]
            self.camera[i].attach_tools()
            self.camera[i]._plot.request_redraw()
            for j in range(len(self.camera[i]._quiverplots)):
                self.camera[i]._plot.remove(self.camera[i]._quiverplots[j])
            self.camera[i]._quiverplots=[]


    def _button_edit_ori_files_fired(self):
        editor = codeEditor(path=self.par_path)
        editor.edit_traits(kind = 'livemodal')


    def drawcross(self,str_x,str_y,x,y,color1,size1):
        for i in range(len(self.camera)):
            self.camera[i].drawcross(str_x,str_y,x[i],y[i],color1,size1)

    def backup_ori_files(self):
        # backup ORI/ADDPAR files to the backup_cal directory
        calOriParams = par.CalOriParams(len(self.camera), path = self.par_path)
        calOriParams.read()
        for f in calOriParams.img_ori:
            shutil.copyfile(f,f+'.bck')
            g = f.replace('ori','addpar')
            shutil.copyfile(g, g + '.bck')

    def restore_ori_files(self):
        # backup ORI/ADDPAR files to the backup_cal directory
        calOriParams = par.CalOriParams(len(self.camera), path = self.par_path)
        calOriParams.read()

        for f in calOriParams.img_ori:
            print "restored %s " % f
            shutil.copyfile(f+'.bck',f)
            g = f.replace('ori','addpar')
            shutil.copyfile(g + '.bck', g)

    def protect_ori_files(self):
        # backup ORI/ADDPAR files to the backup_cal directory
        calOriParams = par.CalOriParams(len(self.camera), path = self.par_path)
        calOriParams.read()
        for f in calOriParams.img_ori:
            d = file(f,'r').read().split()
            if not np.all(np.isfinite(np.asarray(d).astype('f'))):
                print "protected ORI file %s " % f
                shutil.copyfile(f+'.bck',f)



    def load_init(self):
        calOriParams = par.CalOriParams(len(self.camera), path = self.par_path)
        calOriParams.read()
        (fixp_name, img_cal_name, img_ori, tiff_flag, pair_flag, chfield) = \
            (calOriParams.fixp_name, calOriParams.img_cal_name, calOriParams.img_ori, calOriParams.tiff_flag, calOriParams.pair_flag, calOriParams.chfield)
        self.ori_img_name = img_cal_name
        for i in range (len(self.camera)):
            print ("reading "+self.ori_img_name[i])
            try:
                img1=imread(self.ori_img_name[i]).astype(np.ubyte)
            except:
                print("Error reading image "+self.ori_img_name[i])
                break
            self.ori_img.append(img1)
            if self.camera[i]._plot is not None:
                self.camera[i]._plot.delplot(*self.camera[i]._plot.plots.keys()[0:])
            self.camera[i]._plot_data.set_data('imagedata',self.ori_img[i].astype(np.byte))
            self.camera[i]._img_plot=self.camera[i]._plot.img_plot('imagedata',colormap=gray)[0]

            self.camera[i]._x=[]
            self.camera[i]._y=[]
            self.camera[i]._plot.overlays=[]
            self.camera[i]._img_plot.tools=[]
            self.camera[i].attach_tools()
            self.camera[i]._plot.request_redraw()


            self.ptv.py_set_img(self.ori_img[i],i)

        f.close()
# Loading manual parameters here
        #TODO: rewrite using Parameters subclass
        man_ori_path = os.path.join(os.getcwd(),'parameters','man_ori.par')
        f=open(man_ori_path, 'r')
        if f==None:
            printf('\nError loading man_ori.par')
        else:
            for i in range (len(self.camera)):
                for j in range(4):
                    self.camera[i].man_ori[j]=int(f.readline().strip())


#
#   def drawcross(self,str_x,str_y,x,y,color1,size1):
#           for i in range(len(self.camera)):
#                   self.camera[i].drawcross(str_x,str_y,x[i],y[i],color1,size1)

    def update_plots(self,images,is_float=0):
        for i in range(len(images)):
            self.camera[i].update_image(images[i],is_float)


    #---------------------------------------------------
    # Constructor
    #---------------------------------------------------
    def __init__(self, par_path):
        super(calibration_gui, self).__init__() # this is needed according to chaco documentation
        self.need_reset=0
        self.par_path = par_path
        self.ptv = ptv
        # self.ptv.py_init_proc_c() - likely that this created memory overflow on Mac
