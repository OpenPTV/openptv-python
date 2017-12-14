from threading import Thread

import numpy as np
from chaco.api import Plot, ArrayPlotData, gray, GridPlotContainer
from chaco.tools.api import ZoomTool, PanTool
from enable.component_editor import ComponentEditor
from scipy.misc import imread, imresize
from traits.api import HasTraits, Str, Instance, Button, List, Bool
from traitsui.api import Item, View


class DemoWindow (HasTraits):
    """ DemoWindow class contains the relevant information and functions for a single camera window: image, zoom, pan
    important members:
        _plot_data  - contains image data to display (used by update_image)
        _plot - instance of Plot class to use with _plot_data
        """
    _plot_data = Instance(ArrayPlotData)
    _plot = Instance(Plot)

    name = Str
    view = View(Item(name='_plot', editor=ComponentEditor(), show_label=False))

    def __init__(self):
        """ Initialization of plot system
        """
        padd = 25
        self._plot_data = ArrayPlotData()
        self._plot = Plot(self._plot_data, default_origin="top left")
        self._plot.padding_left = padd
        self._plot.padding_right = padd
        self._plot.padding_top = padd
        self._plot.padding_bottom = padd
        self.right_p_x0, self.right_p_y0, self.right_p_x1, self.right_p_y1, self._quiverplots = [], [], [], [], []

    def attach_tools(self):
        """ attach_tools(self) contains the relevant tools: clicker, pan, zoom
        """
        pan = PanTool(self._plot, drag_button='middle')
        zoom_tool = ZoomTool(self._plot, tool_mode="box", always_on=False)
#       zoom_tool = BetterZoom(component=self._plot, tool_mode="box", always_on=False)
        zoom_tool.max_zoom_out_factor = 1.0  # Disable "bird view" zoom out
        self._img_plot.overlays.append(zoom_tool)
        self._img_plot.tools.append(pan)

    def update_image(self, image, is_float):
        """ update_image - displays/updates image in the curren camera window
        parameters:
            image - image data
            is_float - if true, displays an image as float array,
            else displays as byte array (B&W or gray)
        example usage:
            update_image(image,is_float=False)
        """
        if is_float:
            # self._plot_data.set_data('imagedata',image.astype(np.float))

            self._plot_data.set_data('imagedata', image.astype(np.float))

        else:
            # self._plot_data.set_data('imagedata',image.astype(np.byte))
            self._plot_data.set_data('imagedata', image)

        if not hasattr(self, '_img_plot'):  # make a new plot if there is nothing to update
            print "inside 1"
            # self._img_plot=Instance(ImagePlot)
            self._img_plot = self._plot.img_plot('imagedata', colormap=gray)[0]
            # self.attach_tools()

           # self._plot.request_redraw()


class DemoThread(Thread):

    def run(self):
        print "starting thread"
        seq_first = self.seq_first
        seq_last = self.seq_last
        base_name = self.base_name

        # main loop - format image name, read it and call
        # v.py_sequence_loop(..) for current step
        for i in range(seq_first, seq_last + 1):
            if i < 10:
                seq_ch = "%01d" % i
            elif i < 100:
                seq_ch = "%02d" % i
            else:
                seq_ch = "%03d" % i
            self.temp_img.append([])
            self.can_continue = False
            for j in range(self.n_camera):
                img_name = base_name[j] + seq_ch
                print ("Setting image: ", img_name)
                try:
                    img = imread(img_name).astype(np.float)
                    self.temp_img[-1].append(imresize(img, 0.5))
                except:
                    print "Error reading file"
        print "Thread finished"
#           self.DemoGUI._update_thread_plot_changed()
#               self.DemoGUI.camera_list[j]._plot.request_redraw()

#           GUI.invoke_later(setattr, self.DemoGUI, 'update_thread_plot', True)
#           print "a1"
#           while not self.can_continue: # wait while plotting of the current step is finished, then continue for the next step
#               pass
#               #self.camera_list[j].update_image(temp_img,is_float=1)
#

        print "tracking with display thread finished"


class DemoGUI (HasTraits):
    camera_list = List
    containter = Instance(GridPlotContainer)
    update_thread_plot = Bool(False)

    button_play = Button()

    # Defines GUI view --------------------------
    traits_view = View(Item('container', editor=ComponentEditor(), show_label=False),
                       Item('button_play', label='Play', show_label=False),
                       width=1000, height=600, resizable=True, title="Demo")

    #---------------------------------------------------
    # Constructor and Chaco windows initialization
    #---------------------------------------------------
    def __init__(self, ptv=None, exp1=None):
        super(DemoGUI, self).__init__()
        self.exp1 = exp1
        self.ptv1 = ptv
        self.n_camera = self.exp1.active_params.m_params.Num_Cam
        print self.n_camera
        self.orig_image = []
        self.hp_image = []
        if self.n_camera == 1:
            shape = (1, 1)
        else:
            shape = (2, int(self.n_camera / 2))
        self.container = GridPlotContainer(shape=shape)

        for i in range(self.n_camera):
            self.camera_list.append(DemoWindow())
            self.container.add(self.camera_list[i]._plot)
            self.orig_image.append(np.array([], dtype=np.ubyte))
            self.hp_image.append(np.array([]))

    def _button_play_fired(self):
        self.tr_thread = DemoThread()
        self.tr_thread.seq_first = self.exp1.active_params.m_params.Seq_First
        self.tr_thread.seq_last = self.exp1.active_params.m_params.Seq_Last
        self.tr_thread.n_camera = self.n_camera
        self.tr_thread.temp_img = []
        base_name = []
        for i in range(self.n_camera):
            exec(
                "base_name.append(self.exp1.active_params.m_params.Basename_%d_Seq)" % (i + 1))
            print base_name[i]
        self.tr_thread.base_name = base_name
        self.tr_thread.start()
        while (self.tr_thread and (self.tr_thread.isAlive() or len(self.tr_thread.temp_img) > 0)):
            # print len(self.tr_thread.temp_img)
            if len(self.tr_thread.temp_img) > 0:
                if len(self.tr_thread.temp_img[0]) == self.n_camera:
                    # print len(self.tr_thread.temp_img)
                    for i in range(self.n_camera):
                        self.camera_list[i].update_image(
                            self.tr_thread.temp_img[0][i], is_float=1)
                        # self.camera_list[i]._plot.request_redraw()
                    self.container.window.control.Update()

        del self.tr_thread.temp_img[0]


# def _update_thread_plot_changed(self):
# print "inside up_thread"
# if self.update_thread_plot and self.tr_thread:
# print "inside up_thread2"
# for i in range (self.n_camera):
# self.camera_list[i].update_image(self.tr_thread.temp_img[i],is_float=1)
# self.camera_list[i]._plot.request_redraw()
# print "Changed"
# self.tr_thread.can_continue=True
# self.update_thread_plot=False
