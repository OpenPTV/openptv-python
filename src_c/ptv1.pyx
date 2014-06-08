import numpy as np
cimport numpy as np

from tracking_run_py cimport tracking_run, TrackingRun

ctypedef extern struct n_tupel:
    int     p[4]
    double  corr


ctypedef extern struct target:
    int pnr
    double  x, y
    int     n, nx, ny, sumg
    int     tnr



ctypedef extern struct coord_2d:
    int pnr
    double x, y

cdef extern from "stdlib.h":
    void *memcpy(void *dst, void *src, long n)
    void free(void *ptr)
    enum: NULL

cdef extern from "parameters.h":
    ctypedef struct control_par:
        int num_cams
    ctypedef struct volume_par:
        pass
    volume_par* read_volume_par(char* vpar_fname)

# Apologies. This is needed until orientation overhaul begins.
cdef enum:
    nmax = 20240

cdef extern from "globals.h": # to lose the declspec
    void prepare_eval(int n_img, int *n_fix)

    void highpass(unsigned char *img, unsigned char *img_hp, int dim_lp, int filter_hp, int field)
    int init_proc_c()
    int start_proc_c()
    int pre_processing_c ()
    int detection_proc_c() 
    int correspondences_proc_c() 
    int calibration_proc_c(int sel) 

    int sequence_proc_c(int dumb_flag)
    int sequence_proc_loop_c(int dumbell, int i)

    tracking_run* trackcorr_c_init()
    int trackcorr_c_loop (tracking_run *run_info, int step, int display)
    int trackcorr_c_finish(tracking_run *run_info, int step)
    int trackback_c ()
    int trajectories_c(int i, int num_cams)
    void read_ascii_data(int filenumber)
    int determination_proc_c (int dumbbell)

    int mouse_proc_c (int click_x, int click_y, int kind, int num_image, volume_par *vpar, control_par *cpar)
    target *p[4]
    target pix[4][20240]
    coord_2d geo[4][20240]
    coord_2d crd[4][20240]
    n_tupel con[20240] 
    int x_calib[4][1000]
    int y_calib[4][1000]
    int z_calib[4][1000]
    int ncal_points[4]  
    int  orient_x1[4][1000]
    int  orient_y1[4][1000]
    int  orient_x2[4][1000]
    int  orient_y2[4][1000]
    int  orient_n[4]    
    int intx0_tr[4][10000],intx1_tr[4][10000],intx2_tr[4][10000],inty0_tr[4][10000],inty1_tr[4][10000],inty2_tr[4][10000],pnr1_tr[4][10000],pnr2_tr[4][10000],m1_tr
    float pnr3_tr[4][10000]
    int n_img
    int num[4]
    int zoom_x[], zoom_y[], zoom_f[]
    int rclick_intx1[4],rclick_inty1[4],rclick_intx2[4],rclick_inty2[4], rclick_points_x1[4][10000],rclick_points_y1[4][10000],rclick_count[4]
    int rclick_points_intx1, rclick_points_inty1

    int imgsize
    int imx
    int imy
    int dumbbell_pyptv
    int match4_g,match3_g,match2_g,match1_g
    unsigned char *img[4]
    int seq_step_shake
    int nfix 

# New jw_ptv-global configuration:
cdef extern from "globals.h":
    control_par *cpar

def py_set_imgdimensions(size,imx1,imy1):
    global imgsize,imx,imy
    imgsize=<int>size
    imx=<int>imx1
    imy=<int>imy1

def py_highpass(np.ndarray img1, np.ndarray img2, dim_lp1, filter_lp1, field1 ):
    highpass(<unsigned char *>img1.data, <unsigned char *>img2.data, dim_lp1, filter_lp1, field1)
    

def py_set_img(np.ndarray img_one, i):
    global img

    cdef int img_size=img_one.size
    cdef unsigned char *img_dest=<unsigned char *>img_one.data
    cdef int i1=i
#   for i1 in range(50):
#       print("img0 ", img_dest[i1])
    memcpy(img[i],<unsigned char *>img_one.data,img_size*sizeof(unsigned char))

def py_get_img(np.ndarray img_one, i):
    global img, imgsize
    print ("img_size=",imgsize)
    #cdef int i1=i
    
    memcpy(img_one.data,img[i],imgsize*sizeof(unsigned char))
    cdef unsigned char *img_dest=<unsigned char *>img_one.data
    cdef int i1=i
    for i1 in range(50):
        print("img1 ", img_one[i1])


    
def py_start_proc_c():
    start_proc_c()
        
        
        
def py_init_proc_c():
    init_proc_c()  #initialize general globals 
    
def py_pre_processing_c():
    pre_processing_c()

def py_detection_proc_c():
    detection_proc_c()

def py_read_attributes(a):
    global imgsize, imx, imy
    a.append(imgsize)
    a.append(imx)
    a.append(imy)
    
def py_get_pix(x,y):
    global pix,n_img
    cdef int i,j
    for i in range(n_img):
        x1=[]
        y1=[]
        for j in range(num[i]):
            x1.append(pix[i][j].x)
            y1.append(pix[i][j].y)
        x.append(x1)
        y.append(y1)
    
def py_calibration(sel):
    calibration_proc_c(sel) 
    
def py_correspondences_proc_c(quadruplets,triplets,pairs, unused):
    global pix,n_img,match4_g,match3_g,match2_g,match1_g,geo,con,p

    correspondences_proc_c()
#  get quadruplets ---------------------------  
    cdef int i,j
    quadruplets_x=[]
    quadruplets_y=[]
    for j in range(n_img):
        x1=[]
        y1=[]
        for i in range (match4_g):
            p1 = geo[j][con[i].p[j]].pnr
            if (p1>-1):
                x1.append(pix[j][p1].x)
                y1.append(pix[j][p1].y)
        quadruplets_x.append(x1)
        quadruplets_y.append(y1)
    quadruplets.append(quadruplets_x)
    quadruplets.append(quadruplets_y)
# get triplets -----------------------------
    
    triplets_x=[]
    triplets_y=[]
    for j in range(n_img):
        x1=[]
        y1=[]
        for i in range (match4_g,match4_g+match3_g):
            p1 = geo[j][con[i].p[j]].pnr
            if (p1>-1 and con[i].p[j] > -1):
                x1.append(pix[j][p1].x)
                y1.append(pix[j][p1].y)
        triplets_x.append(x1)
        triplets_y.append(y1)
    triplets.append(triplets_x)
    triplets.append(triplets_y)
#get pairs -----------------------------------------

    pairs_x=[]
    pairs_y=[]
    for j in range(n_img):
        x1=[]
        y1=[]
        for i in range (match4_g+match3_g,match4_g+match3_g+match2_g):
            p1 = geo[j][con[i].p[j]].pnr
            if (p1>-1 and con[i].p[j] > -1):
                x1.append(pix[j][p1].x)
                y1.append(pix[j][p1].y)
        pairs_x.append(x1)
        pairs_y.append(y1)
    pairs.append(pairs_x)
    pairs.append(pairs_y)
#get unused -----------------------------------------


    unused_x=[]
    unused_y=[]
    
    for j in range (n_img):
        x1=[]
        y1=[]
        for i in range(num[j]):
            p1 = pix[j][i].tnr
            if p1 == -1 :
                x1.append(pix[j][i].x)
                y1.append(pix[j][i].y)
        unused_x.append(x1)
        unused_y.append(y1)
    unused.append(unused_x)
    unused.append(unused_y)

def py_get_from_calib(x,y):
    global x_calib,y_calib,ncal_points  
    cdef int i,j
    for i in range(n_img):
        x1=[]
        y1=[]
        for j in range(ncal_points[i]):
            x1.append(x_calib[i][j])
            y1.append(y_calib[i][j])
        x.append(x1)
        y.append(y1)

def py_get_from_sortgrid(x,y,pnr):
    global x_calib,y_calib,z_calib,ncal_points,pix  
    cdef int i,j
    for i in range(n_img):
        x1=[]
        y1=[]
        pnr1=[]
        for j in range(ncal_points[i]):
            if (z_calib[i][j]>=0):
                x1.append(pix[i][j].x)
                y1.append(pix[i][j].y)
                pnr1.append(z_calib[i][j])
            
        x.append(x1)
        y.append(y1)
        pnr.append(pnr1)
        
def py_get_from_orient(x1,y1,x2,y2):
    global orient_x1,orient_y1,orient_x2,orient_y2,orient_n

    cdef int i,j
    for i in range(n_img):
        x_1=[]
        y_1=[]
        x_2=[]
        y_2=[]
        for j in range(orient_n[i]+1):
            x_1.append(orient_x1[i][j])
            y_1.append(orient_y1[i][j])
            x_2.append(orient_x2[i][j])
            y_2.append(orient_y2[i][j])
            
        x1.append(x_1)
        y1.append(y_1)
        x2.append(x_2)
        y2.append(y_2)
        
def  py_sequence_init(dumbflag=0):
    sequence_proc_c(<int>dumbflag)

# set dumbell=1, if dumbflag=3, see jw_ptv.c
def py_sequence_loop(dumbell,i):
    sequence_proc_loop_c(<int>dumbell,<int>i)

def py_get_from_sequence_init():
    global seq_step_shake
    return seq_step_shake
    
def py_trackcorr_init():
    cdef tracking_run *tr = trackcorr_c_init()
    cdef TrackingRun ret = TrackingRun()
    ret.tr = tr
    return ret
    
def py_trackcorr_loop(TrackingRun run_info, int step, int display):

    global intx0_tr,intx1_tr,intx2_tr,inty0_tr,inty1_tr,inty2_tr,pnr1_tr,pnr2_tr,pnr3_tr,m1_tr
    trackcorr_c_loop(run_info.tr, step, display)
    cdef int i,j
    if display:
        intx0,intx1,intx2,inty0,inty1,inty2,pnr1,pnr2,pnr3=[],[],[],[],[],[],[],[],[]
        print m1_tr
        
        for i in range(n_img):
            intx0_t,intx1_t,intx2_t,inty0_t,inty1_t,inty2_t,pnr1_t,pnr2_t,pnr3_t=[],[],[],[],[],[],[],[],[]
            for j in range (m1_tr):
                intx0_t.append(intx0_tr[i][j])
                inty0_t.append(inty0_tr[i][j])
                intx1_t.append(intx1_tr[i][j])
                inty1_t.append(inty1_tr[i][j])
                intx2_t.append(intx2_tr[i][j])
                inty2_t.append(inty2_tr[i][j])
                if pnr1_tr[i][j]>-1:
                    pnr1_t.append(pnr1_tr[i][j])
                if pnr2_tr[i][j]>-1:
                    pnr2_t.append(pnr2_tr[i][j])
                if pnr3_tr[i][j]>-1:
                    pnr3_t.append(pnr3_tr[i][j])
            intx0.append(intx0_t)
            intx1.append(intx1_t)
            intx2.append(intx2_t)
            inty0.append(inty0_t)
            inty1.append(inty1_t)
            inty2.append(inty2_t)
            pnr1.append(pnr1_t)
            pnr2.append(pnr2_t)
            pnr3.append(pnr3_t)
        return intx0,intx1,intx2,inty0,inty1,inty2,pnr1,pnr2,pnr3,m1_tr
    return 0
    
    
def py_trackcorr_finish(TrackingRun run_info, int step):
    trackcorr_c_finish(run_info.tr, step)
    
def py_trackback_c():
    trackback_c ()
    
def py_get_mark_track_c(i_img):
    global imx,imy,zoom_x,zoom_y,zoom_f
    return imx,imy,zoom_x[i_img],zoom_y[i_img],zoom_f[i_img]

def py_traject_loop(seq):
    global intx1_tr,intx2_tr,inty1_tr,inty2_tr,m1_tr
    trajectories_c(seq, cpar[0].num_cams)
    intx1,intx2,inty1,inty2=[],[],[],[]
    for i in range(n_img):
        intx1_t,intx2_t,inty1_t,inty2_t=[],[],[],[]
        for j in range(m1_tr):
            intx1_t.append(intx1_tr[i][j])
            inty1_t.append(inty1_tr[i][j])
            intx2_t.append(intx2_tr[i][j])
            inty2_t.append(inty2_tr[i][j])
        intx1.append(intx1_t)
        inty1.append(inty1_t)
        intx2.append(intx2_t)
        inty2.append(inty2_t)
    return intx1,inty1,intx2,inty2,m1_tr
        
def py_ptv_set_dumbbell(dumbbell):
    global dumbbell_pyptv
    dumbbell_pyptv=<int>dumbbell
    
def py_right_click(int coord_x, int coord_y, n_image):
    global rclick_intx1,rclick_inty1,rclick_intx2,rclick_inty2, rclick_points_x1,rclick_points_y1,rclick_count,rclick_points_intx1, rclick_points_inty1
    x2_points,y2_points,x1,y1,x2,y2=[],[],[],[],[],[]
    
    cdef volume_par *vpar = read_volume_par("parameters/criteria.par")
    r = mouse_proc_c (coord_x, coord_y, 3, n_image, vpar, cpar)
    free(vpar)
    
    if r==-1:
        return -1,-1,-1,-1,-1,-1,-1,-1
    for i in range(n_img):
        x2_temp,y2_temp=[],[]
        for j in range(rclick_count[i]):
            x2_temp.append(rclick_points_x1[i][j])
            y2_temp.append(rclick_points_y1[i][j])
        x2_points.append(x2_temp)
        y2_points.append(y2_temp)
        x1.append(rclick_intx1[i])
        y1.append(rclick_inty1[i])
        x2.append(rclick_intx2[i])
        y2.append(rclick_inty2[i])
    return  x1,y1,x2,y2,x2_points,y2_points,rclick_points_intx1, rclick_points_inty1
            
def py_determination_proc_c(dumbbell):
    determination_proc_c (<int>dumbbell)

def py_rclick_delete(coord_x,coord_y,n_image):
    mouse_proc_c(<int>coord_x, <int> coord_y, 4,<int>n_image, NULL, NULL)

def py_get_pix_N(x,y,n_image):
    global pix,n_img
    cdef int i,j
    i=n_image
    x1=[]
    y1=[]
    for j in range(num[i]):
      x1.append(pix[i][j].x)
      y1.append(pix[i][j].y)
      x.append(x1)
      y.append(y1)

def get_pix_crd(num_cams):
    """
    For testing purposes, while pix and crd arrays exist, return them as
    numpy arrays.
    
    Arguments:
    num_cams - number of cameras in the PTV system.
    
    Returns:
    pix_arr - a (num_cams, nmax, 2) array, copy of the pix structre-array
        that is filled-in by prepare_eval.
    crd_arr - a (num_cams, nmax, 3) array, same for the crd struct-array.
    """
    pix_arr = np.zeros((num_cams, nmax, 2))
    crd_arr = np.zeros((num_cams, nmax, 3))
    
    for part, img in np.ndindex((nfix, num_cams)):
        pix_arr[img, part, 0] = pix[img][part].x
        pix_arr[img, part, 1] = pix[img][part].y
        
        crd_arr[img, part, 0] = crd[img][part].x
        crd_arr[img, part, 1] = crd[img][part].y
        crd_arr[img, part, 2] = crd[img][part].pnr
    
    return pix_arr, crd_arr
    
def py_prepare_eval(num_cams):
    """
    Wrapper around prepare_eval for regression-testing purposes.
    
    Arguments:
    num_cams - number of cameras in the PTV system.
    
    Returns:
    pix_arr - a (num_cams, nmax, 2) array, copy of the pix structre-array
        that is filled-in by prepare_eval.
    crd_arr - a (num_cams, nmax, 3) array, same for the crd struct-array.
    """
    prepare_eval(num_cams, NULL) # the second argument is never used within.
    return get_pix_crd(num_cams)

