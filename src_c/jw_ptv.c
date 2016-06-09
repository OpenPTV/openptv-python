/****************************************************************************
 *****************************************************************************
 
 Author/Copyright:   Hans-Gerd Maas / Jochen Willneff
 
 Address:	      	Institute of Geodesy and Photogrammetry
 ETH - Hoenggerberg
 CH - 8093 Zurich
 
 Creation Date:	    took a longer time ...
 
 Description:	    target detection, correspondences and
 positioning with tclTk display
 -> 4 camera version
 
 Routines contained:    	only start_proc_c, init_proc_c, detection_proc_c left, pre_processing_c, pre_processing_c, Denis - 26/10/2010 
 
 ****************************************************************************/
#include "ptv.h"

#include <optv/calibration.h>
#include <optv/parameters.h>
#include <optv/trafo.h>

#include "tools.h"
#include "image_processing.h"
#include "orientation.h"

#define nmax 20240

/*  global declarations for ptv  */
/*-------------------------------------------------------------------------*/

int determination_proc_c();

volume_par *vpar;
control_par *cpar;

int pair_flag=0;					/*flag for accept pair */
int	nfix;	       	       	       	/* no. of control points */
int	num[4];	       		       	    /* no. of targets per image */
int numc[4];                        /* no. of targets in current image */
int nump[4];                        /* no. of targets in previous image */
int numn[4];                        /* no. of targets in next image */
int n_trac[4];	           	/* no. of tracks */
int	match=0;		      	/* no. of matches */
int	match2=0;	      	       	/* no. of matches in 2nd pass */
int match4_g, match3_g, match2_g, match1_g;

int	nr[4][4];		     	/* point numbers for man. ori */
int	imgsize;	      	/* image size */
int	zoom_x[4],zoom_y[4],zoom_f[4];  /* zoom parameters */
int	pp1=0, pp2=0, pp3=0, pp4=0,pp5=0;   	/* for man. orientation */
int	seq_first, seq_last;	       	/* 1. and last img of seq */
int	max_shake_points, max_shake_frames, step_shake;
int	demo_nr;		      	/* for demo purposes */
int	examine = 0;		       	/* for more detailed output */
int	dump_for_rdb;		       	/* # of dumpfiles for rdb */
int cr_sz;                          /* size of crosses */
int display;                        /* display flag */
int corp, corc, corn;
int mask;						/*checkmark for subtract mask*/

double seq_slice_step,seq_slicethickness,seq_zdim,seq_dummy;
double 	rmsX, rmsY, rmsZ, mean_sigma0;		/* a priori rms */
double  db_scale;               /*dumbbell length, Beat Mai 2010*/  

FILE	*fp1, *fp2, *fp3, *fp4, *fpp;

char	img_name[4][256];      	/* original image names */
char   	img_lp_name[4][256]; 	/* lowpass image names */
char   	img_hp_name[4][256];   	/* highpass image names */
char   	img_cal[4][128];       	/* calibrayion image names */
char   	img_ori[4][128];       	/* image orientation data */
char   	img_ori0[4][128];      	/* orientation approx. values */
char   	img_addpar[4][128];    	/* image additional parameters */
char   	safety[4][128];
char    safety_addpar[4][128];
char   	img_addpar0[4][128];   	/* ap approx. values */
char    seq_ch[128], seq_name[4][128];
char	img_mask_name[4][256];	/* mask image names*/
char	img_mask_path[256];
char   	track_dir[128];	       	/* directory with dap track data */
char    fixp_name[128];
char   	res_name[128];	      	/* result destination */
char   	buf[256], val[256];	       	/* buffer */
char    name[128];  //Beat Dez 08
double  xp, yp; //Beat Dez 08



unsigned char	*img[4];      	/* image data */
unsigned char	*img_mask[4];	/* mask data */
unsigned char	*img_new[4];	/* image data for reducing mask */
unsigned char	*img0[4];      	/* image data for filtering etc */
unsigned char	*zoomimg;     	/* zoom image data */

Exterior       	Ex[4],sEx[4];	      	/* exterior orientation */
Interior       	I[4],sI[4];	       	/* interior orientation */
Glass       	G[4],sG[4];	       	/* glass orientation */
ap_52	       	ap[4],sap[4];	       	/* add. parameters k1,k2,k3,p1,p2,scx,she */
target	       	pix[4][nmax]; 	/* target pixel data */
target	       	pix0[4][12];    	/* pixel data for man_ori points */

// Intermediate measure while removing other globals:
Calibration glob_cal[4];
#define UPDATE_CALIBRATION(num_cam, extp, intp, gp, imgo, app, imga, fn) \
  read_ori((extp), (intp), (gp), (imgo), (app), (imga), (fn));\
  memcpy(&(glob_cal[num_cam].ext_par), (extp), sizeof(Exterior));\
  memcpy(&(glob_cal[num_cam].int_par), (intp), sizeof(Interior));\
  memcpy(&(glob_cal[num_cam].glass_par), (gp), sizeof(Glass));\
  memcpy(&(glob_cal[num_cam].added_par), (app), sizeof(ap_52));\

int             x_calib[4][1000];
int             y_calib[4][1000];
int             z_calib[4][1000];
int 		ncal_points[4];
int orient_x1[4][1000], orient_y1[4][1000]; /* Saves leg coordinates of residual arrows */

coord_2d       	crd[4][nmax];  	/* (distorted) metric coordinates */
coord_2d       	geo[4][nmax];  	/* corrected metric coordinates */
coord_3d       	fix[20096];     	/* testfield points coordinates */ //Beat changed it on 090325
n_tupel	       	con[nmax];     	/* list of correspondences */

mm_LUT	       	mmLUT[4];     	/* LUT for multimedia radial displacement */
int lut_inited = 0;  /* temp. until integration of multimedia changes */
coord_3d        *p_c3d;

/***************************************************************************/

int init_proc_c()
{
    int  i;
    double dummy;
    
    puts ("\n Multimedia Particle Positioning and Tracking Software \n");
    
    fpp = fopen ("parameters/pft_version.par", "r");
    if (!fpp){
        fpp = fopen ("parameters/pft_version.par", "w");
        fprintf(fpp,"%d\n", 0);
        fclose(fpp);
    }
    
    /*  read from main parameter file  */
    cpar = read_control_par("parameters/ptv.par");
    cpar->mm->nlay = 1;
    
    for (i=0; i<cpar->num_cams; i++)
    {
        strncpy(img_cal[i], cpar->cal_img_base_name[i], 128);
    }
    
    /* read illuminated layer data */
    vpar = read_volume_par("parameters/criteria.par");
    
    imgsize = cpar->imx * cpar->imy;
    for (i = 0; i < cpar->num_cams; i++)
    {
        /* initialize zoom parameters and image positions */
        num[i] = 0;
        zoom_x[i] = cpar->imx/2; zoom_y[i] = cpar->imy/2; zoom_f[i] = 1;
        
        /* allocate memory for images */
        img[i] = (unsigned char *) calloc (imgsize, 1);
        if ( ! img[i])
        {
            printf ("calloc for img%d --> error\n", i);
            exit (1);
        }
        
        img_mask[i] = (unsigned char *) calloc (imgsize, 1);
        if ( ! img_mask[i])
        {
            printf ("calloc for img_mask%d --> error\n", i);
            exit (1);
        }
        
        img0[i] = (unsigned char *) calloc (imgsize, 1);
        if ( ! img0[i])
        {
            printf ("calloc for img0%d --> error\n", i);
            exit (1);
        }
        
        img_new[i] = (unsigned char *) calloc (imgsize, 1);
        if ( ! img_new[i])
        {
            printf ("calloc for img_new%d --> error\n", i);
            exit (1);
        }
    }
    
    zoomimg = (unsigned char *) calloc (imgsize, 1);
    if ( ! zoomimg)
    {
        printf ("calloc for zoomimg --> error\n");
        // return TCL_ERROR; //denis 26-10-2010
        return 1;
    }
    
    //parameter_panel_init(interp);
    //cr_sz = atoi(Tcl_GetVar2(interp, "mp", "pcrossize",  TCL_GLOBAL_ONLY));
    
    display = 1;
    // return TCL_OK;
    
    // globals for correspondences, denis 01/11/2010
    match4_g=0;
    match3_g=0;
    match2_g=0;
    match1_g=0;
    //----------------------------------
    
    return 0;
    
}


int start_proc_c()
{
    int  i, k;
    unsigned char *im0 = img[0];
    
    /*  read from main parameter file  */
    cpar = read_control_par("parameters/ptv.par");
    cpar->mm->nlay = 1;
    
    for (i=0; i<cpar->num_cams; i++)
    {
        strncpy(img_cal[i], cpar->cal_img_base_name[i], 128);
    }
    
    /* read illuminated layer data */
    vpar = read_volume_par("parameters/criteria.par");
    
    for (i = 0; i < cpar->num_cams; i++)
    {
        /*  create file names  */
        strcpy (img_lp_name[i], cpar->img_base_name[i]); 
        strcat (img_lp_name[i], "_lp");
        
        strcpy (img_hp_name[i], cpar->img_base_name[i]);
        strcat (img_hp_name[i], "_hp");
        
        strcpy (img_ori[i], img_cal[i]);  strcat (img_ori[i], ".ori");
        strcpy (img_addpar[i], img_cal[i]); strcat (img_addpar[i],".addpar");
        
        /*  read orientation and additional parameters  */
        UPDATE_CALIBRATION(i, &Ex[i], &I[i], &G[i], img_ori[i], &(ap[i]),
            img_addpar[i], NULL)
        rotation_matrix(&(Ex[i]));
    
    }
    
	return 0;
}

/* pre_processing_c() performs the image processing that makes the image ready for 
   particle detection.
   
   Arguments:
   int y_remap_mode - a flag denoting how to treat interlaced cameras. Not used 
     anymore so should be 0. Consult trafo.c for more detail.
*/
int pre_processing_c(int y_remap_mode)
{
    int i_img, sup, i;
    
    sprintf(val, "Filtering with Highpass");
    
    /* read support of unsharp mask */
    fpp = fopen ("parameters/unsharp_mask.par", "r");
    if ( fpp == 0) { sup = 12;}
    else	{ fscanf (fpp, "%d\n", &sup); fclose (fpp); }
    
    //_____________________Matthias subtract mask__________________________
    
    
    /* Matthias JULI 08 read checkmark for masks and create mask names*/
    
    fpp = fopen_r ("parameters/targ_rec.par");
    for (i=0; i<14; i++){
        fscanf (fpp, "%d", &mask);      /*checkmark for subtract mask */
    }
    fscanf (fpp, "%s\n", img_mask_path);
    fclose (fpp);
    /*read mask names*/
    strcpy (img_mask_name[0], img_mask_path); strcat (img_mask_name[0], ".0");
    strcpy (img_mask_name[1], img_mask_path); strcat (img_mask_name[1], ".1");
    strcpy (img_mask_name[2], img_mask_path); strcat (img_mask_name[2], ".2");
    strcpy (img_mask_name[3], img_mask_path); strcat (img_mask_name[3], ".3");
    
    /*if the checkmark is set, read mask-image and subtract it from the filtered-original image.*/
    if (mask==1)
    {//read mask image
        for (i_img=0; i_img < cpar->num_cams; i_img++)
        {
            highpass (img[i_img], img[i_img], sup, 0, cpar);
            subtract_mask (img[i_img], img_mask[i_img], img_new[i_img]); //subtract mask from original image
            memcpy(img[i_img], img_new[i_img], imgsize);
            
            // sprintf(val, "newimage %d", i_img+1);
        }
    }//end if
    
    if (mask==2)//Beat April 090402 was ==0
    {
        for (i_img=0; i_img < cpar->num_cams; i_img++)
        {
            highpass (img[i_img], img[i_img], sup, 0, cpar);//highpass original image
        }
    }//end if
    
    /*------------------------------------------------------------*/
    
    
    return 0;
    
}


int detection_proc_c(char **image_names) 
{
    int	       	i, i_img, j;
    int	       	xmin, pft_version=3;
    char val[256];
    char filename[256];
    FILE	*FILEIN;
    
    /* process info */
    sprintf(val, "Detection of Particles");
    strcpy(val, "");
    
    /* xmin set to 10 so v_line is not included in detection, in future xmin should
     be set to 0, peakfitting has to be changed too */
    xmin=0;
    
    /*  read pft version  */
    fpp = fopen ("parameters/pft_version.par", "r");
    if (fpp){
        fscanf (fpp, "%d\n", &pft_version);
        pft_version = pft_version + 3;
        printf(" Peak fitting version is %d\n", pft_version);
        fclose (fpp);
    }
    else{
        fpp = fopen ("parameters/pft_version.par", "w");
        fprintf(fpp,"%d\n", 0);
        fclose(fpp);
    }
    
    
    /* reset zoom values */
    for (i_img = 0; i_img < cpar->num_cams; i_img++)
    {
        zoom_x[i_img] = cpar->imx/2;
        zoom_y[i_img] = cpar->imy/2;
        zoom_f[i_img] = 1;
        /*copy images because the target recognition will set greyvalues to 0*/
        memcpy(img0[i_img], img[i_img], imgsize);
    }
    
    /* target recognition */
    for (i_img = 0; i_img < cpar->num_cams; i_img++)
    {
        switch (pft_version)
        {
            case 3:	/* pft with profile and distance check */
                /* newest version */
                xmin=0; /* vertical line restriction */
                
                num[i_img] = peak_fit_new (img[i_img],
                    "parameters/targ_rec.par",
                    xmin, cpar->imx, 1, cpar->imy, pix[i_img], i_img, cpar);
                break;
                
            case 0:	/* without peak fitting technique */
                simple_connectivity (img[i_img], img0[i_img],
                    "parameters/targ_rec.par",
                    xmin, cpar->imx, 1, cpar->imy, pix[i_img], i_img, &num[i_img], cpar);
                break;
                
            case 1:	/* with old (but fast) peak fitting technique */
                targ_rec (img[i_img], img0[i_img],
                    "parameters/targ_rec.par",
                    xmin, cpar->imx, 1, cpar->imy, pix[i_img], i_img, &num[i_img], cpar);
                break;
                
            case 4: /* new option for external image processing routines */
                /* added by Alex, 19.04.10 */
                /* this works here only for the pre-processing stage */
                
                num[i_img] = read_targets(pix[i_img], image_names[i_img], 0);
                                
                // printf("pix.x0=%f\n",pix[i_img][0].x);
                // printf("pix.y0=%f\n",pix[i_img][0].y);
                
                break;
        }
        
        // printf("pix.x0=%f\n",pix[0][0].x);
        // printf("pix.y0=%f\n",pix[0][0].y);
        // sprintf (buf,"%d: %d,  ", i_img+1, num[i_img]);
        // strcat(val, buf);
        
        /* proper sort of targets in y-direction for later binary search */
        /* and for dimitris' tracking */
        quicksort_target_y (pix[i_img], num[i_img]);
        
        /* reorganize target numbers */
        for (i=0; i<num[i_img]; i++)  pix[i_img][i].pnr = i;
    }
    printf("pix.x01=%f\n",pix[0][0].x);
	printf("pix.y01=%f\n",pix[0][0].y);
    
    sprintf (buf, "Number of detected particles per image");
    printf("%s\n", val);
    return 1;
}

/* Arguments:
   img_base_names - per-camera name of image without the frame number.
   int frame - frame number to use when composing file names (0 for none).
*/
int correspondences_proc_c (char **img_base_names, int frame)
{
    int	i, i_img;
    double x,y;
    
    puts ("\nTransformation to metric coordinates\n");
    
    for (i_img = 0; i_img < cpar->num_cams; i_img++) {
        for (i=0; i<num[i_img]; i++) {
            /* rearrange point numbers after manual deletion of points */
            pix[i_img][i].pnr = i;
            
            /* transformations pixel coordinates -> metric coordinates */
            /* transformations metric coordinates -> corrected metric coordinates */
            pixel_to_metric(&crd[i_img][i].x, &crd[i_img][i].y, 
                pix[i_img][i].x, pix[i_img][i].y, cpar);
            crd[i_img][i].pnr = pix[i_img][i].pnr;
            
            x = crd[i_img][i].x - I[i_img].xh;
            y = crd[i_img][i].y - I[i_img].yh;
            correct_brown_affin (x, y, ap[i_img], &geo[i_img][i].x, &geo[i_img][i].y);
            
            geo[i_img][i].pnr = crd[i_img][i].pnr;
        }
        
        /* sort coordinates for binary search in correspondences_proc */
        quicksort_coord2d_x (geo[i_img], num[i_img]);
    }
    
    /* init multimedia radial displacement LUTs */
    /* ======================================== */
    
    if ( !lut_inited && (cpar->mm->n1 != 1 || cpar->mm->n2[0] != 1 || cpar->mm->n3 != 1))
    {
        puts ("Init multimedia displacement LUTs");
        for (i_img = 0; i_img < cpar->num_cams; i_img++) 
            init_mmLUT(i_img, glob_cal + i_img, cpar);
        lut_inited = 1;
    }
    
    correspondences_4 (vpar, cpar);
    
    /* --------------- */
    /* save pixel coords for tracking */
    for (i_img = 0; i_img < cpar->num_cams; i_img++) {
        write_targets(pix[i_img], num[i_img], img_base_names[i_img], frame);
    }
    
    return 0;
}


int calibration_proc_c (int sel)
{
    int i, j,  i_img, k, n, sup,dummy,multi,planes;
    int prev, next; 
    int chfield;       		       	/* flag for field mode */
    
    double resid_x[1000], resid_y[1000]; /* raw residuals */
    int pixnr[1000]; /* Array for numbers of points used by the end 
                        orientation. Waits for a redesign. */
    
    double dummy_float;
    int intx1, inty1, intx2, inty2;
    coord_2d    	apfig1[11][11];	/* regular grid for ap figures */
    coord_2d     	apfig2[11][11];	/* ap figures */
    coord_3d     	fix4[4];       	/* object points for preorientation */
    coord_2d     	crd0[4][12];    	/* image points for preorientation */
    char	       	multi_filename[10][256],filename[256], val[256];
    const char *valp;
    
    FILE	*FILEIN;
    char	filein[256];
    FILE	*FILEIN_ptv;
    char	filein_ptv[256];
    FILE	*FILEIN_T;
    char	filein_T[256];
    int filenumber;
    int dumy,frameCount,currentFrame;
    int a[4],a1[4],a2[4],success=1;
    double residual;
    
    /* read support of unsharp mask */
    fp1 = fopen ("parameters/unsharp_mask.par", "r");
    if (! fp1)	sup = 12;
    else	{ fscanf (fp1, "%d\n", &sup); fclose (fp1); }
    
    /* Beat Mai 2007 to set the variable examine for mulit-plane calibration*/
    fp1 = fopen_r ("parameters/examine.par");
    fscanf (fp1,"%d\n", &dummy);
    fscanf (fp1,"%d\n", &multi);
    fclose (fp1);
    if (dummy==1){
        examine=4;
    }
    else{
        examine=0;
    }
    // printf("after 1\n");
    /*Oswald Juni 2008 accept pairs-------------------------------*/
    
    
    fp1 = fopen_r ("parameters/cal_ori.par");
    fscanf (fp1,"%s\n", fixp_name);
    for (i=0; i<4; i++)
	{
        fscanf (fp1, "%s\n", img_name[i]);
        fscanf (fp1, "%s\n", img_ori0[i]);
	}
    fscanf (fp1, "%d\n", &dummy_float);
    fscanf (fp1, "%d\n", &pair_flag);
    fscanf (fp1, "%d\n", &chfield);
    fclose (fp1);
    
    if (pair_flag==1){
        int OSWALDDUMMY=1;
    }
    else{
        int OSWALDDUMMY=0;
    }
    
    // printf("after 4\n");
    ///////////////////////////////////////////////////////////////////////////////
    
    switch (sel)
    {
        case 1: /*  read calibration parameter file  */
            /* But this is always done. So skip. */
            
            /*  create file names  */
            for (i=0; i < cpar->num_cams; i++)
            {
                strcpy (img_ori[i], img_name[i]);
                strcat (img_ori[i], ".ori");
                strcpy (img_addpar0[i], img_name[i]);
                strcat (img_addpar0[i], ".addpar0");
                strcpy (img_addpar[i], img_name[i]);
                strcat (img_addpar[i], ".addpar");
                strcpy (img_hp_name[i], img_name[i]);
                strcat (img_hp_name[i], "_hp");
            }
            strcpy (safety[0], "safety_0");
            strcat (safety[0], ".ori");
            strcpy (safety[1], "safety_1");
            strcat (safety[1], ".ori");
            strcpy (safety[2], "safety_2");
            strcat (safety[2], ".ori");
            strcpy (safety[3], "safety_3");
            strcat (safety[3], ".ori");
            strcpy (safety_addpar[0], "safety_0");
            strcat (safety_addpar[0], ".addpar");
            strcpy (safety_addpar[1], "safety_1");
            strcat (safety_addpar[1], ".addpar");
            strcpy (safety_addpar[2], "safety_2");
            strcat (safety_addpar[2], ".addpar");
            strcpy (safety_addpar[3], "safety_3");
            strcat (safety_addpar[3], ".addpar");
            
            /* commented out the print used for debugging
            for (i=0; i<50; i++)
            {
                printf("img0=%d\n",img[0][i]);
            }
            */
            break;
            
            
        case 2: // puts ("Detection procedure"); strcpy(val,"");
            
            printf("Detection procedure\n");
            
            /* Highpass Filtering */
            pre_processing_c (chfield);
            
            /* reset zoom values */
            for (i = 0; i < cpar->num_cams; i++)
            {
                zoom_x[i] = cpar->imx/2; zoom_y[i] = cpar->imy/2; zoom_f[i] = 1;
            }
            
            /* copy images because the target recognition
             will set greyvalues to zero */
            printf("\n after high pass inside detection");
            for (i = 0; i < cpar->num_cams; i++)
            {
                memcpy(img0[i], img[i], imgsize);
            }
            
            /* target recognition */
            for (i = 0; i < cpar->num_cams; i++)
            {
                targ_rec (img[i], img0[i], "parameters/detect_plate.par",
                          0, cpar->imx, 1, cpar->imy, pix[i], i, &num[i], cpar);
                
                
                // sprintf (buf,"image %d: %d,  ", i+1, num[i]);
                // strcat(val, buf);
                printf("image %d: %d,  \n", i+1, num[i]);
                
                if (num[i] > nmax)  exit (1);
            }
            printf("\n after targ_rec inside detection");
            /* save pixel coord as approx. for template matching */
            if (examine) for (i = 0; i < cpar->num_cams; i++)
            {
                sprintf (filename, "%s_pix", img_name[i]);
                fp1 = fopen (filename, "w");
                for (j=0; j<num[i]; j++)
                    fprintf (fp1, "%4d  %8.3f  %8.3f\n",
                             pix[i][j].pnr, pix[i][j].x, pix[i][j].y);
                
                fclose (fp1);
            }
            
            break;
            
        case 4: /* read pixel coordinates of older pre-orientation */
            
            /* read point numbers of pre-clicked points */
            fp1 = fopen_r ("parameters/man_ori.par");
            for (i = 0; i < cpar->num_cams; i++)
            {
                fscanf (fp1, "%d %d %d %d \n",
                        &nr[i][0], &nr[i][1], &nr[i][2], &nr[i][3]);
            }
            fclose (fp1);
            
            /* read coordinates of pre-clicked points */
            fp1 = fopen ("man_ori.dat", "r");
            if (! fp1)	break;
            for (i_img = 0; i_img < cpar->num_cams; i_img++) for (i=0; i<4; i++)
            {
                fscanf (fp1, "%lf %lf\n",
                        &pix0[i_img][i].x, &pix0[i_img][i].y);
                
            }
            fclose (fp1);
            
            break;
        case 9: puts ("Plot initial guess");
            for (i=0; i < cpar->num_cams; i++)
            {
                /* read control point coordinates for man_ori points */
                fp1 = fopen_r (fixp_name);
                k = 0;
                while ( fscanf (fp1, "%d %lf %lf %lf", &fix[k].pnr,
                                &fix[k].x, &fix[k].y, &fix[k].z) != EOF) k++;
                fclose (fp1);
                nfix = k;
                
                /* read initial guess orientation from the ori files, no add params */
                read_ori (&Ex[i], &I[i], &G[i], img_ori0[i], &(ap[i]), 
                    img_addpar0[i], "addpar.raw");
                
                /* presenting detected points by back-projection */
                just_plot (Ex[i], I[i], G[i], ap[i], nfix, fix, i, cpar);
                
                /*write artifical images*/
                
                
            }
            
            break;
        case 5: puts ("Sort grid points - Alex test 18.5.11");
            for (i = 0; i < cpar->num_cams; i++)
            {
                /* read control point coordinates for man_ori points */
                fp1 = fopen_r (fixp_name);
                k = 0;
                while ( fscanf (fp1, "%d %lf %lf %lf", &fix[k].pnr,
                                &fix[k].x, &fix[k].y, &fix[k].z) != EOF) k++;
                fclose (fp1);
                nfix = k;
                
                /* take clicked points from control point data set */
                for (j=0; j<4; j++)	for (k=0; k<nfix; k++)
                {
                    if (fix[k].pnr == nr[i][j])	fix4[j] = fix[k];
                }
                
                /* get approx for orientation and ap */
                UPDATE_CALIBRATION(i, &Ex[i], &I[i], &G[i], img_ori0[i], &(ap[i]),
                    img_addpar0[i], "addpar.raw")
                
                /* transform clicked points */
                for (j=0; j<4; j++)
                {
                    pixel_to_metric(&crd0[i][j].x, &crd0[i][j].y,
                        pix0[i][j].x, pix0[i][j].y, cpar);
                    correct_brown_affin (crd0[i][j].x, crd0[i][j].y, ap[i],
                                         &crd0[i][j].x, &crd0[i][j].y);
                }
                
                /* raw orientation with 4 points */
                raw_orient_v3 (Ex[i], I[i], G[i], ap[i], *(cpar->mm), 4, fix4, crd0[i], 
                    &Ex[i], &G[i], i, 0);
                sprintf (filename, "raw%d.ori", i);
                write_ori (Ex[i], I[i], G[i], ap[i], filename, NULL); /*ap ignored*/
                
                /* sorting of detected points by back-projection */
                sortgrid_man (Ex[i], I[i], G[i], ap[i], nfix, fix, num[i], 
                    pix[i], i, cpar);
                
                /* adapt # of detected points */
                num[i] = nfix;
                
                
                
                ncal_points[i]=nfix;
                for (j=0; j<nfix; j++)
                {
                    intx1 = (int) pix[i][j].x ;
                    inty1 = (int) pix[i][j].y ;
                }
            }
            
            /* dump dataset for rdb */
            if (examine == 4)
            {
                /* create filename for dumped dataset */
                sprintf (filename, "dump_for_rdb");
                fp1 = fopen (filename, "w");
                
                /* write # of points to file */
                fprintf (fp1, "%d\n", nfix);
                
                /* write point and image coord to file */
                for (i=0; i<nfix; i++)
                {
                    fprintf (fp1, "%4d %10.3f %10.3f %10.3f   %d    ",
                             fix[i].pnr, fix[i].x, fix[i].y, fix[i].z, 0);
                    for (i_img=0; i_img < cpar->num_cams; i_img++)
                    {
                        if (pix[i_img][i].pnr >= 0)
                        {
                            /* transform pixel coord to metric */
                            pixel_to_metric(&crd[i_img][i].x, &crd[i_img][i].y, 
                                pix[i_img][i].x, pix[i_img][i].y, cpar);
                            fprintf (fp1, "%4d %8.5f %8.5f    ",
                                     pix[i_img][i].pnr,
                                     crd[i_img][i].x, crd[i_img][i].y);
                        }
                        else
                        {
                            fprintf (fp1, "%4d %8.5f %8.5f    ",
                                     pix[i_img][i].pnr, 0.0, 0.0);
                        }
                    }
                    fprintf (fp1, "\n");
                }
                fclose (fp1);
                printf ("dataset dumped into %s\n", filename);
            }
            break;
            
        case 14: puts ("Sortgrid = initial guess");
            
            for (i=0; i < cpar->num_cams; i++)
            {
                /* read control point coordinates for man_ori points */
                fp1 = fopen_r (fixp_name);
                k = 0;
                while ( fscanf (fp1, "%d %lf %lf %lf", &fix[k].pnr,
                                &fix[k].x, &fix[k].y, &fix[k].z) != EOF) k++;
                fclose (fp1);
                nfix = k;
                
                /* get approx for orientation and ap */
                UPDATE_CALIBRATION(i, &Ex[i], &I[i], &G[i], img_ori0[i], &(ap[i]),
                    img_addpar0[i], "addpar.raw");
                
                /* sorting of detected points by back-projection */
                sortgrid_man (Ex[i], I[i], G[i], ap[i], nfix, fix, num[i], 
                    pix[i], i, cpar);
                
				/* adapt # of detected points */
				num[i] = nfix;
                
				for (j=0; j<nfix; j++)
				{
					if (pix[i][j].pnr < 0) z_calib[i][j]=pix[i][j].pnr;
				    else  z_calib[i][j]=fix[j].pnr;
                    intx1 = (int) pix[i][j].x ;
                    inty1 = (int) pix[i][j].y ;
				}
            }
            
            break;  
            
            
            
        case 6: puts ("Orientation"); strcpy(buf, "");
            
            strcpy (safety[0], "safety_0");
            strcat (safety[0], ".ori");
            strcpy (safety[1], "safety_1");
            strcat (safety[1], ".ori");
            strcpy (safety[2], "safety_2");
            strcat (safety[2], ".ori");
            strcpy (safety[3], "safety_3");
            strcat (safety[3], ".ori");
            strcpy (safety_addpar[0], "safety_0");
            strcat (safety_addpar[0], ".addpar");
            strcpy (safety_addpar[1], "safety_1");
            strcat (safety_addpar[1], ".addpar");
            strcpy (safety_addpar[2], "safety_2");
            strcat (safety_addpar[2], ".addpar");
            strcpy (safety_addpar[3], "safety_3");
            strcat (safety_addpar[3], ".addpar");
            
            for (i_img = 0; i_img < cpar->num_cams; i_img++)
            {
                for (i=0; i<nfix ; i++)
                {
                    pixel_to_metric(&crd[i_img][i].x, &crd[i_img][i].y,
                        pix[i_img][i].x, pix[i_img][i].y, cpar);
                    crd[i_img][i].pnr = pix[i_img][i].pnr;
                }
                
                /* save data for special use of resection routine */
                if (examine == 4 && multi==0)
                {
                    printf ("try write resection data to disk\n");
                    /* point coordinates */
                    //sprintf (filename, "resect_%s.fix", img_name[i_img]);
                    sprintf (filename, "%s.fix", img_name[i_img]);
                    write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img],
                        img_ori[i_img], NULL); /* ap ignored */
                    fp1 = fopen (filename, "w");
                    for (i=0; i<nfix; i++)
                        fprintf (fp1, "%3d  %10.5f  %10.5f  %10.5f\n",
                                 fix[i].pnr, fix[i].x, fix[i].y, fix[i].z);
                    fclose (fp1);
                    
                    /* metric image coordinates */
                    //sprintf (filename, "resect_%s.crd", img_name[i_img]);
                    sprintf (filename, "%s.crd", img_name[i_img]);
                    fp1 = fopen (filename, "w");
                    for (i=0; i<nfix; i++)
                        fprintf (fp1,
                                 "%3d  %9.5f  %9.5f\n", crd[i_img][i].pnr,
                                 crd[i_img][i].x, crd[i_img][i].y);
                    fclose (fp1);
                    
                    /* orientation and calibration approx data */
                    write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img],
                        "resect.ori0", "resect.ap0");
                    printf ("resection data written to disk\n");
                }
                
                
                /* resection routine */
                /* ================= */
                printf ("examine=%d\n",examine);
                if (examine != 4) {
                    orient_v3 (Ex[i_img], I[i_img], G[i_img], ap[i_img], *(cpar->mm),
                               nfix, fix, crd[i_img],
                               &Ex[i_img], &I[i_img], &G[i_img], &ap[i_img], i_img,
                               resid_x, resid_y, pixnr, orient_n + i_img);
                    for (i = 0; i < orient_n[i_img]; i++) {
                        orient_x1[i_img][i] = pix[i_img][pixnr[i]].x;
                        orient_y1[i_img][i] = pix[i_img][pixnr[i]].y;
                        orient_x2[i_img][i] = pix[i_img][pixnr[i]].x + 5000*resid_x[i];
                        orient_y2[i_img][i] = pix[i_img][pixnr[i]].y + 5000*resid_y[i];
                    }
                }    
                /* ================= */
                
                
                /* resection with dumped datasets */
                if (examine == 4)
                {
                    printf("Resection with dumped datasets? (y/n)");
                    //scanf("%s",buf);
                    //if (buf[0] != 'y')	continue;
                    //strcpy (buf, "");
                    if (multi == 0)	continue;
                    
                    /* read calibration frame datasets */
                    //sprintf (multi_filename[0],"img/calib_a_cam");
                    //sprintf (multi_filename[1],"img/calib_b_cam");
                    
                    fp1 = fopen_r ("parameters/multi_planes.par");
                    fscanf (fp1,"%d\n", &planes);
                    for(i=0;i<planes;i++) 
                        fscanf (fp1,"%s\n", multi_filename[i]);
                        //fscanf (fp1,"%s\n", &multi_filename[i]);
                    fclose(fp1);
                    for (n=0, nfix=0, dump_for_rdb=0; n<10; n++)
                    {
                        //sprintf (filename, "resect.fix%d", n);
                        
                        sprintf (filename, "%s%d.tif.fix", multi_filename[n],i_img+1);
                        
                        fp1 = fopen (filename, "r");
                        if (! fp1)	continue;
                        
                        printf("reading file: %s\n", filename);
                        k = 0;
                        while ( fscanf (fp1, "%d %lf %lf %lf",
                                        &fix[nfix+k].pnr, &fix[nfix+k].x,
                                        &fix[nfix+k].y, &fix[nfix+k].z)
                               != EOF) k++;
                        fclose (fp1);
                                                
                        /* read metric image coordinates */
                        //sprintf (filename, "resect_%d.crd%d", i_img, n);
                        sprintf (filename, "%s%d.tif.crd", multi_filename[n],i_img+1);
                        printf("reading file: %s\n", filename);
                        fp1 = fopen (filename, "r");
                        if (! fp1)	continue;
                        
                        for (i=nfix; i<nfix+k; i++)
                            fscanf (fp1, "%d %lf %lf",
                                    &crd[i_img][i].pnr,
                                    &crd[i_img][i].x, &crd[i_img][i].y);
                        nfix += k;
                        fclose (fp1);
                    }
                    
                    printf("nfix = %d\n",nfix);
                    
                    /* resection */
                    /*Beat Mai 2007*/
                    sprintf (filename, "raw%d.ori", i_img);
                    UPDATE_CALIBRATION(i_img, &Ex[i_img], &I[i_img], &G[i_img], filename,
                        &(ap[i_img]), "addpar.raw", NULL);
                    
                    /* markus 14.05.2007 show coordinates combined */
                    for (i=0; i<nfix ; i++)			  
                    {
                        /* first crd->pix */
                        metric_to_pixel(&pix[i_img][i].x, &pix[i_img][i].y, 
                            crd[i_img][i].x, crd[i_img][i].y, cpar);
                    }
                    
                    
                    orient_v3 (Ex[i_img], I[i_img], G[i_img], ap[i_img], *(cpar->mm),
                               nfix, fix, crd[i_img],
                               &Ex[i_img], &I[i_img], &G[i_img], &ap[i_img], i_img,
                               resid_x, resid_y, pixnr,
                               orient_n + i_img);
                    for (i = 0; i < orient_n[i_img]; i++) {
                        orient_x1[i_img][i] = pix[i_img][pixnr[i]].x;
                        orient_y1[i_img][i] = pix[i_img][pixnr[i]].y;
                        orient_x2[i_img][i] = pix[i_img][pixnr[i]].x + 5000*resid_x[i];
                        orient_y2[i_img][i] = pix[i_img][pixnr[i]].y + 5000*resid_y[i];
                    }
                    ///////////////////////////////////////////
                    
                    
                }
                
                
                /* save orientation and additional parameters */
                write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img],
                    img_ori[i_img], NULL);
                fp1 = fopen( img_ori[i_img], "r" );
                if(fp1 != NULL) {
                    fclose(fp1);
                    UPDATE_CALIBRATION(i_img, &sEx[i_img], &sI[i_img], &sG[i_img], 
                        img_ori[i_img], &(sap[i_img]), img_addpar0[i_img],
                        "addpar.raw")
                    
                    write_ori (sEx[i_img], sI[i_img], sG[i_img], sap[i_img], 
                        safety[i_img], safety_addpar[i_img]);
                }
                else{
                    write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img],
                        safety[i_img], safety_addpar[i_img]);
                }
                write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img], img_ori[i_img],
                    img_addpar[i_img]);
            }
            break;
            
        case 10: 
        	printf ("Orientation from particles \n"); 
			
			strcpy (safety[0], "safety_0");
			strcat (safety[0], ".ori");
			strcpy (safety[1], "safety_1");
			strcat (safety[1], ".ori");
			strcpy (safety[2], "safety_2");
			strcat (safety[2], ".ori");
			strcpy (safety[3], "safety_3");
			strcat (safety[3], ".ori");
			strcpy (safety_addpar[0], "safety_0");
			strcat (safety_addpar[0], ".addpar");
			strcpy (safety_addpar[1], "safety_1");
			strcat (safety_addpar[1], ".addpar");
			strcpy (safety_addpar[2], "safety_2");
			strcat (safety_addpar[2], ".addpar");
			strcpy (safety_addpar[3], "safety_3");
			strcat (safety_addpar[3], ".addpar");
			
            cpar = read_control_par("parameters/ptv.par");
            prepare_eval_shake(cpar);
            
			for (i_img = 0; i_img < cpar->num_cams; i_img++)
			{
                orient_v3 (Ex[i_img], I[i_img], G[i_img], ap[i_img], *(cpar->mm),
                           nfix, fix, crd[i_img],
                           &Ex[i_img], &I[i_img], &G[i_img], &ap[i_img], i_img,
                           resid_x, resid_y, pixnr, orient_n + i_img);
                for (i = 0; i < orient_n[i_img]; i++) {
                    orient_x1[i_img][i] = pix[i_img][pixnr[i]].x;
                    orient_y1[i_img][i] = pix[i_img][pixnr[i]].y;
                    orient_x2[i_img][i] = pix[i_img][pixnr[i]].x + 5000*resid_x[i];
                    orient_y2[i_img][i] = pix[i_img][pixnr[i]].y + 5000*resid_y[i];
                }
				
				/* save orientation and additional parameters */
				//make safety copy of ori files
				
				fp1 = fopen( img_ori[i_img], "r" );
				if(fp1 != NULL) {
					fclose(fp1);
					UPDATE_CALIBRATION(i_img, &sEx[i_img], &sI[i_img], &sG[i_img],
                        img_ori[i_img], &(sap[i_img]), img_addpar0[i_img],
                        "addpar.raw");
					
					write_ori (sEx[i_img], sI[i_img], sG[i_img], sap[i_img],
                        safety[i_img], safety_addpar[i_img]);
				}
				else{
					write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img],
                        safety[i_img], safety_addpar[i_img]);
				}
				write_ori (Ex[i_img], I[i_img], G[i_img], ap[i_img],
                    img_ori[i_img], img_addpar[i_img]);
			}
			break;
            
        case 12: puts ("Orientation from dumbbells"); strcpy(buf, "");
			
            prepare_eval(cpar, &nfix); //goes and looks up what sequence is defined and takes all cord. from rt_is
            orient_v5 (cpar, nfix, glob_cal);
			
            for(i_img = 0; i_img < cpar->num_cams; i_img++){
                write_ori (glob_cal[i_img].ext_par, glob_cal[i_img].int_par, 
                    glob_cal[i_img].glass_par, glob_cal[i_img].added_par,
                    img_ori[i_img], img_addpar[i_img]);
            }
			
            break;
            
    }
    
    return 0;
}

int sequence_proc_c  (int dumb_flag)
{
    int     i, j, ok, k, nslices=19, slicepos=0, pft_version = 3;
    int dumbbell=0;
    double dummy;
    
    seq_step_shake=1;
    fpp = fopen_r ("parameters/sequence.par");
    for (i=0; i<4; i++)
        fscanf (fpp, "%s\n", seq_name[i]);     /* name of sequence */
    fscanf (fpp,"%d\n", &seq_first);
    fscanf (fpp,"%d\n", &seq_last);
    fclose (fpp);
    
    
    //display = atoi(argv[1]); 
    //Beat Mai 2010 for dumbbell
    if (dumb_flag==3){
        dumbbell=1;
        display=0;
    }
    
    /* scanning ptv ************** */
    printf("\nObject volume is scanned in %d slices!\n", nslices);
    slicepos=0;
    
    /* read illuminated Volume */
    vpar = read_volume_par("parameters/criteria.par");
    
    /* read illuminated layer data */
    if (dumbbell==1){
        fpp = fopen ("parameters/dumbbell.par", "r");
        if (fpp){
            fscanf (fpp, "%lf", &(vpar->eps0));
            fscanf (fpp, "%lf", &seq_dummy);
            fscanf (fpp, "%lf", &seq_dummy);
            fscanf (fpp, "%lf", &seq_dummy);
            fscanf (fpp, "%d", &seq_step_shake);
            fclose (fpp);
        }
        else{
            fpp = fopen ("parameters/dumbbell.par", "w");
            fprintf(fpp,"%lf\n", 5.0);
            fprintf(fpp,"%lf\n", 46.5);
            fprintf(fpp,"%lf\n", 0.5);
            fprintf(fpp,"%lf\n", 2.);
            fprintf(fpp,"%d\n", 2);
            fprintf(fpp,"%d\n",500);
            fprintf(fpp,"\n\n");
            fprintf(fpp,"explanation for parameters:\n");
            fprintf(fpp,"\n");
            fprintf(fpp,"1: eps (mm)\n");
            fprintf(fpp,"2: dunbbell scale\n");
            fprintf(fpp,"3: gradient descent factor\n");
            fprintf(fpp,"4: weight for dumbbell penalty\n");
            fprintf(fpp,"5: step size through sequence\n");
            fprintf(fpp,"6: num iterations per click\n");
            fclose(fpp);
            vpar->eps0 = 10;
        }
    }
    
    
    cpar->mm->nlay = 1;
    
    seq_zdim = vpar->Zmax_lay[0] - vpar->Zmin_lay[0];
    seq_slice_step= seq_zdim/nslices;
    seq_slicethickness=5.0;
    
    printf("\nzdim: %f, max: %f, min: %f, st: %f\n", seq_zdim,
        vpar->Zmax_lay[0], vpar->Zmin_lay[0], seq_slice_step);
    
    return 0;
}

int sequence_proc_loop_c  (int dumbbell,int i)
{
    double slice_step,slicethickness,zdim,dummy;
    
    int step_shake;
    int j,k,pft_version = 3,ok;
    char *seq_name_ptrs[4] = {seq_name[0], seq_name[1], seq_name[2], seq_name[3]};
    
    slice_step=seq_slice_step;
    slicethickness=seq_slicethickness;
    zdim=seq_zdim;
    dummy=seq_dummy;
    step_shake=seq_step_shake;
    
    if (i < 10)             sprintf (seq_ch, "%1d", i);
    else if (i < 100)       sprintf (seq_ch, "%2d",  i);
    else       sprintf (seq_ch, "%3d",  i);
    
    for (j = 0; j < cpar->num_cams; j++)
	{
        sprintf (img_lp_name[j], "%s%s_lp", seq_name[j], seq_ch);
        sprintf (img_hp_name[j], "%s%s_hp", seq_name[j], seq_ch);
	}
    
    //Beat Mai 2010 for dumbbell
    if (dumbbell==0){
        if (cpar->chfield == 0)
            sprintf (res_name, "res/rt_is.%s", seq_ch);
        else
            sprintf (res_name, "res/rt_is.%s_%1d", seq_ch, cpar->chfield);
    }
    else{
        if (cpar->chfield == 0)
            sprintf (res_name, "res/db_is.%s", seq_ch);
        else
            sprintf (res_name, "res/db_is.%s_%1d", seq_ch, cpar->chfield);
    }
    
    /* calling function for each sequence-n-tupel */
    
    /* read and display original images */
    
    /*  read pft version  */
    /* added by Alex for external detection procedure, 19.4.10 */
    fpp = fopen ("parameters/pft_version.par", "r");
    if (fpp)
    {
        fscanf (fpp, "%d\n", &pft_version);
        pft_version=pft_version+3;
        fclose (fpp);
    }
    
    if (cpar->hp_flag) {
        pre_processing_c (cpar->chfield);
        puts("\nHighpass switched on\n");
    } else { puts("\nHighpass switched off\n"); }
    /*      if (display) {Tcl_Eval(interp, "update idletasks");}*/
    /**************************************************************************************/
    /* pft_version = 4 means external detection, Alex, 19.4.10 */
    
    if ( pft_version == 4) { 
		for (k = 0; k < cpar->num_cams; k++) {
            num[k] = read_targets(pix[k], seq_name[k], i);
            
            /* proper sort of targets in y-direction for later binary search */
            /* and for dimitris' tracking */
            quicksort_target_y (pix[k], num[k]);
            /* reorganize target numbers */
            for (j=0; j<num[k]; j++)  pix[k][j].pnr = j;
        }
    } 
    /***************************************************************************************/
    else {
		detection_proc_c (NULL); // added i to the detection_proc_c to get 'filenumber' for external API, Alex, 19.04.10
    }
    
    correspondences_proc_c (seq_name_ptrs, i);
    
    if (cpar->num_cams > 1) {
		determination_proc_c (dumbbell);
    }
    
    /* delete unneeded files */
    for (j=0; j < cpar->num_cams; j++)
	{
        ok = remove (img_lp_name[j]);
        ok = remove (img_hp_name[j]);
	}
    
    /* reset of display flag */
    display = 1;
    
    return 0;
}


int determination_proc_c (int dumbbell)
{
    int  	i, j, n,dummy;
    int  	p[4];
    double  x[4], y[4], X,Y,Z;
    double  Zlo = 1e20, Zhi = -1e20;
    //int dumbbell=0,i1,i2;
    int i1,i2;
    double x1,y1,z1,x2,y2,z2,dist,mx,my,mz,nx,ny,nz;
    int a1[4],a2[4],checksum_1,checksum_2;
    
    
    printf ("Determinate \n");
    
    sprintf (buf, "Point positioning (mid_point in 3d)");
    
    /* Beat Mai 2007 to set the variable examine for mulit-plane calibration*/
    fp1 = fopen_r ("parameters/examine.par");
    fscanf (fp1,"%d\n", &dummy);
    fclose (fp1);
    if (dummy==1){
        examine=4;
    }
    else{
        examine=0;
    }
    //////////////////////////////////
    
    
    fp1 = fopen (res_name, "w");
    
    if ( ! fp1)
    {
        sprintf(res_name,"res/dt_lsq");
        fp1 = fopen (res_name, "w");
    }
    if ( ! fp1)
    {
        printf ("WARNING! Cannot find dir: /res,  data written to dt_lsq in same dir \n");
        
        sprintf (res_name, "dt_lsq");
        fp1 = fopen (res_name, "w");
    }
    /* create dump file for rdb */
    if (examine == 4)
    {
        /* create filename for dumped dataset */
        sprintf (res_name, "dump_for_rdb");
        printf ("dataset dumped into %s\n", res_name);
        fp2 = fopen (res_name, "w");
        
        /* write # of points to file */
        fprintf (fp2, "%d\n", match);
    }
    /* first line to be updated in res_name file */
    fprintf (fp1, "%4d\n", match);
    /* least squares determination for triplets */
    
    rmsX = 0; rmsY = 0; rmsZ = 0;	mean_sigma0 = 0;
    
    for (i=0; i<match; i++)
    {
        for (j=0; j<4; j++)
            if (con[i].p[j] >= 0)	p[j] = geo[j][con[i].p[j]].pnr;
            else		       	p[j] = -1;
        
        for (j=0, n=0; j<4; j++)
        {
            if (p[j] > -1)
            {
                x[j] = crd[j][p[j]].x;	y[j] = crd[j][p[j]].y;
                n++;
            }
            else
            {
                x[j] = -1e10;	y[j] = -1e10;
                if (p[j] == -2)	n = -100;
            }
        }
        
        /* take only points which are matched in all images */
        /* or triplets/quadruplets which result from object model */
        /* e.g.: quad -> n=4; model triplet -> n=3; model pair -> n=2;
         unrestricted triplet -> n<0; unrestricted pair -> n<0 */
        /*     if (cpar->num_cams > 2  &&  n < 3)	continue; */
        
        /* ################################# */
        /* take only points which are matched in all images */
        /* or triplets/quadruplets which result from object model */
        /* e.g.: quad -> n=4; model triplet -> n=3; model pair -> n=2;
         unrestricted triplet -> n<0; unrestricted pair -> n<0 */
        if ((cpar->num_cams > 2 && num[0]>64 && num[1]>64 && num[2]>64 && num[3]>64)
            &&  n < 3)	continue;
        
        /* hack due to problems with approx in det_lsq: */
        X = 0.0; Y = 0.0;
        Z = (vpar->Zmin_lay[0] + vpar->Zmax_lay[0])/2.0;
        
        for (j = 0; j < cpar->num_cams; j++) { X += Ex[j].x0; Y += Ex[j].y0; }
        X /= cpar->num_cams; Y /= cpar->num_cams;
        /* ******************************** */
        
        det_lsq_3d (glob_cal, *(cpar->mm),
                    x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3], &X, &Y, &Z,
                    cpar->num_cams);
        
        
        /* write a sequential point number,
         sumg, if the point was used, and the 3D coordinates */
        fprintf (fp1, "%4d", i+1);
        
        fprintf (fp1, " %9.3f %9.3f %9.3f", X, Y, Z);
        if (p[0] > -1)	fprintf (fp1, " %4d", pix[0][p[0]].pnr);
        else			fprintf (fp1, " %4d", -1);
        if (p[1] > -1)	fprintf (fp1, " %4d", pix[1][p[1]].pnr);
        else			fprintf (fp1, " %4d", -1);
        if (p[2] > -1)	fprintf (fp1, " %4d", pix[2][p[2]].pnr);
        else			fprintf (fp1, " %4d", -1);
        if (p[3] > -1)	fprintf (fp1, " %4d\n", pix[3][p[3]].pnr);
        else			fprintf (fp1, " %4d\n", -1);
        
        /* write data as new points to dump for rdb */
        if (examine == 4)
        {
            fprintf (fp2, "%d %10.3f %10.3f %10.3f   %d    ", i, X, Y, Z, 3);
            for (j = 0; j < cpar->num_cams; j++)
                if (x[j] != -1e10)
                    fprintf (fp2, "%4d %8.5f %8.5f    ", i, x[j], y[j]);
                else
                    fprintf (fp2, "%4d %8.5f %8.5f    ", -999, x[j], y[j]);
            fprintf (fp2, "\n");
        }
        
        if (Z < Zlo)  Zlo = Z;   if (Z > Zhi)  Zhi = Z;
    }
    
    if (examine == 4) fclose (fp2);
    fclose (fp1);
    
    //Beat Mai 2010: now we should open the file db_is.* again, check
    //               if it has exactly two points, rescale them, write them again and close the file.
    if (dumbbell==1) {display=0;} //Denis
    
    if (dumbbell==1){
        fpp = fopen ("parameters/dumbbell.par", "r");
        if (fpp){
            fscanf (fpp, "%lf", &(vpar->eps0));
            fscanf (fpp, "%lf", &db_scale);
            fclose (fpp);
        }
        
        fpp = fopen (res_name, "r");
        fscanf (fpp, "%d\n", &match);
        if(match==2){
            fscanf(fpp, "%d %lf %lf %lf %d %d %d %d\n",
                   &i1, &x1, &y1, &z1,
                   &a1[0], &a1[1], &a1[2], &a1[3]);
            fscanf(fpp, "%d %lf %lf %lf %d %d %d %d\n",
                   &i2, &x2, &y2, &z2,
                   &a2[0], &a2[1], &a2[2], &a2[3]);
            //now adapt x,y,z
            /*dist=pow(pow(x2-x1,2.)+pow(y2-y1,2.)+pow(z2-z1,2.),0.5);
             mx=0.5*(x1+x2);
             my=0.5*(y1+y2);
             mz=0.5*(z1+z2);
             nx=(x2-x1)/dist;
             ny=(y2-y1)/dist;
             nz=(z2-z1)/dist;
             x1=mx-0.5*db_scale*nx;
             x2=mx+0.5*db_scale*nx;
             y1=my-0.5*db_scale*ny;
             y2=my+0.5*db_scale*ny;
             z1=mz-0.5*db_scale*nz;
             z2=mz+0.5*db_scale*nz;*/
            
            //check if reasonable
            /*dist=pow(pow(x2-x1,2.)+pow(y2-y1,2.)+pow(z2-z1,2.),0.5);
             if (fabs(dist-38)>1){
			 match=0;
             }*/
            
            //check if all quadruplets or triplets
            checksum_1=0;
            checksum_2=0;
            for(j=0;j<4;j++){
                if(a1[1]<0){
                    checksum_1++;
                }
                if(a2[1]<0){
                    checksum_2++;
                }
            }
            if(checksum_1>1 || checksum_2>1){
                match=0;
            }
            //end of check if all quadruplets or triplets
        }
        else{
            match=0;
        }
        fclose (fpp);
        fpp = fopen (res_name, "w");
        if(match==2){
            fprintf (fpp, "%4d\n", match);
            fprintf (fpp, " %4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n", i1,x1,y1,z1,a1[0],a1[1],a1[2],a1[3]);
            fprintf (fpp, " %4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n", i2,x2,y2,z2,a2[0],a2[1],a2[2],a2[3]);
        }
        else{
            fprintf (fpp, "%4d\n", 0);
        }
        fclose (fpp);
        
    }
    //end of dumbbell treatment
    
    rmsX = sqrt(rmsX/match); rmsY = sqrt(rmsY/match); rmsZ = sqrt(rmsZ/match);
    mean_sigma0 = sqrt (mean_sigma0/match);
    
    sprintf (buf, "Match: %d, => rms = %4.2f micron, rms_x,y,z = %5.3f/%5.3f/%5.3f mm", match, mean_sigma0*1000, rmsX, rmsY, rmsZ);
    puts (buf);
    
    /* sort coordinates for binary search in epi line segment drawing */
    for (i = 0; i < cpar->num_cams; i++) quicksort_coord2d_x (geo[0], num[0]);
    
    printf (" Determinate done\n");
    
    return 0;
    
}


