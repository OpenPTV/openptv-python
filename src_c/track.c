/*******************************************************************

Routine:	      	track.c

Author/Copyright:     	Jochen Willneff

Address:	       	Institute of Geodesy and Photogrammetry
		       	ETH - Hoenggerberg
	               	CH - 8093 Zurich

Creation Date:		Beginning: February '98
                        End: far away

Description:   	        Tracking of particles in image- and objectspace

Routines contained:    	trackcorr_c

*******************************************************************/

/* References:
[1] http://en.wikipedia.org/wiki/Gradian
*/

#include "ptv.h"
#include <optv/tracking_frame_buf.h>
#include "tracking_run.h"
#include "vec_utils.h"
#include "parameters.h"
#include "tools.h"
#include "ttools.h"

/* Global variables marked extern in 'globals.h' and not defined elsewhere: */
int intx0_tr[4][10000], inty0_tr[4][10000], intx1_tr[4][10000],\
    inty1_tr[4][10000], intx2_tr[4][10000], inty2_tr[4][10000], \
    pnr1_tr[4][10000], pnr2_tr[4][10000], m1_tr;
int seq_step_shake;
double pnr3_tr[4][10000];
double npart, nlinks;

/* The buffer space required for this algorithm: 

Note that MAX_TARGETS is taken from the global M, but I want a separate
definition because the fb created here should be local, not used outside
this file. 

MAX_CANDS is the max number of candidates sought in search volume for next 
link.
*/
#define BUFSPACE 4
#define MAX_TARGETS 20000
#define MAX_CANDS 4

tracking_run* trackcorr_c_init() {
    int step;
    tracking_run *ret;
    
    /* Remaining globals:
    see below for communication globals.
    */
    
    ret = (tracking_run *) malloc(sizeof(tracking_run));
    tr_init(ret, "parameters/sequence.par", "parameters/track.par",
        "parameters/criteria.par", "parameters/ptv.par");
    
    fb_init(ret->fb, 4, ret->cpar->num_cams, MAX_TARGETS, 
        "res/rt_is", "res/ptv_is", "res/added", ret->seq_par->img_base_name);

    /* Prime the buffer with first frames */
    for (step = ret->seq_par->first; step < ret->seq_par->first + 3; step++) {
        fb_read_frame_at_end(ret->fb, step, 0);
        fb_next(ret->fb);
    }
    fb_prev(ret->fb);
    
    ret->lmax = norm((ret->tpar->dvxmin - ret->tpar->dvxmax), \
        (ret->tpar->dvymin - ret->tpar->dvymax), \
        (ret->tpar->dvzmin - ret->tpar->dvzmax));
    volumedimension (&(ret->vpar->X_lay[1]), &(ret->vpar->X_lay[0]), &(ret->ymax), 
        &(ret->ymin), &(ret->vpar->Zmax_lay[1]), &(ret->vpar->Zmin_lay[0]),
        ret->fb->num_cams);

    // Denis - globals below are used in trackcorr_finish
    npart=0;
    nlinks=0;
    
    return ret;
}

/* reset_foundpix_array() sets default values for foundpix objects in an array.
 *
 * Arguments:
 * foundpix *arr - the array to reset
 * int arr_len - array length
 * int num_cams - number of places in the whichcam member of foundpix.
 */
void reset_foundpix_array(foundpix *arr, int arr_len, int num_cams) {
    int i, cam;
    for (i = 0; i < arr_len; i++) {
	    arr[i].ftnr = -1;
        arr[i].freq = 0;
        
        for(cam = 0; cam < num_cams; cam++) {
            arr[i].whichcam[cam] = 0;
	    }
    }
}

/* copy_foundpix_array() copies foundpix objects from one array to another.
 *
 * Arguments:
 * foundpix *dest, *src - src is the array to copy, dest receives it.
 * int arr_len - array length
 * int num_cams - number of places in the whichcam member of foundpix.
 */
void copy_foundpix_array(foundpix *dest, foundpix *src, int arr_len, 
    int num_cams) 
{
    int i, cam;
    for (i = 0; i < arr_len; i++) {
        dest[i].ftnr = src[i].ftnr;
        dest[i].freq = src[i].freq;
        for (cam = 0; cam < num_cams; cam++) {
            dest[i].whichcam[cam] = src[i].whichcam[cam];
        }
    }
}

/* register_closest_neighbs() finds candidates for continuing a particle's
 * path in the search volume, and registers their data in a foundpix array
 * that is later used by the tracking algorithm.
 * TODO: the search area can be in a better data structure.
 *
 * Arguments:
 * target *targets - the targets list to search.
 * int num_targets - target array length.
 * int cam - the index of the camera we're working on.
 * double cent_x, cent_y - image coordinates of search area, [pixel]
 * double dl, dr, du, dd - respectively the left, right, up, down distance to
 *   the search area borders from its center, [pixel]
 * foundpix *reg - an array of foundpix objects, one for each possible 
 *   neighbour. Output array.
 */
void register_closest_neighbs(target *targets, int num_targets, int cam,
    double cent_x, double cent_y, double dl, double dr, double du, double dd,
    foundpix *reg)
{
    int cand, all_cands[MAX_CANDS];
    
    candsearch_in_pix (targets, num_targets, cent_x, cent_y, dl, dr, 
        du, dd, all_cands);
        
    for (cand = 0; cand < MAX_CANDS; cand++) {
        if(all_cands[cand] == -999) {
            reg[cand].ftnr = -1;
        } else {
            reg[cand].whichcam[cam] = 1;
            reg[cand].ftnr = targets[all_cands[cand]].tnr;
        }
    }
}

/* search_volume_center_moving() finds the position of the center of the search
 * volume for a moving particle using the velocity of last step.
 * 
 * Arguments:
 * pos3d prev_pos - previous position
 * pos3d curr_pos - current position
 * pos3d *output - output variable, for the  calculated 
 *   position.
 */
void search_volume_center_moving(pos3d prev_pos, pos3d curr_pos, pos3d output)
{
    int dim;
    
    for (dim = 0; dim < 3; dim++) {
        output[dim] = 2*curr_pos[dim] - prev_pos[dim];
    }
}

/* pos3d_in_bounds() checks that all components of a pos3d are in their
   respective bounds taken from a track_par object.
   
   Arguments:
   pos3d pos - the 3-component array to check.
   track_par *bounds - the struct containing the bounds specification.
   
   Returns:
   True if all components in bounds, false otherwise.
 */
int pos3d_in_bounds(pos3d pos, track_par *bounds) {
    return (
        bounds->dvxmin < pos[0] && pos[0] < bounds->dvxmax &&
        bounds->dvymin < pos[1] && pos[1] < bounds->dvymax &&
        bounds->dvzmin < pos[2] && pos[2] < bounds->dvzmax ); 
}

/* angle_acc() calculates the angle between the (1st order) numerical velocity
   vectors to the predicted next position and to the candidate actual position.
   The angle is calculated in [gon], see [1].
   The predicted position is the position if the particle continued at current
   velocity.
   
   Arguments:
   pos3d start, pred, cand - the particle start position, predicted position,
      and possible actual position, respectively.
   double *angle - output variable, the angle between the two velocity
      vectors, [gon]
   double *acc - output variable, the 1st-order numerical acceleration embodied
      in the deviation from prediction.
 */
void angle_acc(pos3d start, pos3d pred, pos3d cand, double *angle, double *acc)
{
    pos3d v0, v1;
    
    subst_pos3d(pred, start, v0);
    subst_pos3d(cand, start, v1);
    
    *acc = diff_norm_pos3d(v0, v1);
    
    if ((v0[0] == -v1[0]) && (v0[1] == -v1[1]) && (v0[2] == -v1[2])) {
        *angle = 200;
    } else {
        *angle = (200./M_PI) * acos(dot_pos3d(v0, v1) / norm(v0[0], v0[1], v0[2]) \
            / norm(v1[0], v1[1], v1[2]));
    }
}

void trackcorr_c_loop (tracking_run *run_info, int step, int display)
{
   /* sequence loop */
    char  val[256], buf[256];
    int j, h, k, mm, kk, invol=0;
    int zaehler1, zaehler2, philf[4][MAX_CANDS];
    int count1=0, count2=0, count3=0, zusatz=0;
    int intx0, intx1, inty0, inty1;
    int intx2, inty2;
    int quali=0;
    pos3d diff_pos, X[7]; /* 7 reference points used in the algorithm, TODO: check if can reuse some */
    double x1[4], y1[4], x2[4], y2[4], angle, acc, angle0, acc0,  dl;
    double xr[4], xl[4], yd[4], yu[4], angle1, acc1;
    double xp[4], yp[4], xc[4], yc[4], xn[4], yn[4];
    double rr;
    int flag_m_tr=0;
    
    /* Shortcuts to inside current frame */
    P *curr_path_inf, *ref_path_inf;
    corres *curr_corres, *ref_corres;
    target **curr_targets, **ref_targets;
    int _ix; /* For use in any of the complex index expressions below */
    int _frame_parts; /* number of particles in a frame */

    /* Shortcuts into the tracking_run struct */
    framebuf *fb;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;
    
    /* Remaining globals:
    all those in trackcorr_c_init.
    calibration globals.
    */
    
    foundpix *w, *wn, p16[4*MAX_CANDS];
    sprintf (buf, "Time step: %d, seqnr: %d, Particle info:",
        step - run_info->seq_par->first, step);
    count1=0; zusatz=0;
    
    fb = run_info->fb;
    tpar = run_info->tpar;
    vpar = run_info->vpar;
    cpar = run_info->cpar;
    curr_targets = fb->buf[1]->targets;
    
    /* try to track correspondences from previous 0 - corp, variable h */
    for (h = 0; h < fb->buf[1]->num_parts; h++) {
        for (j = 0; j < 7; j++) init_pos3d(X[j]);
        
        curr_path_inf = &(fb->buf[1]->path_info[h]);
        curr_corres = &(fb->buf[1]->correspond[h]);
        
	    curr_path_inf->inlist = 0;
        reset_foundpix_array(p16, 16, fb->num_cams);
        
	    /* 3D-position */
	    copy_pos3d(X[1], curr_path_inf->x);

	    /* use information from previous to locate new search position
	       and to calculate values for search area */
	    if (curr_path_inf->prev >= 0) {
            ref_path_inf = &(fb->buf[0]->path_info[curr_path_inf->prev]);
	        copy_pos3d(X[0], ref_path_inf->x);
            search_volume_center_moving(ref_path_inf->x, curr_path_inf->x, X[2]);
            
	        for (j = 0; j < fb->num_cams; j++) {
                img_coord (X[2][0], X[2][1], X[2][2], Ex[j],I[j], G[j], ap[j],
                    mmp, &xn[j], &yn[j]);
		        metric_to_pixel (xn[j], yn[j], imx,imy, pix_x,pix_y, &x1[j], &y1[j], chfield);
	        }
	    } else {  
            copy_pos3d(X[2], X[1]);
	        for (j=0; j < fb->num_cams; j++) {
	            if (curr_corres->p[j] == -1) {
                    img_coord (X[2][0], X[2][1], X[2][2], Ex[j],I[j], G[j],
                        ap[j], mmp, &xn[j], &yn[j]);
                    metric_to_pixel (xn[j], yn[j], imx, imy, pix_x, pix_y,
                        &x1[j], &y1[j], chfield);
	            } else {
                    _ix = curr_corres->p[j];
                    x1[j] = curr_targets[j][_ix].x;
                    y1[j] = curr_targets[j][_ix].y;
                }
            }
	    } 
        
	    /* calculate searchquader and reprojection in image space */
	    searchquader(X[2][0], X[2][1], X[2][2], xr, xl, yd, yu, tpar, cpar);

	    /* search in pix for candidates in next time step */
	    for (j = 0; j < fb->num_cams; j++) {
            register_closest_neighbs(fb->buf[2]->targets[j],
                fb->buf[2]->num_targets[j], j, x1[j], y1[j],
                xl[j], xr[j], yu[j], yd[j], &p16[j*MAX_CANDS]);
	    }
        
	    /* fill and sort candidate struct */
	    sortwhatfound(p16, &zaehler1, fb->num_cams);
	    w = (foundpix *) calloc (zaehler1, sizeof (foundpix));
        
	    if (zaehler1 > 0) count2++;
        copy_foundpix_array(w, p16, zaehler1, fb->num_cams);
	    /*end of candidate struct */

	    /* check for what was found */
	    for (mm=0; mm<zaehler1;mm++) { /* zaehler1-loop */
	        /* search for found corr of current the corr in next
		    with predicted location */

            reset_foundpix_array(p16, 16, fb->num_cams);

	        /* found 3D-position */
            ref_path_inf = &(fb->buf[2]->path_info[w[mm].ftnr]);
            copy_pos3d(X[3], ref_path_inf->x);

	        if (curr_path_inf->prev >= 0) {
                for (j = 0; j < 3; j++) 
                    X[5][j] = 0.5*(5.0*X[3][j] - 4.0*X[1][j] + X[0][j]);
	        } else {
                search_volume_center_moving(X[1], X[3], X[5]);
            }
            searchquader(X[5][0], X[5][1], X[5][2], xr, xl, yd, yu, tpar, cpar);

	        for (j = 0; j < fb->num_cams; j++) {
                img_coord (X[5][0], X[5][1], X[5][2], Ex[j],I[j], G[j], ap[j],
                    mmp, &xn[j], &yn[j]);
		        metric_to_pixel (xn[j], yn[j], imx,imy, pix_x,pix_y, &x2[j], &y2[j], chfield);
	        }

	        /* search for candidates in next time step */
	        for (j=0; j < fb->num_cams; j++) {
	            zaehler2 = candsearch_in_pix (fb->buf[3]->targets[j], 
                    fb->buf[3]->num_targets[j], x1[j], y1[j],
					xl[j], xr[j], yu[j], yd[j], philf[j]);

		        for(k = 0; k < 4; k++) {
				    if (philf[j][k] == -999) {
                        p16[j*4+k].ftnr=-1;
				    } else {
				        if (fb->buf[3]->targets[j][philf[j][k]].tnr != -1) {
                            _ix = philf[j][k];
                            p16[j*4+k].ftnr = fb->buf[3]->targets[j][_ix].tnr;
                            p16[j*4+k].whichcam[j] = 1;
					    }
				    }
		        }
		    }
	        /* end of search in pix */

	        /* fill and sort candidate struct */
	        sortwhatfound(p16, &zaehler2, fb->num_cams);
	        wn = (foundpix *) calloc (zaehler2, sizeof (foundpix));
	        if (zaehler2 > 0) count3++;
            copy_foundpix_array(wn, p16, zaehler2, fb->num_cams);

	        /*end of candidate struct */
	        /* ************************************************ */
	        for (kk=0; kk < zaehler2; kk++)  { /* zaehler2-loop */
                ref_path_inf = &(fb->buf[3]->path_info[wn[kk].ftnr]);
                copy_pos3d(X[4], ref_path_inf->x);

                subst_pos3d(X[4], X[3], diff_pos);
                if ( pos3d_in_bounds(diff_pos, tpar)) { 
                    angle_acc(X[3], X[4], X[5], &angle1, &acc1);

                    if (curr_path_inf->prev >= 0) {
                        angle_acc(X[1], X[2], X[3], &angle0, &acc0);
		            } else {
                        acc0=acc1; angle0=angle1;
                    }

                    acc=(acc0+acc1)/2; angle=(angle0+angle1)/2;
                    quali=wn[kk].freq+w[mm].freq;

                    if ((acc < tpar->dacc && angle < tpar->dangle) || \
                        (acc < tpar->dacc/10)) 
                    {
                        dl = (diff_norm_pos3d(X[1], X[3]) + 
                            diff_norm_pos3d(X[4], X[3]) )/2;
                        rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle)/(quali);
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr);
                    }
		        }
	        }   /* end of zaehler2-loop */

	        /* creating new particle position */
	        /* *************************************************************** */
	        for (j = 0;j < fb->num_cams; j++) {
                img_coord (X[5][0], X[5][1], X[5][2], Ex[j],I[j], G[j], ap[j],
                    mmp, &xn[j], &yn[j]);
		        metric_to_pixel (xn[j], yn[j], imx,imy, pix_x,pix_y, &xn[j], &yn[j], chfield);
	        }

	        /* reset img coord because of num_cams < 4 */
	        for (j=0;j < fb->num_cams; j++) { x2[j]=-1e10; y2[j]=-1e10; }

	        /* search for unused candidates in next time step */
	        for (j = 0;j < fb->num_cams; j++) {
		        /* use fix distance to define xl, xr, yu, yd instead of searchquader */
		        xl[j]= xr[j]= yu[j]= yd[j] = 3.0;

	            zaehler2 = candsearch_in_pixrest (fb->buf[3]->targets[j], 
                    fb->buf[3]->num_targets[j], xn[j], yn[j],
					xl[j], xr[j], yu[j], yd[j], philf[j]);
		        if(zaehler2>0 ) {
                    _ix = philf[j][0];
		            x2[j] = fb->buf[3]->targets[j][_ix].x;
                    y2[j] = fb->buf[3]->targets[j][_ix].y;
		        }
		    }
	        quali=0;

	        for (j = 0; j < fb->num_cams; j++) {
		        if (x2[j] != -1e10 && y2[j] != -1e10) {
		        pixel_to_metric (x2[j],y2[j], imx,imy, pix_x,pix_y, &x2[j],&y2[j], chfield); quali++;
		        }
		    }

	        if ( quali >= 2) {
                copy_pos3d(X[4], X[5]);
		        invol=0; 

		        det_lsq_3d (Ex, I, G, ap, mmp, x2[0], y2[0], x2[1], y2[1], 
                    x2[2], y2[2], x2[3], y2[3],
                    &(X[4][0]), &(X[4][1]), &(X[4][2]), fb->num_cams);

		        /* volume check */
                if ( vpar->X_lay[0] < X[4][0] && X[4][0] < vpar->X_lay[1] &&
		            run_info->ymin < X[4][1] && X[4][1] < run_info->ymax &&
		            vpar->Zmin_lay[0] < X[4][2] && X[4][2] < vpar->Zmax_lay[1]) {invol=1;}

                subst_pos3d(X[3], X[4], diff_pos);
                if ( invol == 1 && pos3d_in_bounds(diff_pos, tpar) ) { 
                    angle_acc(X[3], X[4], X[5], &angle, &acc);

                    if ((acc < tpar->dacc && angle < tpar->dangle) || \
                        (acc < tpar->dacc/10)) 
                    {
                        dl=(diff_norm_pos3d(X[1], X[3]) + 
                            diff_norm_pos3d(X[4], X[3]) )/2;
                        rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle) /
                            (quali+w[mm].freq);
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr);

                        if (tpar->add) {
                            ref_path_inf = &(fb->buf[3]->path_info[
                                fb->buf[3]->num_parts]);
                            copy_pos3d(ref_path_inf->x, X[4]);
                            reset_links(ref_path_inf);

                            _frame_parts = fb->buf[3]->num_parts;
                            ref_corres = &(fb->buf[3]->correspond[_frame_parts]);
                            ref_targets = fb->buf[3]->targets;
			                for (j = 0; j < fb->num_cams; j++) {
				                ref_corres->p[j]=-1;
                                
				                if(philf[j][0]!=-999) {
                                    _ix = philf[j][0];
                                    ref_targets[j][_ix].tnr = _frame_parts;
                                    ref_corres->p[j] = _ix;
                                    ref_corres->nr = _frame_parts;
				                }
			                }
			                fb->buf[3]->num_parts++;
                            zusatz++; 
                        }
                    }
                }
		        invol=0;
	        }
	        quali=0;

	        /* end of creating new particle position */
	        /* *************************************************************** */
            
	        /* try to link if kk is not found/good enough and prev exist */
	        if ( curr_path_inf->inlist == 0 && curr_path_inf->prev >= 0 ) {
                subst_pos3d(X[3], X[1], diff_pos);
                
                if (pos3d_in_bounds(diff_pos, tpar)) {
                    angle_acc(X[1], X[2], X[3], &angle, &acc);

                    if ( (acc < tpar->dacc && angle < tpar->dangle) || \
                        (acc < tpar->dacc/10) )
                    {
                        quali = w[mm].freq;
                        dl = (diff_norm_pos3d(X[1], X[3]) + 
                            diff_norm_pos3d(X[0], X[1]) )/2;
                        rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle)/(quali);
                        register_link_candidate(curr_path_inf, rr, w[mm].ftnr);
			        }
		        }
	        }

	        free(wn);
        } /* end of zaehler1-loop */
        
	    /* begin of inlist still zero */
	    if (tpar->add) {
	        if ( curr_path_inf->inlist == 0 && curr_path_inf->prev >= 0 ) {
                for (j = 0; j < fb->num_cams; j++) {
                    img_coord (X[2][0], X[2][1], X[2][2], Ex[j],I[j], G[j],
                        ap[j], mmp, &xn[j], &yn[j]);
		            metric_to_pixel (xn[j], yn[j], imx,imy, pix_x,pix_y, &xn[j], &yn[j], chfield);
		            x2[j]=-1e10;
                    y2[j]=-1e10;
                } 

    		    /* search for unused candidates in next time step */
		        for (j = 0; j < fb->num_cams; j++) {
		            /*use fix distance to define xl, xr, yu, yd instead of searchquader */
		            xl[j]= xr[j]= yu[j]= yd[j] = 3.0;
	                zaehler2 = candsearch_in_pixrest (fb->buf[2]->targets[j], 
                        fb->buf[2]->num_targets[j], xn[j], yn[j],
					    xl[j], xr[j], yu[j], yd[j], philf[j]);
		            if(zaehler2 > 0) {
                        _ix = philf[j][0];
	    	            x2[j] = fb->buf[2]->targets[j][_ix].x;
                        y2[j] = fb->buf[2]->targets[j][_ix].y;
		            }
		        }
		        quali=0;

		        for (j = 0; j < fb->num_cams; j++) {
		            if (x2[j] !=-1e10 && y2[j] != -1e10) {
		                pixel_to_metric (x2[j],y2[j], imx,imy, pix_x,pix_y, &x2[j],&y2[j], chfield); quali++;
		            }
		        }

		        if (quali>=2) {
                    copy_pos3d(X[3], X[2]);
		            invol=0; 
    
	    	        det_lsq_3d (Ex, I, G, ap, mmp,
                        x2[0], y2[0], x2[1], y2[1], x2[2], y2[2], x2[3], y2[3],
                        &(X[3][0]), &(X[3][1]), &(X[3][2]), fb->num_cams);

		            /* in volume check */
		            if ( vpar->X_lay[0] < X[3][0] && X[3][0] < vpar->X_lay[1] &&
                        run_info->ymin < X[3][1] && X[3][1] < run_info->ymax &&
                        vpar->Zmin_lay[0] < X[3][2] && 
                        X[3][2] < vpar->Zmax_lay[1]) {invol = 1;}

                    subst_pos3d(X[2], X[3], diff_pos);
                    if ( invol == 1 && pos3d_in_bounds(diff_pos, tpar) ) { 
                        angle_acc(X[1], X[2], X[3], &angle, &acc);

                        if ( (acc < tpar->dacc && angle < tpar->dangle) || \
                            (acc < tpar->dacc/10) ) 
                        {
                            dl = (diff_norm_pos3d(X[1], X[3]) + 
                                diff_norm_pos3d(X[0], X[1]) )/2;
                            rr = (dl/run_info->lmax + acc/tpar->dacc + angle/tpar->dangle)/(quali);
                            
                            ref_path_inf = &(fb->buf[2]->path_info[
                                fb->buf[2]->num_parts]);
                            copy_pos3d(ref_path_inf->x, X[3]);
                            reset_links(ref_path_inf);

                            _frame_parts = fb->buf[2]->num_parts;
                            register_link_candidate(curr_path_inf, rr, _frame_parts);
                            
                            ref_corres = &(fb->buf[2]->correspond[_frame_parts]);
                            ref_targets = fb->buf[2]->targets;
			                for (j = 0;j < fb->num_cams; j++) {
                                ref_corres->p[j]=-1;
                                
                                if(philf[j][0]!=-999) {
                                    _ix = philf[j][0];
                                    ref_targets[j][_ix].tnr = _frame_parts;
                                    ref_corres->p[j] = _ix;
                                    ref_corres->nr = _frame_parts;
                                }
                            }
                            fb->buf[2]->num_parts++;
                            zusatz++;
                        }
                    }
		            invol=0;
		        } // if quali >= 2
            }
        }
	    /* end of inlist still zero */
	    /***********************************/

	    free(w);
	} /* end of h-loop */
    
    /* sort decis and give preliminary "finaldecis"  */
    for (h = 0; h < fb->buf[1]->num_parts; h++) {
        curr_path_inf = &(fb->buf[1]->path_info[h]);
        
	    if(curr_path_inf->inlist > 0 ) {
	        sort(curr_path_inf->inlist, (float *) curr_path_inf->decis,
                curr_path_inf->linkdecis);
      	    curr_path_inf->finaldecis = curr_path_inf->decis[0];
	        curr_path_inf->next = curr_path_inf->linkdecis[0];
	    }
	}

    /* create links with decision check */
    for (h = 0;h < fb->buf[1]->num_parts; h++) {
        curr_path_inf = &(fb->buf[1]->path_info[h]);

	    if(curr_path_inf->inlist > 0 ) {
            ref_path_inf = &(fb->buf[2]->path_info[curr_path_inf->next]);
            
	        if (ref_path_inf->prev == -1) {	
	            /* best choice wasn't used yet, so link is created */
                ref_path_inf->prev = h; 
            } else {
	            /* best choice was already used by mega[2][mega[1][h].next].prev */
	            /* check which is the better choice */
	            if ( fb->buf[1]->path_info[ref_path_inf->prev].finaldecis > \
                    curr_path_inf->finaldecis) 
                {
		            /* remove link with prev */
		            fb->buf[1]->path_info[ref_path_inf->prev].next = NEXT_NONE;
                    ref_path_inf->prev = h; 
		        } else {
		            curr_path_inf->next = NEXT_NONE;
	            }
	        }
        }
        if (curr_path_inf->next != -2 ) count1++;
    } 
    /* end of creation of links with decision check */
    /* ******** Draw links now ******** */
    m1_tr = 0;
    
    if (display) {
        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);
            curr_corres = &(fb->buf[1]->correspond[h]);
            ref_corres = &(fb->buf[2]->correspond[curr_path_inf->next]);
            
            if (curr_path_inf->next != -2 ) {
                strcpy(buf,"");
                sprintf(buf ,"green");
	
                for (j = 0; j < fb->num_cams; j++) {
                    if (curr_corres->p[j] > 0 && ref_corres->p[j] > 0) {
                        flag_m_tr=1;  
                        xp[j] = curr_targets[j][curr_corres->p[j]].x;
               		    yp[j] = curr_targets[j][curr_corres->p[j]].y;
               		    xc[j] = fb->buf[2]->targets[j][ref_corres->p[j]].x;
               		    yc[j] = fb->buf[2]->targets[j][ref_corres->p[j]].y;
               		    predict (xp[j], yp[j], xc[j], yc[j], &xn[j], &yn[j]);
                        
               		    if ( ( fabs(xp[j]-zoom_x[j]) < imx/(2*zoom_f[j]))
                            && (fabs(yp[j]-zoom_y[j]) < imy/(2*zoom_f[j])))
                        {
	                        strcpy(val,"");
                           	sprintf(val ,"orange");

                        	intx0 = (int)(imx/2+zoom_f[j]*(xp[j]-zoom_x[j]));
	                        inty0 = (int)(imy/2+zoom_f[j]*(yp[j]-zoom_y[j]));
                            intx1 = (int)(imx/2+zoom_f[j]*(xc[j]-zoom_x[j]));
	                        inty1 = (int)(imy/2+zoom_f[j]*(yc[j]-zoom_y[j]));
	                        intx2 = (int)(imx/2+zoom_f[j]*(xn[j]-zoom_x[j]));
	                        inty2 = (int)(imy/2+zoom_f[j]*(yn[j]-zoom_y[j]));

	                        intx0_tr[j][m1_tr]=intx0;
	                        inty0_tr[j][m1_tr]=inty0;
	                        intx1_tr[j][m1_tr]=intx1;
	                        inty1_tr[j][m1_tr]=inty1;
	                        intx2_tr[j][m1_tr]=intx2;
	                        inty2_tr[j][m1_tr]=inty2;
	                        pnr1_tr[j][m1_tr]=-1;
	                        pnr2_tr[j][m1_tr]=-1;
	                        pnr3_tr[j][m1_tr]=-1;
		
	                        if (curr_path_inf->finaldecis > 0.2) {
	            	            pnr1_tr[j][m1_tr] = h;
		                        pnr2_tr[j][m1_tr] = curr_path_inf->next;
		                        pnr3_tr[j][m1_tr] = curr_path_inf->finaldecis;
		                    }
                        }
                    }

                    if (flag_m_tr==0)  {
                        intx0_tr[j][m1_tr]=0;
    	                inty0_tr[j][m1_tr]=0;
	                    intx1_tr[j][m1_tr]=0;
	                    inty1_tr[j][m1_tr]=0;
	                    intx2_tr[j][m1_tr]=0;
	                    inty2_tr[j][m1_tr]=0;
	                    pnr1_tr[j][m1_tr]=-1;
	                    pnr2_tr[j][m1_tr]=-1;
	                    pnr3_tr[j][m1_tr]=-1; 
                    }
                    flag_m_tr=0;
                }
                m1_tr++;
            }
        }
    }
    /* ******** End of Draw links now ******** */
    sprintf (buf, "step: %d, curr: %d, next: %d, links: %d, lost: %d, add: %d",
        step, fb->buf[1]->num_parts, fb->buf[2]->num_parts, count1, 
        fb->buf[1]->num_parts - count1, zusatz);

    /* for the average of particles and links */
    npart = npart + fb->buf[1]->num_parts;
    nlinks = nlinks + count1;

    fb_next(fb);
    fb_write_frame_from_start(fb, step);
    if(step < run_info->seq_par->last - 2) {
        fb_read_frame_at_end(fb, step + 3, 0); 
    }
} /* end of sequence loop */

void trackcorr_c_finish(tracking_run *run_info, int step)
{
  int range = run_info->seq_par->last - run_info->seq_par->first;
  
  /* average of all steps */
  npart /= range;
  nlinks /= range;
  printf ("Average over sequence, particles: %5.1f, links: %5.1f, lost: %5.1f\n",
	  npart, nlinks, npart-nlinks);

  fb_next(run_info->fb);
  fb_write_frame_from_start(run_info->fb, step);
  
  fb_free(run_info->fb);
    
  /* reset of display flag */
  display = 1;
}

/*     track backwards */
void trackback_c ()
{
    char  buf[256];
    int i, j, h, k, step, invol=0;
    int zaehler1, philf[4][MAX_CANDS];
    int count1=0, count2=0, zusatz=0;
    int quali=0;
    double x2[4], y2[4], angle, acc, lmax, dl;
    double xr[4], xl[4], yd[4], yu[4];
    pos3d diff_pos, X[7]; /* 7 reference points used in the algorithm, TODO: check if can reuse some */
    double xn[4], yn[4];
    double rr, Ymin=0, Ymax=0;
    double npart=0, nlinks=0;
    foundpix *w, p16[4*MAX_CANDS];

    sequence_par *seq_par;
    track_par *tpar;
    volume_par *vpar;
    control_par *cpar;
    framebuf *fb;
    
    /* Shortcuts to inside current frame */
    P *curr_path_inf, *ref_path_inf;
    corres *ref_corres;
    target **ref_targets;
    int _ix; /* For use in any of the complex index expressions below */
    int _frame_parts; /* number of particles in a frame */
    
    display = 1; 
    /* read data */
    seq_par = read_sequence_par("parameters/sequence.par");
    tpar = read_track_par("parameters/track.par");
    vpar = read_volume_par("parameters/criteria.par");
    cpar = read_control_par("parameters/ptv.par");

    fb = (framebuf *) malloc(sizeof(framebuf));
    fb_init(fb, 4, cpar->num_cams, MAX_TARGETS, 
        "res/rt_is", "res/ptv_is", "res/added", seq_par->img_base_name);

    /* Prime the buffer with first frames */
    for (step = seq_par->last; step > seq_par->last - 4; step--) {
        fb_read_frame_at_end(fb, step, 1);
        fb_next(fb);
    }
    fb_prev(fb);
    
    lmax = norm((tpar->dvxmin - tpar->dvxmax), (tpar->dvymin - tpar->dvymax),
	    (tpar->dvzmin - tpar->dvzmax));
    volumedimension (&(vpar->X_lay[1]), &(vpar->X_lay[0]), &Ymax,
        &Ymin, &(vpar->Zmax_lay[1]), &(vpar->Zmin_lay[0]), fb->num_cams);

    /* sequence loop */
    for (step = seq_par->last - 1; step > seq_par->first; step--) {
        sprintf (buf, "Time step: %d, seqnr: %d, Particle info:",
            step - seq_par->first, step);
        
        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);
            
            /* We try to find link only if the forward search failed to. */ 
            if ((curr_path_inf->next < 0) || (curr_path_inf->prev != -1)) continue;

            for (j = 0; j < 7; j++) init_pos3d(X[j]);
            curr_path_inf->inlist = 0;
            reset_foundpix_array(p16, 16, fb->num_cams);
            
            /* 3D-position of current particle */
            copy_pos3d(X[1], curr_path_inf->x);
            
            /* use information from previous to locate new search position
            and to calculate values for search area */
            ref_path_inf = &(fb->buf[0]->path_info[curr_path_inf->next]);
	        copy_pos3d(X[0], ref_path_inf->x);
            search_volume_center_moving(ref_path_inf->x, curr_path_inf->x, X[2]);

            for (j=0; j < fb->num_cams; j++) {   
                img_coord (X[2][0], X[2][1], X[2][2], Ex[j],I[j], G[j], ap[j],
                    mmp, &xn[j], &yn[j]);
                metric_to_pixel (xn[j], yn[j], imx,imy, pix_x,pix_y, &xn[j],
                    &yn[j], chfield);
            }

            /* calculate searchquader and reprojection in image space */
            searchquader(X[2][0], X[2][1], X[2][2], xr, xl, yd, yu, tpar, cpar);

            for (j = 0; j < fb->num_cams; j++) {
                zaehler1 = candsearch_in_pix (
                    fb->buf[2]->targets[j], fb->buf[2]->num_targets[j], xn[j], yn[j],
                    xl[j], xr[j], yu[j], yd[j], philf[j]);

                for(k=0; k<4; k++) {
                    if( zaehler1>0) {
                        if (philf[j][k] == -999){
                            p16[j*4+k].ftnr=-1;
                        } else {
                            p16[j*4+k].ftnr=fb->buf[3]->targets[j][philf[j][k]].tnr;
                            p16[j*4+k].whichcam[j]=1;
                        }
                    }
                }
            }
            /* search in pix for candidates in next time step */
            //for (j = 0; j < fb->num_cams; j++) { 
            //    register_closest_neighbs(fb->buf[2]->targets[j],
            //        fb->buf[2]->num_targets[j], j, xn[j], yn[j],
            //        xl[j], xr[j], yu[j], yd[j], &p16[j*MAX_CANDS]);
            //}

            /* fill and sort candidate struct */
            sortwhatfound(p16, &zaehler1, fb->num_cams);
            w = (foundpix *) calloc (zaehler1, sizeof (foundpix));

            /*end of candidate struct */
            if (zaehler1 > 0) count2++;
            copy_foundpix_array(w, p16, zaehler1, fb->num_cams);

            for (i = 0; i < zaehler1; i++) {
                ref_path_inf = &(fb->buf[2]->path_info[w[i].ftnr]);
                copy_pos3d(X[3], ref_path_inf->x);

                subst_pos3d(X[1], X[3], diff_pos);
                if (pos3d_in_bounds(diff_pos, tpar)) {
                    angle_acc(X[1], X[2], X[3], &angle, &acc);

                    /* *********************check link *****************************/
                    if ((acc < tpar->dacc && angle < tpar->dangle) || \
                        (acc < tpar->dacc/10))
                    {
                        dl = (diff_norm_pos3d(X[1], X[3]) + 
                            diff_norm_pos3d(X[0], X[1]) )/2;
                        quali=w[i].freq;
                        rr = (dl/lmax + acc/tpar->dacc + angle/tpar->dangle)/quali;
                        register_link_candidate(curr_path_inf, rr, w[i].ftnr);
                    }
                }
            }

            free(w);
            /******************/
            quali=0;

            /* reset img coord because num_cams < 4 */
            for (j=0;j<4;j++) { x2[j]=-1e10; y2[j]=-1e10;}

            /* if old wasn't found try to create new particle position from rest */
            if (tpar->add) {
                if ( curr_path_inf->inlist == 0) {
                    for (j = 0; j < fb->num_cams; j++) {
                        /* use fix distance to define xl, xr, yu, yd instead of searchquader */
                        xl[j]= xr[j]= yu[j]= yd[j] = 3.0;

                        zaehler1 = candsearch_in_pixrest (fb->buf[2]->targets[j], 
                            fb->buf[2]->num_targets[j], xn[j], yn[j],
                            xl[j], xr[j], yu[j], yd[j], philf[j]);
                        if(zaehler1 > 0) {
                            _ix = philf[j][0];
                            x2[j] = fb->buf[2]->targets[j][_ix].x;
                            y2[j] = fb->buf[2]->targets[j][_ix].y;
                        }
                    }

                    for (j = 0; j < fb->num_cams; j++) {
                        if (x2[j] !=-1e10 && y2[j] != -1e10) {
                            pixel_to_metric (x2[j],y2[j], imx,imy, pix_x,pix_y, &x2[j],&y2[j], chfield); quali++;
                        }
                    }

                    if (quali>=2) {
                        copy_pos3d(X[3], X[2]);
                        invol=0;

                        det_lsq_3d (Ex, I, G, ap, mmp,
                            x2[0], y2[0], x2[1], y2[1], x2[2], y2[2], x2[3], y2[3],
                            &(X[3][0]), &(X[3][1]), &(X[3][2]), fb->num_cams);

                        /* volume check */
                        if ( vpar->X_lay[0] < X[3][0] && X[3][0] < vpar->X_lay[1] &&
                            Ymin < X[3][1] && X[3][1] < Ymax &&
                            vpar->Zmin_lay[0] < X[3][2] && X[3][2] < vpar->Zmax_lay[1]) 
                                {invol = 1;}

                        subst_pos3d(X[1], X[3], diff_pos);
                        if (invol == 1 && pos3d_in_bounds(diff_pos, tpar)) { 
                            angle_acc(X[1], X[2], X[3], &angle, &acc);

                            if ( (acc<tpar->dacc && angle<tpar->dangle) || \
                                (acc<tpar->dacc/10) ) 
                            {
                                dl = (diff_norm_pos3d(X[1], X[3]) + 
                                    diff_norm_pos3d(X[0], X[1]) )/2;
                                rr =(dl/lmax+acc/tpar->dacc + angle/tpar->dangle)/(quali);

                                ref_path_inf = &(fb->buf[2]->path_info[
                                    fb->buf[2]->num_parts]);
                                copy_pos3d(ref_path_inf->x, X[3]);
                                reset_links(ref_path_inf);

                                _frame_parts = fb->buf[2]->num_parts;
                                register_link_candidate(curr_path_inf, rr,
                                    _frame_parts);
                                
                                ref_corres = &(fb->buf[2]->correspond[_frame_parts]);
                                ref_targets = fb->buf[2]->targets;

                                for (j = 0;j < fb->num_cams; j++) {
                                    ref_corres->p[j]=-1;
                                    
                                    if(philf[j][0]!=-999) {
                                        _ix = philf[j][0];
                                        ref_targets[j][_ix].tnr = _frame_parts;
                                        ref_corres->p[j] = _ix;
                                        ref_corres->nr = _frame_parts;
                                    }
                                }
                                fb->buf[2]->num_parts++;
                            }
                        }
                        invol=0;
                    }
                }
            } /* end of if old wasn't found try to create new particle position from rest */
        } /* end of h-loop */

        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);
        
            if(curr_path_inf->inlist > 0 ) {
                sort(curr_path_inf->inlist, (float *)curr_path_inf->decis,
                    curr_path_inf->linkdecis);
            }
        }

        /* create links with decision check */
        count1=0; zusatz=0;
        for (h = 0; h < fb->buf[1]->num_parts; h++) {
            curr_path_inf = &(fb->buf[1]->path_info[h]);
            
            if (curr_path_inf->inlist > 0 ) {
                /* if old/new and unused prev == -1 and next == -2 link is created */
                ref_path_inf = &(fb->buf[2]->path_info[curr_path_inf->linkdecis[0]]);
                
                if ( ref_path_inf->prev == PREV_NONE && \
                    ref_path_inf->next == NEXT_NONE )
                {
                    curr_path_inf->finaldecis = curr_path_inf->decis[0];
                    curr_path_inf->prev = curr_path_inf->linkdecis[0];
                    fb->buf[2]->path_info[curr_path_inf->prev].next = h;
                    zusatz++;
                }

                /* old which link to prev has to be checked */
                if ((ref_path_inf->prev != PREV_NONE) && \
                    (ref_path_inf->next == NEXT_NONE) )
                {
                    copy_pos3d(X[0], fb->buf[0]->path_info[curr_path_inf->next].x);
                    copy_pos3d(X[1], curr_path_inf->x);
                    copy_pos3d(X[3], ref_path_inf->x);
                    copy_pos3d(X[4], fb->buf[3]->path_info[ref_path_inf->prev].x);
                    for (j = 0; j < 3; j++) 
                        X[5][j] = 0.5*(5.0*X[3][j] - 4.0*X[1][j] + X[0][j]);

                    angle_acc(X[3], X[4], X[5], &angle, &acc);
                    
                    if ( (acc<tpar->dacc && angle<tpar->dangle) ||  (acc<tpar->dacc/10) ) {
                        curr_path_inf->finaldecis = curr_path_inf->decis[0];
                        curr_path_inf->prev = curr_path_inf->linkdecis[0];
                        fb->buf[2]->path_info[curr_path_inf->prev].next = h;
                        zusatz++;
                    }
                }
            }

            if (curr_path_inf->prev != -1 ) count1++;
        } /* end of creation of links with decision check */

        sprintf (buf, "step: %d, curr: %d, next: %d, links: %d, lost: %d, add: %d",
            step, fb->buf[1]->num_parts, fb->buf[2]->num_parts, count1,
            fb->buf[1]->num_parts - count1, zusatz);

        /* for the average of particles and links */
        npart = npart + fb->buf[1]->num_parts;
        nlinks = nlinks + count1;

        fb_next(fb);
        fb_write_frame_from_start(fb, step);
        if(step > seq_par->first + 2) { fb_read_frame_at_end(fb, step - 3, 1); }
    } /* end of sequence loop */

    /* average of all steps */
    npart /= (seq_par->last - seq_par->first - 1);
    nlinks /= (seq_par->last - seq_par->first - 1);

    printf ("Average over sequence, particles: %5.1f, links: %5.1f, lost: %5.1f\n",
    npart, nlinks, npart-nlinks);

    fb_next(fb);
    fb_write_frame_from_start(fb, step);
    
    fb_free(fb);
    free(fb);
    free(tpar);

    /* reset of display flag */
    display = 1;
}

