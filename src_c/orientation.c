/*******************************************************************

Routine:      	orientation.c

Author/Copyright:      	Hans-Gerd Maas

Address:      	Institute of Geodesy and Photogrammetry
                ETH - Hoenggerberg
	       	CH - 8093 Zurich

Creation Date:	summer 1988

Description:   	computes 6 parameters of exterior orientation
	     	and different sets of additional parameters
	      	from given approximate values, fixpoint- and image
	    	coordinates in ASCII files

Functiones used:   	img_affin, adjlib.a

Routines contained:

Related routines:

******************************************************************/
#include <optv/calibration.h>
#include <optv/tracking_frame_buf.h>
#include <optv/parameters.h>
#include <optv/trafo.h>
#include <optv/ray_tracing.h>
#include <optv/lsqadj.h>

#include <optv/vec_utils.h>
#include "ptv.h"

#define MAX_TARGETS 20000

/* Conversion radians -> gradians */
#include <math.h>
#define ro 200 / M_PI

/* Only needed while orientation funcions still do I/O */
#include <stdio.h>

/* Variables declared extern in 'globals.h' and not defined elsewhere: */
int orient_x2[4][1000], orient_y2[4][1000], orient_n[4];

void prepare_eval (control_par *cpar, int *n_fix) {
    int     i_img, i, filenumber, step_shake, count = 0;
	double  dummy;
    sequence_par *seq_par;
    FILE *fpp;
    
    int part_pointer; /* Holds index of particle later */
    
    frame frm;
    frame_init(&frm, cpar->num_cams, MAX_TARGETS);
    
	seq_par = read_sequence_par("parameters/sequence.par", cpar->num_cams);

	fpp = fopen ("parameters/dumbbell.par", "r");
    if (fpp){
        fscanf (fpp, "%lf", &dummy);
		fscanf (fpp, "%lf", &dummy);
	    fscanf (fpp, "%lf", &dummy);
	    fscanf (fpp, "%lf", &dummy);
		fscanf (fpp, "%d", &step_shake);
        fclose (fpp);
    }

    for (filenumber = seq_par->first; filenumber <= seq_par->last; \
        filenumber += step_shake)
    {
        read_frame(&frm, "res/db_is", NULL, NULL, seq_par->img_base_name,
            filenumber);

		for (i = 0; i < frm.num_parts; i++) {
            for (i_img = 0; i_img < cpar->num_cams; i_img++) {
                part_pointer = frm.correspond[i].p[i_img];
                if (part_pointer != CORRES_NONE) {
                    pix[i_img][count].x = frm.targets[i_img][part_pointer].x;
                    pix[i_img][count].y = frm.targets[i_img][part_pointer].y;
                } else {
                    pix[i_img][count].x = -999;
                    pix[i_img][count].y = -999;
                }
                
				if(pix[i_img][count].x>-999 && pix[i_img][count].y>-999){
				   pixel_to_metric (&crd[i_img][count].x, &crd[i_img][count].y,
                       pix[i_img][count].x, pix[i_img][count].y, cpar);
				}
				else{
                   crd[i_img][count].x=-1e10;
				   crd[i_img][count].y=-1e10;
				}
				crd[i_img][count].pnr=count;
			}
			count ++;
		}
	}
    free_frame(&frm);
	nfix=count;
}

/* This is very similar to prepare_eval, but it is sufficiently different in
   small but devious ways, not only parameters, that for now it'll be a 
   different function. */
void prepare_eval_shake(control_par *cpar) {
    FILE *fpp;
    char* target_file_base[4];
    int i_img, i, filenumber, step_shake, count = 0, frame_count = 0;
    int seq_first, seq_last;
    int frame_used, part_used;
    int max_shake_points, max_shake_frames;
	double  dummy;
    sequence_par *seq_par;
    
    int part_pointer; /* Holds index of particle later */
    
    frame frm;
    frame_init(&frm, cpar->num_cams, MAX_TARGETS);
    
	seq_par = read_sequence_par("parameters/sequence.par", cpar->num_cams);

	fpp = fopen ("parameters/shaking.par", "r");
    fscanf (fpp,"%d\n", &seq_first);
    fscanf (fpp,"%d\n", &seq_last);
    fscanf (fpp,"%d\n", &max_shake_points);
    fscanf (fpp,"%d\n", &max_shake_frames);
    fclose (fpp);
    
    step_shake = (int)((double)(seq_last - seq_first + 1) /
        (double)max_shake_frames + 0.5);
    printf ("shake step is %d\n", step_shake);
    
    for (filenumber = seq_first + 2; filenumber < seq_last - 1; \
        filenumber += step_shake)
    {
        frame_used = 0;
        read_frame(&frm, "res/rt_is", "res/ptv_is", NULL,
            seq_par->img_base_name, filenumber);
        
        printf("read frame %d\n", filenumber);
        printf("found %d particles \n", frm.num_parts);

        for (i = 0; i < frm.num_parts; i++) {
            if ((frm.path_info[i].prev == PREV_NONE) || \
                (frm.path_info[i].next == NEXT_NONE) ) continue;
            
            part_used = 1;
            
            for (i_img = 0; i_img < cpar->num_cams; i_img++) {
                part_pointer = frm.correspond[i].p[i_img];
                if (part_pointer == CORRES_NONE) {
                    part_used = 0;
                    break;
                }
            }
            
            if (part_used == 1) {
                frame_used = 1;
                
                for (i_img = 0; i_img < cpar->num_cams; i_img++) {
                    part_pointer = frm.correspond[i].p[i_img];
                    pix[i_img][count].x = frm.targets[i_img][part_pointer].x;
                    pix[i_img][count].y = frm.targets[i_img][part_pointer].y;
                    pix[i_img][count].pnr = count;
                                
                    pixel_to_metric(&crd[i_img][count].x, &crd[i_img][count].y,
                        pix[i_img][count].x, pix[i_img][count].y, cpar);
                    crd[i_img][count].pnr = count;
                }
                
                fix[count].x = frm.path_info[i].x[0];
                fix[count].y = frm.path_info[i].x[1];
                fix[count].z = frm.path_info[i].x[2];
                fix[count].pnr = count;
                count++;
            }
            if (count > max_shake_points) break;
        }
        if (frame_used == 1) frame_count++;
        if ((count >= max_shake_points) || (frame_count > max_shake_frames))
            break;
    }
    free_frame(&frm);
    nfix = count;
    printf("End of prepare_eval_shake, nfix = %d \n", nfix);
}

void cross(Ax,Ay,Az,Bx,By,Bz,Cx,Cy,Cz)

double Ax,Ay,Az,Bx,By,Bz,*Cx,*Cy,*Cz;

{
    *Cx=Ay*Bz-Az*By;
	*Cy=Az*Bx-Ax*Bz;
	*Cz=Ax*By-Ay*Bx;
}

void dotP(Ax,Ay,Az,Bx,By,Bz,d)

double Ax,Ay,Az,Bx,By,Bz,*d;

{
    *d=Ax*Bx+Ay*By+Az*Bz;
}

void mid_point(A1x,A1y,A1z,Ux,Uy,Uz,B1x,B1y,B1z,Vx,Vy,Vz,dist,XX,YY,ZZ)

double A1x,A1y,A1z,Ux,Uy,Uz,B1x,B1y,B1z,Vx,Vy,Vz;
double *dist,*XX,*YY,*ZZ;

{
	double Wx,Wy,Wz;
	double P1x,P1y,P1z,P2x,P2y,P2z;
	double cx,cy,cz,tmp1,tmp2;
    
	// taken from: Distance between 2 lines in 3d space 
    // http://groups.google.pl/group/comp.soft-sys.matlab/browse_thread/thread/a81803de728c9684/602e6bbf4c755565?hl=pl#602e6bbf4c755565
	
	//W = cross(U,V)
	cross(Ux,Uy,Uz,Vx,Vy,Vz,&Wx,&Wy,&Wz);

    //P1 = A1 + dot(cross(B1-A1,V),W)/dot(W,W)*U;
	cross(B1x-A1x,B1y-A1y,B1z-A1z,Vx,Vy,Vz,&cx,&cy,&cz);
	dotP(cx,cy,cz,Wx,Wy,Wz,&tmp1);
	dotP(Wx,Wy,Wz,Wx,Wy,Wz,&tmp2);	
	tmp1=tmp1/tmp2;
	P1x=A1x+tmp1*Ux;
	P1y=A1y+tmp1*Uy;
	P1z=A1z+tmp1*Uz;

    //P2 = B1 + dot(cross(B1-A1,U),W)/dot(W,W)*V;
	cross(B1x-A1x,B1y-A1y,B1z-A1z,Ux,Uy,Uz,&cx,&cy,&cz);
	dotP(cx,cy,cz,Wx,Wy,Wz,&tmp1);
	dotP(Wx,Wy,Wz,Wx,Wy,Wz,&tmp2);	
	tmp1=tmp1/tmp2;
	P2x=B1x+tmp1*Vx;
	P2y=B1y+tmp1*Vy;
	P2z=B1z+tmp1*Vz;

    //dist = norm(Y-X)
	*dist=pow(pow(P2x-P1x,2.)+pow(P2y-P1y,2.)+pow(P2z-P1z,2.),0.5);
	*XX=0.5*(P1x+P2x);
	*YY=0.5*(P1y+P2y);
	*ZZ=0.5*(P1z+P2z);

}

void eval_ori_v2 (cal, db_scale,weight_scale,n_img, nfix, d_outer,av_dist_error,residual, mmp)
Calibration *cal;
double db_scale;
double weight_scale;
int n_img,nfix;
double *d_outer;
double *av_dist_error;
double *residual;
mm_np mmp;

{
	int     i_img,i,count_inner=0,count_outer=0,pair_count=0,count_dist=0,n,m;
	double  xa,ya,xb,yb,temp,d_inner=0.,av_dist=0.,x,y;
	double X[4][3], a[4][3];
    double dist,dist_error,X_pos,Y_pos,Z_pos,XX,YY,ZZ,X1,Y1,Z1,X2,Y2,Z2;
	double tmp_d=0.,tmp_dist=0.;
	*d_outer=0.;
	*av_dist_error=0.;

   
	for(i=0;i<nfix;i++){
		//new det_lsq function, bloody fast!
		if(crd[0][i].x>-999){
			x = crd[0][i].x - cal[0].int_par.xh;
	        y = crd[0][i].y - cal[0].int_par.yh;
	        //correct_brown_affin (x, y, ap[0], &x, &y);
		    ray_tracing(x,y, &(cal[0]), mmp, X[0], a[0]);
		}		
		if(crd[1][i].x>-999){
			x = crd[1][i].x - cal[1].int_par.xh;
	        y = crd[1][i].y - cal[1].int_par.yh;
	        //correct_brown_affin (x, y, ap[1], &x, &y);
		    ray_tracing(x,y, &(cal[1]), mmp, X[1], a[1]);
		}		
		if(crd[2][i].x>-999){
			x = crd[2][i].x - cal[2].int_par.xh;
	        y = crd[2][i].y - cal[2].int_par.yh;
	        //correct_brown_affin (x, y, ap[2], &x, &y);
		    ray_tracing(x,y, &(cal[2]), mmp, X[2], a[2]);
		}		
		if(crd[3][i].x>-999){
			x = crd[3][i].x - cal[3].int_par.xh;
	        y = crd[3][i].y - cal[3].int_par.yh;
	        //correct_brown_affin (x, y, ap[3], &x, &y);
		    ray_tracing(x,y, &(cal[3]), mmp, X[3], a[3]);
		}

		count_inner=0;
		X_pos=0.;Y_pos=0.;Z_pos=0.;
		for (n=0;n<n_img;n++){
			for(m=n+1;m<n_img;m++){
				if(crd[n][i].x>-999 && crd[m][i].x>-999){
                    mid_point(X[n][0], X[n][1], X[n][2],
                        a[n][0], a[n][1], a[n][2], 
                        X[m][0], X[m][1], X[m][2],
                        a[m][0], a[m][1], a[m][2], &dist,&XX,&YY,&ZZ);
					count_inner++;
                    d_inner += dist;
					X_pos+=XX;Y_pos+=YY;Z_pos+=ZZ;
				}
			}
		}
        d_inner/=(double)count_inner;
		X_pos/=(double)count_inner;Y_pos/=(double)count_inner;Z_pos/=(double)count_inner;
		//end of new det_lsq

		if(Z_pos > cal[0].glass_par.vec_z){
            d_inner = Z_pos - cal->glass_par.vec_z+d_inner;
		}
		*d_outer +=d_inner;
		
		count_outer++;
		if(pair_count==0) {
            X1=X_pos;Y1=Y_pos;Z1=Z_pos;
		}
		if(pair_count==1) {
            X2=X_pos;Y2=Y_pos;Z2=Z_pos;
		}
		///here I introduce penalty for scale
		pair_count++;
		if(pair_count==2){
			pair_count=0;			
		    dist=sqrt(pow(X2-X1,2.)+pow(Y2-Y1,2.)+pow(Z2-Z1,2.));
			av_dist+=dist;
			if(dist<db_scale){
			    dist_error=1-dist/db_scale;
			}
			else{
                dist_error=1-db_scale/dist;
			}
			*av_dist_error+=dist_error;
			count_dist++;
		}
		///end of eval
	}
	av_dist /=(double)count_dist;
	*d_outer /=(double)count_outer;
	*d_outer/=av_dist;
	*av_dist_error /=(double)count_dist;
	*residual=*d_outer+weight_scale*(*av_dist_error);
}

void orient_v5 (control_par *cpar, int nfix, Calibration cal[])

/*
Exterior	*Ex;	 exterior orientation, approx and result 
Interior	*I;		 interior orientation, approx and result 
Glass   	*G;		 glass orientation, approx and result 
ap_52		*ap;	 add. parameters, approx and result 
int	       	nfix;		# of object points 
*/

{
    FILE *fp1, *fpp;
    int  	i,j,itnum,max_itnum,i_img,dummy;
    double       	residual, best_residual, old_val, dm, drad, sens, factor, weight_scale;   
    double 	Xp, Yp, Zp, xp, yp, xpd, ypd, r, qq;
	double db_scale,eps0,epi_miss, dist;
	int  	useflag, ccflag, scxflag, sheflag, interfflag, xhflag, yhflag,
    k1flag, k2flag, k3flag, p1flag, p2flag;
    
    dm = 1.0;
    drad = 1.0;
    printf("dm = %f, drad = %f\n", dm, drad);

	fpp = fopen ("parameters/dumbbell.par", "r");
    if (fpp){
        fscanf (fpp, "%lf", &eps0);
		fscanf (fpp, "%lf", &db_scale);
	    fscanf (fpp, "%lf", &factor);
	    fscanf (fpp, "%lf", &weight_scale);
		fscanf (fpp, "%d", &dummy);
		fscanf (fpp, "%d", &max_itnum);
        fclose (fpp);
    }

	fp1 = fopen ("parameters/orient.par", "r");
    fscanf (fp1,"%d", &useflag);
    fscanf (fp1,"%d", &ccflag);
    fscanf (fp1,"%d", &xhflag);
    fscanf (fp1,"%d", &yhflag);
    fscanf (fp1,"%d", &k1flag);
    fscanf (fp1,"%d", &k2flag);
    fscanf (fp1,"%d", &k3flag);
    fscanf (fp1,"%d", &p1flag);
    fscanf (fp1,"%d", &p2flag);
    fscanf (fp1,"%d", &scxflag);
    fscanf (fp1,"%d", &sheflag);
    fscanf (fp1,"%d", &interfflag);
    fclose (fp1);
	
  puts ("\n\nbegin of iterations");
  itnum = 0;  
  while (itnum < max_itnum){
    //printf ("\n\n%2d. iteration\n", ++itnum);
    itnum++;

    eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
        &epi_miss, &dist, &residual, *(cpar->mm));
	best_residual=residual;
    
    for (i_img = 0; i_img < cpar->num_cams; i_img++) {
	     
		 cal[i_img].ext_par.x0 += dm;
	     eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix,
            &epi_miss, &dist, &residual, *(cpar->mm));
         
		 if(best_residual-residual < 0){ //then try other direction
			 cal[i_img].ext_par.x0 -= 2*dm;
	         eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
                &epi_miss, &dist, &residual, *(cpar->mm));
             
			 if(best_residual-residual < 0){// then leave it unchanged
                  cal[i_img].ext_par.x0 += dm;
			 }
			 else{ // it was a success!
                  best_residual=residual;
			 }
		 }
		 else{ // it was a success!
			 best_residual=residual;
		 }
	     

		 cal[i_img].ext_par.y0 += dm;
	     eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
            &epi_miss, &dist, &residual, *(cpar->mm));

		 if(best_residual-residual < 0){ //then try other direction
			 cal[i_img].ext_par.y0 -= 2*dm;
	         eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
                &epi_miss, &dist, &residual, *(cpar->mm));
			 if(best_residual-residual < 0){// then leave it unchanged
                  cal[i_img].ext_par.y0 += dm;
			 }
			 else{ // it was a success!
                  best_residual=residual;
			 }
		 }
		 else{ // it was a success!
			 best_residual=residual;
		 }

		 cal[i_img].ext_par.z0 += dm;
	     eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
            &epi_miss, &dist, &residual, *(cpar->mm));
		 if(best_residual-residual < 0){ //then try other direction
			 cal[i_img].ext_par.z0 -= 2*dm;
	         eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix,
                &epi_miss, &dist, &residual, *(cpar->mm));
			 if(best_residual-residual < 0){// then leave it unchanged
                  cal[i_img].ext_par.z0 += dm;
			 }
			 else{ // it was a success!
                  best_residual=residual;
			 }
		 }
		 else{ // it was a success!
			 best_residual=residual;
		 }

		 cal[i_img].ext_par.omega += drad;
		 rotation_matrix (&(cal[i_img].ext_par));
	     eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
            &epi_miss, &dist, &residual, *(cpar->mm));
		 if(best_residual-residual < 0){ //then try other direction
			 cal[i_img].ext_par.omega -= 2*drad;
			 rotation_matrix (&(cal[i_img].ext_par));
	         eval_ori_v2(cal, db_scale, weight_scale, cpar->mm, nfix, 
                &epi_miss, &dist, &residual, *(cpar->mm));
			 if(best_residual-residual < 0){// then leave it unchanged
                  cal[i_img].ext_par.omega += drad;
				  rotation_matrix(&(cal[i_img].ext_par));
			 }
			 else{ // it was a success!
                  best_residual=residual;
			 }
		 }
		 else{ // it was a success!
			 best_residual=residual;
		 }

		 cal[i_img].ext_par.phi += drad;
		 rotation_matrix (&(cal[i_img].ext_par));
	     eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
            &epi_miss, &dist, &residual, *(cpar->mm));
		 if(best_residual-residual < 0){ //then try other direction
			 cal[i_img].ext_par.phi -= 2*drad;
			 rotation_matrix(&(cal[i_img].ext_par));
	         eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix,
                &epi_miss, &dist, &residual, *(cpar->mm));
			 if(best_residual-residual < 0){// then leave it unchanged
                  cal[i_img].ext_par.phi += drad;
				  rotation_matrix(&(cal[i_img].ext_par));
			 }
			 else{ // it was a success!
                  best_residual=residual;
			 }
		 }
		 else{ // it was a success!
			 best_residual=residual;
		 }
		 
		 cal[i_img].ext_par.kappa += drad;
		 rotation_matrix(&(cal[i_img].ext_par));
	     eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix,
            &epi_miss, &dist, &residual, *(cpar->mm));
		 if(best_residual-residual < 0){ //then try other direction
			 cal[i_img].ext_par.kappa -= 2*drad;
			 rotation_matrix(&(cal[i_img].ext_par));
	         eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix,
                &epi_miss, &dist, &residual, *(cpar->mm));
			 if(best_residual-residual < 0){// then leave it unchanged
                  cal[i_img].ext_par.kappa+= drad;
				  rotation_matrix(&(cal[i_img].ext_par));
			 }
			 else{ // it was a success!
                  best_residual=residual;
			 }
		 }
		 else{ // it was a success!
			 best_residual=residual;
		 }
		 
		 if(ccflag==1){
			 if (1<2){
		    cal[i_img].int_par.cc += dm;
	        eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
                &epi_miss, &dist, &residual, *(cpar->mm));
		    if(best_residual-residual < 0){ //then try other direction
			    cal[i_img].int_par.cc -= 2*dm;
	            eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
                    &epi_miss, &dist, &residual, *(cpar->mm));
			    if(best_residual-residual < 0){// then leave it unchanged
                    cal[i_img].int_par.cc += dm;
			    }
			    else{ // it was a success!
                    best_residual=residual;
			    }
		    }
		    else{ // it was a success!
			    best_residual=residual;
		    }
			 }
			 else{
	        cal[0].int_par.cc += dm;
			cal[1].int_par.cc  =cal[0].int_par.cc;
			cal[2].int_par.cc  =cal[0].int_par.cc;
			cal[3].int_par.cc  =cal[0].int_par.cc;
            eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
                    &epi_miss, &dist, &residual, *(cpar->mm));
		    if(best_residual-residual < 0){ //then try other direction
			    cal[0].int_par.cc -= 2*dm;
				cal[1].int_par.cc  =cal[0].int_par.cc;
			    cal[2].int_par.cc  =cal[0].int_par.cc;
			    cal[3].int_par.cc  =cal[0].int_par.cc;
                eval_ori_v2(cal, db_scale, weight_scale, cpar->num_cams, nfix, 
                    &epi_miss, &dist, &residual, *(cpar->mm));
			    if(best_residual-residual < 0){// then leave it unchanged
                    cal[0].int_par.cc += dm;
					cal[1].int_par.cc  =cal[0].int_par.cc;
			        cal[2].int_par.cc  =cal[0].int_par.cc;
			        cal[3].int_par.cc  =cal[0].int_par.cc;
			    }
			    else{ // it was a success!
                    best_residual=residual;
			    }
		    }
		    else{ // it was a success!
			    best_residual=residual;
		    }

			 }
		 }
	}
	
	printf ("eps_tot: %8.7f, eps_epi: %7.5f, eps_dist: %7.5f, step: %d\n",best_residual, epi_miss,dist,itnum);

      
 }



  
}


/*  num_deriv_exterior() calculates the partial numerical derivative of image
    coordinates of a given 3D position, over each of the 6 exterior orientation
    parameters (3 position parameters, 3 rotation angles).
    
    Arguments:
    int cam - number of camera currently processed. Only needed if the LUT 
        structure is used, otherwise ignored.
    Exterior ext - an object describing current exterior orientation. Changed
        in place during run, but restored back.
    Interior I0, Glass G0, ap_52 ap0, mm_np mm - rest of calibration 
        parameters, passed directly to img_coord(). Interim measure before
        we can pass a pointer to a Calibration object.
    double dpos, double dang - the step size for numerical differentiation,
        dpos for the metric variables, dang for the angle variables. Units
        are same as the units of the variables derived.
    vec3d pos - the current 3D position represented on the image.
    
    Return parameters:
    double x_ders[6], y_ders[6] respectively the derivatives of the x and y
        image coordinates as function of each of the orientation parameters.
 */
void num_deriv_exterior(int cam, Exterior ext, Interior I0, Glass G0, ap_52 ap0, 
    mm_np mm, double dpos, double dang, vec3d pos,
    double x_ders[6], double y_ders[6])
{
    int pd; 
    double step, xs, ys, xpd, ypd;
    double* vars[6];
    
    vars[0] = &(ext.x0);
    vars[1] = &(ext.y0);
    vars[2] = &(ext.z0);
    vars[3] = &(ext.omega);
    vars[4] = &(ext.phi);
    vars[5] = &(ext.kappa);
    
    /* Starting image position */
	img_coord (cam, pos[0], pos[1], pos[2], ext, I0, G0, ap0, mm, &xs, &ys);
    
    for (pd = 0; pd < 6; pd++) {
        step = (pd > 2) ? dang : dpos;
        
        *(vars[pd]) += step;
        if (pd > 2) rotation_matrix(&ext);
        img_coord(cam, pos[0], pos[1], pos[2], ext, I0, G0, ap0, mm, &xpd, &ypd);
        x_ders[pd] = (xpd - xs) / step;
        y_ders[pd] = (ypd - ys) / step;
        *(vars[pd]) -= step;
    }
    rotation_matrix(&ext);
}

/*
    Returns:
    1 if iterative solution found, 0 otherwise.
*/
int orient_v3 (Ex0, I0, G0, ap0, mm, nfix, fix, crd, Ex, I, G, ap, nr, resid_x, resid_y,
    pixnr, num_used)
Exterior	Ex0, *Ex;	/* exterior orientation, approx and result */
Interior	I0, *I;		/* interior orientation, approx and result */
Glass   	G0, *G;		/* glass orientation, approx and result */
ap_52		ap0, *ap;	/* add. parameters, approx and result */
mm_np		mm;    		/* multimedia parameters */
int	       	nfix;		/* # of object points */
coord_3d	fix[];		/* object point data */
coord_2d	crd[];		/* image coordinates */
int	       	nr;  		/* image number for residual display */
double      resid_x[], resid_y[]; /* return array for residuals */
int         pixnr[]; /* indices of used detected points */
int         *num_used;  /* Number of points used for orientation */
{
  int  	i,j,n, itnum, stopflag, n_obs=0,convergeflag;
  int  	useflag, ccflag, scxflag, sheflag, interfflag, xhflag, yhflag,
    k1flag, k2flag, k3flag, p1flag, p2flag;
  int  	intx1, intx2, inty1, inty2;
  double dm = 0.00001,  drad = 0.0000001;
  double       	X[1800][19], Xh[1800][19], y[1800], yh[1800], ident[10],
    XPX[19][19], XPy[19], beta[19], Xbeta[1800],
    resi[1800], omega=0, sigma0, sigmabeta[19],
    P[1800], p, sumP;
  double xp, yp, xpd, ypd, r, qq;
  FILE 	*fp1;
  int dummy, multi,numbers;
  double al,be,ga,nGl,e1_x,e1_y,e1_z,e2_x,e2_y,e2_z,n1,n2,safety_x,safety_y,safety_z;
  
  vec3d pos;


  /* read, which parameters shall be used */
  printf("inside orient_v3\n");
  fp1 = fopen ("parameters/orient.par","r");
  if (fp1){
  fscanf (fp1,"%d", &useflag);
  fscanf (fp1,"%d", &ccflag);
  fscanf (fp1,"%d", &xhflag);
  fscanf (fp1,"%d", &yhflag);
  fscanf (fp1,"%d", &k1flag);
  fscanf (fp1,"%d", &k2flag);
  fscanf (fp1,"%d", &k3flag);
  fscanf (fp1,"%d", &p1flag);
  fscanf (fp1,"%d", &p2flag);
  fscanf (fp1,"%d", &scxflag);
  fscanf (fp1,"%d", &sheflag);
  fscanf (fp1,"%d", &interfflag);
  fclose (fp1);
  }
  else{
  printf("problem opening parameters/orient.par\n");
  }


  //if(interfflag){
      nGl=sqrt(pow(G0.vec_x,2.)+pow(G0.vec_y,2.)+pow(G0.vec_z,2.));
	  e1_x=2*G0.vec_z-3*G0.vec_x;
	  e1_y=3*G0.vec_x-1*G0.vec_z;
	  e1_z=1*G0.vec_y-2*G0.vec_y;
	  n1=sqrt(pow(e1_x,2.)+pow(e1_y,2.)+pow(e1_z,2.));
	  e1_x=e1_x/n1;
	  e1_y=e1_y/n1;
	  e1_z=e1_z/n1;
	  e2_x=e1_y*G0.vec_z-e1_z*G0.vec_x;
	  e2_y=e1_z*G0.vec_x-e1_x*G0.vec_z;
	  e2_z=e1_x*G0.vec_y-e1_y*G0.vec_y;
	  n2=sqrt(pow(e2_x,2.)+pow(e2_y,2.)+pow(e2_z,2.));
	  e2_x=e2_x/n2;
	  e2_y=e2_y/n2;
	  e2_z=e2_z/n2;
	  al=0;
	  be=0;
	  ga=0;
  //}


  printf("\n Inside orient_v3, initialize memory \n");

  /* init X, y (set to zero) */
  for (i=0; i<1800; i++)
    {
      for (j=0; j<19; j++) {
        X[i][j] = 0.0;
      }
      y[i] = 0.0;  P[i] = 0.0;
    }

   printf("\n Memory is initialized, \n");

  /* init identities */
  ident[0] = I0.cc;  ident[1] = I0.xh;  ident[2] = I0.yh;
  ident[3]=ap0.k1; ident[4]=ap0.k2; ident[5]=ap0.k3;
  ident[6]=ap0.p1; ident[7]=ap0.p2;
  ident[8]=ap0.scx; ident[9]=ap0.she;

  /* main loop, program runs through it, until none of the beta values
     comes over a threshold and no more points are thrown out
     because of their residuals */


  safety_x=G0.vec_x;
  safety_y=G0.vec_y;
  safety_z=G0.vec_z;

  printf("\n\n start iterations, orient_v3 \n");
  itnum = 0;  stopflag = 0;
  while ((stopflag == 0) && (itnum < 80))
    {
      
      itnum++;
      
      printf ("\n\n %2d. iteration \n", itnum);
      
      for (i=0, n=0; i<nfix; i++) if (crd[i].pnr == fix[i].pnr) 
	  {
	  /* use only certain points as control points */
	  switch (useflag)
	    {
	    case 1: if ((fix[i].pnr % 2) == 0)  continue;  break;
	    case 2: if ((fix[i].pnr % 2) != 0)  continue;  break;
	    case 3: if ((fix[i].pnr % 3) == 0)  continue;  break;
	    }

	  /* check for correct correspondence */
	  if (crd[i].pnr != fix[i].pnr)	continue;


	  pixnr[n/2] = i;		/* for drawing residuals */
	  pos[0] = fix[i].x;
      pos[1] = fix[i].y;
      pos[2] = fix[i].z;
	  rotation_matrix (&Ex0);
	  img_coord (nr, pos[0], pos[1], pos[2], Ex0, I0, G0, ap0, mm, &xp,&yp);


	  /* derivatives of add. parameters */

	  r = sqrt (xp*xp + yp*yp);

	  X[n][7] = ap0.scx;
	  X[n+1][7] = sin(ap0.she);

	  X[n][8] = 0;
	  X[n+1][8] = 1;

	  X[n][9] = ap0.scx * xp * r*r;
	  X[n+1][9] = yp * r*r;

	  X[n][10] = ap0.scx * xp * pow(r,4.0);
	  X[n+1][10] = yp * pow(r,4.0);

	  X[n][11] = ap0.scx * xp * pow(r,6.0);
	  X[n+1][11] = yp * pow(r,6.0);

	  X[n][12] = ap0.scx * (2*xp*xp + r*r);
	  X[n+1][12] = 2 * xp * yp;

	  X[n][13] = 2 * ap0.scx * xp * yp;
	  X[n+1][13] = 2*yp*yp + r*r;

	  qq =  ap0.k1*r*r; qq += ap0.k2*pow(r,4.0);
	  qq += ap0.k3*pow(r,6.0);
	  qq += 1;
	  X[n][14] = xp * qq + ap0.p1 * (r*r + 2*xp*xp) + 2*ap0.p2*xp*yp;
	  X[n+1][14] = 0;

	  X[n][15] = -cos(ap0.she) * yp;
	  X[n+1][15] = -sin(ap0.she) * yp;


	  /* numeric derivatives */
      num_deriv_exterior(nr, Ex0, I0, G0, ap0, mm, dm, drad, pos, X[n], X[n + 1]);

	  I0.cc += dm;
	  rotation_matrix (&Ex0);
	  img_coord (nr, pos[0], pos[1], pos[2], Ex0,I0, G0, ap0, mm, &xpd, &ypd);
	  X[n][6]      = (xpd - xp) / dm;
	  X[n+1][6] = (ypd - yp) / dm;
	  I0.cc -= dm;
      
	  //G0.vec_x += dm;
	  //safety_x=G0.vec_x;
	  //safety_y=G0.vec_y;
	  //safety_z=G0.vec_z;
	  al +=dm;
	  G0.vec_x+=e1_x*nGl*al;G0.vec_y+=e1_y*nGl*al;G0.vec_z+=e1_z*nGl*al;
	  img_coord (nr, pos[0], pos[1], pos[2], Ex0,I0, G0, ap0, mm, &xpd, &ypd);
	  X[n][16]      = (xpd - xp) / dm;
	  X[n+1][16] = (ypd - yp) / dm;
	  //G0.vec_x -= dm;
	  //G0.vec_x-=e1_x*nGl*al;G0.vec_y-=e1_y*nGl*al;G0.vec_z-=e1_z*nGl*al;
	  al-=dm;
	  G0.vec_x=safety_x;
	  G0.vec_y=safety_y;
	  G0.vec_z=safety_z;

	  //G0.vec_y += dm;
	  be +=dm;
	  G0.vec_x+=e2_x*nGl*be;G0.vec_y+=e2_y*nGl*be;G0.vec_z+=e2_z*nGl*be;
	  img_coord (nr, pos[0], pos[1], pos[2], Ex0,I0, G0, ap0, mm, &xpd, &ypd);
	  X[n][17]      = (xpd - xp) / dm;
	  X[n+1][17] = (ypd - yp) / dm;
	  //G0.vec_y -= dm;
	  //G0.vec_x-=e2_x*nGl*be;G0.vec_y-=e2_y*nGl*be;G0.vec_z-=e2_z*nGl*be;
	  be-=dm;
	  G0.vec_x=safety_x;
	  G0.vec_y=safety_y;
	  G0.vec_z=safety_z;

	  //G0.vec_y += dm;
	  ga +=dm;
	  G0.vec_x+=G0.vec_x*nGl*ga;G0.vec_y+=G0.vec_y*nGl*ga;G0.vec_z+=G0.vec_z*nGl*ga;
	  img_coord (nr, pos[0], pos[1], pos[2], Ex0,I0, G0, ap0, mm, &xpd, &ypd);
	  X[n][18]      = (xpd - xp) / dm;
	  X[n+1][18] = (ypd - yp) / dm;
	  //G0.vec_y -= dm;
	  //G0.vec_x-=G0.vec_x*nGl*ga;G0.vec_y-=G0.vec_y*nGl*ga;G0.vec_z-=G0.vec_z*nGl*ga;
	  ga-=dm;
	  G0.vec_x=safety_x;
	  G0.vec_y=safety_y;
	  G0.vec_z=safety_z;
	  
	  y[n]   = crd[i].x - xp;
	  y[n+1] = crd[i].y - yp;

	  n += 2;
	} // end if crd == fix
      
      
      n_obs = n;
      
      printf(" n_obs = %d\n", n_obs); 


      /* identities */

      for (i=0; i<10; i++)  X[n_obs+i][6+i] = 1;

      y[n_obs+0] = ident[0] - I0.cc;		y[n_obs+1] = ident[1] - I0.xh;
      y[n_obs+2] = ident[2] - I0.yh;		y[n_obs+3] = ident[3] - ap0.k1;
      y[n_obs+4] = ident[4] - ap0.k2;		y[n_obs+5] = ident[5] - ap0.k3;
      y[n_obs+6] = ident[6] - ap0.p1;		y[n_obs+7] = ident[7] - ap0.p2;
      y[n_obs+8] = ident[8] - ap0.scx;		y[n_obs+9] = ident[9] - ap0.she;



      /* weights */
      for (i=0; i<n_obs; i++)  P[i] = 1;
      if ( ! ccflag)  P[n_obs+0] = 1e20;
      if ( ! xhflag)  P[n_obs+1] = 1e20;
      if ( ! yhflag)  P[n_obs+2] = 1e20;
      if ( ! k1flag)  P[n_obs+3] = 1e20;
      if ( ! k2flag)  P[n_obs+4] = 1e20;
      if ( ! k3flag)  P[n_obs+5] = 1e20;
      if ( ! p1flag)  P[n_obs+6] = 1e20;
      if ( ! p2flag)  P[n_obs+7] = 1e20;
      if ( ! scxflag) P[n_obs+8] = 1e20;
      if ( ! sheflag) P[n_obs+9] = 1e20;


      n_obs += 10;  sumP = 0;
      for (i=0; i<n_obs; i++)	       	/* homogenize */
	{
	  p = sqrt (P[i]);
	  for (j=0; j<19; j++)  Xh[i][j] = p * X[i][j];
	  yh[i] = p * y[i];  sumP += P[i];
	}



      /* Gauss Markoff Model */
	  numbers=16;
	  if(interfflag){
         numbers=18;
	  }
	  
	  ata ((double *) Xh, (double *) XPX, n_obs, numbers, 19 );
      matinv ((double *) XPX, numbers, 19);
      atl ((double *) XPy, (double *) Xh, yh, n_obs, numbers, 19);
      matmul ((double *) beta, (double *) XPX, (double *) XPy, numbers,numbers,1,19,19);
	  
      stopflag = 1;
	  convergeflag = 1;
      //puts ("\n==> beta :\n");
      for (i=0; i<numbers; i++)
	{
	  printf (" beta[%d] = %10.6f\n  ",i, beta[i]);
	  if (fabs (beta[i]) > 0.0001)  stopflag = 0;	/* more iterations */////Achtung
	  if (fabs (beta[i]) > 0.01)  convergeflag = 0;
	}
      //printf ("\n\n");
	  
	  if ( ! ccflag) beta[6]=0;
      if ( ! xhflag) beta[7]=0;
      if ( ! yhflag) beta[8]=0;
      if ( ! k1flag) beta[9]=0;
      if ( ! k2flag) beta[10]=0;
      if ( ! k3flag) beta[11]=0;
      if ( ! p1flag) beta[12]=0;
      if ( ! p2flag) beta[13]=0;
      if ( ! scxflag)beta[14]=0;
      if ( ! sheflag) beta[15]=0;
      
      Ex0.x0 += beta[0];  Ex0.y0 += beta[1];  Ex0.z0 += beta[2];
      Ex0.omega += beta[3];  Ex0.phi += beta[4];  Ex0.kappa += beta[5];
      I0.cc += beta[6];  I0.xh += beta[7];  I0.yh += beta[8];
      ap0.k1 += beta[9];  ap0.k2 += beta[10];  ap0.k3 += beta[11];
      ap0.p1 += beta[12];  ap0.p2 += beta[13];
      ap0.scx += beta[14];  ap0.she += beta[15];
	  if(interfflag){
	  //G0.vec_x += beta[16];	  
	  //G0.vec_y += beta[17];
      G0.vec_x+=e1_x*nGl*beta[16];G0.vec_y+=e1_y*nGl*beta[16];G0.vec_z+=e1_z*nGl*beta[16];
	  G0.vec_x+=e2_x*nGl*beta[17];G0.vec_y+=e2_y*nGl*beta[17];G0.vec_z+=e2_z*nGl*beta[17];
	  //G0.vec_x+=G0.vec_x*nGl*beta[18];G0.vec_y+=G0.vec_y*nGl*beta[18];G0.vec_z+=G0.vec_z*nGl*beta[18];
	  }
	  beta[0]=beta[0];
    } // end of while iterations and stopflag



  /* compute residuals etc. */

  matmul( (double *) Xbeta, (double *) X, (double *) beta, n_obs, numbers, 1, n_obs, 19);
  omega = 0;
  for (i=0; i<n_obs; i++)
    {
      resi[i] = Xbeta[i] - y[i];  omega += resi[i] * P[i] * resi[i];
    }
  sigma0 = sqrt (omega / (n_obs - numbers));

  for (i=0; i<numbers; i++)  sigmabeta[i] = sigma0 * sqrt(XPX[i][i]);


  /* correlations between parameters */
  /*if (examine)	for (i=0; i<18; i++)
    {
      for (j=0; j<18; j++)
	printf ("%6.2f",
		XPX[i][j] / (sqrt(XPX[i][i]) * sqrt(XPX[j][j])));
      printf ("\n");
    }*/


  /* print results */
  printf ("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
  printf ("\n\nResults after %d iterations:\n\n", itnum);
  printf ("sigma0 = %6.2f micron\n", sigma0*1000);
  printf ("X0 =    %8.3f   +/- %8.3f\n", Ex0.x0, sigmabeta[0]);
  printf ("Y0 =    %8.3f   +/- %8.3f\n", Ex0.y0, sigmabeta[1]);
  printf ("Z0 =    %8.3f   +/- %8.3f\n", Ex0.z0, sigmabeta[2]);
  printf ("omega = %8.4f   +/- %8.4f\n", Ex0.omega*ro, sigmabeta[3]*ro);
  printf ("phi   = %8.4f   +/- %8.4f\n", Ex0.phi*ro, sigmabeta[4]*ro);
  printf ("kappa = %8.4f   +/- %8.4f\n", Ex0.kappa*ro, sigmabeta[5]*ro);
  if(interfflag){
  printf ("G0.vec_x = %8.4f   +/- %8.4f\n", G0.vec_x/nGl, (sigmabeta[16]+sigmabeta[17]));
  printf ("G0.vec_y = %8.4f   +/- %8.4f\n", G0.vec_y/nGl, (sigmabeta[16]+sigmabeta[17]));
  printf ("G0.vec_z = %8.4f   +/- %8.4f\n", G0.vec_z/nGl, (sigmabeta[16]+sigmabeta[17]));
  }
  //printf ("vec_z = %8.4f   +/- %8.4f\n", G0.vec_z, sigmabeta[18]);
  printf ("camera const  = %8.5f   +/- %8.5f\n", I0.cc, sigmabeta[6]);
  printf ("xh            = %8.5f   +/- %8.5f\n", I0.xh, sigmabeta[7]);
  printf ("yh            = %8.5f   +/- %8.5f\n", I0.yh, sigmabeta[8]);
  printf ("k1            = %8.5f   +/- %8.5f\n", ap0.k1, sigmabeta[9]);
  printf ("k2            = %8.5f   +/- %8.5f\n", ap0.k2, sigmabeta[10]);
  printf ("k3            = %8.5f   +/- %8.5f\n", ap0.k3, sigmabeta[11]);
  printf ("p1            = %8.5f   +/- %8.5f\n", ap0.p1, sigmabeta[12]);
  printf ("p2            = %8.5f   +/- %8.5f\n", ap0.p2, sigmabeta[13]);
  printf ("scale for x'  = %8.5f   +/- %8.5f\n", ap0.scx, sigmabeta[14]);
  printf ("shearing      = %8.5f   +/- %8.5f\n", ap0.she*ro, sigmabeta[15]*ro);


  /* show original images with residual vectors (requires globals) */
  for (i = 0; i < n_obs - 10; i += 2) {
    n = pixnr[i/2];
    //intx2 = intx1 + resi[i]*5000;
    //inty2 = inty1 + resi[i+1]*5000;
    resid_x[n]=resi[i];
	resid_y[n]=resi[i+1];
  }
  *num_used = (n_obs - 10)/2; 


  if (convergeflag){
      rotation_matrix (&Ex0);
      *Ex = Ex0;	*I = I0;	*ap = ap0; *G = G0;
      return 1;
  }
  else{	
	  puts ("orientation does not converge");
      return 0;
  }
}


/*
    Returns:
    1 if iterative solution found, 0 otherwise.
*/
int raw_orient_v3 (Exterior Ex0, Interior I, Glass G0, ap_52 ap, mm_np mm, 
    int nfix, coord_3d fix[], coord_2d crd[], Exterior *Ex, Glass *G,
    int nr, int only_show)
{
  double		X[10][6], y[10],
    XPX[6][6], XPy[6], beta[6];
  double xp, yp;
  int     	i,j,n, itnum, stopflag, n_obs=0;
  double		dm = 0.0001,  drad = 0.000001;
  
  vec3d pos;
  
  /* init X, y (set to zero) */
  for (i=0; i<10; i++)
    {
      for (j=0; j<6; j++)  X[i][j] = 0;
      y[i] = 0;
    }

  ap.k1 = 0;	ap.k2 = 0;	ap.k3 = 0;	ap.p1 = 0;	ap.p2 = 0;
  ap.scx = 1; ap.she = 0;


  /* main loop, program runs through it, until none of the beta values
     comes over a threshold and no more points are thrown out
     because of their residuals */

  itnum = 0;  stopflag = 0;

///////////make a menu so one see the raw guess!!!!!
  if(only_show==1) stopflag=1;
/////// Beat Lüthi 9. Mai 2007

  while ((stopflag == 0) && (itnum < 20)) {
    ++itnum;

    for (i = 0, n = 0; i < nfix; i++) if (crd[i].x != -999) {
        pos[0] = fix[i].x;
        pos[1] = fix[i].y;
        pos[2] = fix[i].z;

	    rotation_matrix(&Ex0);
        num_deriv_exterior(nr, Ex0, I, G0, ap, mm, dm, drad, pos, X[n], X[n + 1]);
        
	    img_coord (nr, pos[0], pos[1], pos[2], Ex0, I, G0, ap, mm, &xp, &yp);
	    y[n]   = crd[i].x - xp;
	    y[n+1] = crd[i].y - yp;

	    n += 2;
	}
    n_obs = n;

    /* Gauss Markoff Model */

    ata ((double *) X, (double *) XPX, n_obs, 6, 6);
    matinv ((double *) XPX, 6, 6);
    atl ((double *) XPy, (double *) X, y, n_obs, 6, 6);
    matmul ((double *) beta, (double *) XPX, (double *) XPy, 6, 6, 1, 6, 6);

    stopflag = 1;
	for (i = 0; i < 6; i++){
        if (fabs (beta[i]) > 0.1 )  stopflag = 0;
	}

    Ex0.x0 += beta[0];  Ex0.y0 += beta[1];  Ex0.z0 += beta[2];
    Ex0.omega += beta[3];  Ex0.phi += beta[4];
    Ex0.kappa += beta[5];
	//G0.vec_x += beta[6];G0.vec_y += beta[7];//G0.vec_z += beta[8];
	stopflag =stopflag ;	  
  }

  if (stopflag)
    {
      *Ex = Ex0;
	  *G  = G0;
      rotation_matrix(Ex);
    }
  else {
	  puts ("raw orientation impossible");
    }
    return stopflag;
}

