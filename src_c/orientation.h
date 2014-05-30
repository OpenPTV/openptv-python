/* Forward declarations for calibration (camera orientation) functions found in
orientation.c */

#ifndef ORIENTATION_H
#define ORIENTATION_H

void prepare_eval(int n_img, int *n_fix);
void orient_v3(Exterior Ex0, Interior I0, Glass G0, ap_52 ap0,
    mm_np mm, int nfix, coord_3d fix[], coord_2d crd[], 
    Exterior *Ex, Interior *I, Glass *G, ap_52 *ap, int nr);
void raw_orient_v3(Exterior Ex0, Interior I, Glass G0, ap_52 ap,
    mm_np mm, int nfix, coord_3d fix[], coord_2d crd[], 
    Exterior *Ex, Glass *G, int only_show);
void orient_v5(int n_img, int nfix, Exterior *Ex, Interior *I, Glass *G, ap_52 *ap);


#endif
