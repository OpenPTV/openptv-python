
/*  Unit-test suit for the C core of PyPTV. Uses the Check framework:
    http://check.sourceforge.net/
    
    To run it, type "make check" when in the top C directory, src_c/
*/

#include <check.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>

#include <optv/vec_utils.h>
#include <optv/parameters.h>

#include "../src_c/ptv.h"
#include "../src_c/orientation.h"

/* Generate a calibration object with example values matching those in the
   files read by test_read_ori.
*/
Calibration test_cal(void) {
    Exterior correct_ext = {
        105.2632, 102.7458, 403.8822,
        -0.2383291, 0.2442810, 0.0552577, 
        {{0.9688305, -0.0535899, 0.2418587}, 
        {-0.0033422, 0.9734041, 0.2290704},
        {-0.2477021, -0.2227387, 0.9428845}}};
    Interior correct_int = {-2.4742, 3.2567, 100.0000};
    Glass correct_glass = {0.0001, 0.00001, 150.0};
    ap_52 correct_addp = {0., 0., 0., 0., 0., 1., 0.};
    Calibration correct_cal = {correct_ext, correct_int, correct_glass, 
        correct_addp};
    rotation_matrix(&(correct_cal.ext_par));
    
    return correct_cal;
}

/* Regression test for reading orientation files. Just reads a sample file and
   makes sure that nothing crashes and the orientation structures are filled
   out correctly.
*/
START_TEST(test_read_ori)
{
    Calibration correct_cal, *cal;
    correct_cal = test_cal();
    
    char ori_file[] = "testing_fodder/cal/cam1.tif.ori";
    char add_file[] = "testing_fodder/cal/cam1.tif.addpar";
    
    fail_if((cal = read_calibration(ori_file, add_file, NULL)) == NULL);
    fail_unless(compare_calib(cal, &correct_cal));
}
END_TEST

/* Unit test for writing orientation files. Writes a sample calibration,
   reads it back and compares.
*/
START_TEST(test_write_ori)
{
    Calibration correct_cal, *cal;
    correct_cal = test_cal();
    char ori_file[] = "testing_fodder/test.ori";
    char add_file[] = "testing_fodder/test.addpar";
    
    write_ori(correct_cal.ext_par, correct_cal.int_par,
        correct_cal.glass_par, correct_cal.added_par, ori_file, add_file);
    fail_if((cal = read_calibration(ori_file, add_file, NULL)) == NULL);
    fail_unless(compare_calib(cal, &correct_cal));
    
    remove(ori_file);
    remove(add_file);
}
END_TEST

/* Exterior jacobian in the case of no glass or water is easy to verify
   analytically.
*/
START_TEST(test_num_deriv_exterior)
{
    int der;
    
    Calibration cal = {
        .ext_par = {0., 0., 400., 0., 0., 0., 
            {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}} // rotation matrix
        },
        .int_par = {0., 0., 1.},
        .glass_par = {0., 0., 200.},
        .added_par = {0., 0., 0., 0., 0., 1., 0.}
    };
    mm_np trivial_mm = {1., 1., {1., 0., 0.}, {.1, 0., 0.}, 1., 0.};
    vec3d pos = {0., 0., 100.};
    
    double x_ders[6], y_ders[6];
    double dpos = 1., dang = M_PI/6.;
    
    double x_ders_correct[] = {-1./300, 0., 0., 0., tan(dang)/dang, 0.};
    double y_ders_correct[] = {0., -1./300, 0., -tan(dang)/dang, 0., 0.};
    
    num_deriv_exterior(0, cal.ext_par, cal.int_par, cal.glass_par, cal.added_par,
        trivial_mm, dpos, dang, pos, x_ders, y_ders);
    
    for (der = 0; der < 6; der++) {
        fail_unless(fabs(x_ders[der] - x_ders_correct[der]) < 1e-6);
        fail_unless(fabs(y_ders[der] - y_ders_correct[der]) < 1e-6);
    }
}
END_TEST

START_TEST(test_prepshake)
{
    
    /* replace res/ with a fuller results dir. This will be undone 
       after the test.
    */
    chdir("testing_fodder/");
    control_par *cpar = read_control_par("parameters/ptv.par");
    fail_unless(rename("res", "_res") == 0);
    symlink("sample_res", "res");
    
    prepare_eval_shake(cpar);
    for (int i = 0; i < nfix; i++) {
        fail_unless(crd[0][i].pnr == fix[i].pnr);
    }
    
    /* undo res switch. */
    unlink("res");
    fail_unless(rename("_res", "res") == 0);
}
END_TEST

Suite* ptv_suite(void) {
    Suite *s = suite_create ("PTV");

    TCase *tc_rori = tcase_create ("Read orientation file");
    tcase_add_test(tc_rori, test_read_ori);
    suite_add_tcase (s, tc_rori);

    TCase *tc_wori = tcase_create ("Write orientation file");
    tcase_add_test(tc_wori, test_write_ori);
    suite_add_tcase (s, tc_wori);

    TCase *tc = tcase_create ("Milkshake brings all boys to yard");
    tcase_add_test(tc, test_prepshake);
    suite_add_tcase (s, tc);

    TCase *tc_numder = tcase_create ("Numeric derivative of an Exterior object");
    tcase_add_test(tc_numder, test_num_deriv_exterior);
    suite_add_tcase (s, tc_numder);
    
    return s;
}

int main(void) {
    int number_failed;
    Suite *s = ptv_suite ();
    SRunner *sr = srunner_create (s);
    srunner_run_all (sr, CK_VERBOSE);
    number_failed = srunner_ntests_failed (sr);
    srunner_free (sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

