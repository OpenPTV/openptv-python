# Synthetic Cavity Case

This fixture is a deterministic synthetic variant of `test_cavity` with known ground truth.

Use it for calibration, correspondence, and tracking regression tests.

For the unified project overview, engine selection, and versioning rules, see the top-level [README.md](../../../README.md).

Key contents:

- `cal/`: working calibrations recovered from synthetic calibration-body targets
- `ground_truth/cal/`: exact camera models used to generate the synthetic data
- `calibration_targets/`: synthetic target files for the calibration body
- `img_orig/`: synthetic particle target files for two frames
- `res_orig/`: reference `rt_is`, `ptv_is`, and `added` outputs
- `ground_truth/particles/`: exact 3D particle coordinates per frame
