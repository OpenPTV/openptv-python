import os
import shutil
import unittest
from pathlib import Path

from openptv_python.calibration import Calibration, read_calibration
from openptv_python.parameters import read_control_par
from openptv_python.track import (
    track_forward_start,
    trackcorr_c_finish,
    trackcorr_c_loop,
)
from openptv_python.tracking_run import tr_new


def remove_directory(directory_path):
    """Remove a directory and its contents."""
    # Convert the input to a Path object
    path = Path(directory_path)

    # Iterate over all items in the directory
    for item in path.iterdir():
        if item.is_file():
            # Remove files
            item.unlink()
        elif item.is_dir():
            # Remove subdirectories and their contents
            remove_directory(item)

    # Remove the directory itself
    path.rmdir()


def copy_directory(source_path, destination_path):
    """Copy the contents of the source directory to the destination directory."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)

    # Create the destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)

    # Copy the contents from the source to the destination
    for item in source_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, destination_path / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination_path / item.name)


def read_all_calibration(num_cams: int = 4) -> list[Calibration]:
    """Read all calibration files."""
    ori_tmpl = "cal/cam%d.tif.ori"
    added_tmpl = "cal/cam%d.tif.addpar"

    calib = []

    for cam in range(num_cams):
        ori_name = ori_tmpl % (cam + 1)
        added_name = added_tmpl % (cam + 1)
        calib.append(read_calibration(Path(ori_name), Path(added_name)))

    return calib


class TestBurgers(unittest.TestCase):
    """Test the Burgers vortex case."""

    def test_burgers(self):
        """Test the Burgers vortex case."""
        current_directory = Path.cwd()
        print(f"working from {current_directory}")
        directory = Path("tests/testing_fodder/burgers")
        parameters_path = (directory / "parameters").resolve(strict=True)

        os.chdir(directory)

        # print(os.path.abspath(os.curdir))
        # # print(Path.cwd())

        if Path("res/").exists():
            remove_directory("res/")

        if Path("img/").exists():
            remove_directory("img/")

        copy_directory("res_orig/", "res/")
        copy_directory("img_orig/", "img/")

        cpar = read_control_par(parameters_path / "ptv.par")
        self.assertIsNotNone(cpar)

        calib = read_all_calibration(cpar.num_cams)

        print("----------------------------")
        print("Test Burgers vortex case")

        run = tr_new(
            parameters_path / "sequence.par",
            parameters_path / "track.par",
            parameters_path / "criteria.par",
            parameters_path / "ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            10000.0,
        )

        # print("num cams in run is", run.cpar.num_cams)
        # print("add particle is", run.tpar.add)

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        # print(f"total num parts is {run.npart}, num links is {run.nlinks}")

        self.assertEqual(
            run.npart, 19, f"Was expecting npart == 19 but found {run.npart}"
        )
        self.assertEqual(
            run.nlinks, 17, f"Was expecting nlinks == 17 but found {run.nlinks}"
        )

        run = tr_new(
            parameters_path / "sequence.par",
            parameters_path / "track.par",
            parameters_path / "criteria.par",
            parameters_path / "ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            10000.0,
        )

        # run.tpar = run.tpar._replace(add=1)
        run.tpar = run.tpar._replace(add=1)
        print("changed add particle to", run.tpar.add)

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)

        trackcorr_c_finish(run, run.seq_par.last)
        print(f"total num parts is {run.npart}, num links is {run.nlinks}")

        self.assertEqual(
            run.npart, 20, f"Was expecting npart == 19 but found {run.npart}"
        )
        self.assertEqual(
            run.nlinks, 20, f"Was expecting nlinks == 17 but found {run.nlinks}"
        )

        remove_directory("res/")
        remove_directory("img/")

        os.chdir(current_directory)


if __name__ == "__main__":
    unittest.main()
