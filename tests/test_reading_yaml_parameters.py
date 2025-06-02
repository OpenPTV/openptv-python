"""Test the reading of the parameters from a yaml file."""

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import yaml

# dictionary that connects the .par file names and the respective parameter classes
from openptv_python.parameters import (
    CalibrationPar,
    ControlPar,
    MultiPlanesPar,
    OrientPar,
    SequencePar,
    TargetPar,
    TrackPar,
    VolumePar,
)


def read_parameters_from_yaml(file_path: Path):
    """Read the parameters from a yaml file and returns a dictionary with the parameters."""
    with open(file_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params


def write_parameters_to_yaml(file_path: Path, params: Dict):
    """Write the parameters to a yaml file."""
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(params, file, default_flow_style=False)


# %%


par_dict = {
    "cal_ori": CalibrationPar,
    "detect_plate": TargetPar,
    "track": TrackPar,
    "multi_planes": MultiPlanesPar,
    "ptv": ControlPar,
    "criteria": VolumePar,
    "sequence": SequencePar,
    "targ_rec": TargetPar,
    "orient": OrientPar,
}
# par_dict


parameters = read_parameters_from_yaml(
    Path("tests/testing_fodder/parameters/merged_parameters.yaml")
)

for key in parameters:
    par_class = par_dict[key]
    instance = par_class().from_dict(parameters[key])
    print(instance)
    print(asdict(instance))
