"""Test the reading of the parameters from a yaml file."""
from typing import Dict

import yaml

# dictionary that connects the .par file names and the respective parameter classes
from openptv_python.parameters import (
    CalibrationPar,
    ControlPar,
    MultimediaPar,
    MultiPlanesPar,
    OrientPar,
    SequencePar,
    TargetPar,
    TrackPar,
    VolumePar,
)


def read_parameters_from_yaml(file_path):
    """Read the parameters from a yaml file and returns a dictionary with the parameters."""
    with open(file_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params


def write_parameters_to_yaml(file_path, params: Dict):
    """Write the parameters to a yaml file."""
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(params, file, default_flow_style=False)


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
    "mm": MultimediaPar,
}
# par_dict


parameters = read_parameters_from_yaml(
    "tests/testing_fodder/parameters/merged_parameters.yaml"
)

for key in parameters:
    par_class = par_dict[key]
    instance = par_class().from_dict(parameters[key])
    print(instance)
    print(instance.to_dict())
