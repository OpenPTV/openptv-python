import pathlib
from dataclasses import asdict

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

par_dict = {'cal_ori': CalibrationPar,
            'detect_plate':TargetPar,
            'track': TrackPar,
            'multi_planes':MultiPlanesPar,
            'ptv': ControlPar,
            'criteria':VolumePar,
            'sequence':SequencePar,
            'targ_rec': TargetPar,
            'orient' : OrientPar
            }

# print(par_dict)

# Define the directory containing the .par files
directory_path = pathlib.Path('tests/testing_fodder/parameters')
output_yaml_file = 'tests/testing_fodder/parameters/merged_parameters.yaml'

# Initialize an empty dictionary to store the merged data
merged_data = {}

ptv_par = directory_path / 'ptv.par'
par_class = par_dict['ptv']
par_content = par_class().from_file(ptv_par)
num_cams = par_content.num_cams



# Find all .par files in the specified directory
par_files = directory_path.glob('*.par')
# print(par_files)

# Iterate through each .par file
for file_path in par_files:
    # Extract the title from the file name (assuming the file name is something like "title.par")
    title = file_path.name
    # print(title)

    # Read the content of the .par file and convert it to a dictionary
    # with open(file_path, 'r') as file:
    #     par_content = yaml.safe_load(file)

    # Here we need to read the content into a respective class and
    # store it with the paramater name as the key

    if title in par_dict:
        par_class = par_dict[title]
        if par_class is CalibrationPar or par_class is SequencePar:
            par_content = par_class().from_file(file_path, num_cams)
        else:
            par_content = par_class().from_file(file_path)
        # par_content = par_class(**par_content)
        # print(par_content)

        # Add the dictionary to the merged data with the title as the key
        merged_data[title] = asdict(par_content)

# Write the merged data to a single YAML file
with open(output_yaml_file, 'w', encoding='utf-8') as output_file:
    yaml.dump(merged_data, output_file, default_flow_style=False)

print(f'Merged data written to {output_yaml_file}')
