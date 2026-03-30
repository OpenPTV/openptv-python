import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import yaml


    def read_parameters_from_yaml(file_path):
        with open(file_path, "r") as file:
            parameters = yaml.safe_load(file)
        return parameters


    def write_parameters_to_yaml(file_path, parameters):
        with open(file_path, "w") as file:
            yaml.dump(parameters, file, default_flow_style=False)


    # Assuming the parameters are stored in a dictionary
    parameters = {
        "param1": 4,
        "param2": [
            "dumbbell/cam1_Scene77_4085",
            "cal/cam1.tif",
            "dumbbell/cam2_Scene77_4085",
            "cal/cam2.tif",
            "dumbbell/cam3_Scene77_4085",
            "cal/cam3.tif",
            "dumbbell/cam4_Scene77_4085",
            "cal/cam4.tif",
        ],
        "param3": [1, 0, 1, 1280, 1024, 0.017, 0.017, 0, 1, 1.49, 1.33, 5],
    }

    # Example paths for the parameter file
    parameter_file_path = "parameters.yaml"

    # Writing parameters to YAML file
    write_parameters_to_yaml(parameter_file_path, parameters)

    # Reading parameters from YAML file
    read_parameters = read_parameters_from_yaml(parameter_file_path)

    # Displaying the read parameters
    print("Read parameters:")
    print(read_parameters)
    return (yaml,)


@app.cell
def _(yaml):
    class Parameters:

        def __init__(self, param1, param2, param3):
            self.param1 = param1
            self.param2 = param2
            self.param3 = param3
    parameters_instance = Parameters(param1=4, param2=['dumbbell/cam1_Scene77_4085', 'cal/cam1.tif', 'dumbbell/cam2_Scene77_4085', 'cal/cam2.tif', 'dumbbell/cam3_Scene77_4085', 'cal/cam3.tif', 'dumbbell/cam4_Scene77_4085', 'cal/cam4.tif'], param3=[1, 0, 1, 1280, 1024, 0.017, 0.017, 0, 1, 1.49, 1.33, 5])
    # Create an instance of the Parameters class
    parameters_dict = {'param1': parameters_instance.param1, 'param2': parameters_instance.param2, 'param3': parameters_instance.param3}
    yaml_file_path = 'parameters_instance.yaml'
    with open(yaml_file_path, 'w') as _yaml_file:
        yaml.dump(parameters_dict, _yaml_file, default_flow_style=False)
    with open(yaml_file_path, 'r') as _yaml_file:
        _read_parameters_dict = yaml.safe_load(_yaml_file)
    print('Read parameters from YAML file:')
    # Convert the instance to a dictionary
    # Define the path for the YAML file
    # Write the class instance to the YAML file
    # Read the YAML file back into a dictionary
    # Display the read parameters
    print(_read_parameters_dict)
    return


@app.cell
def _():
    from openptv_python.parameters import ControlPar

    cpar = ControlPar()
    return ControlPar, cpar


@app.cell
def _(cpar):
    cpar.from_file("../tests/testing_fodder/parameters/ptv.par")
    return


@app.cell
def _(cpar):
    cpar
    return


@app.cell
def _(cpar, yaml):
    yaml_file_path_1 = 'ptv_par.yaml'
    with open(yaml_file_path_1, 'w') as _yaml_file:
        yaml.dump(cpar.to_dict(), _yaml_file, default_flow_style=False)
    return (yaml_file_path_1,)


@app.cell
def _(cpar):
    cpar.__dict__
    return


@app.cell
def _(cpar):
    cpar.to_dict()
    return


@app.cell
def _(yaml, yaml_file_path_1):
    with open(yaml_file_path_1, 'r') as _yaml_file:
        _read_parameters_dict = yaml.safe_load(_yaml_file)
    print('Read parameters from YAML file:')
    print(_read_parameters_dict)
    return


@app.cell
def _(yaml):
    def merge_yaml_files(file1_path, file2_path, merged_file_path, title1, title2):
        # Read YAML files
        with open(file1_path, "r") as file1:
            data1 = yaml.safe_load(file1)
        with open(file2_path, "r") as file2:
            data2 = yaml.safe_load(file2)

        # Create a dictionary with titles
        merged_data = {title1: data1, title2: data2}

        # Write merged data to a new YAML file
        with open(merged_file_path, "w") as merged_file:
            yaml.dump(merged_data, merged_file, default_flow_style=False)


    # Example paths for the YAML files
    yaml_file1_path = "parameters.yaml"
    yaml_file2_path = "ptv_par.yaml"
    merged_yaml_file_path = "merged_file.yaml"

    # Example titles for the YAML files
    title1 = "Title1"
    title2 = "Title2"

    # Merge the YAML files
    merge_yaml_files(
        yaml_file1_path, yaml_file2_path, merged_yaml_file_path, title1, title2
    )

    # Read the merged YAML file back into a dictionary
    with open(merged_yaml_file_path, "r") as merged_file:
        read_merged_data = yaml.safe_load(merged_file)

    # Display the read merged data
    print(f"Read merged data from YAML file with titles {title1} and {title2}:")
    print(read_merged_data)
    return


@app.cell
def _(ControlPar):
    # Example usage
    control_par_dict = {
        "num_cams": 4,
        "img_base_name": [
            "dumbbell/cam1_Scene77_4085",
            "dumbbell/cam2_Scene77_4085",
            "dumbbell/cam3_Scene77_4085",
            "dumbbell/cam4_Scene77_4085",
        ],
        "cal_img_base_name": [
            "cal/cam1.tif",
            "cal/cam2.tif",
            "cal/cam3.tif",
            "cal/cam4.tif",
        ],
        "hp_flag": 1,
        "all_cam_flag": 0,
        "tiff_flag": 1,
        "imx": 1280,
        "imy": 1024,
        "pix_x": 0.017,
        "pix_y": 0.017,
        "chfield": 0,
        "mm": {"nlay": 1, "n1": 1.0, "n2": [1.49], "d": [5.0], "n3": 1.33},
    }

    # Convert the dictionary back to ControlPar instance
    control_par_instance = ControlPar.from_dict(control_par_dict)

    # Print the resulting ControlPar instance
    print(control_par_instance)
    return


if __name__ == "__main__":
    app.run()
