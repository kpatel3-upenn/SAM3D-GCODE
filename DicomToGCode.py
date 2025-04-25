import subprocess
import os
import xml.etree.ElementTree as ET


def generate_params_xml(file_path, folder_name, output_file,
                        xy_res=0.2, z_res=0.1, extrusion_res=0.0011,
                        moving_speed=900, extruding_speed=360, crosshatch=True,
                        angle=0, plane=0, fill=1):
    # Create the root element
    root = ET.Element("parameters")

    # Define the parameters with defaults
    params = {
        "folderName": folder_name,
        "outputFile": output_file,
        "xyRes": str(xy_res),
        "zRes": str(z_res),
        "extrusionRes": str(extrusion_res),
        "movingSpeed": str(moving_speed),
        "extrudingSpeed": str(extruding_speed),
        "crosshatch": str(crosshatch).lower(),
        "angle": str(angle),
        "plane": str(plane),  # 0=XY 1=YZ 2=XZ
        "fill": str(fill)
    }

    # Add parameters as sub-elements
    for key, value in params.items():
        element = ET.SubElement(root, key)
        element.text = value

    # Create a tree object
    tree = ET.ElementTree(root)

    # Write the tree to an XML file
    tree.write(file_path, encoding='utf-8', xml_declaration=True)


def call_dicom_2_gcode_java(java_code_path, params_file_path):
    # Ensure the paths are absolute
    java_code_path = os.path.abspath(java_code_path)
    params_file_path = os.path.abspath(params_file_path)

    # Change the current working directory to the directory containing the params.xml file
    params_dir = os.path.dirname(params_file_path)

    # Compile the Java program if the .class file doesn't exist
    class_file = os.path.join(java_code_path, "Dicom2GCode.class")
    if not os.path.exists(class_file):
        print(f"{class_file} not found, compiling the Java program...")

        compile_command = [
            "javac",
            "-classpath", f"{java_code_path}:{os.path.join(java_code_path, 'ij.jar')}",
            os.path.join(java_code_path, "Dicom2GCode.java")
        ]

        try:
            compile_process = subprocess.run(compile_command, check=True)
            print("Compilation successful!")
        except subprocess.CalledProcessError as e:
            print("Compilation failed!", e)
            return
    else:
        print(f"{class_file} already exists, skipping compilation.")

    # Run the Java program from the directory containing the params.xml file
    run_command = [
        "java",
        "-classpath", f"{java_code_path}:{os.path.join(java_code_path, 'ij.jar')}",
        "Dicom2GCode",
        "params.xml"  # Use the relative path "params.xml" since the working directory is changed
    ]

    try:
        run_process = subprocess.run(run_command, cwd=params_dir, check=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        print("Output:", run_process.stdout.decode())
        if run_process.stderr:
            print("Error:", run_process.stderr.decode())
    except subprocess.CalledProcessError as e:
        print("Running the Java program failed!", e)
        print("Error output:", e.stderr.decode())

