import numpy as np
import os
import pydicom
from RescaleMaskedDICOM import simple_rescale_dicom_array_to_16bit


def load_dicom_series(folder_path):
    dicom_files = [pydicom.dcmread(os.path.join(folder_path, f), force=True) for f in os.listdir(folder_path)]
    dicom_files = [df for df in dicom_files if hasattr(df, 'ImagePositionPatient')]
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    dicom_array = np.stack([df.pixel_array for df in dicom_files])
    return dicom_array, dicom_files


def save_dicom_series(dicom_array, original_series, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Undoing the transpose from load_dicom_series
    curr_index = 0
    for dicom_file in original_series:
        if hasattr(dicom_file, 'ImagePositionPatient'):
            dicom_file.PixelData = dicom_array[curr_index, :, :].tobytes()
            dicom_file.save_as(os.path.join(output_folder, f"slice_{curr_index}.dcm"))
            curr_index += 1


def save_dicom_series_removing_empty_slices(dicom_array, original_series, output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    curr_index = 0  # Index for the DICOM files
    saved_index = 0  # Index for the saved files without empty slices

    for dicom_file in original_series:
        current_slice = dicom_array[curr_index, :, :]
        if np.any(current_slice):
            dicom_file.PixelData = current_slice.tobytes()
            dicom_file.save_as(os.path.join(output_folder, f"slice_{saved_index}.dcm"))
            saved_index += 1
        curr_index += 1


def main():
    input_folder = (
        "/Users/krunalpatel/Medicine/Rajapakse_Lab/BioPrinting/dicom2gcode/data/Example_Spinal_Cage/0000E5C9")
    output_folder = (
        "/Users/krunalpatel/Medicine/Rajapakse_Lab/BioPrinting/dicom2gcode/data/Example_Spinal_Cage"
        "/0000E5C9_rescaled_averaged")
    rescaled_mask_output_folder = ("/Users/krunalpatel/Medicine/Rajapakse_"
                                   "Lab/BioPrinting/dicom2gcode/data/2_inch_femur_segment/"
                                   "rescaled_mask")
    density_output_folder = ("/Users/krunalpatel/Medicine/Rajapakse_"
                             "Lab/BioPrinting/dicom2gcode/data/density_masked_femur_4")
    mask_dicom_folder = ("/Users/krunalpatel/Medicine/Rajapakse_"
                         "Lab/BioPrinting/dicom2gcode/data/Example_Spinal_Cage/masked")
    nifti_file = ("/Users/krunalpatel/Medicine/Rajapakse_"
                  "Lab/BioPrinting/dicom2gcode/data/Example_Spinal_Cage/Segmentation_0000E5C9.nii")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load DICOM series and arrange pixel data
    dicom_array, original_series = load_dicom_series(input_folder)

    mask_dicom_array, mask_dicom_series = load_dicom_series(mask_dicom_folder)

    rescaled_dicom_array = simple_rescale_dicom_array_to_16bit(dicom_array)
    rescaled_mask_array = simple_rescale_dicom_array_to_16bit(mask_dicom_array)
    # save_dicom_series(rescaled_mask_array, original_series, rescaled_mask_output_folder)

    # Process through threshold_and_skip
    # processed_array = threshold_and_skip_within_mask(rescaled_dicom_array, n_tiles=4, mask_tensor=rescaled_mask_array)
    # processed_array = local_average_density(rescaled_dicom_array, rescaled_mask_array)
    # processed_array = local_average_density(rescaled_dicom_array, nifti_array)
    # processed_array = local_averaging_with_edge_detection(rescaled_dicom_array, rescaled_mask_array)
    # processed_array = local_averaging_with_canny_edge_preservation_3d(dicom_array, mask_dicom_array)
    # processed_array = local_averaging_with_mask_edge_preservation_on_gpu(rescaled_dicom_array, nifti_array)
    # processed_array = local_averaging_with_mask_edge_preservation_on_m1_hardware(rescaled_dicom_array, nifti_array)

    # Save as DICOM series
    # save_dicom_series(processed_array, original_series, output_folder)


if __name__ == "__main__":
    main()
