import numpy as np
import os
from VariableDensityAlgorithms import load_dicom_series, save_dicom_series
from RescaleMaskedDICOM import simple_rescale_dicom_array_to_16bit


def create_region_based_variable_density_roi_fractional_skip_with_block_size(bone_roi, low_density_roi, block_size=2):
    output_tensor = np.copy(bone_roi)

    high_density = np.mean(bone_roi[(low_density_roi <= 0) & (bone_roi > 0)])
    low_density = np.mean(bone_roi[low_density_roi > 0])

    density_proportion = low_density / high_density if high_density > 0 else 0
    skip_fraction = (1 - density_proportion)

    accumulated_skip = 0.0

    for x in range(0, bone_roi.shape[2], block_size):
        for y in range(0, bone_roi.shape[1], block_size):
            accumulated_skip += skip_fraction

            if accumulated_skip >= 1.0:
                x_end = min(x + block_size, bone_roi.shape[2])
                y_end = min(y + block_size, bone_roi.shape[1])
                low_density_block_mask = low_density_roi[:, y:y_end, x:x_end] > 0
                output_tensor[:, y:y_end, x:x_end][low_density_block_mask] = 0

                accumulated_skip -= 1.0  # Reset part of the accumulated value for skipping

    return output_tensor


if __name__ == '__main__':
    # Example usage
    original_input_folder = (
        "/Users/krunalpatel/Medicine/Rajapakse_Lab/BioPrinting/dicom2gcode/data/00000304")
    roi_input_folder = (
        "/Users/krunalpatel/Medicine/Rajapakse_Lab/BioPrinting/dicom2gcode/data/2_inch_femur_segment/"
        "roi_from_segmentation")
    low_density_roi_input_folder = (
        "/Users/krunalpatel/Medicine/Rajapakse_Lab/BioPrinting/dicom2gcode/data/2_inch_femur_segment"
        "/low_density_roi_mask")
    output_folder = (
        "/Users/krunalpatel/Medicine/Rajapakse_Lab/BioPrinting/dicom2gcode/data/2_inch_femur_segment"
        "/variable_density_roi")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    roi_dicom_array, original_roi_dicom_series = load_dicom_series(roi_input_folder)
    low_density_roi_dicom_array, original_low_density_roi_dicom_series = load_dicom_series(low_density_roi_input_folder)

    region_based_roi = create_region_based_variable_density_roi_fractional_skip_with_block_size(
        roi_dicom_array,
        simple_rescale_dicom_array_to_16bit(low_density_roi_dicom_array)
    )
    save_dicom_series(region_based_roi, original_roi_dicom_series, output_folder)
