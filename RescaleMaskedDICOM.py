from pydicom import dcmread
import numpy as np
import os


def load_dicom_series(dicom_folder):
    dicom_files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    dicom_files.sort()  # Make sure to sort the files in the correct order
    dicom_series = [dcmread(f) for f in dicom_files]
    dicom_array = np.stack([ds.pixel_array for ds in dicom_series])
    return dicom_array, dicom_series


def simple_rescale_dicom_array_to_16bit(dicom_array):
    # Next line we rescale the array by shifting down by the minimum value in the array, so that the lowest value is 0
    min_value = np.min(dicom_array)
    rescaled_array = np.copy(dicom_array)
    if min_value > 0:
        rescaled_array -= min_value
    if min_value < 0:
        rescaled_array[rescaled_array < 0] = 0

    return rescaled_array.astype(np.uint16)  # Convert to 16-bit integer


def save_rescaled_dicom(rescaled_array, original_series, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, ds in enumerate(original_series):
        slice_array = rescaled_array[i, :, :].astype(np.int16)
        ds.PixelData = slice_array.tobytes()
        ds.save_as(os.path.join(output_folder, f"rescaled_{i}.dcm"), write_like_original=True)


def main():
    input_dicom_folder = "C:/Grace/Lab/variable-density-bioprinting-py/data/00000304"
    output_dicom_folder = "C:/Grace/Lab/variable-density-bioprinting-py/data/410RMDtest"

    # Load the masked DICOM series
    masked_dicom_array, masked_dicom_series = load_dicom_series(input_dicom_folder)

    # Rescale the DICOM array
    # rescaled_dicom_array = rescale_dicom_array(masked_dicom_array, new_min, new_max)
    # rescaled_dicom_array = rescale_dicom_array_to_16bit(masked_dicom_array, masked_dicom_series)
    rescaled_dicom_array = simple_rescale_dicom_array_to_16bit(masked_dicom_array)

    # Save the rescaled DICOM series
    save_rescaled_dicom(rescaled_dicom_array, masked_dicom_series, output_dicom_folder)


if __name__ == "__main__":
    main()
