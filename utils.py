import numpy as np
from PIL import Image
import os
import math
import mrcfile
import pydicom as dicom
import matplotlib.pyplot as plt


def _dicom_zcoord(ds):
    if 'ImagePositionPatient' in ds:
        return float(ds.ImagePositionPatient[2])
    if 'SliceLocation' in ds:
        return float(ds.SliceLocation)
    return 0.0

def load3dmatrix(folder, datatype):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.%s' % datatype)]
    filepaths.sort()
    if datatype == "png":
        images = [Image.open(f) for f in filepaths]
    if datatype == "dcm":
        filepaths = sorted(
            filepaths, 
            key=lambda fp: _dicom_zcoord(dicom.dcmread(fp, stop_before_pixels=True)))
        images = [dicom.dcmread(f).pixel_array for f in filepaths]
    image = np.stack(images, axis=-1)
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image)) * 255
    return image


def load_dicom_series(folder_path):
    dicom_files = [dicom.dcmread(os.path.join(folder_path, f), force=True) for f in os.listdir(folder_path)]
    dicom_files = [df for df in dicom_files if hasattr(df, 'ImagePositionPatient')]
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    dicom_array = np.stack([df.pixel_array for df in dicom_files])
    return dicom_array, dicom_files


def padtocube(array):
    shape = array.shape
    max_dim = max(shape)
    left_pad1 = (max_dim - shape[0]) // 2
    right_pad1 = max_dim - shape[0] -  left_pad1
    left_pad2 = (max_dim - shape[1]) // 2
    right_pad2 = max_dim - shape[1] -  left_pad2
    left_pad3 = (max_dim - shape[2]) // 2
    right_pad3 = max_dim - shape[2] -  left_pad3
    padded_array = np.pad(array, ((left_pad1, right_pad1), (left_pad2, right_pad2), (left_pad3, right_pad3)), mode='constant', constant_values=0)
    return padded_array


def remove_symmetrical_cube_padding(original_shape, padded_array):
    x_start = (padded_array.shape[0] - original_shape[0]) // 2 if padded_array.shape[0] > original_shape[0] else 0
    y_start = (padded_array.shape[1] - original_shape[1]) // 2 if padded_array.shape[1] > original_shape[1] else 0
    z_start = (padded_array.shape[2] - original_shape[2]) // 2 if padded_array.shape[2] > original_shape[2] else 0
    
    unpadded_array = padded_array[
        x_start:x_start + original_shape[0],
        y_start:y_start + original_shape[1],
        z_start:z_start + original_shape[2]
    ]

    return unpadded_array


def extract_dicom_metadata(folder):
    filepaths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.dcm')]
    filepaths.sort()
    
    first_dicom = dicom.dcmread(filepaths[0])
    pixel_spacing = first_dicom.PixelSpacing
    pixel_width = first_dicom.PixelWidth
    pixel_height = first_dicom.PixelHeight
    
    pixel_depth = None
    if hasattr(first_dicom, 'SliceThickness'):
        pixel_depth = first_dicom.SliceThickness
    elif hasattr(first_dicom, 'SpacingBetweenSlices'):
        pixel_depth = first_dicom.SpacingBetweenSlices

    x_size = int(first_dicom.Columns)
    y_size = int(first_dicom.Rows)
    z_size = len(filepaths)

    metadata = {
        'pixel_spacing': pixel_spacing,
        'pixel_width': pixel_width,
        'pixel_height': pixel_height,
        'pixel_depth': pixel_depth,
        'x_size': x_size,
        'y_size': y_size,
        'z_size': z_size
    }

    return metadata


def save_mrc(array, filepath):
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(array.astype(np.float32))


def load_mrc(filepath):
    with mrcfile.open(filepath) as mrc:
        mrc_data = mrc.data
        array = np.array(mrc_data)
    return array