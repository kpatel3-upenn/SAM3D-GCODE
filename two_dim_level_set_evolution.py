import numpy as np
import os
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_gradient_magnitude, binary_closing
from VariableDensityAlgorithms import load_dicom_series, save_dicom_series


def crop_to_bounding_box(volume, padding=5):
    """
    Crop the 3D volume to the bounding box of non-zero elements with added padding.

    Parameters:
    - volume: 3D numpy array representing the volume.
    - padding: Integer value specifying the padding amount to add around the bounding box.

    Returns:
    - cropped_volume: The cropped volume with added padding.
    - min_coords: Tuple of minimum z, y, x coordinates of the bounding box before padding.
    - max_coords: Tuple of maximum z, y, x coordinates of the bounding box before padding.
    """
    non_empty = np.where(volume != 0)
    min_z, max_z = non_empty[0].min(), non_empty[0].max()
    min_y, max_y = non_empty[1].min(), non_empty[1].max()
    min_x, max_x = non_empty[2].min(), non_empty[2].max()

    # Apply padding, ensuring indices are within the volume bounds
    min_z = max(min_z - padding, 0)
    max_z = min(max_z + padding, volume.shape[0] - 1)
    min_y = max(min_y - padding, 0)
    max_y = min(max_y + padding, volume.shape[1] - 1)
    min_x = max(min_x - padding, 0)
    max_x = min(max_x + padding, volume.shape[2] - 1)

    cropped_volume = volume[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
    return cropped_volume, (min_z, min_y, min_x), (max_z, max_y, max_x)


def two_d_gradient_based_speed_function(image):
    """
    Defines a speed function for level set evolution that focuses on areas of maximum gradient magnitude.

    Parameters:
    - image: The input image (2D or 3D) where the level set evolution is performed.

    Returns:
    - A speed function (same shape as 'image') that promotes evolution in areas of high gradient magnitude.
    """
    # Compute the gradients of the image
    grad_x, grad_y = np.gradient(image)
    image_gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Here, we invert the gradient magnitude to slow down the evolution in high gradient areas,
    # aiming to "stop" the level set evolution at these boundaries.
    # Note: Adding a small constant (epsilon) to avoid division by zero.
    epsilon = 1e-5
    F = 1 / (1 + image_gradient_magnitude + epsilon)

    return F


def reinitialize_phi(phi, iterations=10, dt=0.1):
    """
    Reinitialize the level set function phi to a signed distance function using an iterative method.

    Parameters:
    - phi: The level set function to be reinitialized.
    - iterations: Number of iterations to perform.
    - dt: Time step for the iterative reinitialization process.

    Returns:
    - Reinitialized level set function.
    """
    # Compute the signed distance transform of the positive and negative regions of phi
    phi_positive = distance_transform_edt(phi > 0)  # Distance transform of the positive region
    phi_negative = distance_transform_edt(phi < 0)  # Distance transform of the negative region

    # Compute the initial signed distance function
    phi_sdf = phi_positive - phi_negative

    # Iteratively reinitialize phi
    phi_reinitialized = phi_sdf.copy()
    for _ in range(iterations):
        grad_phi = np.gradient(phi_reinitialized)
        grad_phi_norm = np.sqrt(sum(g ** 2 for g in grad_phi))

        # Update phi using the reinitialization equation approximation
        phi_reinitialized -= dt * (grad_phi_norm - 1) * np.sign(phi_sdf)

    return phi_reinitialized


def normalize_image(image):
    """
    Normalize the image to have values between 0 and 1.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


def raster_fill_slice_high_density_regions(phi_mask, epsilon):
    """
    Fill high density region

    Parameters:
    - phi_mask: The phi resulting from 2d level set evolution - negative values represent outline/border
    - epsilon: threshold for border vs region values

    Returns:
    - Modified mask slice with high density regions as 0 and everything else as 1
    """
    filled_slice = np.full_like(phi_mask, fill_value=1, dtype=np.float32)

    for i in range(phi_mask.shape[0]):
        mask_row = phi_mask[i, :]
        # high density from left

        # find left edge of starting border
        if np.any((mask_row < epsilon)):
            start_left = np.argmax((mask_row < epsilon))
        else:
            continue

        # find right edge of starting border
        if np.any((mask_row[start_left + 1:] >= epsilon)):
            start_right = start_left + np.argmax((mask_row[start_left + 1:] >= epsilon))
        else:
            continue

        # find left edge of ending border
        if np.any((mask_row[start_right + 1:] < epsilon)):
            end_left = start_right + 1 + np.argmax((mask_row[start_right + 1:] < epsilon))
        else:
            continue

        # find right edge of ending border
        if np.any((mask_row[end_left + 1:] >= epsilon)):
            end_right = end_left + np.argmax((mask_row[end_left + 1:] >= epsilon))
        else:
            continue

        filled_slice[i, start_left:end_right] = 0

        # high density from right
        # find right edge of starting border
        if np.any(mask_row[phi_mask.shape[1] - 1:end_left - 2:-1] < epsilon):
            r_start_right = phi_mask.shape[1] - 1 - np.argmax(mask_row[phi_mask.shape[1] - 1:end_left - 2:-1] < epsilon)
        else:
            continue

        # find left edge of starting border
        if np.any((mask_row[r_start_right - 1:end_left - 2:-1] >= epsilon)):
            r_start_left = r_start_right - np.argmax(mask_row[r_start_right - 1:end_left - 2:-1] >= epsilon)
        else:
            continue

        # find right edge of ending border
        if np.any((mask_row[r_start_left - 1:end_left - 2:-1] < epsilon)):
            r_end_right = r_start_left - 1 - np.argmax((mask_row[r_start_left - 1:end_left - 2:-1] < epsilon))
        else:
            continue

        # find left edge of ending border
        if np.any((mask_row[r_end_right - 1:end_left - 2:-1] >= epsilon)):
            r_end_left = r_end_right - np.argmax((mask_row[r_end_right - 1:end_left - 2:-1] >= epsilon))
        else:
            continue

        filled_slice[i, r_end_left:r_start_right] = 0

        # low density
        if r_end_left > end_right:
            filled_slice[i, end_right: r_end_left] = 2

    return filled_slice


def raster_fill_high_density_regions(phi_mask, epsilon=0.0):
    """
        Apply the raster fill operation to each slice in the volume based on a corresponding mask.

        Parameters:
        - phi_mask:  3D mask
        - epsilon: threshold for the fill operation

        Returns:
        - The modified 3D volume with filled values.
        """
    output_mask = np.full_like(phi_mask, fill_value=1)
    for slice_index in range(phi_mask.shape[0]):
        output_mask[slice_index, :, :] = raster_fill_slice_high_density_regions(phi_mask[slice_index, :, :], epsilon)
    return output_mask


def evolve_level_set(image, iterations, speed_function, regularize_surface, timestep=0.05):
    # Crop the image and initial surface to the bounding box of the initial surface
    cropped_image, min_coords, max_coords = crop_to_bounding_box(image, padding=0)
    cropped_image = normalize_image(cropped_image)
    # cropped_image = raster_fill_volume(cropped_image)

    cropped_phi = np.full_like(cropped_image, fill_value=1, dtype=np.float32)
    for slice_index in range(cropped_image.shape[0]):
        cropped_image_slice = cropped_image[slice_index, :, :]
        phi_slice = initialize_phi_on_gradients(cropped_image[slice_index, :, :], threshold=0.3)
        evolved_phi_slice = two_d_level_set_evolution(
            cropped_image_slice,
            phi_slice,
            iterations,
            speed_function,
            regularize_surface,
            timestep=timestep
        )
        cropped_phi[slice_index, :, :] = evolved_phi_slice

    cropped_raster_phi = raster_fill_high_density_regions(cropped_phi, 0)

    # Prepare an output phi array of the original size filled with a default value (1 because we're treating most
    # of the volume as the background)
    phi = np.full_like(image, fill_value=1, dtype=np.float32)
    min_z, min_y, min_x = min_coords
    max_z, max_y, max_x = max_coords

    raster_phi = np.full_like(image, fill_value=1)
    raster_phi[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = cropped_raster_phi

    # Embed the evolved phi back into the phi array of original size
    phi[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = cropped_phi

    return phi, raster_phi


def two_d_level_set_evolution(image, phi, iterations, speed_function, regularize_shape, timestep=0.05):
    evolved_phi = phi
    for i in range(iterations):
        # Evolution logic on cropped volumes
        dphi_dx, dphi_dy = np.gradient(evolved_phi)
        gradient_magnitude = np.sqrt(dphi_dx ** 2 + dphi_dy ** 2)
        F = speed_function(image)
        evolved_phi = evolved_phi + timestep * F * gradient_magnitude
    # evolved_phi = regularize_shape(evolved_phi)

    return evolved_phi


def initialize_phi_on_gradients(image, sigma=1, threshold=0.3):
    """
    Initialize the level set function based on the gradients of the image.

    Parameters:
    - image: The input image (2D or 3D).
    - sigma: Standard deviation for Gaussian gradient magnitude calculation.
    - threshold: Threshold value to select high-gradient regions.

    Returns:
    - phi: Initialized level set function.
    """
    # Calculate the gradient magnitude using a Gaussian filter for smoothing
    grad_mag = gaussian_gradient_magnitude(image, sigma=sigma)

    # Threshold the gradient magnitude to find regions of interest
    mask = (grad_mag > (threshold * np.max(grad_mag)))

    # Compute the signed distance transform for the interior and exterior of the mask
    phi_inside = -distance_transform_edt(mask)  # Negative inside the mask
    phi_outside = distance_transform_edt(~mask)  # Positive outside the mask

    # Combine inside and outside distances to form phi
    phi = phi_inside + phi_outside

    return phi


def apply_raster_mask(density_mask, image):
    dicom_mask = np.zeros_like(density_mask, dtype=np.uint16)
    dicom_mask[density_mask == 2] = 1
    return np.multiply(image, dicom_mask)


def apply_level_set_evolution_mask(dicom_pixel_array):
    iterations = 5
    evolved_phi, raster_phi = evolve_level_set(
        dicom_pixel_array,
        iterations,
        two_d_gradient_based_speed_function,
        reinitialize_phi
    )
    # This just returns the low density region mask
    return apply_raster_mask(raster_phi, dicom_pixel_array)


if __name__ == "__main__":

    # Example usage
    input_folder = "/Grace/Lab/variable-density-bioprinting-py/data/2_inch_femur_segment/roi_from_segmentation_1"
    # mask_input_folder = "/Grace/Lab/variable-density-bioprinting-py/data/2_inch_femur_segment/masked"
    # density_segmentation_lines = ("/Grace/Lab/variable-density-bioprinting-py/data/2_inch_femur_segment"
    #                               "/2d_density_segmentation_lines")
    output_folder = "/Grace/Lab/variable-density-bioprinting-py/data/2_inch_femur_segment/low_density_roi_mask"
    # density_segmentation_regions_folder = ("/Grace/Lab/variable-density-bioprinting-py/data/2_inch_femur_segment"
    #                                       "/2d_density_segmentation_regions")

    # if not os.path.exists(density_segmentation_lines):
    #     os.mkdir(density_segmentation_lines)

    # if not os.path.exists(density_segmentation_regions_folder):
    #    os.mkdir(density_segmentation_regions_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    dicom_array, original_dicom_series = load_dicom_series(input_folder)
    # mask_array, mask_original_series = load_dicom_series(mask_input_folder)

    # threshold = 0.3
    # initial_phi = initial_phi_viewer(dicom_array, threshold)
    # density_segmentation_regions_initial = create_density_segmentation_regions(initial_phi)
    # save_dicom_series(density_segmentation_regions_initial, original_dicom_series,
    #                   "/Grace/Lab/variable-density-bioprinting-py/data/2_inch_femur_segment/initial")
    #
    # initial_surface = create_initial_surface(dicom_array)
    iterations = 5
    evolved_phi, raster_phi = evolve_level_set(
        dicom_array,
        iterations,
        two_d_gradient_based_speed_function,
        reinitialize_phi
    )

    # density_segmentation_line_mask = create_density_segmentation_line_mask(evolved_phi)
    # save_dicom_series(density_segmentation_line_mask, original_dicom_series, density_segmentation_lines)

    # output low_density_roi to be used in RegionBasedVariableDensityAlgorithm
    low_density_roi = apply_raster_mask(raster_phi, dicom_array)
    save_dicom_series(low_density_roi, original_dicom_series, output_folder)
    # '''
    '''
    test_phi = np.array([[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                          [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                          [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                          [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                          [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]]])

    raster_filled_regions = raster_fill_high_density_regions(test_phi, epsilon=0.3)
    raster_mask = convert_raster_mask(raster_filled_regions)
    save_dicom_series(raster_mask, original_dicom_series, "/Grace/Lab/variable-density-bioprinting-py/data"
                                                          "/raster_fill_test")
    '''
