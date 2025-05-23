import os
import math
import numpy as np
from tqdm import tqdm
from VariableDensityAlgorithms import load_dicom_series


def rotate_coord(x, y, angle_deg):
    rad = math.radians(angle_deg)
    return (
        x * math.cos(rad) + y * math.sin(rad),
        y * math.cos(rad) - x * math.sin(rad),
    )


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def is_between(val, bound1, bound2):
    return bound1 < val < bound2 or bound2 < val < bound1


def get_dicom_resolution(folder):
    _, series = load_dicom_series(folder)
    first = series[0]
    x_res, y_res = map(float, first.PixelSpacing)
    z_res = float(getattr(first, 'SliceThickness', x_res))
    return x_res, y_res, z_res


class GCodeWriter:
    CONNECT = 0.0
    STOP = 1.0

    def __init__(
        self, voxels, x_res, y_res, z_res, output_path,
        print_xy_res=0.2, print_z_res=0.1,
        extrusion_res=0.0011, move_speed=900, extrude_speed=360,
        fill=1.0, rotate_angle_per_layer=0.0
    ):
        """
        :param voxels: input array
        :param x_res: voxel's x resolution
        :param y_res: voxel's y resolution
        :param z_res: voxel's z resolution
        :param output_path: file location to save the output
        :param print_xy_res: printer's x and y resolution
        :param print_z_res: printer's z resolution
        :param extrusion_res: printer's extrusion resolution
        :param move_speed: printer's movement speed
        :param extrude_speed: printer's extrusion speed
        :param fill: intended fill percentage
        :param rotate_angle_per_layer: rotation angle between layers for stability
        """
        # params
        self.input_voxels = voxels
        self.x_res, self.y_res, self.z_res = x_res, y_res, z_res
        self.output_path = output_path
        self.print_z_res = print_z_res
        self.extrusion_res = extrusion_res
        self.move_speed = move_speed
        self.extrude_speed = extrude_speed
        self.fill = fill
        self.angle_step = rotate_angle_per_layer

        # skipping lines and slices based on resolution differences
        self.skip_lines = (print_xy_res / self.x_res) / fill
        self.skip_slices = print_z_res / self.z_res

        # keeps track of the range of pixels written to
        self.x_max = self.y_max = 0
        self.x_min = voxels.shape[2]
        self.y_min = voxels.shape[1]

        # list of lists, one for each layer of all printer paths
        self.paths_per_layer = []
        # current angle
        self.current_angle = 0.0

    def run(self):
        self._prepare_voxel_mask()
        self._extract_paths()
        self._write_gcode()

    def _prepare_voxel_mask(self):
        """
        Preprocesses the original voxel array input
        Pads the array, converts to 0-1 mask, and resizes based on resolutions
        """
        padded = np.pad(self.input_voxels, ((0, 0), (1, 1), (1, 1)), mode='constant')
        padded[padded > 0] = 1

        z_new = int(self.input_voxels.shape[0] // self.skip_slices)
        self.voxels = np.zeros((z_new, *padded.shape[1:]))

        for i in tqdm(range(z_new), desc="Downsampling slices"):
            self.voxels[i] = padded[int(i * self.skip_slices)]

    def _extract_paths(self):
        """
        For each layer, rotates the slice (for stability), extracts paths, and adds them to the paths lists
        """
        for layer in tqdm(self.voxels, desc="Extracting paths"):
            rotated = self._rotate_layer(layer)
            paths = self._plan_layer_paths(rotated)
            self.paths_per_layer.append(paths)
            self.current_angle = (self.current_angle + self.angle_step) % 360

    def _rotate_layer(self, layer):
        """
        Rotates a 2D matrix to self.current_angle using a rotation matrix
        """
        y_indices, x_indices = np.nonzero(layer)
        if len(x_indices) == 0:
            return np.zeros_like(layer)

        coords = np.vstack((x_indices, y_indices)).T
        theta = math.radians(self.current_angle)
        R = np.array([[math.cos(theta), math.sin(theta)],
                      [-math.sin(theta), math.cos(theta)]])

        rotated_coords = coords @ R.T
        x_min, y_min = np.floor(rotated_coords.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(rotated_coords.max(axis=0)).astype(int)

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        result = np.zeros((height, width), dtype=np.uint8)

        # Shift and round coordinates
        shifted_coords = np.round(rotated_coords - [x_min, y_min]).astype(int)
        shifted_coords = shifted_coords[(shifted_coords[:, 1] < height) & (shifted_coords[:, 0] < width)]

        result[shifted_coords[:, 1], shifted_coords[:, 0]] = 1
        self.x_shift = x_min
        self.y_shift = y_min
        return result

    def _plan_layer_paths(self, layer):
        """
        For a given layer, iterate over rows to create all paths
        """
        height, width = layer.shape
        paths = []
        prev_first = prev_last = -1

        # determines which rows to scan (downsampling via skip_lines)
        skip = max(int(self.skip_lines), 1)
        sampled_y_indices = np.arange(0, height, skip)  # y-indices representing physical spacing

        for (i, y) in enumerate(sampled_y_indices):
            row = layer[y]

            # determines locations where 1s start, and 1s end
            diff = np.diff(np.concatenate(([0], row, [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            if i % 2 == 1:  # Reverse every other line for zigzag pattern
                starts = starts[::-1]
                ends = ends[::-1]

            for (first_x, last_x) in zip(starts, ends):
                x1, x2 = (first_x, last_x)
                path = self._create_path(x1, x2, y, prev_first, prev_last)
                paths.append(path)
                prev_first, prev_last = x1, x2

        return paths

    def _create_path(self, x1, x2, y, prev_first, prev_last):
        """
        Given the coordinates of the path, rotate back to original coordinates, and return
        """
        p1 = rotate_coord(x1 + self.x_shift, y + self.y_shift, -self.current_angle)
        p2 = rotate_coord(x2 + self.x_shift, y + self.y_shift, -self.current_angle)

        # update global max and min
        self.x_max = max(self.x_max, p1[0], p2[0])
        self.x_min = min(self.x_min, p1[0], p2[0])
        self.y_max = max(self.y_max, p1[1], p2[1])
        self.y_min = min(self.y_min, p1[1], p2[1])

        status = self.CONNECT if is_between(x1, prev_first, prev_last) or is_between(prev_last, x1, x2) else self.STOP
        return [p1[0] * self.x_res, p1[1] * self.y_res, p2[0] * self.x_res, p2[1] * self.y_res, status]

    def _write_gcode(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # center the print based on global max and min
        x_shift = (self.x_max + self.x_min) / 2
        y_shift = (self.y_max + self.y_min) / 2

        extrude_amt = 0.0
        with open(self.output_path, 'w') as f:
            f.write("""M104 S200 ; set temperature
                G28 ; home all axes
                M109 S200 ; wait for temperature
                G21 ; mm units
                G90 ; absolute coordinates
                M82 ; absolute extrusion
                G92 E0
            """)
            for z_idx, paths in enumerate(tqdm(self.paths_per_layer, desc="Writing G-code")):
                z_pos = self.print_z_res * (z_idx + 1)
                f.write(f"\nG1 Z{z_pos:.3f} F{self.move_speed}")
                prev_x = prev_y = 0.0

                for x1, y1, x2, y2, status in paths:
                    x1 -= x_shift
                    y1 -= y_shift
                    x2 -= x_shift
                    y2 -= y_shift

                    if status == self.CONNECT:
                        extrude_amt += distance(prev_x, prev_y, x1, y1) * self.extrusion_res
                        f.write(f"\nG1 X{x1:.5f} Y{y1:.5f} F{self.extrude_speed} E{extrude_amt:.5f}")
                    else:
                        f.write(f"\nG1 X{x1:.5f} Y{y1:.5f} F{self.move_speed}")

                    extrude_amt += distance(x1, y1, x2, y2) * self.extrusion_res
                    f.write(f"\nG1 X{x2:.5f} Y{y2:.5f} F{self.extrude_speed} E{extrude_amt:.5f}")

                    prev_x, prev_y = x2, y2


# Entry point
if __name__ == '__main__':
    folder = "C:/Grace/Lab/variable-density-bioprinting-py/data/4.18testing6/temp/density_region_based_roi"
    out_path = "C:/Grace/Lab/variable-density-bioprinting-py/data/5.8_v2g2_test12/output.gcode"
    voxels, _ = load_dicom_series(folder)
    x_res, y_res, z_res = get_dicom_resolution(folder)

    gcode = GCodeWriter(voxels, x_res, y_res, z_res, out_path, rotate_angle_per_layer=30)
    gcode.run()
