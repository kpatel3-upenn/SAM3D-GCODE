import numpy as np
import scipy.ndimage as ndi


# ─────────────────────────────────────────────────────────────
# 1.  MASK-AWARE NORMALISATION  (★ keep 12-bit precision)     │
# ─────────────────────────────────────────────────────────────
def normalize_inside_mask(volume: np.ndarray,
                          mask:   np.ndarray,
                          out_min: int = 0,
                          out_max: int = 4095,
                          dtype=np.uint16) -> np.ndarray:
    """
    Rescale intensities inside `mask` to [out_min, out_max].
    Voxels outside mask are set to out_min.

    Using 0-4095 + uint16 keeps ≈12 bits of contrast and avoids
    the posterised / speckled look you saw in ITK-SNAP.
    """
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must have identical shapes")

    mask = mask.astype(bool)
    vol  = volume.astype(np.float32)

    if not mask.any():
        raise ValueError("Mask is empty")

    lo, hi = vol[mask].min(), vol[mask].max()
    if hi == lo:
        raise ValueError("Mask has zero intensity range")

    scaled = (vol - lo) / (hi - lo) * (out_max - out_min) + out_min
    scaled[~mask] = out_min              # keep background flat
    return scaled.astype(dtype)


# ─────────────────────────────────────────────────────────────
# 2.  OTSU LIMITED TO THE MASK                                │
# ─────────────────────────────────────────────────────────────
def otsu_low_density(volume: np.ndarray,
                     mask:   np.ndarray,
                     nbins: int = 4096,
                     ) -> np.ndarray:
    """
    Return a binary mask of voxels BELOW Otsu threshold
    (i.e. the low-density part of the masked anatomy).

    `volume` is assumed to be the *normalised* image.
    """
    if volume.shape != mask.shape:
        raise ValueError("volume and mask must match")

    roi = volume[mask.astype(bool)]
    if roi.size == 0:
        raise ValueError("Mask has no voxels")

    hist, edges = np.histogram(roi,
                               bins=nbins,
                               range=(roi.min(), roi.max()))
    hist = hist.astype(np.float64)
    hist /= hist.sum()                    # → probability mass

    cumsum  = np.cumsum(hist)
    cummean = np.cumsum(hist * edges[:-1])
    global_mean = cummean[-1]

    valid = (cumsum > 0) & (cumsum < 1)
    sigma_b2 = np.zeros_like(cumsum)
    num   = (global_mean * cumsum[valid] - cummean[valid]) ** 2
    sigma_b2[valid] = num / (cumsum[valid] * (1.0 - cumsum[valid]))

    threshold = edges[sigma_b2.argmax()]

    out = np.zeros_like(volume, dtype=np.int16)
    within = mask.astype(bool)
    out[within] = (volume[within] <= threshold).astype(np.int16)
    return out


# ─────────────────────────────────────────────────────────────
# 3.  MORPHOLOGY + CC FILTER (★ get ONE clean object)         │
# ─────────────────────────────────────────────────────────────
def _ball_struct(r: int) -> np.ndarray:
    if r < 1:
        return np.ones((1, 1, 1), dtype=bool)
    x, y, z = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (x*x + y*y + z*z) <= r*r


def clean_binary(mask: np.ndarray,
                 radius: int = 2,
                 keep_largest: bool = True,
                 min_voxels: int | None = None) -> np.ndarray:
    """
    Closing  →  connected-components  →  hole filling.
    """
    mask = mask.astype(bool)

    # closing
    # if radius > 0:
    #     mask = ndi.binary_closing(mask, structure=_ball_struct(radius))

    # CC analysis
    lbl, n = ndi.label(mask, structure=np.ones((3, 3, 3), int))
    if n == 0:
        return np.zeros_like(mask, dtype=np.int16)

    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0  # background

    if keep_largest:
        mask = lbl == sizes.argmax()
    elif min_voxels is not None:
        good = np.where(sizes >= min_voxels)[0]
        mask = np.isin(lbl, good)

    # fill holes slice-wise or in 3-D
    # mask = ndi.binary_fill_holes(mask)
    return mask.astype(np.int16)


# ─────────────────────────────────────────────────────────────
# 4.  ONE-LINER PIPELINE WRAPPER                              │
# ─────────────────────────────────────────────────────────────
def low_density_region(volume: np.ndarray,
                       seg_mask: np.ndarray,
                       *,
                       norm_minmax: tuple[int, int] = (0, 4095),
                       norm_dtype = np.uint16,
                       cc_radius: int = 2,
                       cc_min_vox: int = 5_000,
                       ) -> np.ndarray:
    """
    Full pipeline:  normalise → Otsu → morphology clean-up.

    Returns a uint8 mask (1 = low-density) aligned with `volume`.
    """
    norm_vol = normalize_inside_mask(volume, seg_mask,
                                     out_min=norm_minmax[0],
                                     out_max=norm_minmax[1],
                                     dtype=norm_dtype)

    low_dense_raw = otsu_low_density(norm_vol, seg_mask)
    low_dense_cls = clean_binary(low_dense_raw,
                                 radius=cc_radius,
                                 keep_largest=True,
                                 min_voxels=cc_min_vox)
    return low_dense_cls  # uint8


def force_high_density_shell(low_density_mask: np.ndarray,
                             roi_mask: np.ndarray,
                             *,
                             shell_mm: float = 1.0,
                             voxel_spacing: tuple[float, float, float] = (1., 1., 1.)
                             ) -> np.ndarray:
    """
    Parameters
    ----------
    low_density_mask : (Z,Y,X) binary array
        1 = low‑density voxels after Otsu + clean‑up
    roi_mask         : (Z,Y,X) binary array
        The original anatomical segmentation (full patella volume)
    shell_mm         : float
        Thickness of the protective shell you always want printed at
        high density, expressed in millimetres.
    voxel_spacing    : (sz, sy, sx)
        Voxel sizes in millimetres (CT spacing); used to convert the
        physical thickness to voxel units.

    Returns
    -------
    np.ndarray  (uint8)
        New low‑density mask in which the outer `shell_mm` of voxels
        has been **cleared** (set to 0 → high density).
    """
    if low_density_mask.shape != roi_mask.shape:
        raise ValueError("Masks must have identical shape.")

    # -- 1. Compute physical distance from every voxel to the ROI boundary --
    # SignedMaurer gives +dist inside ROI, −dist outside
    dist = ndi.distance_transform_edt(roi_mask,
                                      sampling=voxel_spacing)

    # -- 2. Identify voxels within shell_mm of the surface ------------------
    shell_voxels = (roi_mask.astype(bool)) & (dist <= shell_mm)

    # -- 3. Remove those from the low‑density region ------------------------
    out = low_density_mask.copy()
    out[shell_voxels] = 0        # 0 → will be treated as high density
    return out.astype(low_density_mask.dtype)