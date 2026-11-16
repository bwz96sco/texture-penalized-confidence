"""HYPSO-1 captures: calibration with the `hypso` package and mapping to Landsat 8 bands."""
import contextlib, io

import numpy as np

# Landsat 8 OLI bands B1-B4 (nm), as fixed in eval-spec.md.
LANDSAT_BANDS = {"B1": (430, 450), "B2": (450, 510), "B3": (530, 590), "B4": (640, 670)}
COEFF_TYPE = "moved"  # the hypso1-calibration package's documented default


def toa_reflectance(l1a_path: str, quiet: bool = True) -> dict:
    """Calibrate one L1A capture. Returns the TOA reflectance cube (H, W, 120), centre
    wavelengths (120,), the L1A DN cube, the mean solar zenith angle and the capture time."""
    import matplotlib
    matplotlib.use("Agg")  # hypso imports matplotlib; never open GUI windows
    from hypso import Hypso1

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        h = Hypso1(l1a_path, verbose=False)
        dn = np.asarray(h.l1a_cube)
        h.generate_l1b_cube(coeff_type=COEFF_TYPE)
        h.run_direct_georeferencing()
        h.generate_l1d_cube(use_direct_georef=True)
    return {"toa": np.asarray(h.l1d_cube), "wavelengths": np.asarray(h.wavelengths, dtype=float), "dn": dn,
            "sza_mean": float(np.nanmean(h.solar_zenith_angles_direct)), "iso_time": str(h.iso_time)}


def band_channels(wavelengths: np.ndarray) -> dict:
    """Channel indices whose centre wavelength falls inside each Landsat band."""
    return {b: np.flatnonzero((wavelengths >= lo) & (wavelengths < hi)).tolist()
            for b, (lo, hi) in LANDSAT_BANDS.items()}


def to_landsat_bands(cube: np.ndarray, channels: dict) -> np.ndarray:
    """Mean over each band's channels, float32 (H, W, 4)."""
    return np.stack([cube[..., channels[b]].mean(axis=-1) for b in LANDSAT_BANDS], axis=-1).astype(np.float32)
