"""SPARCS (Landsat 8) subscenes: DN to top-of-atmosphere reflectance, labels and location."""
import re

import numpy as np
import pyproj
import tifffile

# OLI reflectance rescaling, identical for bands 1-9 in Landsat 8 Level-1 metadata.
REFLECTANCE_MULT, REFLECTANCE_ADD = 2.0e-5, -0.1
# `_data.tif` holds 10 bands in order (B1-B7, B9-B11); the first four are B1-B4.
BANDS = slice(0, 4)
# SPARCS codes: 0 cloud shadow, 1 cloud shadow over water, 2 water, 3 ice/snow,
# 4 land, 5 clouds, 6 flooded. Merged to 0 water, 1 land, 2 cloud; 255 = not trained on.
MERGE = np.array([255, 0, 0, 255, 1, 2, 0], dtype=np.uint8)
CLOUD_CODE = 5


def _epsg(page) -> int:
    keys = page.tags[34735].value
    for i in range(4, len(keys), 4):
        if keys[i] == 3072 and keys[i + 3] != 32767:  # ProjectedCSTypeGeoKey, not user-defined
            return int(keys[i + 3])
    # User-defined: read the zone from the citation, e.g. "WGS 84 / UTM zone 22N|WGS 84|".
    m = re.search(r"UTM zone (\d+)([NS])", page.tags[34737].value)
    if not m:
        raise ValueError("no UTM zone in the GeoTIFF citation")
    return (32600 if m.group(2) == "N" else 32700) + int(m.group(1))


def load(sparcs_dir: str, subscene: str):
    """Return DN array (H, W, 4) uint16 and raw label codes (H, W) uint8."""
    with tifffile.TiffFile(f"{sparcs_dir}/{subscene}_data.tif") as t:
        dn = t.pages[0].asarray()[..., BANDS]
    labels = tifffile.imread(f"{sparcs_dir}/{subscene}_labels.tif")
    return dn, labels


def centre_latlon(sparcs_dir: str, subscene: str):
    """Latitude and longitude of the subscene centre (UTM subscenes only)."""
    with tifffile.TiffFile(f"{sparcs_dir}/{subscene}_data.tif") as t:
        page = t.pages[0]
        sx, sy, _ = page.tags[33550].value
        _, _, _, x0, y0, _ = page.tags[33922].value
        h, w = page.shape[:2]
        epsg = _epsg(page)
    cx, cy = x0 + sx * w / 2, y0 - sy * h / 2
    lon, lat = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True).transform(cx, cy)
    return lat, lon


def toa_reflectance(dn: np.ndarray, sun_elevation_deg: float) -> np.ndarray:
    """USGS Level-1 DN to TOA reflectance with sun-angle correction, float32."""
    rho = REFLECTANCE_MULT * dn.astype(np.float64) + REFLECTANCE_ADD
    return (rho / np.sin(np.radians(sun_elevation_deg))).astype(np.float32)
