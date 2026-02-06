

#!/usr/bin/env python3
"""
Post-process downloaded MOD10A1 .hdf tiles:
- extract the 'NDSI_Snow_Cover' SDS
- mosaic all tiles for each day (if multiple)
- warp to EPSG:4326 and clip to bbox
- stack into a single NetCDF (time, lat, lon)

Run:
  python mod10a1_clip_to_netcdf.py \
    --in_dir downloads \
    --out_nc MOD10A1_v061_bbox_20001001_20170930.nc
"""

from __future__ import annotations
import os
import re
import argparse
import datetime as dt
from collections import defaultdict

import numpy as np
import netCDF4 as nc
from osgeo import gdal

gdal.UseExceptions()

# BBOX (lon/lat)
MIN_LON, MIN_LAT = -109.0985, 36.810326
MAX_LON, MAX_LAT = -105.09583, 40.317627

# Output resolution in degrees (adjust if desired)
OUT_RES_DEG = 0.005  # ~500m-ish at mid-lats


def parse_modis_date_from_filename(fn: str) -> dt.date | None:
    """
    MOD10A1.AYYYYDDD.hXXvYY.061....hdf  -> date from YYYY + DDD
    """
    m = re.search(r"\.A(\d{4})(\d{3})\.", fn)
    if not m:
        return None
    year = int(m.group(1))
    doy = int(m.group(2))
    return dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)


def pick_subdataset(hdf_path: str, sds_contains: str = "NDSI_Snow_Cover") -> str:
    ds = gdal.Open(hdf_path)
    if ds is None:
        raise RuntimeError(f"GDAL could not open: {hdf_path}")
    subs = ds.GetSubDatasets()
    if not subs:
        raise RuntimeError(f"No subdatasets found in: {hdf_path}")
    needle = sds_contains.lower()
    matches = [name for name, desc in subs if needle in name.lower() or needle in desc.lower()]
    if not matches:
        preview = "\n".join([f"  - {name}" for name, _ in subs[:12]])
        raise RuntimeError(
            f"Could not find SDS containing '{sds_contains}' in {hdf_path}\n"
            f"First subdatasets:\n{preview}"
        )
    return matches[0]


def build_vrt(sds_paths: list[str], vrt_path: str) -> str:
    vrt = gdal.BuildVRT(vrt_path, sds_paths)
    if vrt is None:
        raise RuntimeError("gdal.BuildVRT failed")
    vrt.FlushCache()
    return vrt_path


def warp_clip_to_geotiff(src: str, out_tif: str) -> str:
    opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        outputBounds=(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT),
        xRes=OUT_RES_DEG,
        yRes=OUT_RES_DEG,
        resampleAlg="near",
        format="GTiff",
        multithread=True,
    )
    out = gdal.Warp(out_tif, src, options=opts)
    if out is None:
        raise RuntimeError(f"gdal.Warp failed for {src}")
    out.FlushCache()
    out = None
    return out_tif


def read_geotiff(tif: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = gdal.Open(tif)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.int16)

    gt = ds.GetGeoTransform()
    origin_x, px_w, _, origin_y, _, px_h = gt
    nx, ny = ds.RasterXSize, ds.RasterYSize

    lons = origin_x + (np.arange(nx) + 0.5) * px_w
    lats = origin_y + (np.arange(ny) + 0.5) * px_h  # px_h negative for north-up

    ds = None
    return arr, lats, lons


def days_since_1970(d: dt.date) -> int:
    return (d - dt.date(1970, 1, 1)).days


def init_nc(out_nc: str, lats: np.ndarray, lons: np.ndarray) -> nc.Dataset:
    root = nc.Dataset(out_nc, "w", format="NETCDF4")
    root.createDimension("time", None)
    root.createDimension("lat", len(lats))
    root.createDimension("lon", len(lons))

    vlat = root.createVariable("lat", "f4", ("lat",))
    vlon = root.createVariable("lon", "f4", ("lon",))
    vtime = root.createVariable("time", "i4", ("time",))

    vlat.units = "degrees_north"
    vlon.units = "degrees_east"
    vtime.units = "days since 1970-01-01"
    vtime.calendar = "standard"

    vlat[:] = lats
    vlon[:] = lons

    vsnow = root.createVariable(
        "NDSI_Snow_Cover",
        "i2",
        ("time", "lat", "lon"),
        zlib=True,
        complevel=4,
        chunksizes=(1, min(512, len(lats)), min(512, len(lons))),
        fill_value=-32768,
    )
    vsnow.long_name = "MOD10A1 NDSI Snow Cover (integer-coded)"
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing downloaded .hdf files")
    ap.add_argument("--out_nc", required=True, help="Output NetCDF path")
    ap.add_argument("--tmp_dir", default="tmp_mod10a1", help="Temp working directory")
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    # Group files by day
    by_day: dict[dt.date, list[str]] = defaultdict(list)
    for fn in os.listdir(args.in_dir):
        if not fn.lower().endswith(".hdf"):
            continue
        d = parse_modis_date_from_filename(fn)
        if d is None:
            continue
        by_day[d].append(os.path.join(args.in_dir, fn))

    if not by_day:
        raise RuntimeError("No MOD10A1 .hdf files found in input directory.")

    nc_ds = None
    expected_shape = None

    for i, day in enumerate(sorted(by_day.keys())):
        files = by_day[day]
        print(f"{day.isoformat()}  ({len(files)} tile(s))")

        sds_paths = []
        for f in files:
            try:
                sds_paths.append(pick_subdataset(f, "NDSI_Snow_Cover"))
            except Exception as e:
                print(f"  warning: {e}")

        if not sds_paths:
            print("  no NDSI_Snow_Cover SDS found; skipping")
            continue

        vrt_path = os.path.join(args.tmp_dir, f"mosaic_{day.isoformat()}.vrt")
        tif_path = os.path.join(args.tmp_dir, f"clip_{day.isoformat()}.tif")

        build_vrt(sds_paths, vrt_path)
        warp_clip_to_geotiff(vrt_path, tif_path)

        arr, lats, lons = read_geotiff(tif_path)

        if nc_ds is None:
            nc_ds = init_nc(args.out_nc, lats, lons)

        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise RuntimeError(f"Grid changed on {day}: {arr.shape} vs {expected_shape}")

        t = len(nc_ds.dimensions["time"])
        nc_ds.variables["time"][t] = days_since_1970(day)
        nc_ds.variables["NDSI_Snow_Cover"][t, :, :] = arr
        nc_ds.sync()

    if nc_ds is not None:
        nc_ds.close()

    print(f"Done: {args.out_nc}")


if __name__ == "__main__":
    main()


