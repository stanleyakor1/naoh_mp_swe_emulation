import numpy as np
import xarray as xr
import rioxarray as rxr
import py3dep
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import logging
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
import pandas as pd

import logging
import os
import sys


def setup_logging(level="INFO"):
    """
    Send logs to stdout (captured by Slurm/PBS job output) with immediate flush.
    """
    # If the scheduler sets a level, respect it; else use INFO
    level = os.environ.get("LOGLEVEL", level).upper()

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers that may have been added by libraries or previous runs
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)  
    handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)

    root.addHandler(handler)

    
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


setup_logging("INFO")

logger = logging.getLogger(__name__)


def write_array_to_netcdf(
    data_3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    start_date: str,
    end_date: str,
    out_nc: str,
    var_name: str = "fsca_bin",
    units: str = "1",
    long_name: str = "Binary snow / no-snow / cloud mask",
    calendar: str = "standard",
    time_freq: str = "D",
    compress_level: int = 4,
):
    """
    Write (time, lat, lon) numpy array to a NetCDF file.

    Parameters
    ----------
    data_3d : np.ndarray
        Array with shape (T, nlat, nlon). Can include np.nan.
    lat : np.ndarray
        1D array length nlat (e.g., 701).
    lon : np.ndarray
        1D array length nlon (e.g., 801).
    start_date, end_date : str
        Date strings parseable by pandas, e.g. "2002-10-01".
        Inclusive endpoints.
    out_nc : str
        Output NetCDF filename (e.g., "snow_mask.nc").
    var_name : str
        Variable name in NetCDF.
    """

    if data_3d.ndim != 3:
        raise ValueError(f"data_3d must be 3D (time, lat, lon); got shape {data_3d.shape}")

    T, nlat, nlon = data_3d.shape

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("lat and lon must be 1D arrays.")

    if len(lat) != nlat:
        raise ValueError(f"lat length {len(lat)} does not match data lat dim {nlat}")
    if len(lon) != nlon:
        raise ValueError(f"lon length {len(lon)} does not match data lon dim {nlon}")

    # Build time coordinate from start/end (inclusive)
    time = pd.date_range(start=start_date, end=end_date, freq=time_freq)

    if len(time) != T:
        raise ValueError(
            f"Time coordinate length {len(time)} does not match data time dim {T}. "
            f"Check start_date/end_date or data length."
        )

    da = xr.DataArray(
        data_3d.astype(np.float32, copy=False),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": ("lat", lat.astype(np.float32, copy=False)),
            "lon": ("lon", lon.astype(np.float32, copy=False)),
        },
        name=var_name,
        attrs={
            "long_name": long_name,
            "units": units,
            "missing_value": np.float32(np.nan),
        },
    )

    ds = xr.Dataset({var_name: da})

    # CF-ish coordinate metadata
    ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["time"].attrs.update({"standard_name": "time"})

    #
    ds.attrs.update(
        {
            "Conventions": "CF-1.8",
            "history": f"Created {pd.Timestamp.now(tz='UTC').isoformat()}",
        }
    )

    # Compression settings
    encoding = {
        var_name: {
            "zlib": True,
            "complevel": int(compress_level),
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }

    try:
        ds.to_netcdf(out_nc, engine="netcdf4", encoding=encoding)
    except Exception:
        ds.to_netcdf(out_nc, engine="scipy") 

    return out_nc

class NSTFSnowGapFiller:
    """
    Non-local Spatio-Temporal Filtering (NSTF) for MODIS snow cover gap-filling.

    Input:
      snow_stack: (time, rows, cols) with values:
        1 = snow, 0 = no snow, NaN = cloud/gap (or <0 treated as gap)

    Output:
      filled_stack: same shape, gaps filled where possible
      quality_flag:
        0 = original / unfilled
        1 = high confidence
        2 = medium confidence
        3 = low confidence
    """

    def __init__(
        self,
        spatial_window=15,
        temporal_window=7,
        elevation_weight=0.5,
        slope_weight=0.2,
        aspect_weight=0.15,
        spatial_weight=0.1,
        temporal_weight=0.05,
        similarity_threshold=0.7,
        normalize_weights=True,
        elev_scale_m=100.0,
        slope_scale_deg=10.0,
        aspect_scale_deg=45.0,
        temporal_scale_days=3.0,
    ):
        self.spatial_window = int(spatial_window)
        self.temporal_window = int(temporal_window)

        self.elevation_weight = float(elevation_weight)
        self.slope_weight = float(slope_weight)
        self.aspect_weight = float(aspect_weight)
        self.spatial_weight = float(spatial_weight)
        self.temporal_weight = float(temporal_weight)

        self.similarity_threshold = float(similarity_threshold)

        # Characteristic scales for exponential kernels
        self.elev_scale_m = float(elev_scale_m)
        self.slope_scale_deg = float(slope_scale_deg)
        self.aspect_scale_deg = float(aspect_scale_deg)
        self.temporal_scale_days = float(temporal_scale_days)

        # keep similarity meaning consistent (0–1) by normalizing weights
        if normalize_weights:
            self._normalize_weights()

    def _normalize_weights(self):
        w = np.array(
            [
                self.elevation_weight,
                self.slope_weight,
                self.aspect_weight,
                self.spatial_weight,
                self.temporal_weight,
            ],
            dtype=float,
        )
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("NSTF weights must sum to a positive finite value.")
        w /= w_sum
        (
            self.elevation_weight,
            self.slope_weight,
            self.aspect_weight,
            self.spatial_weight,
            self.temporal_weight,
        ) = w.tolist()

    def fill_gaps(
        self,
        snow_stack,
        dem,
        slope=None,
        aspect=None,
        cloud_mask=None,
        dem_spacing=None,
        verbose=True,
    ):
        """
        Fill gaps in snow cover time series.

        Parameters
        ----------
        snow_stack : ndarray
            3D array (time, rows, cols) of snow cover data (0/1/NaN).
        dem : ndarray
            2D array (rows, cols) of elevation data in meters (recommended).
        slope : ndarray, optional
            2D array of slope in degrees. If None, computed from dem.
        aspect : ndarray, optional
            2D array of aspect in degrees [0, 360). If None, computed from dem.
        cloud_mask : ndarray, optional
            3D boolean array True=gap/cloud. If None, NaN or <0 treated as gap.
        dem_spacing : tuple or None
            Fix #4: spacing for DEM gradient (dy, dx). Supply in meters per pixel if possible.
            Example: dem_spacing=(dy_m, dx_m). If None, assumes unit spacing (less physical).
        verbose : bool
            If True, log progress.

        Returns
        -------
        filled_stack : ndarray
        quality_flag : ndarray (uint8)
        """
        snow_stack = np.asarray(snow_stack)
        dem = np.asarray(dem)

        if snow_stack.ndim != 3:
            raise ValueError("snow_stack must be 3D (time, rows, cols).")
        if dem.ndim != 2:
            raise ValueError("dem must be 2D (rows, cols).")

        n_times, n_rows, n_cols = snow_stack.shape
        if dem.shape != (n_rows, n_cols):
            raise ValueError("dem shape must match snow_stack spatial dimensions (rows, cols).")

        filled_stack = snow_stack.copy()
        quality_flag = np.zeros_like(snow_stack, dtype=np.uint8)

        # Create cloud mask if not provided
        if cloud_mask is None:
            cloud_mask = np.isnan(snow_stack) | (snow_stack < 0)
        else:
            cloud_mask = np.asarray(cloud_mask, dtype=bool)
            if cloud_mask.shape != snow_stack.shape:
                raise ValueError("cloud_mask must have the same shape as snow_stack.")

        # Calculate slope/aspect if not provided
        if slope is None:
            slope = self._calculate_slope(dem, spacing=dem_spacing)
        else:
            slope = np.asarray(slope)

        if aspect is None:
            aspect = self._calculate_aspect(dem, spacing=dem_spacing)
        else:
            aspect = np.asarray(aspect)

        if slope.shape != dem.shape or aspect.shape != dem.shape:
            raise ValueError("slope and aspect must match dem shape (rows, cols).")

        # Valid pixels mask across time
        valid_mask_all = ~cloud_mask

        # Process each time step
        for t in range(n_times):
            if verbose:
                logger.info(f"Processing time step {t+1}/{n_times}")

            gaps = cloud_mask[t]
            if not np.any(gaps):
                continue

            
            gap_indices = np.argwhere(gaps)

            for (i, j) in gap_indices:
                filled_value, quality = self._fill_pixel(
                    i, j, t,
                    filled_stack,
                    dem, slope, aspect,
                    cloud_mask,
                    valid_mask_all
                )

                if filled_value is not None:
                    filled_stack[t, i, j] = filled_value
                    quality_flag[t, i, j] = quality

        return filled_stack, quality_flag

    def _fill_pixel(self, i, j, t, snow_stack, dem, slope, aspect, cloud_mask, valid_mask_all):
        """Fill a single gap pixel using multiple methods."""
        # Method 1: Temporal interpolation (highest quality)
        value, quality = self._temporal_interpolation(i, j, t, snow_stack, cloud_mask)
        if value is not None:
            return value, quality

        # Method 2: Spatio-temporal with terrain similarity
        value, quality = self._nstf_fill(
            i, j, t, snow_stack, dem, slope, aspect, cloud_mask
        )
        if value is not None:
            return value, quality

        # Method 3: Spatial only with terrain similarity
        value, quality = self._spatial_terrain_fill(
            i, j, t, snow_stack, dem, slope, aspect, valid_mask_all
        )
        if value is not None:
            return value, quality

        return None, 0

    def _temporal_interpolation(self, i, j, t, snow_stack, cloud_mask):
        """
        Fix #2: Temporal interpolation that actually uses before_val/after_val.

        Strategy:
          - Search up to 3 days before and after for the nearest valid value.
          - If both sides exist and total gap length <= 3 days:
              * if they agree -> return that value (quality 1)
              * if they disagree -> inverse-distance weighted probability -> threshold at 0.5 (quality 2)
          - Otherwise return None.
        """
        n_times = snow_stack.shape[0]

        before_val = after_val = None
        before_dist = after_dist = None

        # Search backward (1..3 days)
        for dt in range(1, min(4, t + 1)):
            if not cloud_mask[t - dt, i, j]:
                before_val = snow_stack[t - dt, i, j]
                before_dist = dt
                break

        # Search forward (1..3 days)
        for dt in range(1, min(4, n_times - t)):
            if not cloud_mask[t + dt, i, j]:
                after_val = snow_stack[t + dt, i, j]
                after_dist = dt
                break

        if before_val is not None and after_val is not None:
            if before_dist + after_dist <= 3:
                # Both are expected to be 0/1; cast safely
                b = float(before_val)
                a = float(after_val)

                if b == a:
                    return int(round(b)), 1  # high

                # Inverse-distance weights: closer observation has more influence
                w_b = 1.0 / before_dist
                w_a = 1.0 / after_dist
                p_snow = (w_b * b + w_a * a) / (w_b + w_a)

                filled_value = 1 if p_snow >= 0.5 else 0
                return filled_value, 2  # medium

        return None, 0

    def _nstf_fill(self, i, j, t, snow_stack, dem, slope, aspect, cloud_mask):
        """
        Non-local spatio-temporal filtering.
        """
        n_times, n_rows, n_cols = snow_stack.shape

        target_elev = dem[i, j]
        target_slope = slope[i, j]
        target_aspect = aspect[i, j]

        # Spatial window
        half_window = self.spatial_window // 2
        i_min = max(0, i - half_window)
        i_max = min(n_rows, i + half_window + 1)
        j_min = max(0, j - half_window)
        j_max = min(n_cols, j + half_window + 1)

        # Temporal window
        t_min = max(0, t - self.temporal_window)
        t_max = min(n_times, t + self.temporal_window + 1)

        similarities = []
        snow_values = []

        for tt in range(t_min, t_max):
            for ii in range(i_min, i_max):
                for jj in range(j_min, j_max):
                    # candidate must be valid at (tt,ii,jj)
                    if cloud_mask[tt, ii, jj]:
                        continue

                    # skip the target pixel at the target time (the actual missing one)
                    if tt == t and ii == i and jj == j:
                        continue

                    sim = self._calculate_similarity(
                        target_elev, target_slope, target_aspect,
                        dem[ii, jj], slope[ii, jj], aspect[ii, jj],
                        abs(ii - i), abs(jj - j), abs(tt - t)
                    )

                    if sim >= self.similarity_threshold:
                        similarities.append(sim)
                        snow_values.append(snow_stack[tt, ii, jj])

        if similarities:
            similarities = np.asarray(similarities, dtype=float)
            snow_values = np.asarray(snow_values, dtype=float)

            # Weighted vote -> probability of snow
            denom = np.sum(similarities)
            if denom <= 0 or not np.isfinite(denom):
                return None, 0

            snow_prob = np.sum(similarities * snow_values) / denom
            filled_value = 1 if snow_prob >= 0.5 else 0

            # Quality heuristic
            mean_sim = float(np.mean(similarities))
            n = len(similarities)
            if n >= 10 and mean_sim > 0.8:
                quality = 1
            elif n >= 5:
                quality = 2
            else:
                quality = 3

            return filled_value, quality

        return None, 0

    def _spatial_terrain_fill(self, i, j, t, snow_stack, dem, slope, aspect, valid_mask_all):
        """Spatial filling using terrain + spatial distance at the current time."""
        n_rows, n_cols = snow_stack.shape[1], snow_stack.shape[2]

        target_elev = dem[i, j]
        target_slope = slope[i, j]
        target_aspect = aspect[i, j]

        # Larger window for spatial-only
        half_window = self.spatial_window
        i_min = max(0, i - half_window)
        i_max = min(n_rows, i + half_window + 1)
        j_min = max(0, j - half_window)
        j_max = min(n_cols, j + half_window + 1)

        similarities = []
        snow_values = []

        for ii in range(i_min, i_max):
            for jj in range(j_min, j_max):
                if (ii == i and jj == j) or (not valid_mask_all[t, ii, jj]):
                    continue

                sim = self._calculate_similarity(
                    target_elev, target_slope, target_aspect,
                    dem[ii, jj], slope[ii, jj], aspect[ii, jj],
                    abs(ii - i), abs(jj - j), 0
                )

                if sim >= (self.similarity_threshold - 0.1):
                    similarities.append(sim)
                    snow_values.append(snow_stack[t, ii, jj])

        if similarities:
            similarities = np.asarray(similarities, dtype=float)
            snow_values = np.asarray(snow_values, dtype=float)

            denom = np.sum(similarities)
            if denom <= 0 or not np.isfinite(denom):
                return None, 0

            snow_prob = np.sum(similarities * snow_values) / denom
            filled_value = 1 if snow_prob >= 0.5 else 0
            return filled_value, 3  # low

        return None, 0

    def _calculate_similarity(
        self,
        elev1, slope1, aspect1,
        elev2, slope2, aspect2,
        spatial_dist_i, spatial_dist_j, temporal_dist
    ):
        """Similarity score in [0,1] (if weights are normalized)."""

        # Elevation similarity
        elev_diff = abs(elev1 - elev2)
        elev_sim = np.exp(-elev_diff / self.elev_scale_m)

        # Slope similarity
        slope_diff = abs(slope1 - slope2)
        slope_sim = np.exp(-slope_diff / self.slope_scale_deg)

        # Aspect similarity (circular distance)
        d = abs(aspect1 - aspect2)
        aspect_diff = min(d, 360.0 - d)
        aspect_sim = np.exp(-aspect_diff / self.aspect_scale_deg)

        # Spatial distance decay
        spatial_dist = np.sqrt(float(spatial_dist_i) ** 2 + float(spatial_dist_j) ** 2)
        spatial_sim = np.exp(-spatial_dist / max(1.0, float(self.spatial_window)))

        # Temporal distance decay
        temporal_sim = np.exp(-float(temporal_dist) / self.temporal_scale_days)

        similarity = (
            self.elevation_weight * elev_sim
            + self.slope_weight * slope_sim
            + self.aspect_weight * aspect_sim
            + self.spatial_weight * spatial_sim
            + self.temporal_weight * temporal_sim
        )

        
        if not np.isfinite(similarity):
            return 0.0

        return float(similarity)


    def _calculate_slope(self, dem, spacing=None):
        """Calculate slope from DEM in degrees."""
        if spacing is None:
            dy, dx = np.gradient(dem)
        else:
            dy_m, dx_m = spacing
            dy, dx = np.gradient(dem, float(dy_m), float(dx_m))

        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        return slope

    def _calculate_aspect(self, dem, spacing=None):
        """Calculate aspect from DEM in degrees [0, 360)."""
        if spacing is None:
            dy, dx = np.gradient(dem)
        else:
            dy_m, dx_m = spacing
            dy, dx = np.gradient(dem, float(dy_m), float(dx_m))

        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.mod(aspect, 360.0)
        return aspect

if __name__ == "__main__":

    WY = 2007 # from wy2010 to wy2017
    logger.info(f'Working on Water Year {WY}')
    logger.info("Loading datasets...")
    
    dem = xr.open_dataset('/bsuscratch/stanleyakor/nscid/dem/dem_on_modis_grid.nc')['elevation'].values
    
    
    snow_stack_path = f'/bsuscratch/stanleyakor/nscid/wy/wy{WY}/cloud_corrected_snow_data_wy{WY}.nc'
    snow_stack_with_gaps =  xr.open_dataset(snow_stack_path)['snow_mask']
    
    # Initialize and run gap filler
    logger.info("Initializing NSTF gap filler...")
    filler = NSTFSnowGapFiller(
        spatial_window=15, 
        temporal_window=7, 
        similarity_threshold=0.65
    )
    
    logger.info("Filling gaps...")
    filled_stack, quality_flag = filler.fill_gaps(
        snow_stack_with_gaps,
        dem
    )

    logger.info("Writing to netcdf...")
    data = xr.open_dataset(snow_stack_path)

    lat = data['lat'].values
    lon = data ['lon'].values

    start_date = f'{WY-1}-10-01'
    end_date   = f'{WY}-09-30'
    out_nc=f"/bsuscratch/stanleyakor/nscid/wy/wy{WY}/full_corrected_snow_data_wy{WY}.nc"
    
    out_file = write_array_to_netcdf(
    data_3d=filled_stack,
    lat=lat,
    lon=lon,
    start_date=start_date,
    end_date=end_date ,
    out_nc=out_nc,
    var_name="snow_mask",
    long_name="Snow mask (1=snow, 0=no snow, NaN=cloud)",
    )


