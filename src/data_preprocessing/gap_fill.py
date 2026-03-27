import numpy as np
import xarray as xr
import numpy as np
import pandas as pd
import logging
import os
import sys

class TerrainIDWSnowGapFiller:
    """
    Temporal-then-terrain gap filler for MODIS snow cover data.

    Fill strategy (applied per gap pixel, in order):

      1. Temporal interpolation (fast, pixel-local):
           Search up to `temporal_interp_days` days before and after.
           - Both sides found and agree  → return that value directly.
           - Both sides found, disagree  → inverse-distance weighted blend,
                                           threshold at 0.5 → 0 or 100.
           - One side only              → return that nearest valid value.
           - Neither side found         → fall through to step 2.

      2. Terrain-aware spatial IDW (fallback):
           Search an expanding sequence of spatial windows
           (e.g. half-widths 3 → 7 → 15, giving 7×7 → 15×15 → 31×31)
           until at least `min_neighbors` terrain-similar valid pixels are
           found at the current time step.
           - Found enough neighbors → weighted average.
           - Still cannot fill      → pixel remains NaN.

    Input
    -----
    snow_stack : (time, rows, cols)
        SCF values in [0, 100]; NaN or <0 treated as cloud/gap.

    Output
    ------
    filled_stack : same shape, gaps filled where possible.
    """

    # Default adaptive window half-widths (pixels).
    # Window sizes in pixels: 2*h+1  →  7×7, 15×15, 31×31
    DEFAULT_WINDOWS = (3, 7, 15)

    def __init__(
        self,
        spatial_windows=None,
        temporal_interp_days=3,
        elev_scale_m=100.0,
        slope_scale_deg=10.0,
        aspect_scale_deg=45.0,
        idw_power=1.0,
        min_neighbors=3,
    ):
        """
        Parameters
        ----------
        spatial_windows : sequence of int, optional
            Half-widths of the search windows tried in order, e.g. (3, 7, 15)
            gives window sizes 7×7, 15×15, 31×31.
            Defaults to (3, 7, 15).
        temporal_interp_days : int
            Maximum number of days to look forward/backward when attempting
            temporal interpolation. Default is 3.
        elev_scale_m : float
            Maximum elevation difference (m) for a neighbor to be accepted.
        slope_scale_deg : float
            Maximum slope difference (degrees) for a neighbor to be accepted.
        aspect_scale_deg : float
            Maximum aspect difference (degrees, circular) for acceptance.
        idw_power : float
            Distance decay exponent p in w ~ T / d^p.
        min_neighbors : int
            Minimum number of accepted neighbors required to produce a fill.
            If a window tier yields fewer, the next (larger) tier is tried.
        """
        if spatial_windows is None:
            spatial_windows = self.DEFAULT_WINDOWS

        # Validate and sort windows ascending so we always try small first
        self.spatial_windows      = tuple(sorted(int(w) for w in spatial_windows))
        if len(self.spatial_windows) == 0:
            raise ValueError("spatial_windows must contain at least one value.")

        self.temporal_interp_days = int(temporal_interp_days)
        self.elev_scale_m         = float(elev_scale_m)
        self.slope_scale_deg  = float(slope_scale_deg)
        self.aspect_scale_deg = float(aspect_scale_deg)
        self.idw_power        = float(idw_power)
        self.min_neighbors    = int(min_neighbors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        Fill gaps using temporal interpolation first, terrain IDW as fallback.

        Parameters
        ----------
        snow_stack : ndarray (time, rows, cols)
            SCF data in [0, 100]; NaN or <0 = gap.
        dem : ndarray (rows, cols)
            Elevation in metres.
        slope : ndarray (rows, cols), optional
            Slope in degrees. Computed from dem if not supplied.
        aspect : ndarray (rows, cols), optional
            Aspect in degrees [0, 360). Computed from dem if not supplied.
        cloud_mask : ndarray (time, rows, cols) bool, optional
            True = gap/cloud. Derived from NaN/<0 if not supplied.
        dem_spacing : tuple (dy_m, dx_m), optional
            Physical pixel spacing in metres for slope/aspect computation.
            If None, unit spacing is assumed (less accurate).
        verbose : bool
            Log per-timestep progress if True.

        Returns
        -------
        filled_stack : ndarray (time, rows, cols), float32
            Input stack with gaps filled where possible. Pixels that could
            not be filled by either method remain NaN.
        """
        snow_stack = np.asarray(snow_stack)
        dem        = np.asarray(dem)

        if snow_stack.ndim != 3:
            raise ValueError("snow_stack must be 3D (time, rows, cols).")
        if dem.ndim != 2:
            raise ValueError("dem must be 2D (rows, cols).")

        n_times, n_rows, n_cols = snow_stack.shape
        if dem.shape != (n_rows, n_cols):
            raise ValueError("dem shape must match snow_stack spatial dims (rows, cols).")

        filled_stack  = snow_stack.copy().astype(np.float32)
        if cloud_mask is None:
            cloud_mask = np.isnan(snow_stack) | (snow_stack < 0)
        else:
            cloud_mask = np.asarray(cloud_mask, dtype=bool)
            if cloud_mask.shape != snow_stack.shape:
                raise ValueError("cloud_mask must have the same shape as snow_stack.")

        # --- terrain ---
        if slope is None:
            slope = self._calculate_slope(dem, spacing=dem_spacing)
        else:
            slope = np.asarray(slope)

        if aspect is None:
            aspect = self._calculate_aspect(dem, spacing=dem_spacing)
        else:
            aspect = np.asarray(aspect)

        if slope.shape != dem.shape or aspect.shape != dem.shape:
            raise ValueError("slope and aspect must match dem shape.")

        valid_mask_all = ~cloud_mask

        # --- main loop ---
        for t in range(n_times):
            if verbose:
                logger.info(f"Processing time step {t+1}/{n_times}")

            gaps = cloud_mask[t]
            if not np.any(gaps):
                continue

            gap_indices = np.argwhere(gaps)

            for (i, j) in gap_indices:
                # --- Step 1: temporal interpolation ---
                filled_value = self._temporal_interpolation(
                    i, j, t, filled_stack, cloud_mask
                )

                # --- Step 2: terrain IDW fallback ---
                if filled_value is None:
                    filled_value = self._adaptive_fill_pixel(
                        i, j, t,
                        filled_stack,
                        dem, slope, aspect,
                        valid_mask_all,
                        n_rows, n_cols,
                    )

                if filled_value is not None:
                    filled_stack[t, i, j] = filled_value

        return filled_stack

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _temporal_interpolation(self, i, j, t, snow_stack, cloud_mask):
        """
        Fill a single gap pixel using the nearest valid observations in time.

        Strategy
        --------
        Search up to `temporal_interp_days` steps before and after t for the
        nearest cloud-free value at the same (i, j) location.

        - Both before and after found, and they agree:
            return that value directly.
        - Both found but disagree:
            inverse-distance weighted blend of the two, threshold at 0.5
            (i.e. whichever observation is closer in time dominates).
            Result is returned as 0 or 100 (binary snow presence scaled to SCF).
        - Only one side found:
            return that nearest valid value as-is.
        - Neither side found within the window:
            return None → caller falls back to terrain IDW.

        Parameters
        ----------
        i, j : int
            Pixel row/column.
        t : int
            Current (gap) time index.
        snow_stack : ndarray (time, rows, cols)
            Working filled stack (may already contain previously filled values).
        cloud_mask : ndarray (time, rows, cols) bool
            Original cloud/gap mask — used to identify valid observations.
            Previously filled pixels are NOT treated as valid here, preventing
            filled values from propagating into subsequent temporal fills.

        Returns
        -------
        float or None
        """
        n_times  = snow_stack.shape[0]
        max_days = self.temporal_interp_days

        before_val = after_val = None
        before_dist = after_dist = None

        # Search backward (1 .. max_days)
        for dt in range(1, min(max_days + 1, t + 1)):
            if not cloud_mask[t - dt, i, j]:
                before_val  = snow_stack[t - dt, i, j]
                before_dist = dt
                break

        # Search forward (1 .. max_days)
        for dt in range(1, min(max_days + 1, n_times - t)):
            if not cloud_mask[t + dt, i, j]:
                after_val  = snow_stack[t + dt, i, j]
                after_dist = dt
                break

        if before_val is not None and after_val is not None:
            b = float(before_val)
            a = float(after_val)
            if b == a:
                return b
            # Inverse-distance weighted blend; closer observation dominates
            w_b = 1.0 / before_dist
            w_a = 1.0 / after_dist
            p_snow = (w_b * b + w_a * a) / (w_b + w_a)
            
            return  p_snow 

        if before_val is not None:
            return float(before_val)
        if after_val is not None:
            return float(after_val)

        return None

    def _adaptive_fill_pixel(self, i, j, t, snow_stack, dem, slope, aspect,
                              valid_mask_all, n_rows, n_cols):
        """
        Try each window half-width in ascending order.
        Returns the interpolated SCF value, or None if no window succeeded.

        The adaptive logic is:
          for half_width in spatial_windows:
              collect neighbors in (2*half_width+1)^2 window
              if len(neighbors) >= min_neighbors:
                  return weighted_average
          return None   # could not fill
        """
        for half_w in self.spatial_windows:
            filled_value = self._spatial_terrain_fill(
                i, j, t,
                snow_stack,
                dem, slope, aspect,
                valid_mask_all,
                n_rows, n_cols,
                half_window=half_w,
            )
            if filled_value is not None:
                return filled_value

        return None

    def _spatial_terrain_fill(self, i, j, t, snow_stack, dem, slope, aspect,
                               valid_mask_all, n_rows, n_cols, half_window=None):
        """
        Terrain-weighted IDW interpolation within a single fixed window.

        Weight:
            w_k = (E_k * S_k * A_k) / (d_k + eps)^p

        where E, S, A are piecewise-linear terrain similarity factors in [0,1]
        for elevation, slope, and aspect respectively.

        Returns the interpolated SCF (float, clipped to [0,100]) if at least
        min_neighbors valid terrain-similar neighbors exist, else None.
        """
        if half_window is None:
            half_window = self.spatial_windows[-1]

        target_elev   = dem[i, j]
        target_slope  = slope[i, j]
        target_aspect = aspect[i, j]

        i_min = max(0,       i - half_window)
        i_max = min(n_rows,  i + half_window + 1)
        j_min = max(0,       j - half_window)
        j_max = min(n_cols,  j + half_window + 1)

        weights    = []
        scf_values = []
        eps        = 1e-6

        for ii in range(i_min, i_max):
            for jj in range(j_min, j_max):
                if (ii == i and jj == j) or (not valid_mask_all[t, ii, jj]):
                    continue

                di = np.hypot(float(ii - i), float(jj - j))
                if di <= 0:
                    continue

                # --- elevation factor ---
                elev_diff = abs(float(target_elev) - float(dem[ii, jj]))
                if elev_diff > self.elev_scale_m:
                    continue
                elev_factor = 1.0 - (elev_diff / max(self.elev_scale_m, eps))

                # --- slope factor ---
                slope_diff = abs(float(target_slope) - float(slope[ii, jj]))
                if slope_diff > self.slope_scale_deg:
                    continue
                slope_factor = 1.0 - (slope_diff / max(self.slope_scale_deg, eps))

                # --- aspect factor (circular difference) ---
                aspect_raw_diff = abs(float(target_aspect) - float(aspect[ii, jj]))
                aspect_diff     = min(aspect_raw_diff, 360.0 - aspect_raw_diff)
                if aspect_diff > self.aspect_scale_deg:
                    continue
                aspect_factor = 1.0 - (aspect_diff / max(self.aspect_scale_deg, eps))

                # --- combined terrain + distance weight ---
                terrain_factor = (max(elev_factor,  0.0)
                                  * max(slope_factor, 0.0)
                                  * max(aspect_factor, 0.0))
                wi = terrain_factor / np.power(di + eps, self.idw_power)

                if np.isfinite(wi) and wi > 0:
                    weights.append(wi)
                    scf_values.append(float(snow_stack[t, ii, jj]))

        if len(weights) >= self.min_neighbors:
            weights    = np.asarray(weights,    dtype=float)
            scf_values = np.asarray(scf_values, dtype=float)

            denom = np.sum(weights)
            if denom <= 0 or not np.isfinite(denom):
                return None

            scf_interp = float(np.clip(
                np.sum(weights * scf_values) / denom, 0.0, 100.0
            ))
            return scf_interp

        return None

    def _calculate_slope(self, dem, spacing=None):
        """Slope from DEM in degrees."""
        if spacing is None:
            dy, dx = np.gradient(dem)
        else:
            dy_m, dx_m = spacing
            dy, dx = np.gradient(dem, float(dy_m), float(dx_m))
        return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    def _calculate_aspect(self, dem, spacing=None):
        """Aspect from DEM in degrees [0, 360)."""
        if spacing is None:
            dy, dx = np.gradient(dem)
        else:
            dy_m, dx_m = spacing
            dy, dx = np.gradient(dem, float(dy_m), float(dx_m))
        aspect = np.degrees(np.arctan2(-dx, dy))
        return np.mod(aspect, 360.0)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    WY = 2010
    logger.info(f"Working on Water Year {WY}")
    logger.info("Loading datasets...")

    dem = xr.open_dataset(
        "nscid/dem/dem_on_modis_grid.nc"
    )["elevation"].values

    snow_stack_path = (
        f"/nscid/wy/wy{WY}/"
        f"cloud_corrected_snow_data_wy{WY}.nc"
    )

    start_date = f'{WY-1}-10-01'
    end_date   = f'{WY}-04-15'
    out_nc=f'new_val.nc'
    
    snow_stack_with_gaps = sca.sel(time = slice(start_date, end_date))

    # -----------------------------------------------------------------------
    # Adaptive windows:
    #   half-width  3  →  7×7   window  (tight local fill)
    #   half-width  7  →  15×15 window  (intermediate)
    #   half-width 15  →  31×31 window  (last resort, dense cloud)
    # -----------------------------------------------------------------------
    logger.info("Initializing terrain-aware adaptive gap filler...")
    filler = TerrainIDWSnowGapFiller(
        spatial_windows=(3, 7, 15),   # half-widths → 7×7, 15×15, 31×31
        temporal_interp_days=3,
        elev_scale_m=100.0,
        slope_scale_deg=10.0,
        aspect_scale_deg=45.0,
        idw_power=1.0,
        min_neighbors=3,
    )

    logger.info("Filling gaps...")
    filled_stack = filler.fill_gaps(
        snow_stack_with_gaps,
        dem,
        verbose=True,
    )

    # --- write filled snow stack ---
    logger.info("Writing filled snow stack to NetCDF...")
    data       = xr.open_dataset(snow_stack_path)
    lat        = data["lat"].values
    lon        = data["lon"].values

    write_array_to_netcdf(
        data_3d    = filled_stack,
        lat        = lat,
        lon        = lon,
        start_date = start_date,
        end_date   = end_date,
        out_nc     = out_nc,
        var_name   = "snow_mask",
        long_name  = "Snow cover fraction (0-100, NaN=cloud)",
    )

    logger.info("Done.")
