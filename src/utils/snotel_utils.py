import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional


class SNOTEL:
    def __init__(
        self,
        var: str = "SNOW",
        path_to_header: Optional[str] = None,
        path_to_csv: Optional[str] = None,
        path_to_geog: Optional[str] = None,
        path_to_wrf_file: Optional[xr.Dataset] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        var : str, default="SNOW"
            Variable of interest from the WRF output file.
        path_to_header : str
            Path to NRC SNOTEL location information.
        path_to_csv : str
            Path to SNOTEL datasets for the WRF domain.
        path_to_geog : str
            Path to WRF intermediary file (geo_d02.nc).
        path_to_wrf_file : xr.Dataset
            WRF input dataset.
        start : str
            Start date of analysis (YYYY-MM-DD).
        end : str
            End date of analysis (YYYY-MM-DD).
        save : bool, default=True
            Flag to control saving of results.
        """
        self.path_to_header: Path = Path(path_to_header) if path_to_header else None
        self.path_to_csv: Path = Path(path_to_csv) if path_to_csv else None
        self.geog: xr.Dataset = xr.open_dataset(path_to_geog) if path_to_geog else None
        self.var: str = var
        self.start: Optional[str] = start
        self.end: Optional[str] = end
        self.wrf: Optional[xr.Dataset] = (
            path_to_wrf_file.sel(XTIME=slice(self.start, self.end))
            if path_to_wrf_file is not None
            else None
        )
        self.wrf_file: Optional[xr.DataArray] = (
            self.wrf[self.var] if self.wrf is not None else None
        )

        self.snotel_indices: Dict[str, Tuple[int, int]] = {}
        self.feat: Dict = {}

    def collect_snotel_info(
        self,
    ) -> Tuple[List[float], List[float], List[str], List[int], List[float]]:
        """
        Collects information about NRC SNOTEL sites within the WRF domain.

        Returns
        -------
        lat : list of float
            Latitudes of the SNOTEL sites.
        lon : list of float
            Longitudes of the SNOTEL sites.
        sta_names : list of str
            Names of the SNOTEL sites.
        sta_id : list of int
            Station IDs of the SNOTEL sites.
        elevation : list of float
            Elevation values of the SNOTEL sites.
        """
        snotel = pd.read_csv(self.path_to_header)
        df = snotel[
            (40.317627 >= snotel["Latitude"]) & (snotel["Latitude"] >= 36.810326)
            & (-105.09583 >= snotel["Longitude"]) & (snotel["Longitude"] >= -109.0985)
        ]

        filtered_df = df[df["State"] == "CO"]

        return (
            filtered_df["Latitude"].tolist(),
            filtered_df["Longitude"].tolist(),
            filtered_df["Station Name"].tolist(),
            filtered_df["Station ID"].tolist(),
            filtered_df["Elevation"].tolist(),
        )

    def extract_var(self, ixlat: int, ixlon: int) -> np.ndarray:
        """
        Extracts precipitation from the WRF input file for a given SNOTEL xy index pair.

        Parameters
        ----------
        ixlat : int
            Site latitude index in WRF grid.
        ixlon : int
            Site longitude index in WRF grid.

        Returns
        -------
        np.ndarray
            Time series of precipitation values.
        """
        return self.wrf_file.isel(south_north=ixlat, west_east=ixlon).values

    def get_wrf_xy(self) -> Dict[str, Tuple[int, int]]:
        """
        Converts SNOTEL latitude/longitude pairs to WRF xy indices.

        Returns
        -------
        dict
            Mapping of SNOTEL station IDs to their (ixlat, ixlon) WRF grid indices.
        """
        xlat = self.geog.XLAT.values[0, :, :]
        xlon = self.geog.XLONG.values[0, :, :]

        lat, lon, _, sta_id, _ = self.collect_snotel_info()

        for lat_, lon_, station in zip(lat, lon, sta_id):
            dist = np.sqrt((xlat - lat_) ** 2 + (xlon - lon_) ** 2)
            ixlat, ixlon = np.unravel_index(np.argmin(dist), dist.shape)
            self.snotel_indices[str(station)] = (ixlat, ixlon)

        return self.snotel_indices

    @staticmethod
    def compute_NSE(obs: np.ndarray, sim: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency (NSE)."""
        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        return 1 - (numerator / denominator)

    @staticmethod
    def compute_KGE(obs: np.ndarray, sim: np.ndarray) -> float:
        """Compute Kling-Gupta Efficiency (KGE)."""
        r, _ = pearsonr(obs, sim)
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    def read_station_csv(self) -> Dict[Tuple[float, float, float], float]:
        """
        Reads station CSVs, filters for the analysis period, and computes NSE values.

        Returns
        -------
        dict
            Mapping of (lat, lon, elevation) to NSE values.
        """
        lat, lon, sta_names, sta_id, elevation = self.collect_snotel_info()
        station_indices = self.get_wrf_xy()

        results: Dict[Tuple[float, float, float], float] = {}
        for i, station_id in enumerate(sta_id):
            file_path = self.path_to_csv / f"df_{station_id}.csv"
            generic_col = (
                f"{sta_names[i]} ({station_id}) Snow Water Equivalent (in) Start of Day Values"
            )

            df = pd.read_csv(file_path)
            df = df[(df["Date"] >= self.start) & (df["Date"] <= self.end)]

            obs = df[generic_col].to_numpy() * 25.4  # convert in → mm
            ixlat, ixlon = station_indices[str(station_id)]
            sim = self.extract_var(ixlat, ixlon)

            nse = self.compute_NSE(obs, sim)
            results[(lat[i], lon[i], elevation[i])] = nse

        return results
