"""Regional carbon intensity grid lookup (gCO2/kWh).

Data sourced from:
- IEA (2023) electricity carbon intensity by country/region
- Cloud provider sustainability reports
- electricityMap.org regional averages
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class GridRegion:
    """A grid region with carbon intensity data."""
    name: str
    country: str
    carbon_intensity_g_per_kwh: float  # gCO2eq/kWh
    renewable_pct: float               # 0-100
    notes: str = ""


# Carbon intensity lookup: cloud region -> GridRegion
# Values in gCO2eq/kWh (grid average, 2022-2023 data)
_CLOUD_REGIONS: Dict[str, GridRegion] = {
    # AWS regions
    "us-east-1":      GridRegion("US East (N. Virginia)",    "US",  379.0, 28.0, "PJM grid"),
    "us-east-2":      GridRegion("US East (Ohio)",           "US",  450.0, 18.0, "PJM grid"),
    "us-west-1":      GridRegion("US West (N. California)",  "US",  210.0, 52.0, "CAISO grid"),
    "us-west-2":      GridRegion("US West (Oregon)",         "US",  130.0, 68.0, "NWPP — high hydro"),
    "eu-west-1":      GridRegion("EU West (Ireland)",        "IE",  290.0, 40.0, "EirGrid"),
    "eu-west-2":      GridRegion("EU West (London)",         "GB",  233.0, 42.0, "National Grid ESO"),
    "eu-west-3":      GridRegion("EU West (Paris)",          "FR",   58.0, 91.0, "RTE — nuclear heavy"),
    "eu-central-1":   GridRegion("EU Central (Frankfurt)",   "DE",  364.0, 46.0, "50Hz/Amprion mix"),
    "eu-north-1":     GridRegion("EU North (Stockholm)",     "SE",   13.0, 98.0, "Svenska Kraftnät — hydro/nuclear"),
    "ap-southeast-1": GridRegion("Asia Pacific (Singapore)", "SG",  408.0, 14.0, "EMA grid"),
    "ap-southeast-2": GridRegion("Asia Pacific (Sydney)",    "AU",  520.0, 25.0, "NEM grid"),
    "ap-northeast-1": GridRegion("Asia Pacific (Tokyo)",     "JP",  463.0, 22.0, "TEPCO area"),
    "ap-south-1":     GridRegion("Asia Pacific (Mumbai)",    "IN",  708.0,  8.0, "WR/IR grid"),
    "ca-central-1":   GridRegion("Canada (Central)",         "CA",  130.0, 67.0, "Hydro-Québec"),
    "sa-east-1":      GridRegion("South America (São Paulo)","BR",   74.0, 84.0, "SIN — hydro dominant"),
    "af-south-1":     GridRegion("Africa (Cape Town)",       "ZA",  928.0,  6.0, "Eskom — coal heavy"),
    "me-south-1":     GridRegion("Middle East (Bahrain)",    "BH",  730.0,  2.0, "GCCIA"),
    # GCP regions (alias to nearest AWS region data)
    "us-central1":    GridRegion("GCP US Central (Iowa)",    "US",  398.0, 27.0, "MISO grid"),
    "europe-west1":   GridRegion("GCP EU West (Belgium)",    "BE",  167.0, 60.0, "Elia grid"),
    "europe-west4":   GridRegion("GCP EU West (Netherlands)","NL",  289.0, 42.0, "TenneT NL"),
    "asia-east1":     GridRegion("GCP Asia East (Taiwan)",   "TW",  509.0, 12.0, "Taipower"),
    # Azure regions
    "eastus":         GridRegion("Azure East US",            "US",  379.0, 28.0, "PJM grid"),
    "westeurope":     GridRegion("Azure West Europe",        "NL",  289.0, 42.0, "TenneT NL"),
    "northeurope":    GridRegion("Azure North Europe",       "IE",  290.0, 40.0, "EirGrid"),
    # Country codes (fallback)
    "US":             GridRegion("United States (average)",  "US",  386.0, 22.0, "EIA national avg"),
    "EU":             GridRegion("European Union (average)", "EU",  233.0, 40.0, "ENTSO-E avg"),
    "CN":             GridRegion("China (average)",          "CN",  555.0, 28.0, "NEA grid"),
    "IN":             GridRegion("India (average)",          "IN",  708.0,  8.0, "CEA grid"),
    "GB":             GridRegion("United Kingdom",           "GB",  233.0, 42.0, "National Grid ESO"),
    "FR":             GridRegion("France",                   "FR",   58.0, 91.0, "RTE — nuclear heavy"),
    "DE":             GridRegion("Germany",                  "DE",  364.0, 46.0, "50Hz/Amprion/TenneT/Amprion"),
    "SE":             GridRegion("Sweden",                   "SE",   13.0, 98.0, "Svenska Kraftnät"),
    "NO":             GridRegion("Norway",                   "NO",   17.0, 98.0, "Statnett — hydro"),
    "BR":             GridRegion("Brazil",                   "BR",   74.0, 84.0, "ONS — hydro dominant"),
    "CA":             GridRegion("Canada (average)",         "CA",  130.0, 67.0, "NEB national avg"),
    "AU":             GridRegion("Australia (average)",      "AU",  520.0, 25.0, "AEMO avg"),
    "JP":             GridRegion("Japan (average)",          "JP",  463.0, 22.0, "METI avg"),
    "SG":             GridRegion("Singapore",                "SG",  408.0, 14.0, "EMA"),
    "ZA":             GridRegion("South Africa",             "ZA",  928.0,  6.0, "Eskom"),
}

# World average fallback
_WORLD_AVERAGE = GridRegion("World average", "WORLD", 436.0, 29.0, "IEA 2022 global average")


class RegionalGrid:
    """Lookup carbon intensity for cloud regions and country codes.

    Example::

        grid = RegionalGrid()
        intensity = grid.get_intensity("us-east-1")   # → 379.0 gCO2/kWh
        region = grid.get_region("eu-west-3")         # → GridRegion(...)
    """

    def __init__(self, custom_regions: Optional[Dict[str, GridRegion]] = None):
        self._regions: Dict[str, GridRegion] = dict(_CLOUD_REGIONS)
        if custom_regions:
            self._regions.update(custom_regions)

    def get_region(self, region_id: str) -> GridRegion:
        """Return GridRegion for a region ID, or world average if unknown."""
        key = region_id.lower() if region_id else ""
        # Try exact match
        if key in self._regions:
            return self._regions[key]
        # Try uppercase (country codes stored uppercase)
        if region_id.upper() in self._regions:
            return self._regions[region_id.upper()]
        return _WORLD_AVERAGE

    def get_intensity(self, region_id: str) -> float:
        """Return carbon intensity in gCO2eq/kWh."""
        return self.get_region(region_id).carbon_intensity_g_per_kwh

    def get_renewable_pct(self, region_id: str) -> float:
        """Return renewable energy percentage (0-100)."""
        return self.get_region(region_id).renewable_pct

    def list_regions(self) -> Dict[str, GridRegion]:
        """Return the full region lookup table."""
        return dict(self._regions)

    def add_region(self, region_id: str, region: GridRegion) -> None:
        """Add or override a region entry."""
        self._regions[region_id] = region
