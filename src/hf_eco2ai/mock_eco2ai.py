"""Mock implementation of eco2ai for testing and development."""

import time
import random
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MockEco2AITracker:
    """Mock eco2ai tracker for development and testing."""
    
    def __init__(self, project_name: str = "test", country: str = "USA", region: str = "CA"):
        self.project_name = project_name
        self.country = country
        self.region = region
        self.start_time = None
        self.total_energy = 0.0
        self.total_co2 = 0.0
        self._is_tracking = False
        
    def start(self):
        """Start tracking energy consumption."""
        self.start_time = time.time()
        self._is_tracking = True
        logger.info(f"Started mock energy tracking for {self.project_name}")
        
    def stop(self):
        """Stop tracking and return consumption data."""
        if not self._is_tracking:
            return {"energy": 0.0, "co2": 0.0}
            
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Mock energy consumption: ~100-500W for duration
        base_power = random.uniform(100, 500)  # watts
        self.total_energy = (base_power * duration) / (1000 * 3600)  # kWh
        
        # Mock CO2: California grid ~230g CO2/kWh
        carbon_intensity = 230  # g CO2/kWh
        self.total_co2 = self.total_energy * carbon_intensity / 1000  # kg CO2
        
        self._is_tracking = False
        
        result = {
            "energy": self.total_energy,
            "co2": self.total_co2,
            "duration": duration,
            "power": base_power
        }
        
        logger.info(f"Mock tracking stopped: {self.total_energy:.3f} kWh, {self.total_co2:.3f} kg CO₂")
        return result
        
    def get_current(self) -> Dict[str, float]:
        """Get current consumption estimate."""
        if not self._is_tracking:
            return {"energy": 0.0, "co2": 0.0, "power": 0.0}
            
        duration = time.time() - self.start_time if self.start_time else 0
        base_power = random.uniform(200, 400)  # watts
        current_energy = (base_power * duration) / (1000 * 3600)  # kWh
        current_co2 = current_energy * 230 / 1000  # kg CO2
        
        return {
            "energy": current_energy,
            "co2": current_co2,
            "power": base_power
        }


# Mock eco2ai module interface
def Tracker(project_name: str, **kwargs) -> MockEco2AITracker:
    """Create a mock tracker instance."""
    return MockEco2AITracker(project_name, **kwargs)


# Mock carbon data
CARBON_INTENSITY_BY_REGION = {
    "USA/CA": 230,  # California
    "USA/TX": 400,  # Texas
    "USA/WA": 90,   # Washington
    "Germany/Bavaria": 411,
    "Norway/Oslo": 20,
    "China/Beijing": 681,
    "India/Maharashtra": 820,
}


def get_carbon_intensity(country: str = "USA", region: str = "CA") -> float:
    """Get carbon intensity for region."""
    key = f"{country}/{region}"
    return CARBON_INTENSITY_BY_REGION.get(key, 400)  # Default 400g CO2/kWh


def get_renewable_percentage(country: str = "USA", region: str = "CA") -> float:
    """Get renewable percentage for region."""
    renewable_data = {
        "USA/CA": 45.0,
        "USA/TX": 28.0, 
        "USA/WA": 89.0,
        "Germany/Bavaria": 52.0,
        "Norway/Oslo": 98.0,
        "China/Beijing": 15.0,
        "India/Maharashtra": 12.0,
    }
    key = f"{country}/{region}"
    return renewable_data.get(key, 25.0)  # Default 25%