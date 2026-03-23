"""GPS waypoint navigation with proportional steering control.

Adapted from the EarthRover SDK waypoint navigation example.
This is the classical control fallback — it gets replaced by the IL model
in later phases, but gives Claude basic "go to that coordinate" capability
from day one.
"""

import math
import asyncio

from .rover_client import RoverClient


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great circle distance between two GPS points in meters."""
    R = 6371000  # Earth's radius in meters

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing from point 1 to point 2 in degrees (0-360)."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def normalize_angle(angle: float) -> float:
    """Normalize angle to -180 to 180 range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


class WaypointNavigator:
    """Navigate to GPS coordinates using proportional steering.

    This is the Phase 1 navigation layer — classical control, no ML.
    Gets the rover from A to B using GPS + compass heading + proportional
    angular correction. Not smooth, not clever, but functional.

    Will be replaced by the IL model in later phases.
    """

    def __init__(
        self,
        rover: RoverClient,
        arrival_threshold: float = 2.0,
        default_speed: float = 0.4,
        max_angular: float = 0.5,
        steering_gain: float = 0.02,
    ):
        self.rover = rover
        self.arrival_threshold = arrival_threshold
        self.default_speed = default_speed
        self.max_angular = max_angular
        self.steering_gain = steering_gain
        self._active = False

    async def get_position(self) -> tuple[float, float, float]:
        """Get current (lat, lon, heading) from rover telemetry."""
        data = await self.rover.get_data()
        lat = float(data.get("latitude", 0))
        lon = float(data.get("longitude", 0))
        heading = float(data.get("orientation", 0))
        return lat, lon, heading

    async def distance_to(self, target_lat: float, target_lon: float) -> float:
        """Distance in meters from current position to target."""
        lat, lon, _ = await self.get_position()
        return haversine_distance(lat, lon, target_lat, target_lon)

    async def navigate_to(
        self,
        target_lat: float,
        target_lon: float,
        speed: float | None = None,
        timeout: float = 120.0,
    ) -> dict:
        """Navigate to a GPS coordinate.

        Returns a status dict with distance, success, and reason.
        """
        speed = speed or self.default_speed
        self._active = True

        start_time = asyncio.get_event_loop().time()

        while self._active:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                await self.rover.stop()
                self._active = False
                return {
                    "success": False,
                    "reason": "timeout",
                    "elapsed": round(elapsed, 1),
                }

            lat, lon, heading = await self.get_position()
            distance = haversine_distance(lat, lon, target_lat, target_lon)

            if distance < self.arrival_threshold:
                await self.rover.stop()
                self._active = False
                return {
                    "success": True,
                    "reason": "arrived",
                    "distance": round(distance, 1),
                    "elapsed": round(elapsed, 1),
                }

            # Proportional steering
            target_bearing = calculate_bearing(lat, lon, target_lat, target_lon)
            heading_error = normalize_angle(target_bearing - heading)

            angular = heading_error * self.steering_gain
            angular = max(-self.max_angular, min(self.max_angular, angular))

            # Slow down when heading is way off
            linear = speed * max(0.3, 1 - abs(heading_error) / 180)

            await self.rover.move(linear, angular)
            await asyncio.sleep(0.1)

        await self.rover.stop()
        return {"success": False, "reason": "cancelled"}

    async def stop(self):
        """Cancel active navigation."""
        self._active = False
        await self.rover.stop()

    @property
    def is_active(self) -> bool:
        return self._active
