"""Intent executor — decouples Claude's decision rate from the rover's control rate.

Claude sets an intent (go forward, follow bearing, navigate to GPS). The executor
maintains a continuous 10Hz control loop, translating intent into motor commands.
Claude only needs to speak up when the intent changes.

This is the translation layer between cognitive intent and physical movement.
Phase 1: proportional steering math.
Phase 2: IL model replaces the math, same interface.
"""

import math
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from .rover_client import RoverClient


# --- Geo helpers (shared with navigation.py, could extract later) ---

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def normalize_angle(angle: float) -> float:
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


# --- Intent types ---

class IntentType(Enum):
    IDLE = "idle"
    GO_FORWARD = "go_forward"
    FOLLOW_BEARING = "follow_bearing"
    NAVIGATE_TO = "navigate_to"
    TURN_TO = "turn_to"


@dataclass
class Intent:
    type: IntentType = IntentType.IDLE
    speed: float = 0.0
    bearing: float = 0.0           # target bearing for follow_bearing / turn_to
    target_lat: float = 0.0        # for navigate_to
    target_lon: float = 0.0        # for navigate_to
    arrival_threshold: float = 2.0  # meters, for navigate_to
    timeout: float = 120.0
    started_at: float = field(default_factory=time.time)


class IntentExecutor:
    """Continuous 10Hz control loop driven by high-level intent.

    Claude sets intent via set_intent(). The executor runs the control loop,
    translating intent into [linear, angular] commands at 10Hz.

    Phase 1: proportional steering.
    Phase 2: IL model slots in here — same intent in, smoother commands out.
    """

    def __init__(
        self,
        rover: RoverClient,
        default_speed: float = 0.4,
        max_angular: float = 0.5,
        steering_gain: float = 0.02,
    ):
        self.rover = rover
        self.default_speed = default_speed
        self.max_angular = max_angular
        self.steering_gain = steering_gain
        self._intent = Intent()
        self._running = False
        self._task: asyncio.Task | None = None
        self._result: dict | None = None

    @property
    def current_intent(self) -> Intent:
        return self._intent

    @property
    def is_active(self) -> bool:
        return self._running and self._intent.type != IntentType.IDLE

    @property
    def last_result(self) -> dict | None:
        return self._result

    def start(self):
        """Start the 10Hz control loop. Call once at server boot."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._control_loop())

    async def stop(self):
        """Stop the control loop entirely."""
        self._running = False
        await self.set_idle()
        if self._task:
            self._task.cancel()
            self._task = None

    async def set_idle(self):
        """Stop moving, clear intent."""
        self._intent = Intent()
        await self.rover.stop()

    async def set_intent(self, intent_type: IntentType, **kwargs) -> dict:
        """Set a new intent. The control loop picks it up immediately."""
        speed = kwargs.get("speed", self.default_speed)
        self._result = None

        self._intent = Intent(
            type=intent_type,
            speed=speed,
            bearing=kwargs.get("bearing", 0.0),
            target_lat=kwargs.get("target_lat", 0.0),
            target_lon=kwargs.get("target_lon", 0.0),
            arrival_threshold=kwargs.get("arrival_threshold", 2.0),
            timeout=kwargs.get("timeout", 120.0),
        )

        return {
            "status": "intent_set",
            "intent": intent_type.value,
            "speed": speed,
        }

    async def _get_telemetry(self) -> tuple[float, float, float]:
        """Get (lat, lon, heading) from rover."""
        data = await self.rover.get_data()
        return (
            float(data.get("latitude", 0)),
            float(data.get("longitude", 0)),
            float(data.get("orientation", 0)),
        )

    def _compute_steering(self, current_heading: float, target_bearing: float, speed: float) -> tuple[float, float]:
        """Phase 1: proportional steering. Phase 2: IL model replaces this."""
        heading_error = normalize_angle(target_bearing - current_heading)
        angular = heading_error * self.steering_gain
        angular = max(-self.max_angular, min(self.max_angular, angular))
        linear = speed * max(0.3, 1 - abs(heading_error) / 180)
        return linear, angular

    async def _control_loop(self):
        """Main 10Hz loop. Reads intent, computes commands, sends to SDK."""
        while self._running:
            try:
                intent = self._intent

                if intent.type == IntentType.IDLE:
                    await asyncio.sleep(0.1)
                    continue

                # Check timeout
                elapsed = time.time() - intent.started_at
                if elapsed > intent.timeout:
                    self._result = {"success": False, "reason": "timeout", "elapsed": round(elapsed, 1)}
                    await self.set_idle()
                    continue

                lat, lon, heading = await self._get_telemetry()

                if intent.type == IntentType.GO_FORWARD:
                    # Maintain current heading and speed
                    await self.rover.move(intent.speed, 0.0)

                elif intent.type == IntentType.FOLLOW_BEARING:
                    linear, angular = self._compute_steering(heading, intent.bearing, intent.speed)
                    await self.rover.move(linear, angular)

                elif intent.type == IntentType.TURN_TO:
                    heading_error = normalize_angle(intent.bearing - heading)
                    if abs(heading_error) < 5.0:
                        self._result = {"success": True, "reason": "facing_target", "heading": round(heading, 1)}
                        await self.set_idle()
                        continue
                    angular = heading_error * self.steering_gain
                    angular = max(-self.max_angular, min(self.max_angular, angular))
                    await self.rover.move(0.0, angular)

                elif intent.type == IntentType.NAVIGATE_TO:
                    distance = haversine_distance(lat, lon, intent.target_lat, intent.target_lon)
                    if distance < intent.arrival_threshold:
                        self._result = {
                            "success": True,
                            "reason": "arrived",
                            "distance": round(distance, 1),
                            "elapsed": round(elapsed, 1),
                        }
                        await self.set_idle()
                        continue

                    target_bearing = calculate_bearing(lat, lon, intent.target_lat, intent.target_lon)
                    linear, angular = self._compute_steering(heading, target_bearing, intent.speed)
                    await self.rover.move(linear, angular)

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)

        await self.rover.stop()
