"""Client for communicating with the EarthRover SDK REST API."""

import base64
import httpx


class RoverClient:
    """Wraps the EarthRover SDK endpoints running on localhost."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=10.0)

    async def close(self):
        await self._client.aclose()

    # -- Perception --

    async def get_front_frame(self) -> bytes | None:
        """Get the front camera frame as raw image bytes."""
        resp = await self._client.get("/v2/front")
        resp.raise_for_status()
        data = resp.json()
        frame_b64 = data.get("front_frame")
        if frame_b64:
            return base64.b64decode(frame_b64)
        return None

    async def get_rear_frame(self) -> bytes | None:
        """Get the rear camera frame as raw image bytes."""
        resp = await self._client.get("/v2/rear")
        resp.raise_for_status()
        data = resp.json()
        frame_b64 = data.get("rear_frame")
        if frame_b64:
            return base64.b64decode(frame_b64)
        return None

    async def get_screenshot(self) -> dict:
        """Get front + rear frames and timestamp."""
        resp = await self._client.get("/v2/screenshot")
        resp.raise_for_status()
        return resp.json()

    async def get_data(self) -> dict:
        """Get rover telemetry: battery, GPS, orientation, speed, IMU."""
        resp = await self._client.get("/data")
        resp.raise_for_status()
        return resp.json()

    # -- Control --

    async def move(self, linear: float, angular: float, lamp: int | None = None) -> dict:
        """Send movement command. linear/angular: -1.0 to 1.0. lamp: 0 or 1."""
        command = {"linear": linear, "angular": angular}
        if lamp is not None:
            command["lamp"] = lamp
        resp = await self._client.post("/control", json={"command": command})
        resp.raise_for_status()
        return resp.json()

    async def stop(self) -> dict:
        """Stop the rover."""
        return await self.move(0.0, 0.0)

    async def set_lamp(self, on: bool) -> dict:
        """Turn headlights on or off."""
        resp = await self._client.post("/control", json={"command": {"lamp": 1 if on else 0}})
        resp.raise_for_status()
        return resp.json()

    async def speak(self, text: str) -> dict:
        """Send text-to-speech through the rover's speaker."""
        resp = await self._client.post("/speak", json={"text": text})
        resp.raise_for_status()
        return resp.json()

    # -- Missions --

    async def start_mission(self) -> dict:
        resp = await self._client.post("/start-mission")
        resp.raise_for_status()
        return resp.json()

    async def end_mission(self) -> dict:
        resp = await self._client.post("/end-mission")
        resp.raise_for_status()
        return resp.json()

    async def get_checkpoints(self) -> dict:
        resp = await self._client.get("/checkpoints-list")
        resp.raise_for_status()
        return resp.json()

    async def checkpoint_reached(self) -> dict:
        resp = await self._client.post("/checkpoint-reached", json={})
        resp.raise_for_status()
        return resp.json()

    async def get_mission_history(self) -> dict:
        resp = await self._client.get("/missions-history")
        resp.raise_for_status()
        return resp.json()

    # -- Interventions --

    async def start_intervention(self) -> dict:
        resp = await self._client.post("/interventions/start")
        resp.raise_for_status()
        return resp.json()

    async def end_intervention(self) -> dict:
        resp = await self._client.post("/interventions/end")
        resp.raise_for_status()
        return resp.json()

    async def get_interventions(self) -> dict:
        resp = await self._client.get("/interventions/history")
        resp.raise_for_status()
        return resp.json()
