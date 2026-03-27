"""Spatial memory — Phase 1.

Named places the rover remembers. Simple coordinate + radius store
backed by a JSON file. Phase 2 integrates with Memoria for
Ebbinghaus decay, Hebbian place-event linking, and visit-based
reinforcement.

For now: mark a spot, give it a name, navigate back to it later.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


DEFAULT_RADIUS = 5.0  # metres
PLACES_FILE = Path.home() / ".groundctl" / "places.json"


@dataclass
class Place:
    """A named location the rover knows."""

    name: str
    lat: float
    lon: float
    radius: float = DEFAULT_RADIUS
    created_at: float = field(default_factory=time.time)
    visit_count: int = 0
    last_visited: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Place":
        return cls(**data)


class PlaceStore:
    """Persistent store of named places.

    Phase 1: JSON file on disk. Dead simple.
    Phase 2: Memoria spatial memory type with decay and linking.
    """

    def __init__(self, path: Path | None = None):
        self.path = path or PLACES_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._places: dict[str, Place] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._places = {
                    name: Place.from_dict(p) for name, p in data.items()
                }
            except (json.JSONDecodeError, TypeError):
                self._places = {}

    def _save(self):
        data = {name: p.to_dict() for name, p in self._places.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def mark(self, name: str, lat: float, lon: float, radius: float = DEFAULT_RADIUS) -> Place:
        """Mark the current location as a named place."""
        place = Place(
            name=name,
            lat=lat,
            lon=lon,
            radius=radius,
            visit_count=1,
            last_visited=time.time(),
        )
        self._places[name.lower()] = place
        self._save()
        return place

    def get(self, name: str) -> Place | None:
        """Get a named place."""
        return self._places.get(name.lower())

    def visit(self, name: str):
        """Record a visit to a named place."""
        place = self._places.get(name.lower())
        if place:
            place.visit_count += 1
            place.last_visited = time.time()
            self._save()

    def list_all(self) -> list[Place]:
        """List all named places."""
        return list(self._places.values())

    def remove(self, name: str) -> bool:
        """Forget a named place."""
        if name.lower() in self._places:
            del self._places[name.lower()]
            self._save()
            return True
        return False

    def find_current(self, lat: float, lon: float) -> Place | None:
        """Find which named place (if any) the rover is currently at."""
        from .navigation import haversine_distance

        for place in self._places.values():
            dist = haversine_distance(lat, lon, place.lat, place.lon)
            if dist <= place.radius:
                return place
        return None

    def to_summary(self) -> str:
        """Human/Claude-readable summary of all known places."""
        if not self._places:
            return "No named places yet."

        lines = []
        for p in sorted(self._places.values(), key=lambda x: x.last_visited, reverse=True):
            visits = f"{p.visit_count} visit{'s' if p.visit_count != 1 else ''}"
            lines.append(f"  {p.name} ({p.radius:.0f}m radius, {visits})")
        return "Known places:\n" + "\n".join(lines)
