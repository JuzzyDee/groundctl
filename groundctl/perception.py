"""Perception module — the rover's eyes.

Takes raw camera frames, runs YOLO detection, and produces structured
scene understanding that feeds into:
- Claude (via MCP status) — "what am I looking at?"
- Intent executor — follow_object tracking, approach targeting
- Safety layer — emergency stop on close obstacles

This is the sensory cortex. Camera in, understanding out.
Phase 1: YOLO pretrained (person, dog, car, bicycle, etc.)
Phase 2: Fine-tuned on Yeppoon environment if needed.
"""

from dataclasses import dataclass, field
from ultralytics import YOLO


# Classes we care about from COCO dataset
# YOLO pretrained knows 80 classes — we only need a subset
PRIORITY_CLASSES = {
    "person",
    "dog",
    "cat",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
    "bird",
    "bench",
    "chair",
    "potted plant",
    "fire hydrant",
    "stop sign",
}

# Classes that trigger safety responses
OBSTACLE_CLASSES = {
    "person",
    "dog",
    "cat",
    "car",
    "truck",
    "bicycle",
    "motorcycle",
}


@dataclass
class Detection:
    """A single detected object in a frame."""

    class_name: str
    confidence: float
    # Bounding box in pixels
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centre_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def centre_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def offset_from_centre(self, frame_width: int) -> float:
        """How far left/right of frame centre this detection is.
        Returns -1.0 (hard left) to 1.0 (hard right).
        0.0 = dead centre.
        """
        frame_centre = frame_width / 2
        return (self.centre_x - frame_centre) / frame_centre

    def size_ratio(self, frame_width: int, frame_height: int) -> float:
        """How much of the frame this detection fills (0.0 to 1.0).
        Proxy for distance — bigger = closer.
        """
        frame_area = frame_width * frame_height
        return self.area / frame_area


@dataclass
class SceneUnderstanding:
    """Structured understanding of a single camera frame."""

    detections: list[Detection] = field(default_factory=list)
    frame_width: int = 1024
    frame_height: int = 576

    @property
    def obstacles_ahead(self) -> list[Detection]:
        """Detections in the centre third of the frame that are obstacle classes."""
        centre_left = self.frame_width / 3
        centre_right = 2 * self.frame_width / 3
        return [
            d
            for d in self.detections
            if d.class_name in OBSTACLE_CLASSES
            and d.centre_x >= centre_left
            and d.centre_x <= centre_right
        ]

    @property
    def closest_obstacle(self) -> Detection | None:
        """Largest obstacle detection (proxy for closest)."""
        obstacles = self.obstacles_ahead
        if not obstacles:
            return None
        return max(obstacles, key=lambda d: d.area)

    @property
    def emergency_stop_needed(self) -> bool:
        """True if an obstacle fills more than 40% of the frame centre.
        That's texture-visible-stop territory.
        """
        closest = self.closest_obstacle
        if closest is None:
            return False
        return closest.size_ratio(self.frame_width, self.frame_height) > 0.4

    def find_by_class(self, class_name: str) -> list[Detection]:
        """Find all detections of a specific class."""
        return [d for d in self.detections if d.class_name == class_name]

    def find_largest(self, class_name: str) -> Detection | None:
        """Find the largest (closest) detection of a class."""
        matches = self.find_by_class(class_name)
        if not matches:
            return None
        return max(matches, key=lambda d: d.area)

    def to_summary(self) -> str:
        """Human/Claude-readable scene summary for MCP status."""
        if not self.detections:
            return "Clear — nothing detected"

        parts = []
        class_counts: dict[str, int] = {}
        for d in self.detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        for cls, count in sorted(class_counts.items()):
            if count == 1:
                # Single detection — include position
                det = self.find_by_class(cls)[0]
                offset = det.offset_from_centre(self.frame_width)
                if offset < -0.3:
                    pos = "left"
                elif offset > 0.3:
                    pos = "right"
                else:
                    pos = "ahead"
                size = det.size_ratio(self.frame_width, self.frame_height)
                if size > 0.2:
                    dist = "close"
                elif size > 0.05:
                    dist = "nearby"
                else:
                    dist = "distant"
                parts.append(f"{cls} ({dist}, {pos})")
            else:
                parts.append(f"{count}x {cls}")

        summary = ", ".join(parts)

        if self.emergency_stop_needed:
            summary = f"⚠ OBSTACLE CLOSE — {summary}"

        return summary


class Perceiver:
    """Runs YOLO on camera frames and produces scene understanding.

    Loads the model once, runs inference on each frame.
    Designed to be called at 10-20fps from the main loop.
    """

    def __init__(self, model_name: str = "yolo11n.pt", confidence_threshold: float = 0.4):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self._frame_width = 1024
        self._frame_height = 576

    def perceive(self, frame) -> SceneUnderstanding:
        """Run YOLO on a frame and return structured understanding.

        Args:
            frame: numpy array (BGR from OpenCV) or image path

        Returns:
            SceneUnderstanding with all detections and derived properties
        """
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)

        detections = []
        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls)]

            # Only keep classes we care about
            if class_name not in PRIORITY_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(box.conf),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        # Get frame dimensions from the result
        h, w = results[0].orig_shape
        self._frame_width = w
        self._frame_height = h

        return SceneUnderstanding(
            detections=detections,
            frame_width=w,
            frame_height=h,
        )

    def perceive_for_follow(self, frame, target_class: str = "person") -> dict:
        """Convenience method for follow intent — returns tracking data.

        Returns:
            {
                "found": bool,
                "offset": float (-1 to 1, left to right),
                "size_ratio": float (0 to 1, proxy for distance),
                "detection": Detection or None,
                "scene_summary": str,
            }
        """
        scene = self.perceive(frame)
        target = scene.find_largest(target_class)

        if target is None:
            return {
                "found": False,
                "offset": 0.0,
                "size_ratio": 0.0,
                "detection": None,
                "scene_summary": scene.to_summary(),
                "emergency_stop": scene.emergency_stop_needed,
            }

        return {
            "found": True,
            "offset": target.offset_from_centre(scene.frame_width),
            "size_ratio": target.size_ratio(scene.frame_width, scene.frame_height),
            "detection": target,
            "scene_summary": scene.to_summary(),
            "emergency_stop": scene.emergency_stop_needed,
        }

    def render_overlay(self, frame, scene: SceneUnderstanding):
        """Draw bounding boxes and labels on a frame — for Claude Cam demo.

        Args:
            frame: numpy array (BGR)
            scene: SceneUnderstanding from perceive()

        Returns:
            frame with overlay drawn (mutates in place)
        """
        import cv2

        for det in scene.detections:
            # Box colour based on class
            if det.class_name in OBSTACLE_CLASSES:
                colour = (0, 0, 255) if scene.emergency_stop_needed else (0, 165, 255)
            else:
                colour = (0, 255, 0)

            # Draw box
            cv2.rectangle(
                frame,
                (int(det.x1), int(det.y1)),
                (int(det.x2), int(det.y2)),
                colour,
                2,
            )

            # Label
            label = f"{det.class_name} {det.confidence:.0%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                frame,
                (int(det.x1), int(det.y1) - label_size[1] - 8),
                (int(det.x1) + label_size[0] + 4, int(det.y1)),
                colour,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (int(det.x1) + 2, int(det.y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Emergency stop warning
        if scene.emergency_stop_needed:
            cv2.putText(
                frame,
                "!! EMERGENCY STOP !!",
                (frame.shape[1] // 2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        return frame
