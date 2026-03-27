"""Perception module — the rover's eyes.

Takes raw camera frames, runs YOLO detection and depth estimation,
producing structured scene understanding that feeds into:
- Claude (via MCP status) — "what am I looking at?"
- Intent executor — follow_object tracking, approach targeting
- Safety layer — emergency stop on close obstacles (depth-based)

Two perception layers:
- YOLO11n at 20fps: object tracking for follow/approach intents
- DepthAnything at 2-3fps: proximity safety, catches EVERYTHING
  regardless of object class (barriers, walls, kangaroos, bins)

This is the sensory cortex. Camera in, understanding out.
"""

from dataclasses import dataclass, field
import numpy as np
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


@dataclass
class DepthResult:
    """Result of a depth-based proximity check."""

    emergency_stop: bool
    obstacle_detected: bool
    discontinuity_strength: float  # 0.0 = smooth ground, 1.0 = wall in face
    centre_close_ratio: float  # what % of centre is "close"
    depth_map: np.ndarray | None = field(default=None, repr=False)

    def to_summary(self) -> str:
        if self.emergency_stop:
            return f"⚠ DEPTH STOP — obstacle detected (discontinuity: {self.discontinuity_strength:.2f}, close: {self.centre_close_ratio:.0%})"
        elif self.obstacle_detected:
            return f"Caution — object ahead (discontinuity: {self.discontinuity_strength:.2f})"
        return "Clear"


class DepthSafety:
    """Depth-based proximity detection — catches what YOLO misses.

    Uses DepthAnything to estimate relative depth from a single camera frame.
    Detects obstacles by finding gradient discontinuities — the ground creates
    a smooth near-to-far gradient from bottom to top. An obstacle breaks that
    gradient with a sharp edge.

    This catches barriers, walls, kerbs, unknown objects — anything that's
    close, regardless of whether YOLO knows what it is.
    """

    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 input_size: int = 518):
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline("depth-estimation", model=model_name)
        self._input_size = input_size

    def check(self, frame, return_depth_map: bool = False) -> DepthResult:
        """Run depth estimation and check for obstacles.

        Args:
            frame: numpy array (BGR from OpenCV) or PIL Image
            return_depth_map: if True, include the raw depth map in result

        Returns:
            DepthResult with emergency_stop flag and diagnostics
        """
        from PIL import Image
        import cv2

        # Convert to PIL Image from whatever input we got
        if isinstance(frame, str):
            pil_image = Image.open(frame)
        elif isinstance(frame, np.ndarray):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
        else:
            pil_image = frame

        # Resize for speed — safety doesn't need full resolution
        w, h = pil_image.size
        scale = self._input_size / max(w, h)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            pil_image = pil_image.resize((new_w, new_h))

        result = self._pipe(pil_image)
        depth_map = np.array(result["depth"])

        # Analyse the centre third for obstacles
        dh, dw = depth_map.shape
        centre_left = dw // 3
        centre_right = 2 * dw // 3
        centre_strip = depth_map[:, centre_left:centre_right]

        # Gradient discontinuity detection:
        # Ground = smooth gradient (depth decreasing from bottom to top)
        # Obstacle = sharp jump where depth suddenly increases going up
        #
        # Sample rows in the centre strip, moving from bottom to top
        n_samples = 20
        row_indices = np.linspace(dh - 1, dh // 4, n_samples, dtype=int)
        row_means = [centre_strip[row, :].mean() for row in row_indices]

        # Find the maximum upward jump (depth increasing while moving up)
        # Normalise to 0-255 range
        max_jump = 0.0
        for i in range(1, len(row_means)):
            jump = row_means[i] - row_means[i - 1]  # positive = depth increased going up
            if jump > max_jump:
                max_jump = jump

        # Normalise discontinuity to 0-1
        depth_range = depth_map.max() - depth_map.min()
        if depth_range > 0:
            discontinuity = max_jump / depth_range
        else:
            discontinuity = 0.0

        # Close ratio — what percentage of centre is in the "close" band
        close_threshold = depth_map.max() * 0.7
        close_pixels = (centre_strip > close_threshold).sum()
        close_ratio = close_pixels / centre_strip.size

        # Decision thresholds
        # Two paths to emergency stop:
        # 1. Gradient break + close = approaching an obstacle
        # 2. Very high close ratio alone = already pressed against something
        obstacle_detected = discontinuity > 0.15 or close_ratio > 0.3
        emergency_stop = (discontinuity > 0.25 and close_ratio > 0.3) or close_ratio > 0.5

        return DepthResult(
            emergency_stop=emergency_stop,
            obstacle_detected=obstacle_detected,
            discontinuity_strength=discontinuity,
            centre_close_ratio=close_ratio,
            depth_map=depth_map if return_depth_map else None,
        )

    def render_depth_overlay(self, frame, depth_result: DepthResult):
        """Render depth map as a semi-transparent overlay — for Claude Cam demo.

        Args:
            frame: numpy array (BGR)
            depth_result: DepthResult with depth_map populated

        Returns:
            frame with depth overlay
        """
        import cv2

        if depth_result.depth_map is None:
            return frame

        depth_map = depth_result.depth_map
        h, w = frame.shape[:2]

        # Normalise and colourise
        depth_norm = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        depth_resized = cv2.resize(depth_colour, (w, h))

        # Blend with original
        blended = cv2.addWeighted(frame, 0.6, depth_resized, 0.4, 0)

        # Warning text
        if depth_result.emergency_stop:
            cv2.putText(
                blended,
                "!! DEPTH STOP !!",
                (w // 2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        return blended
