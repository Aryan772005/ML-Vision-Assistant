import os
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from typing import Optional, Tuple, List

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "hand_landmarker.task"
)

_FINGER_COLORS = {
    "thumb":  (255,  87,  34),   
    "index":  ( 33, 150, 243),   
    "middle": ( 76, 175,  80),   
    "ring":   (156,  39, 176),   
    "pinky":  (244,  67,  54),   
    "palm":   (255, 193,   7),   
}

_CONNECTIONS = [
    ("thumb",  [(0, 1), (1, 2),  (2, 3),   (3, 4)]),
    ("index",  [(0, 5), (5, 6),  (6, 7),   (7, 8)]),
    ("middle", [(0, 9), (9, 10), (10, 11), (11, 12)]),
    ("ring",   [(0, 13),(13, 14),(14, 15), (15, 16)]),
    ("pinky",  [(0, 17),(17, 18),(18, 19), (19, 20)]),
    ("palm",   [(0, 5), (5, 9),  (9, 13),  (13, 17), (17, 0)]),
]

_FINGERTIP_IDS = {4, 8, 12, 16, 20}

class HandDetector:

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        max_hands: int = 2,
        min_hand_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Run the app once with internet access to auto-download it,\n"
                "or manually place hand_landmarker.task in the model/ folder."
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self._result: Optional[mp_vision.HandLandmarkerResult] = None
        self._frame_ts_ms: int = 0   

    def process(self, frame: np.ndarray) -> np.ndarray:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33   
        self._result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        annotated = frame.copy()
        if self._result and self._result.hand_landmarks:
            for hand_lms in self._result.hand_landmarks:
                self._draw_custom_landmarks(annotated, hand_lms)

        return annotated

    def get_features(self) -> Optional[np.ndarray]:

        if not self._result or not self._result.hand_landmarks:
            return None

        features = np.zeros(84, dtype=np.float32)

        for lms, handedness in zip(self._result.hand_landmarks, self._result.handedness):
            
            is_left = (handedness[0].category_name == "Left")

            coords = np.array([[lm.x, -lm.y] for lm in lms], dtype=np.float32)

            coords -= coords[0]

            max_val = np.max(np.abs(coords))
            if max_val > 0:
                coords /= max_val

            flat = coords.flatten()

            if is_left:
                features[0:42] = flat
            else:
                features[42:84] = flat

        return features

    def get_bounding_box(
        self, frame: np.ndarray, padding: int = 20
    ) -> Optional[Tuple[int, int, int, int]]:

        if not self._result or not self._result.hand_landmarks:
            return None

        h, w = frame.shape[:2]
        lms = self._result.hand_landmarks[0]
        xs = [int(lm.x * w) for lm in lms]
        ys = [int(lm.y * h) for lm in lms]

        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(w, max(xs) + padding)
        y2 = min(h, max(ys) + padding)
        return x1, y1, x2, y2

    def is_hand_detected(self) -> bool:
        return bool(self._result and self._result.hand_landmarks)

    def _draw_custom_landmarks(self, frame: np.ndarray, hand_lms: list) -> None:

        h, w = frame.shape[:2]
        pts: List[Tuple[int, int]] = [
            (int(lm.x * w), int(lm.y * h)) for lm in hand_lms
        ]

        for finger, pairs in _CONNECTIONS:
            color = _FINGER_COLORS[finger]
            for a, b in pairs:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

        for i, pt in enumerate(pts):
            radius = 6 if i in _FINGERTIP_IDS else 4
            cv2.circle(frame, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, radius, (30, 30, 30), 1, cv2.LINE_AA)

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        confidence: float = 0.0,
        color: Tuple[int, int, int] = (88, 166, 255),
    ) -> None:

        x1, y1, x2, y2 = bbox

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        clen, th = 18, 3
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * clen, cy), color, th, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * clen), color, th, cv2.LINE_AA)

        if label:
            tag = f" {label}  {confidence * 100:.0f}% "
            (tw, th2), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th2 - 12), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, tag, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 2, cv2.LINE_AA,
            )

    def release(self) -> None:

        try:
            self._landmarker.close()
        except Exception:
            pass
