import os
import urllib.request
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
from typing import Optional, Tuple

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
_DEFAULT_MODEL = os.path.join(_MODEL_DIR, "face_landmarker.task")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def _ensure_model():
    if not os.path.exists(_DEFAULT_MODEL):
        print(f"Downloading face landmarker model to {_DEFAULT_MODEL}...")
        os.makedirs(_MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _DEFAULT_MODEL)

class FaceShapeDetector:
    def __init__(self, model_path: str = _DEFAULT_MODEL):
        _ensure_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._result = None
        self._frame_ts_ms = 0

    def process(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self._frame_ts_ms += 33
        self._result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        annotated = frame.copy()
        if self._result and self._result.face_landmarks:
            self._draw_landmarks(annotated, self._result.face_landmarks[0])

        return annotated

    def get_face_shape(self, frame_shape) -> Tuple[Optional[str], float]:

        if not self._result or not self._result.face_landmarks:
            return None, 0.0

        lms = self._result.face_landmarks[0]
        h, w = frame_shape[:2]

        def get_pt(idx):
            return (lms[idx].x * w, lms[idx].y * h)

        def dist(idx1, idx2):
            p1, p2 = get_pt(idx1), get_pt(idx2)
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        face_length     = dist(10, 152)    
        forehead_width  = dist(54, 284)    
        cheekbone_width = dist(234, 454)   
        jaw_width       = dist(132, 361)   

        cw = max(cheekbone_width, 1.0)
        fw = max(forehead_width, 1.0)

        hw_ratio = face_length / cw

        jaw_cheek_ratio = jaw_width / cw

        jaw_fore_ratio = jaw_width / fw

        if hw_ratio >= 1.45:
            shape = "Oblong"          
        elif hw_ratio >= 1.25:
            if jaw_fore_ratio < 0.78:
                shape = "Heart"       
            else:
                shape = "Oval"        
        else:
            
            if jaw_cheek_ratio >= 0.82:
                shape = "Square"      
            else:
                shape = "Round"       

        return f"{shape} Face", 1.0

    def _draw_landmarks(self, frame: np.ndarray, landmarks) -> None:

        h, w = frame.shape[:2]
        
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 1, (255, 166, 88), -1)

    def release(self):
        try:
            self._landmarker.close()
        except:
            pass
