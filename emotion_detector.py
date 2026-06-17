import threading
import numpy as np
from typing import Optional

EMOTION_EMOJI = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "surprise": "😲",
    "fear":     "😨",
    "disgust":  "🤢",
    "neutral":  "😐",
}

class EmotionDetector:

    def __init__(self, sample_every: int = 20):

        self._sample_every = sample_every
        self._frame_count  = 0
        self._lock         = threading.Lock()

        self._emotion:     str   = "neutral"
        self._confidence:  float = 0.0
        self._all_scores:  dict  = {}
        self._is_busy:     bool  = False
        self._available:   bool  = False

        self._deepface = None
        self._load_thread = threading.Thread(target=self._load_deepface, daemon=True)
        self._load_thread.start()

    def _load_deepface(self):

        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._available = True
            print("[EmotionDetector] DeepFace loaded successfully.")
        except ImportError:
            print("[EmotionDetector] DeepFace not installed. Emotion detection disabled.")
        except Exception as e:
            print(f"[EmotionDetector] Load error: {e}")

    def feed_frame(self, frame: np.ndarray):

        if not self._available or self._is_busy:
            return

        self._frame_count += 1
        if self._frame_count % self._sample_every != 0:
            return

        frame_copy = frame.copy()
        t = threading.Thread(target=self._analyze, args=(frame_copy,), daemon=True)
        t.start()

    def _analyze(self, frame: np.ndarray):
        self._is_busy = True
        try:
            results = self._deepface.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            if results and len(results) > 0:
                r = results[0]
                dominant = r.get("dominant_emotion", "neutral")
                scores   = r.get("emotion", {})
                conf     = scores.get(dominant, 0.0) / 100.0

                with self._lock:
                    self._emotion    = dominant
                    self._confidence = conf
                    self._all_scores = {k: v / 100.0 for k, v in scores.items()}
        except Exception as e:
            pass  
        finally:
            self._is_busy = False

    def get_emotion(self) -> dict:

        with self._lock:
            return {
                "emotion":    self._emotion,
                "confidence": self._confidence,
                "emoji":      EMOTION_EMOJI.get(self._emotion, "😐"),
                "scores":     self._all_scores.copy(),
                "available":  self._available,
            }

    def is_available(self) -> bool:
        return self._available
