import os
import time
from collections import deque, Counter
from typing import Optional, Tuple, List, Dict

import numpy as np
import joblib

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPECIAL_LABELS = ["SPACE", "BACKSPACE", "NOTHING"]
ALL_LABELS = LETTERS + SPECIAL_LABELS

class GestureClassifier:

    def __init__(
        self,
        model_dir: str = "model",
        buffer_size: int = 15,
        min_confidence: float = 0.55,
        majority_threshold: float = 0.60,
        cooldown_seconds: float = 1.5,
    ):
        self.model_dir = model_dir
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        self.majority_threshold = majority_threshold
        self.cooldown_seconds = cooldown_seconds

        self._model = None
        self._encoder = None
        self._buffer: deque = deque(maxlen=buffer_size)
        self._last_commit_time: float = 0.0
        self._last_committed: Optional[str] = None

        self._history: List[Dict] = []

        self._load_model()

    def _load_model(self) -> None:
        model_path   = os.path.join(self.model_dir, "sign_model.pkl")
        encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")

        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            print(
                "[classifier] WARNING: Model files not found. "
                "Run dataset/train_model.py first.\n"
                f"  Expected: {model_path}\n"
                f"           {encoder_path}"
            )
            return

        self._model   = joblib.load(model_path)
        self._encoder = joblib.load(encoder_path)
        print(f"[classifier] Model loaded from '{self.model_dir}'")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._encoder is not None

    def predict_raw(
        self, features: np.ndarray
    ) -> Tuple[Optional[str], float]:

        if not self.is_loaded:
            return None, 0.0

        features = np.asarray(features).flatten()

        n_expected = self._model.n_features_in_
        if features.shape[0] > n_expected:
            
            left_hand = features[:n_expected]
            right_hand = features[n_expected:n_expected*2]
            
            if np.any(left_hand):
                features = left_hand
            else:
                features = right_hand

        features = features.reshape(1, -1)
        probs   = self._model.predict_proba(features)[0]
        idx     = int(np.argmax(probs))
        conf    = float(probs[idx])
        label   = self._encoder.inverse_transform([idx])[0]

        if conf < self.min_confidence:
            return None, conf

        return label, conf

    def update(
        self, features: Optional[np.ndarray]
    ) -> Tuple[Optional[str], float, Optional[str]]:

        if features is None:
            self._buffer.clear()
            return None, 0.0, None

        raw_label, raw_conf = self.predict_raw(features)

        if raw_label is None or raw_label == "NOTHING":
            self._buffer.append(None)
            return raw_label, raw_conf, None

        self._buffer.append(raw_label)

        valid = [x for x in self._buffer if x is not None]
        if len(valid) < self.buffer_size * self.majority_threshold:
            return raw_label, raw_conf, None  

        counter = Counter(valid)
        top_label, top_count = counter.most_common(1)[0]
        majority_frac = top_count / len(self._buffer)

        if majority_frac < self.majority_threshold:
            return raw_label, raw_conf, None

        now = time.perf_counter()
        if (now - self._last_commit_time) < self.cooldown_seconds:
            return raw_label, raw_conf, None

        self._last_commit_time = now
        self._last_committed = top_label
        self._buffer.clear()

        self._history.append({
            "label": top_label,
            "confidence": raw_conf,
            "timestamp": now,
        })
        if len(self._history) > 50:
            self._history.pop(0)

        return raw_label, raw_conf, top_label

    def buffer_fill_ratio(self) -> float:

        if self.buffer_size == 0:
            return 0.0
        valid = sum(1 for x in self._buffer if x is not None)
        return valid / self.buffer_size

    def cooldown_remaining(self) -> float:

        elapsed = time.perf_counter() - self._last_commit_time
        remaining = self.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def cooldown_progress(self) -> float:

        elapsed = time.perf_counter() - self._last_commit_time
        return min(1.0, elapsed / self.cooldown_seconds)

    def get_history(self, n: int = 10) -> List[Dict]:

        return list(reversed(self._history[-n:]))

    def clear_history(self) -> None:
        self._history.clear()

    def reset_buffer(self) -> None:
        self._buffer.clear()

    def reset_cooldown(self) -> None:
        self._last_commit_time = 0.0
