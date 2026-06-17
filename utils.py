import time
import threading
import os
from collections import deque
from typing import List, Optional

import pyperclip
import pyttsx3
import nltk

def _ensure_nltk_words():
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words", quiet=True)

class FPSCounter:

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> float:

        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps(self) -> float:
        return self.tick.__func__(self) if False else self._compute()

    def _compute(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

class TextToSpeech:

    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text: str) -> None:

        if not text.strip():
            return
        if self._busy:
            return  
        thread = threading.Thread(target=self._run, args=(text,), daemon=True)
        thread.start()

    def _run(self, text: str) -> None:
        with self._lock:
            self._busy = True
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 0.9)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
            finally:
                self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

class WordSuggestions:

    def __init__(self, max_suggestions: int = 3):
        self.max_suggestions = max_suggestions
        self._words: List[str] = []
        self._load_words()

    def _load_words(self) -> None:
        _ensure_nltk_words()
        try:
            from nltk.corpus import words as nltk_words
            
            self._words = [
                w.lower()
                for w in nltk_words.words()
                if 2 <= len(w) <= 12 and w.isalpha()
            ]
        except Exception:
            
            self._words = [
                "hello", "help", "have", "here", "how",
                "the", "this", "that", "there", "they",
                "world", "work", "with", "what", "when",
                "yes", "you", "your",
                "no", "not", "now",
                "is", "in", "it",
                "am", "are", "at",
                "be", "by",
                "can", "cat",
                "do", "dog",
            ]

    def suggest(self, prefix: str) -> List[str]:

        if not prefix:
            return []
        prefix = prefix.lower()
        matches = [w for w in self._words if w.startswith(prefix)]
        
        matches.sort(key=lambda w: (len(w), w))
        return matches[: self.max_suggestions]

    def current_word_prefix(self, text: str) -> str:

        if not text:
            return ""
        parts = text.rstrip().split()
        return parts[-1] if parts else ""

def save_text_to_file(text: str, parent=None) -> Optional[str]:

    try:
        from tkinter import filedialog
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Recognised Text",
            initialfile="sign_language_output.txt",
        )
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            return filepath
    except Exception as exc:
        print(f"[utils] save_text_to_file error: {exc}")
    return None

def copy_to_clipboard(text: str) -> bool:

    try:
        pyperclip.copy(text)
        return True
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
            return True
        except Exception:
            return False

class CooldownTimer:

    def __init__(self, cooldown_seconds: float = 1.5):
        self.cooldown = cooldown_seconds
        self._last_trigger: float = 0.0

    def ready(self) -> bool:
        return (time.perf_counter() - self._last_trigger) >= self.cooldown

    def trigger(self) -> None:
        self._last_trigger = time.perf_counter()

    def remaining(self) -> float:

        remaining = self.cooldown - (time.perf_counter() - self._last_trigger)
        return max(0.0, remaining)

    def reset(self) -> None:
        self._last_trigger = 0.0
