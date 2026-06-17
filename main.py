import os
import sys
import threading
import time
import cv2
import numpy as np
from typing import Optional, List
import tkinter as tk
from tkinter import messagebox
from face_detector import FaceShapeDetector
from emotion_detector import EmotionDetector
import ai_assistant

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from detector   import HandDetector
from classifier import GestureClassifier
from utils      import FPSCounter, TextToSpeech, WordSuggestions,                       copy_to_clipboard, save_text_to_file
from gui        import SignLanguageApp

class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        
        self.frame:           Optional[np.ndarray] = None
        self.fps:             float = 0.0
        self.raw_label:       Optional[str] = None
        self.raw_conf:        float = 0.0
        self.committed:       Optional[str] = None
        self.hand_detected:   bool = False
        self.cooldown_prog:   float = 1.0
        self.running:         bool = True
        self.mode:            str = "SIGN_LANGUAGE"
        self.face_shape:      Optional[str] = None
        self.face_prob:       float = 0.0
        
        self.ai_face_text:    Optional[str] = None    
        self.ai_response:     Optional[str] = None    
        
        self.emotion:         str = "neutral"
        self.emotion_emoji:   str = "😐"
        self.emotion_conf:    float = 0.0
        
        self.conv_history:    List[dict] = []

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def get(self, *keys):
        with self._lock:
            return tuple(getattr(self, k) for k in keys)

def _download_task_model(model_dir: str = "model") -> bool:

    task_path = os.path.join(model_dir, "hand_landmarker.task")
    if os.path.exists(task_path):
        return True
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    import urllib.request
    print(f"Downloading hand_landmarker.task …")
    os.makedirs(model_dir, exist_ok=True)
    urllib.request.urlretrieve(url, task_path)
    return os.path.exists(task_path)

def ensure_model(model_dir: str = "model") -> bool:

    _download_task_model(model_dir)
    pkl_path = os.path.join(model_dir, "sign_model.pkl")   
    if os.path.exists(pkl_path):
        return True
    
    print("No trained model found — auto-generating + training …")
    dataset_dir = os.path.join(ROOT, "dataset")
    sys.path.insert(0, dataset_dir)
    from generate_synthetic import generate_dataset
    from train_model import train

    generate_dataset()
    
    train()
    return os.path.exists(pkl_path)

class CaptureThread(threading.Thread):
    def __init__(self, state: AppState, camera_index: int = 0):
        super().__init__(daemon=True, name="CaptureThread")
        self._state    = state
        self._cam_idx  = camera_index

        self._detector      = HandDetector(
            max_hands=2,
            detection_confidence=0.70,
            tracking_confidence=0.60,
        )
        self._face_detector = FaceShapeDetector()
        self._emotion_det   = EmotionDetector(sample_every=20)
        self._classifier    = GestureClassifier(
            model_dir="model",
            buffer_size=10,
            min_confidence=0.45,      
            majority_threshold=0.35,
            cooldown_seconds=1.0,
        )
        self._fps            = FPSCounter(window=30)

        self._face_shape_cache: dict = {}
        self._fetching_face:    bool = False
        self._last_face_shape:  Optional[str] = None

    def run(self):
        cap = cv2.VideoCapture(self._cam_idx)
        if not cap.isOpened():
            print(f"[capture] ERROR: Cannot open camera index {self._cam_idx}")
            self._state.update(running=False)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("[capture] Camera opened successfully.")

        while self._state.running:
            ret, raw_frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(raw_frame, 1)

            self._emotion_det.feed_frame(frame)
            em = self._emotion_det.get_emotion()
            self._state.update(
                emotion=em["emotion"],
                emotion_emoji=em["emoji"],
                emotion_conf=em["confidence"],
            )

            try:
                
                annotated     = self._detector.process(frame)
                features      = self._detector.get_features()
                hand_detected = self._detector.is_hand_detected()

                raw_label, raw_conf, committed = self._classifier.update(features)
                cooldown_prog = self._classifier.cooldown_progress()

                bbox = self._detector.get_bounding_box(frame)
                if bbox and raw_label and raw_label not in ("NOTHING", None):
                    self._detector.draw_bounding_box(
                        annotated, bbox,
                        label=raw_label,
                        confidence=raw_conf,
                    )

                self._draw_cooldown_ring(annotated, cooldown_prog)

                annotated = self._face_detector.process(annotated)
                shape, prob = self._face_detector.get_face_shape(frame.shape)
                
                if shape and shape != self._last_face_shape:
                    self._last_face_shape = shape
                    if shape in self._face_shape_cache:
                        self._state.update(ai_face_text=self._face_shape_cache[shape])
                    elif not self._fetching_face:
                        self._fetching_face = True
                        self._state.update(ai_face_text=f"✨ Analyzing {shape}…")

                        def fetch_style(target_shape):
                            res = ai_assistant.get_style_recommendations(target_shape)
                            self._face_shape_cache[target_shape] = res
                            self._fetching_face = False
                            if self._last_face_shape == target_shape:
                                self._state.update(ai_face_text=res)

                        threading.Thread(
                            target=fetch_style, args=(shape,), daemon=True
                        ).start()

                fps = self._fps.tick()
                self._draw_fps(annotated, fps)

                state_update = dict(
                    frame=annotated,
                    fps=fps,
                    raw_label=raw_label,
                    raw_conf=raw_conf,
                    hand_detected=hand_detected,
                    cooldown_prog=cooldown_prog,
                )
                if shape:
                    state_update["face_shape"] = shape
                    state_update["face_prob"] = prob
                if committed is not None:
                    state_update["committed"] = committed
                self._state.update(**state_update)

            except Exception as e:
                import traceback
                print(f"[capture] ERROR: {e}")
                traceback.print_exc()

            time.sleep(0.005)  

        cap.release()
        self._detector.release()
        self._face_detector.release()
        print("[capture] Thread exiting.")

    @staticmethod
    def _draw_fps(frame: np.ndarray, fps: float):
        h, w = frame.shape[:2]
        color = (63, 185, 80) if fps >= 20 else (255, 193, 7) if fps >= 12 else (248, 81, 73)
        cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_cooldown_ring(frame: np.ndarray, progress: float):
        cx, cy, radius = 36, 36, 24
        angle = int(360 * progress)
        color = (63, 185, 80) if progress >= 0.9 else (255, 193, 7) if progress >= 0.5 else (248, 81, 73)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360, (48, 54, 61), 4, cv2.LINE_AA)
        if angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, color, 4, cv2.LINE_AA)
        label = "RDY" if progress >= 0.99 else f"{int(progress*100)}%"
        cv2.putText(frame, label, (cx - 15, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

class AppController:

    POLL_MS = 33

    def __init__(self):
        self._state    = AppState()
        self._tts      = TextToSpeech()
        self._words    = WordSuggestions(max_suggestions=3)
        self._text     = ""
        self._capture: Optional[CaptureThread] = None

        self._app = SignLanguageApp(
            on_clear       = self._clear,
            on_copy        = self._copy,
            on_save        = self._save,
            on_speak       = self._speak,
            on_delete      = self._delete,
            on_word_select = self._word_select,
            on_mode_change = self._set_mode,
            on_enhance     = self._enhance,
            on_send_to_ai  = self._send_to_ai,
            on_summarize   = self._summarize,
        )

        self._capture = CaptureThread(self._state, camera_index=0)
        self._capture.start()

        self._app.after(self.POLL_MS, self._poll)
        self._app.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self):
        self._app.mainloop()

    def _poll(self):
        if not self._state.running:
            return

        (frame, fps, raw_label, raw_conf,
         committed, hand_detected, cooldown_prog,
         emotion, emotion_emoji, emotion_conf,
         ai_face_text, ai_response, face_shape, face_prob) = self._state.get(
            "frame", "fps", "raw_label", "raw_conf",
            "committed", "hand_detected", "cooldown_prog",
            "emotion", "emotion_emoji", "emotion_conf",
            "ai_face_text", "ai_response", "face_shape", "face_prob",
        )

        if frame is not None:
            self._app.update_frame(frame)

        self._app.update_fps(fps)
        self._app.update_sign(raw_label, raw_conf, hand_detected)
        self._app.update_cooldown(cooldown_prog)
        if face_shape:
            self._app.update_face_shape(face_shape, face_prob)

        self._app.update_emotion(emotion, emotion_emoji, emotion_conf)

        if committed:
            self._handle_committed(committed, raw_conf)
            self._state.update(committed=None)

        if ai_face_text:
            self._app.show_ai_response(ai_face_text)
            self._state.update(ai_face_text=None)

        if ai_response:
            self._app.add_conversation_turn("Tariani AI", ai_response)
            self._app._set_ai_status("✓ Response received")
            if self._app.is_auto_speak():
                threading.Thread(target=lambda: self._tts.speak(ai_response),
                                 daemon=True).start()
            self._state.update(ai_response=None)

        self._app.after(self.POLL_MS, self._poll)

    def _handle_committed(self, label: str, confidence: float):
        if label == "SPACE":
            if self._text and not self._text.endswith(" "):
                if self._app.is_auto_speak():
                    last_word = self._words.current_word_prefix(self._text)
                    if last_word:
                        threading.Thread(target=lambda: self._tts.speak(last_word), daemon=True).start()
                self._text += " "
        elif label == "BACKSPACE":
            self._text = self._text[:-1]
        elif label == "NOTHING":
            pass
        else:
            self._text += label

        self._app.update_text(self._text)
        self._app.flash_letter(
            label if label not in ("SPACE", "BACKSPACE") else
            "⎵" if label == "SPACE" else "⌫"
        )
        self._app.add_history(label, confidence)

        prefix = self._words.current_word_prefix(self._text)
        self._app.update_suggestions(self._words.suggest(prefix))

    def _clear(self):
        self._text = ""
        self._app.update_text("")
        self._app.update_suggestions([])
        self._app.clear_history()

    def _delete(self):
        if self._text:
            self._text = self._text[:-1]
            self._app.update_text(self._text)
            self._app.update_suggestions(
                self._words.suggest(self._words.current_word_prefix(self._text))
            )

    def _copy(self) -> bool:
        return copy_to_clipboard(self._text) if self._text else False

    def _save(self):
        return save_text_to_file(self._text, parent=self._app)

    def _speak(self):
        self._tts.speak(self._text if self._text else "Nothing to speak.")

    def _enhance(self):

        if not self._text.strip():
            return

        def run():
            enhanced = ai_assistant.enhance_sign_language(self._text)
            self._text = enhanced + " "
            self._app.update_text(self._text)

        threading.Thread(target=run, daemon=True).start()

    def _send_to_ai(self):

        typed = self._app.get_pending_chat_message()
        user_msg = typed if typed else self._text.strip()

        if not user_msg:
            return

        emotion  = getattr(self._state, "emotion", "neutral")
        history  = list(getattr(self._state, "conv_history", []))

        if not typed:
            self._app.add_conversation_turn("You", user_msg)

        history.append({"role": "user", "text": user_msg})
        self._state.update(conv_history=history)

        def run():
            response = ai_assistant.chat_with_ai(
                user_msg, emotion=emotion, history=history
            )
            new_history = list(getattr(self._state, "conv_history", []))
            new_history.append({"role": "assistant", "text": response})
            self._state.update(conv_history=new_history, ai_response=response)

        threading.Thread(target=run, daemon=True).start()

        if not typed:
            self._text = ""
            self._app.update_text("")

    def _summarize(self):

        history = list(getattr(self._state, "conv_history", []))
        emotion = getattr(self._state, "emotion", "neutral")

        self._app.show_ai_response("📊 Generating session summary…")

        def run():
            summary = ai_assistant.summarize_session(history, dominant_emotion=emotion)
            self._state.update(ai_response=summary)

        threading.Thread(target=run, daemon=True).start()

    def _word_select(self, word: str):
        words = self._text.rstrip().split()
        if words:
            words[-1] = word
        self._text = " ".join(words) + " "
        self._app.update_text(self._text)

    def _set_mode(self, mode: str):
        self._state.update(mode=mode)
        if mode == "FACE_SHAPE":
            self._app.update_text("Face Shape Mode Active. The AI is analyzing your face shape and styling.")
        elif mode == "CONVERSATION":
            self._app.update_text("Conversation Mode Active. Sign your words, then click 'Send to AI'.")
        else:
            self._app.update_text(self._text)
        self._app.update_suggestions([])

    def _on_close(self):
        self._state.update(running=False)
        time.sleep(0.15)
        self._app.destroy()

def show_loading_splash():
    import customtkinter as ctk
    splash = ctk.CTk()
    splash.title("AI Vision Assistant — Loading")
    splash.geometry("440x220")
    splash.resizable(False, False)
    splash.configure(fg_color="#0d1117")

    ctk.CTkLabel(splash, text="🤖  AI Vision Assistant",
                  font=("Inter", 20, "bold"),
                  text_color="#58a6ff").pack(pady=(35, 8))
    msg = ctk.CTkLabel(splash, text="Initialising AI models…",
                        font=("Inter", 12),
                        text_color="#8b949e")
    msg.pack()

    bar = ctk.CTkProgressBar(splash, mode="indeterminate",
                               fg_color="#30363d", progress_color="#58a6ff")
    bar.pack(padx=40, pady=20, fill="x")
    bar.start()

    splash.update()
    return splash, bar, msg

def main():
    splash, bar, msg_label = show_loading_splash()

    model_ready = False
    error_msg   = None

    def setup_in_background():
        nonlocal model_ready, error_msg
        try:
            model_ready = ensure_model()
        except Exception as exc:
            error_msg = str(exc)

    t = threading.Thread(target=setup_in_background, daemon=True)
    t.start()

    while t.is_alive():
        splash.update()
        time.sleep(0.05)

    bar.stop()
    try:
        splash.destroy()
    except Exception:
        pass

    if not model_ready:
        messagebox.showerror(
            "Startup Error",
            f"Could not initialise the model.\n\n{error_msg or 'Unknown error'}\n\n"
            "Run: python dataset/train_model.py"
        )
        sys.exit(1)

    controller = AppController()
    controller.run()

if __name__ == "__main__":
    main()
