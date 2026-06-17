import os
import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import cv2
from typing import Callable, Optional, List

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

P = {
    "bg":           "#060a12",
    "bg2":          "#0c1220",
    "card":         "#0f1621",
    "card2":        "#141d2e",
    "card3":        "#1a2540",
    "border":       "#1e2d45",
    "border2":      "#2a3f5f",
    "glow":         "#4f46e5",
    "accent":       "#7c3aed",   
    "accent2":      "#a78bfa",   
    "accent3":      "#06b6d4",   
    "gold":         "#f59e0b",   
    "success":      "#10b981",   
    "danger":       "#ef4444",
    "warning":      "#f97316",
    "text":         "#e2e8f0",
    "text2":        "#94a3b8",
    "text3":        "#475569",
    "white":        "#ffffff",
    "ai_bg":        "#130d2e",
    "user_bg":      "#0d2116",
}

FONT_UI   = ("Segoe UI", "Inter", "Arial")
FONT_MONO = ("Cascadia Code", "Consolas", "Courier New")

def _f(family, size, weight="normal"):
    return (family, size, weight)

def _ui(size, weight="normal"):
    return _f(FONT_UI[0], size, weight)

def _mono(size, weight="normal"):
    return _f(FONT_MONO[0], size, weight)

class GlowCard(ctk.CTkFrame):
    def __init__(self, parent, glow_color=None, **kwargs):
        kwargs.setdefault("fg_color", P["card"])
        kwargs.setdefault("corner_radius", 14)
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", glow_color or P["border2"])
        super().__init__(parent, **kwargs)

class PulseLED(ctk.CTkCanvas):
    def __init__(self, parent, size=12, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=P["card"], highlightthickness=0, **kwargs)
        self._size = size
        self._color = P["text3"]
        self._draw()

    def _draw(self):
        self.delete("all")
        s = self._size
        pad = 2
        self.create_oval(pad, pad, s-pad, s-pad,
                         fill=self._color, outline=self._color)

    def set_active(self, active: bool):
        self._color = P["success"] if active else P["text3"]
        self._draw()

class SuggestionBar(ctk.CTkFrame):
    def __init__(self, parent, on_select: Callable, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._on_select = on_select
        self._btns: List[ctk.CTkButton] = []

    def update_suggestions(self, words: List[str]):
        for b in self._btns:
            b.destroy()
        self._btns.clear()
        for w in words[:4]:
            b = ctk.CTkButton(
                self, text=w, height=26, width=0,
                font=_ui(10),
                fg_color="#1a1040",
                text_color=P["accent2"],
                hover_color=P["card3"],
                border_color=P["accent"],
                border_width=1,
                corner_radius=13,
                command=lambda ww=w: self._on_select(ww),
            )
            b.pack(side="left", padx=3)
            self._btns.append(b)

class PredictionHistory(ctk.CTkScrollableFrame):
    MAX = 40

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=P["card2"],
                         corner_radius=8,
                         scrollbar_button_color=P["border2"],
                         **kwargs)
        self._items: List = []

    def add(self, label: str, confidence: float):
        pct = int(confidence * 100)
        color = P["success"] if pct >= 80 else P["gold"] if pct >= 55 else P["danger"]
        lbl = ctk.CTkLabel(self,
                            text=f"  {label:<12}  {pct}%",
                            font=_mono(10),
                            text_color=color,
                            anchor="w")
        lbl.pack(fill="x", padx=4, pady=1)
        self._items.append(lbl)
        if len(self._items) > self.MAX:
            self._items[0].destroy()
            self._items.pop(0)

    def clear(self):
        for i in self._items:
            i.destroy()
        self._items.clear()

class ChatPanel(ctk.CTkScrollableFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=P["ai_bg"],
                         corner_radius=10,
                         scrollbar_button_color=P["border2"],
                         **kwargs)
        self._turns: List = []
        
        self._add_system("👋  Hi! I'm Tariani's AI. You can sign words or type below to chat with me. I also understand your emotions!")

    def _add_system(self, text: str):
        f = ctk.CTkFrame(self, fg_color="#0d1a30", corner_radius=10)
        f.pack(fill="x", padx=8, pady=(8, 4))
        ctk.CTkLabel(f, text=text,
                     font=_ui(11),
                     text_color=P["accent3"],
                     wraplength=310,
                     justify="left",
                     anchor="w").pack(padx=10, pady=8)
        self._turns.append(f)

    def add_turn(self, role: str, text: str):
        is_user = (role == "You")
        bg     = P["user_bg"] if is_user else P["ai_bg"]
        prefix = "🧑  You" if is_user else "🤖  Tariani AI"
        tc     = "#86efac" if is_user else P["accent2"]
        anchor = "e" if is_user else "w"

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="x", padx=6, pady=3)

        bubble = ctk.CTkFrame(outer, fg_color=bg, corner_radius=12,
                               border_width=1,
                               border_color="#2a1860" if not is_user else "#0a2e1a")
        bubble.pack(side="right" if is_user else "left",
                    fill="x", expand=True)

        ctk.CTkLabel(bubble, text=prefix,
                     font=_ui(9, "bold"),
                     text_color=tc,
                     anchor="w").pack(anchor="w", padx=10, pady=(7, 2))

        ctk.CTkLabel(bubble, text=text,
                     font=_ui(12),
                     text_color=P["text"],
                     wraplength=300,
                     justify="left",
                     anchor="w").pack(anchor="w", padx=10, pady=(0, 8))

        self._turns.append(outer)
        try:
            self._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def clear(self):
        for t in self._turns:
            try:
                t.destroy()
            except Exception:
                pass
        self._turns.clear()

class SignLanguageApp(ctk.CTk):
    def __init__(
        self,
        on_clear:       Callable,
        on_copy:        Callable,
        on_save:        Callable,
        on_speak:       Callable,
        on_delete:      Callable,
        on_word_select: Callable,
        on_mode_change: Callable,
        on_enhance:     Callable,
        on_send_to_ai:  Callable,
        on_summarize:   Callable,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._on_clear       = on_clear
        self._on_copy        = on_copy
        self._on_save        = on_save
        self._on_speak       = on_speak
        self._on_delete      = on_delete
        self._on_word_select = on_word_select
        self._on_mode_change = on_mode_change
        self._on_enhance     = on_enhance
        self._on_send_to_ai  = on_send_to_ai
        self._on_summarize   = on_summarize

        self._photo:        Optional[ImageTk.PhotoImage] = None
        self._mic_active:   bool = False
        self._auto_speak:   bool = True

        self.title("✋  Tariani's Detector  |  AI Vision Assistant")
        self.geometry("1440x880")
        self.minsize(1200, 750)
        self.configure(fg_color=P["bg"])

        self._build_ui()

    def _build_ui(self):
        self._build_header()

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        content.grid_columnconfigure(0, weight=52)
        content.grid_columnconfigure(1, weight=40)
        content.grid_columnconfigure(2, weight=38)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=0)

        self._build_camera_col(content)
        self._build_ai_col(content)
        self._build_detection_col(content)

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color=P["card"], corner_radius=0, height=62)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        logo_f = ctk.CTkFrame(hdr, fg_color="transparent")
        logo_f.pack(side="left", padx=18)

        ctk.CTkLabel(logo_f,
                      text="✋",
                      font=_ui(22)).pack(side="left", padx=(0, 6))

        name_f = ctk.CTkFrame(logo_f, fg_color="transparent")
        name_f.pack(side="left")

        ctk.CTkLabel(name_f,
                      text="Tariani's Detector",
                      font=_ui(17, "bold"),
                      text_color=P["accent2"]).pack(anchor="w")
        ctk.CTkLabel(name_f,
                      text="AI Vision Assistant",
                      font=_ui(9),
                      text_color=P["text3"]).pack(anchor="w")

        right_f = ctk.CTkFrame(hdr, fg_color="transparent")
        right_f.pack(side="right", padx=18)

        self._fps_lbl = ctk.CTkLabel(right_f, text="FPS --",
                                      font=_mono(11),
                                      text_color=P["text3"])
        self._fps_lbl.pack(side="right", padx=(12, 0))

        self._emotion_badge = ctk.CTkLabel(right_f,
                                            text="😐 neutral",
                                            font=_ui(11),
                                            fg_color=P["card2"],
                                            text_color=P["text2"],
                                            corner_radius=10,
                                            padx=10, pady=4)
        self._emotion_badge.pack(side="right")

        self._speak_toggle = ctk.CTkSwitch(right_f,
                                            text="Auto-speak AI",
                                            font=_ui(10),
                                            text_color=P["text2"],
                                            progress_color=P["accent"],
                                            button_color=P["accent2"],
                                            command=self._toggle_autospeak)
        self._speak_toggle.select()   
        self._speak_toggle.pack(side="right", padx=18)

        mode_f = ctk.CTkFrame(hdr, fg_color="transparent")
        mode_f.pack(side="left", padx=30)

        self._mode_seg = ctk.CTkSegmentedButton(
            mode_f,
            values=["✋ Sign Language", "👤 Face Shape", "💬 Chat AI"],
            command=self._handle_mode,
            font=_ui(12, "bold"),
            selected_color=P["accent"],
            selected_hover_color="#6d28d9",
            unselected_color=P["card2"],
            unselected_hover_color=P["card3"],
            height=36,
        )
        self._mode_seg.set("✋ Sign Language")
        self._mode_seg.pack()

    def _build_camera_col(self, parent):
        col = ctk.CTkFrame(parent, fg_color="transparent")
        col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        col.grid_rowconfigure(0, weight=1)
        col.grid_rowconfigure(1, weight=0)
        col.grid_columnconfigure(0, weight=1)

        cam_card = GlowCard(col, glow_color=P["glow"])
        cam_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        cam_top = ctk.CTkFrame(cam_card, fg_color=P["card2"],
                                corner_radius=0, height=34)
        cam_top.pack(fill="x")
        cam_top.pack_propagate(False)

        self._cam_led = PulseLED(cam_top)
        self._cam_led.pack(side="left", padx=(10, 4), pady=10)
        self._cam_led.set_active(True)

        ctk.CTkLabel(cam_top, text="LIVE CAMERA FEED",
                      font=_ui(9, "bold"),
                      text_color=P["success"]).pack(side="left")

        self._det_lbl = ctk.CTkLabel(cam_top, text="No hand",
                                      font=_ui(10),
                                      text_color=P["text3"])
        self._det_lbl.pack(side="right", padx=12)

        self._cam_label = tk.Label(cam_card, bg=P["bg"], cursor="none")
        self._cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        cd_strip = ctk.CTkFrame(cam_card, fg_color=P["card2"],
                                 corner_radius=0, height=30)
        cd_strip.pack(fill="x")
        cd_strip.pack_propagate(False)

        ctk.CTkLabel(cd_strip, text="Cooldown",
                      font=_ui(8),
                      text_color=P["text3"]).pack(side="left", padx=10)

        self._cd_bar = ctk.CTkProgressBar(cd_strip, height=5, corner_radius=3,
                                           fg_color=P["border"],
                                           progress_color=P["accent"])
        self._cd_bar.set(1.0)
        self._cd_bar.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=12)

        em_card = GlowCard(col, glow_color=P["accent3"])
        em_card.grid(row=1, column=0, sticky="nsew")
        em_card.configure(height=100)
        em_card.grid_propagate(False)

        top_f = ctk.CTkFrame(em_card, fg_color="transparent")
        top_f.pack(fill="x", padx=14, pady=(10, 4))
        
        ctk.CTkLabel(top_f, text="FACE & EMOTION DETECTION",
                      font=_ui(8, "bold"),
                      text_color=P["text3"]).pack(side="left")
                      
        self._face_lbl = ctk.CTkLabel(top_f, text="Detecting face shape...", font=_ui(10, "bold"), text_color=P["accent"])
        self._face_lbl.pack(side="right")

        em_inner = ctk.CTkFrame(em_card, fg_color="transparent")
        em_inner.pack(fill="x", padx=14)

        self._em_emoji = ctk.CTkLabel(em_inner, text="😐",
                                       font=_ui(26))
        self._em_emoji.pack(side="left", padx=(0, 10))

        em_right = ctk.CTkFrame(em_inner, fg_color="transparent")
        em_right.pack(side="left", fill="x", expand=True)

        self._em_lbl = ctk.CTkLabel(em_right, text="Neutral",
                                     font=_ui(13, "bold"),
                                     text_color=P["accent3"],
                                     anchor="w")
        self._em_lbl.pack(anchor="w")

        self._em_bar = ctk.CTkProgressBar(em_right, height=5, corner_radius=3,
                                           fg_color=P["border"],
                                           progress_color=P["accent3"])
        self._em_bar.set(0.0)
        self._em_bar.pack(fill="x", pady=(4, 0))

    def _build_ai_col(self, parent):
        col = GlowCard(parent, glow_color=P["accent"])
        col.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 8))

        ai_hdr = ctk.CTkFrame(col, fg_color=P["ai_bg"], corner_radius=0, height=36)
        ai_hdr.pack(fill="x")
        ai_hdr.pack_propagate(False)

        ctk.CTkLabel(ai_hdr, text="🤖  TARIANI AI",
                      font=_ui(10, "bold"),
                      text_color=P["accent2"]).pack(side="left", padx=12)

        ctk.CTkButton(ai_hdr, text="🗑 Clear",
                       height=24, width=60,
                       font=_ui(9),
                       fg_color=P["card3"],
                       hover_color=P["card2"],
                       corner_radius=6,
                       command=self._clear_chat).pack(side="right", padx=8, pady=5)

        self._ai_status = ctk.CTkLabel(col, text="Ready • Type, speak or sign to chat",
                                        font=_ui(9),
                                        text_color=P["text3"])
        self._ai_status.pack(anchor="w", padx=12, pady=(6, 2))

        self._chat = ChatPanel(col, height=360)
        self._chat.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        input_card = ctk.CTkFrame(col, fg_color=P["card2"], corner_radius=10)
        input_card.pack(fill="x", padx=10, pady=(0, 8))

        input_label = ctk.CTkLabel(input_card, text="💬  Type or 🎙 Speak to AI",
                                    font=_ui(9, "bold"),
                                    text_color=P["accent2"])
        input_label.pack(anchor="w", padx=10, pady=(8, 4))

        msg_row = ctk.CTkFrame(input_card, fg_color="transparent")
        msg_row.pack(fill="x", padx=8, pady=(0, 8))

        self._chat_input = ctk.CTkEntry(
            msg_row,
            placeholder_text="Type a message…",
            font=_ui(12),
            fg_color=P["card3"],
            text_color=P["text"],
            placeholder_text_color=P["text3"],
            border_color=P["border2"],
            border_width=1,
            corner_radius=8,
            height=38,
        )
        self._chat_input.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._chat_input.bind("<Return>", lambda e: self._send_typed())

        self._mic_btn = ctk.CTkButton(
            msg_row, text="🎙",
            width=38, height=38,
            font=_ui(16),
            fg_color=P["accent"],
            hover_color=P["accent2"],
            corner_radius=8,
            command=self._start_mic,
        )
        self._mic_btn.pack(side="left")

        send_btn = ctk.CTkButton(
            msg_row, text="➤",
            width=38, height=38,
            font=_ui(14, "bold"),
            fg_color=P["accent3"],
            hover_color="#0891b2",
            corner_radius=8,
            command=self._send_typed,
        )
        send_btn.pack(side="left", padx=(6, 0))

        bot_row = ctk.CTkFrame(col, fg_color="transparent")
        bot_row.pack(fill="x", padx=10, pady=(0, 10))

        ctk.CTkButton(bot_row,
                       text="📤  Send Sign Text to AI",
                       height=36,
                       font=_ui(11, "bold"),
                       fg_color=P["accent"],
                       hover_color="#6d28d9",
                       corner_radius=8,
                       command=self._do_send_sign).pack(side="left", fill="x",
                                                         expand=True, padx=(0, 4))

        ctk.CTkButton(bot_row,
                       text="📊",
                       height=36, width=44,
                       font=_ui(14),
                       fg_color=P["card3"],
                       hover_color=P["card2"],
                       border_color=P["border2"],
                       border_width=1,
                       corner_radius=8,
                       command=self._on_summarize).pack(side="left")

    def _build_detection_col(self, parent):
        col = ctk.CTkScrollableFrame(parent, fg_color="transparent",
                                      scrollbar_button_color=P["border2"])
        col.grid(row=0, column=2, rowspan=2, sticky="nsew")

        det_card = GlowCard(col, glow_color=P["accent2"])
        det_card.pack(fill="x", pady=(0, 8))

        self._det_title = ctk.CTkLabel(det_card, text="DETECTED SIGN",
                                        font=_ui(8, "bold"),
                                        text_color=P["text3"])
        self._det_title.pack(anchor="w", padx=14, pady=(12, 2))

        self._big_letter = ctk.CTkLabel(det_card, text="—",
                                         font=_mono(72, "bold"),
                                         text_color=P["accent2"])
        self._big_letter.pack(pady=(0, 4))

        det_status_row = ctk.CTkFrame(det_card, fg_color="transparent")
        det_status_row.pack(fill="x", padx=14, pady=(0, 8))

        self._det_led2 = PulseLED(det_status_row)
        self._det_led2.pack(side="left", padx=(0, 6), pady=4)

        self._det_status_lbl = ctk.CTkLabel(det_status_row,
                                             text="Waiting…",
                                             font=_ui(10),
                                             text_color=P["text3"])
        self._det_status_lbl.pack(side="left")

        conf_f = ctk.CTkFrame(det_card, fg_color="transparent")
        conf_f.pack(fill="x", padx=14, pady=(0, 12))

        ctk.CTkLabel(conf_f, text="CONFIDENCE",
                      font=_ui(7, "bold"),
                      text_color=P["text3"]).pack(anchor="w")

        self._conf_bar = ctk.CTkProgressBar(conf_f, height=6, corner_radius=3,
                                             fg_color=P["border"],
                                             progress_color=P["success"])
        self._conf_bar.set(0.0)
        self._conf_bar.pack(fill="x")

        self._conf_pct = ctk.CTkLabel(conf_f, text="0%",
                                       font=_mono(9),
                                       text_color=P["text3"],
                                       anchor="e")
        self._conf_pct.pack(anchor="e")

        txt_card = GlowCard(col, glow_color=P["accent3"])
        txt_card.pack(fill="x", pady=(0, 8))

        txt_hdr = ctk.CTkFrame(txt_card, fg_color="transparent")
        txt_hdr.pack(fill="x", padx=14, pady=(10, 4))

        ctk.CTkLabel(txt_hdr, text="GENERATED TEXT",
                      font=_ui(8, "bold"),
                      text_color=P["text3"]).pack(side="left")

        self._char_count = ctk.CTkLabel(txt_hdr, text="0 chars",
                                         font=_ui(8),
                                         text_color=P["text3"])
        self._char_count.pack(side="right")

        self._textbox = ctk.CTkTextbox(
            txt_card, height=110,
            font=_mono(14),
            fg_color=P["card2"],
            text_color=P["text"],
            border_color=P["border2"],
            border_width=1,
            corner_radius=8,
            wrap="word",
        )
        self._textbox.pack(fill="x", padx=12, pady=(0, 8))
        self._textbox.configure(state="disabled")

        sugg_row = ctk.CTkFrame(txt_card, fg_color="transparent")
        sugg_row.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(sugg_row, text="💡",
                      font=_ui(11),
                      text_color=P["gold"]).pack(side="left", padx=(0, 4))
        self._sugg_bar = SuggestionBar(sugg_row, on_select=self._on_word_select)
        self._sugg_bar.pack(side="left")

        btn_f = ctk.CTkFrame(txt_card, fg_color="transparent")
        btn_f.pack(fill="x", padx=12, pady=(0, 12))

        btns = [
            ("⌫  Back",       P["card3"],  P["danger"],   self._on_delete),
            ("✕  Clear",       P["card3"],  P["warning"],  self._on_clear),
            ("⧉  Copy",        P["card3"],  P["accent2"],  self._do_copy),
            ("⬇  Save",        P["card3"],  P["success"],  self._do_save),
            ("🔊  Speak",       P["card3"],  P["text"],     self._do_speak),
            ("✨  AI Enhance",  P["accent"], P["white"],    self._do_enhance),
        ]

        for i, (txt, fg, tc, cmd) in enumerate(btns):
            b = ctk.CTkButton(
                btn_f, text=txt, height=34, width=0,
                font=_ui(10, "bold"),
                fg_color=fg, text_color=tc,
                hover_color=P["card2"],
                border_color=P["border2"], border_width=1,
                corner_radius=8, command=cmd,
            )
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)

        btn_f.grid_columnconfigure((0, 1), weight=1)

        hist_card = GlowCard(col)
        hist_card.pack(fill="x")

        ctk.CTkLabel(hist_card, text="PREDICTION LOG",
                      font=_ui(8, "bold"),
                      text_color=P["text3"]).pack(anchor="w", padx=14, pady=(10, 4))

        self._hist = PredictionHistory(hist_card, height=120)
        self._hist.pack(fill="x", padx=8, pady=(0, 10))

    def update_frame(self, frame: np.ndarray):
        lw = self._cam_label.winfo_width() or 640
        lh = self._cam_label.winfo_height() or 480
        h, w = frame.shape[:2]
        if lw > 1 and lh > 1:
            scale = min(lw / w, lh / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            frame = cv2.resize(frame, (nw, nh))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._cam_label.configure(image=photo)
        self._photo = photo

    def update_fps(self, fps: float):
        color = P["success"] if fps >= 20 else P["gold"] if fps >= 12 else P["danger"]
        self._fps_lbl.configure(text=f"FPS {fps:.0f}", text_color=color)

    def update_sign(self, label: Optional[str], confidence: float, hand_detected: bool):
        self._det_led2.set_active(hand_detected)
        if hand_detected:
            self._det_lbl.configure(text="● Hand detected", text_color=P["success"])
        else:
            self._det_lbl.configure(text="No hand", text_color=P["text3"])

        if label and label not in ("NOTHING", None):
            fs = 28 if len(label) > 4 else 72
            self._big_letter.configure(text=label,
                                        text_color=P["accent2"],
                                        font=_mono(fs, "bold"))
            self._det_status_lbl.configure(
                text=f"Detected  ·  {int(confidence*100)}%",
                text_color=P["text2"])
            self._conf_bar.set(confidence)
            cc = P["success"] if confidence >= 0.8 else P["gold"] if confidence >= 0.55 else P["danger"]
            self._conf_bar.configure(progress_color=cc)
            self._conf_pct.configure(text=f"{int(confidence*100)}%")
        else:
            self._big_letter.configure(text="—", text_color=P["text3"],
                                        font=_mono(72, "bold"))
            self._det_status_lbl.configure(
                text="Waiting…" if hand_detected else "No hand in frame",
                text_color=P["text3"])
            self._conf_bar.set(0.0)
            self._conf_pct.configure(text="0%")

    def update_face_shape(self, shape: str, prob: float):
        if shape:
            self._face_lbl.configure(text=f"{shape} ({int(prob*100)}%)")
        else:
            self._face_lbl.configure(text="No face detected")

    def update_emotion(self, emotion: str, emoji: str, confidence: float):
        self._em_emoji.configure(text=emoji)
        self._em_lbl.configure(text=emotion.capitalize())
        self._em_bar.set(confidence)
        self._emotion_badge.configure(
            text=f"{emoji}  {emotion.capitalize()}  {int(confidence*100)}%")
        colors = {
            "happy": P["success"], "sad": P["accent3"], "angry": P["danger"],
            "surprise": P["gold"], "fear": P["accent2"], "neutral": P["text3"],
        }
        c = colors.get(emotion.lower(), P["accent3"])
        self._em_lbl.configure(text_color=c)
        self._em_bar.configure(progress_color=c)

    def update_cooldown(self, progress: float):
        self._cd_bar.set(progress)
        self._cd_bar.configure(
            progress_color=P["success"] if progress >= 0.9 else
                           P["gold"]    if progress >= 0.5 else P["danger"])

    def update_text(self, text: str):
        self._textbox.configure(state="normal")
        self._textbox.delete("1.0", "end")
        self._textbox.insert("1.0", text)
        self._textbox.configure(state="disabled")
        self._textbox.see("end")
        self._char_count.configure(text=f"{len(text)} chars")

    def get_text(self) -> str:
        return self._textbox.get("1.0", "end-1c")

    def show_ai_response(self, text: str):

        self._chat.add_turn("AI", text)
        self._set_ai_status("✓ Response received")

    def add_conversation_turn(self, role: str, text: str):
        self._chat.add_turn(role, text)

    def update_suggestions(self, words: List[str]):
        self._sugg_bar.update_suggestions(words)

    def add_history(self, label: str, confidence: float):
        self._hist.add(label, confidence)

    def clear_history(self):
        self._hist.clear()

    def flash_letter(self, letter: str):
        self._big_letter.configure(text=letter, text_color=P["success"])
        self.after(300, lambda: self._big_letter.configure(text_color=P["accent2"]))

    def _set_ai_status(self, msg: str):
        self._ai_status.configure(text=msg)

    def _send_typed(self):
        msg = self._chat_input.get().strip()
        if not msg:
            return
        self._chat_input.delete(0, "end")
        self._chat.add_turn("You", msg)
        self._set_ai_status("🤖 Tariani AI is thinking…")
        
        self._pending_chat_msg = msg
        self._on_send_to_ai()

    def get_pending_chat_message(self) -> str:

        msg = getattr(self, "_pending_chat_msg", "")
        self._pending_chat_msg = ""
        return msg

    def _start_mic(self):
        if self._mic_active:
            return
        self._mic_active = True
        self._mic_btn.configure(text="🔴", fg_color=P["danger"])
        self._set_ai_status("🎙 Listening… speak now")

        def listen():
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.listen(source, timeout=6, phrase_time_limit=10)
                text = r.recognize_google(audio)
                self.after(0, lambda: self._on_voice_result(text))
            except Exception as e:
                self.after(0, lambda: self._on_voice_result(""))
            finally:
                self.after(0, self._reset_mic_btn)

        threading.Thread(target=listen, daemon=True).start()

    def _on_voice_result(self, text: str):
        if text:
            self._chat_input.delete(0, "end")
            self._chat_input.insert(0, text)
            self._send_typed()
        else:
            self._set_ai_status("❌ Could not hear you. Try again.")

    def _reset_mic_btn(self):
        self._mic_active = False
        self._mic_btn.configure(text="🎙", fg_color=P["accent"])

    def _clear_chat(self):
        self._chat.clear()

    def _handle_mode(self, value: str):
        mode_map = {
            "✋ Sign Language": "SIGN_LANGUAGE",
            "👤 Face Shape":    "FACE_SHAPE",
            "💬 Chat AI":       "CONVERSATION",
        }
        titles = {
            "SIGN_LANGUAGE": "DETECTED SIGN",
            "FACE_SHAPE":    "FACE SHAPE",
            "CONVERSATION":  "CONVERSATION MODE",
        }
        mode = mode_map.get(value, "SIGN_LANGUAGE")
        self._det_title.configure(text=titles[mode])
        self._on_mode_change(mode)

    def _toggle_autospeak(self):
        self._auto_speak = not self._auto_speak

    def is_auto_speak(self) -> bool:
        return self._auto_speak

    def _do_copy(self):
        ok = self._on_copy()
        self._show_toast("✓  Copied!" if ok else "⚠ Nothing to copy", error=not ok)

    def _do_save(self):
        self._on_save()

    def _do_speak(self):
        self._on_speak()
        self._show_toast("🔊  Speaking…")

    def _do_enhance(self):
        self.update_text("✨  AI is enhancing your text…")
        self._on_enhance()

    def _do_send_sign(self):
        self._on_send_to_ai()
        self._set_ai_status("🤖 Tariani AI is thinking…")

    def _show_toast(self, msg: str, error: bool = False, ms: int = 2200):
        try:
            t = ctk.CTkToplevel(self)
            t.overrideredirect(True)
            t.attributes("-topmost", True)
            t.configure(fg_color=P["card3"])
            color = P["danger"] if error else P["success"]
            ctk.CTkLabel(t, text=msg,
                          font=_ui(12, "bold"),
                          text_color=color,
                          padx=16, pady=8).pack()
            self.update_idletasks()
            x = self.winfo_x() + self.winfo_width() - 250
            y = self.winfo_y() + self.winfo_height() - 70
            t.geometry(f"+{x}+{y}")
            t.after(ms, t.destroy)
        except Exception:
            pass
