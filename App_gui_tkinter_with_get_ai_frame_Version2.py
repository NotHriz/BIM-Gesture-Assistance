#!/usr/bin/env python3
"""
Tkinter GUI with a placeholder get_ai_frame(frame) that returns (annotated_frame, prediction).

- Video feed (webcam)
- Output display (prediction text)
- Start/Stop button

Integration: Replace ModelWrapper.get_ai_frame(...) with your real model code.
If your model provides landmarks, return an annotated frame with the skeleton drawn and a (label, confidence) tuple.

Dependencies: pip install opencv-python pillow
Run: python App/gui_tkinter_with_get_ai_frame.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import time
import math
import os

FRAME_UPDATE_MS = 30  # ~30 FPS


class ModelWrapper:
    """
    Demo/placeholder model wrapper.

    Implement get_ai_frame(frame) to:
      - run your model on `frame` (BGR image as numpy.ndarray),
      - produce an annotated BGR frame (draw skeleton / landmarks)
      - return (annotated_frame, (label, confidence))
    """

    def __init__(self):
        self.loaded = False
        self.model_path = None

    def load(self, path: str):
        """Optional: load a model file. In this demo, just mark loaded."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.loaded = True
        self.model_path = path

    def get_ai_frame(self, frame: np.ndarray):
        """
        Placeholder that returns:
          - annotated frame (BGR) with a simulated skeleton overlay
          - prediction tuple (label: str, confidence: float)

        Replace the internals of this function with your model inference:
          1. Preprocess `frame` as needed.
          2. Run model to get landmarks + prediction.
          3. Draw landmarks/skeleton onto a copy of `frame`.
          4. Return (annotated_frame, (label, confidence))
        """
        # Work on a copy to avoid mutating original frame
        out = frame.copy()
        h, w = out.shape[:2]

        # Simulate skeleton landmarks around the center (21 points like a hand)
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 6

        # Generate 21 "landmarks" for a plausible hand layout (demo only)
        landmarks = []
        # wrist
        landmarks.append((cx - radius // 2, cy + radius // 2))
        # palm center
        landmarks.append((cx, cy))
        # base points for 5 fingers
        for f in range(5):
            base_x = cx - radius + (f * (2 * radius // 4))
            base_y = cy - radius // 6
            # finger joints: 3 joints + tip
            for j in range(4):
                # push finger points upward with increasing offset
                offset_y = base_y - (j * (radius // 3))
                offset_x = base_x + int(math.sin((f + j) * 0.6) * (radius // 8))
                landmarks.append((int(offset_x), int(offset_y)))

        # Draw circles for landmarks
        for (lx, ly) in landmarks:
            cv2.circle(out, (lx, ly), 4, (0, 255, 0), -1)
        # Draw simple skeleton lines (connect sequentially in groups)
        # wrist -> palm -> finger bases
        cv2.line(out, landmarks[0], landmarks[1], (0, 180, 0), 2)
        # connect palm to each finger base and along finger
        idx = 2
        for f in range(5):
            # connect palm to base of the finger
            cv2.line(out, landmarks[1], landmarks[idx], (0, 180, 0), 2)
            # connect joints along finger
            for j in range(3):
                cv2.line(out, landmarks[idx + j], landmarks[idx + j + 1], (0, 180, 0), 2)
            idx += 4

        # Example prediction: brightness heuristic (demo)
        avg = float(frame.mean())
        if avg > 100:
            label = "open"
            conf = min(0.99, (avg - 100) / 150.0 + 0.1)
        else:
            label = "closed"
            conf = min(0.99, (100 - avg) / 200.0 + 0.1)

        # Optionally annotate text on frame
        text = f"{label} ({conf:.2f})"
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        return out, (label, conf)


class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BIM Gesture Assistance — Tkinter GUI")

        self.model = ModelWrapper()

        # Camera
        self.cap = None
        self.running = False

        # Last frame for snapshot or debug
        self._last_frame = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Video area (Label)
        self.video_label = ttk.Label(main)
        self.video_label.grid(row=0, column=0, columnspan=3, pady=(0, 8))

        # Prediction display
        ttk.Label(main, text="Prediction:").grid(row=1, column=0, sticky="w")
        self.pred_var = tk.StringVar(value="—")
        self.pred_label = ttk.Label(main, textvariable=self.pred_var, font=("Segoe UI", 12, "bold"))
        self.pred_label.grid(row=1, column=1, sticky="w")

        # Start/Stop button
        self.btn_startstop = ttk.Button(main, text="Start", command=self.toggle_camera)
        self.btn_startstop.grid(row=1, column=2, sticky="e")

        # Configure resizing behavior
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open the webcam.")
            return
        self.running = True
        self.btn_startstop.config(text="Stop")
        self._update_loop()

    def stop_camera(self):
        if not self.running:
            return
        self.running = False
        self.btn_startstop.config(text="Start")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.video_label.config(image="")
        self.pred_var.set("—")

    def _update_loop(self):
        """Read frame, call get_ai_frame, update UI, schedule next."""
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # try again next tick
            self.root.after(FRAME_UPDATE_MS, self._update_loop)
            return

        self._last_frame = frame

        try:
            annotated_frame, prediction = self.model.get_ai_frame(frame)
        except Exception as e:
            # Ensure GUI stays responsive even if model fails
            annotated_frame = frame.copy()
            prediction = ("error", 0.0)
            print("get_ai_frame error:", e)

        # Convert BGR->RGB and to PIL ImageTk for display
        display_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_img)
        # Resize to fit window while maintaining aspect
        display_w = 640
        display_h = int(pil_img.height * (display_w / pil_img.width))
        pil_img = pil_img.resize((display_w, display_h))
        imgtk = ImageTk.PhotoImage(pil_img)

        # Keep reference to avoid GC
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # Update prediction display
        label, conf = prediction
        self.pred_var.set(f"{label} ({conf:.2f})")

        self.root.after(FRAME_UPDATE_MS, self._update_loop)

    def close(self):
        self.stop_camera()
        self.root.quit()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    app.run()