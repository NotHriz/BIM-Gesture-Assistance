#!/usr/bin/env python3
"""
Tkinter GUI integrated with your GestureEngine.

- Expects engine.process_frame(frame, timestamp_ms) -> (hands_landmarks, prediction_str)
  where hands_landmarks is a list of hands, each hand is a list of (x_norm, y_norm) tuples.
- Draws skeletons for detected hands on the video feed.
- Start / Stop camera button, Load Model (calls engine.load_model or engine.load if present),
  Snapshot button, safe shutdown on close.

Save as App/gui_enhanced.py and run with:
    python App/gui_enhanced.py

Make sure your project root is in PYTHONPATH so `from engine import GestureEngine` works.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import time
import os
import traceback

from engine import GestureEngine  # your engine.py provided earlier


# MediaPipe-like default hand connections for 21 keypoints
DEFAULT_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm cross-links
]


def draw_landmarks(frame_bgr, hands_landmarks, connections=None, normalized=True):
    """
    Draw landmarks and connections for one or more hands on a BGR frame.

    - frame_bgr: np.ndarray (H,W,3) BGR image
    - hands_landmarks: list of hands, each hand is list of (x, y) tuples
                       where x,y are normalized (0..1) if normalized=True
    - connections: list of (i,j) index pairs. Defaults to DEFAULT_HAND_CONNECTIONS.
    - normalized: whether landmark coords are in [0..1] range (True) or pixel coords (False)

    Returns an annotated copy of frame_bgr.
    """
    import numpy as np
    import cv2

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if not hands_landmarks:
        return out

    if connections is None:
        connections = DEFAULT_HAND_CONNECTIONS

    colors = [
        (0, 255, 0),    # green
        (0, 200, 255),  # orange
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
    ]

    for hi, hand in enumerate(hands_landmarks):
        pts = []
        for lm in hand:
            # lm expected as (x, y)
            try:
                x, y = lm
            except Exception:
                # skip invalid landmark
                continue
            if normalized:
                px = int(round(x * w))
                py = int(round(y * h))
            else:
                px = int(round(x))
                py = int(round(y))
            pts.append((px, py))

        # Draw points
        for (px, py) in pts:
            cv2.circle(out, (px, py), 4, colors[hi % len(colors)], -1)

        # Draw connections
        for (a, b) in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], colors[hi % len(colors)], 2)

    return out


class GestureApp:
    def __init__(self, window, cam_index=1, canvas_size=(640, 480)):
        self.window = window
        self.window.title("BIM Gesture System")

        # Engine
        self.engine = GestureEngine()

        # Camera state
        self.cam_index = cam_index
        self.cap = None
        self.running = False

        self.canvas_w, self.canvas_h = canvas_size

        # Last processed BGR frame (for snapshot)
        self._last_frame = None

        # Keep reference to PhotoImage to avoid GC
        self._photo_ref = None

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main = ttk.Frame(self.window, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl = ttk.Frame(main)
        ctrl.pack(fill=tk.X, pady=(0, 6))

        self.btn_startstop = ttk.Button(ctrl, text="Start", command=self.toggle_camera)
        self.btn_startstop.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_load = ttk.Button(ctrl, text="Load Model", command=self._on_load_model)
        self.btn_load.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_snapshot = ttk.Button(ctrl, text="Snapshot", command=self._on_snapshot)
        self.btn_snapshot.pack(side=tk.LEFT, padx=(0, 6))

        # Canvas for video
        self.canvas = tk.Canvas(main, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Prediction display
        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(bottom, text="Prediction:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.pred_var = tk.StringVar(value="—")
        self.label_result = ttk.Label(bottom, textvariable=self.pred_var, font=("Arial", 16, "bold"))
        self.label_result.pack(side=tk.LEFT, padx=(8, 0))

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.running:
            return
        # Try the preferred index first; fallback to 0
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera error", f"Could not open webcam (tried indices {self.cam_index} and 0).")
                self.cap = None
                return
        self.running = True
        self.btn_startstop.config(text="Stop")
        self.window.after(10, self._update_frame)

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
        self.canvas.delete("all")
        self.pred_var.set("—")

    def _on_load_model(self):
        path = filedialog.askopenfilename(title="Select model file", filetypes=[("All files", "*.*")])
        if not path:
            return
        # Try to call engine.load_model(path) or engine.load(path)
        try:
            load_fn = getattr(self.engine, "load_model", None) or getattr(self.engine, "load", None)
            if callable(load_fn):
                load_fn(path)
                messagebox.showinfo("Model Loaded", f"Loaded model from:\n{path}")
                self.pred_var.set(f"Loaded: {os.path.basename(path)}")
            else:
                messagebox.showinfo("Load Model", "Engine does not expose load_model(path).")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _on_snapshot(self):
        if self._last_frame is None:
            messagebox.showwarning("No frame", "No frame available to save.")
            return
        path = filedialog.asksaveasfilename(title="Save snapshot", defaultextension=".png",
                                            filetypes=[("PNG image", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            cv2.imwrite(path, self._last_frame)
            messagebox.showinfo("Saved", f"Snapshot saved to {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def get_ai_frame_and_prediction(self, frame_bgr):
        """
        Calls engine.process_frame(frame, timestamp_ms) and interprets the result.

        Expected engine output (as we updated engine.py):
            (hands_landmarks, prediction_str)
        Where hands_landmarks is a list of hands; each hand is list of (x_norm, y_norm) tuples.

        Returns:
            annotated_frame_bgr, prediction_str
        """
        ts = int(time.time() * 1000)
        try:
            result = self.engine.process_frame(frame_bgr, ts)
        except Exception as e:
            # Engine crashed; return original frame and error message
            print("Engine exception:", e)
            traceback.print_exc()
            return frame_bgr, f"Engine error"

        # Expect result to be (hands_landmarks, prediction)
        if isinstance(result, tuple) and len(result) == 2:
            hands_landmarks, prediction = result
            # If hands_landmarks looks like a list of hands, draw them
            if isinstance(hands_landmarks, list):
                # Normalize detection: if engine returns a single hand as flat list of floats,
                # attempt to detect and convert, but our engine returns list-of-hands.
                annotated = draw_landmarks(frame_bgr, hands_landmarks, normalized=True)
                return annotated, prediction
            else:
                # Unexpected format: show original frame
                return frame_bgr, prediction

        # If engine returned something else (legacy), try to interpret:
        # If engine returned only a prediction string, just return frame + string
        return frame_bgr, str(result)

    def _update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # try again soon
            self.window.after(30, self._update_frame)
            return

        # Mirror for user-friendly interaction
        frame = cv2.flip(frame, 1)

        # Save BGR frame for snapshot
        self._last_frame = frame.copy()

        # Get annotated frame and prediction
        annotated_frame, prediction = self.get_ai_frame_and_prediction(frame)

        # Convert BGR -> RGB -> PIL -> ImageTk
        try:
            rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        except Exception:
            # If annotated_frame is invalid, fallback to raw frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)
        # Resize to canvas size while maintaining aspect ratio
        pil_img = pil_img.resize((self.canvas_w, self.canvas_h))
        photo = ImageTk.PhotoImage(pil_img)
        self._photo_ref = photo  # prevent GC

        # Draw on canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        # Update prediction display
        try:
            self.pred_var.set(prediction)
        except Exception:
            self.pred_var.set(str(prediction))

        # Schedule next update
        self.window.after(10, self._update_frame)

    def _on_close(self):
        # Stop camera
        self.stop_camera()
        # Try engine cleanup
        for cleanup in ("close", "shutdown", "release"):
            fn = getattr(self.engine, cleanup, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root, cam_index=1, canvas_size=(640, 480))
    root.mainloop()