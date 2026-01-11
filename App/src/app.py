import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import time
import os
import traceback
import numpy as np

# Ensure your engine.py is in the same folder or PYTHONPATH
from engine import GestureEngine  

# MediaPipe-like default hand connections for 21 keypoints
DEFAULT_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)               # Palm cross-links
]

def draw_landmarks(frame_bgr, hands_landmarks, connections=None, normalized=True):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if not hands_landmarks:
        return out
    if connections is None:
        connections = DEFAULT_HAND_CONNECTIONS

    colors = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]

    for hi, hand in enumerate(hands_landmarks):
        pts = []
        for lm in hand:
            x, y = lm
            px = int(round(x * w)) if normalized else int(round(x))
            py = int(round(y * h)) if normalized else int(round(y))
            pts.append((px, py))

        for (px, py) in pts:
            cv2.circle(out, (px, py), 4, colors[hi % len(colors)], -1)
        for (a, b) in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(out, pts[a], pts[b], colors[hi % len(colors)], 2)
    return out

class GestureApp:
    def __init__(self, window, cam_index=1, canvas_size=(640, 480)):
        self.window = window
        self.window.title("BIM Gesture System - Multi-Model Edition")

        # Initialize Engine
        try:
            self.engine = GestureEngine()
        except Exception as e:
            messagebox.showerror("Engine Error", f"Failed to load models: {e}")
            traceback.print_exc()

        # Camera state
        self.cam_index = cam_index
        self.cap = None
        self.running = False
        self.canvas_w, self.canvas_h = canvas_size
        self._last_frame = None
        self._photo_ref = None

        self._build_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main = ttk.Frame(self.window, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Control Row ---
        ctrl = ttk.Frame(main)
        ctrl.pack(fill=tk.X, pady=(0, 6))

        self.btn_startstop = ttk.Button(ctrl, text="Start Camera", command=self.toggle_camera)
        self.btn_startstop.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(ctrl, text="Source:").pack(side=tk.LEFT, padx=(5, 2))
        available_cams = self._list_cameras()
        self.cam_select = ttk.Combobox(ctrl, values=available_cams, width=5, state="readonly")
        self.cam_select.set(available_cams[0])
        self.cam_select.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(ctrl, text="Model:").pack(side=tk.LEFT, padx=(10, 2))
        self.model_switch = ttk.Combobox(ctrl, values=["Random Forest", "SVM", "CNN"], state="readonly", width=15)
        self.model_switch.current(0)
        self.model_switch.pack(side=tk.LEFT, padx=(0, 6))
        self.model_switch.bind("<<ComboboxSelected>>", self._on_model_change)

        self.btn_snapshot = ttk.Button(ctrl, text="Snapshot", command=self._on_snapshot)
        self.btn_snapshot.pack(side=tk.RIGHT)

        # --- Canvas for Video ---
        self.canvas = tk.Canvas(main, width=self.canvas_w, height=self.canvas_h, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        bottom = ttk.Frame(main)
        bottom.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(bottom, text="Result: ", font=("Arial", 12)).pack(side=tk.LEFT)
        self.pred_var = tk.StringVar(value="Ready")
        self.label_result = ttk.Label(bottom, textvariable=self.pred_var, font=("Arial", 16, "bold"))
        self.label_result.pack(side=tk.LEFT)

    def _list_cameras(self):
        valid_indices = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                valid_indices.append(i)
                cap.release()
        return valid_indices if valid_indices else [0]

    def _on_model_change(self, event):
        selected = self.model_switch.get()
        self.engine.set_model(selected)

    def toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.running: return
        idx = int(self.cam_select.get())
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera.")
            return
        self.running = True
        self.btn_startstop.config(text="Stop Camera")
        self._update_frame()

    def stop_camera(self):
        self.running = False
        self.btn_startstop.config(text="Start Camera")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")
        self.pred_var.set("Ready")

    def _on_snapshot(self):
        if self._last_frame is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            cv2.imwrite(path, self._last_frame)
            messagebox.showinfo("Success", "Snapshot saved!")

    def _update_frame(self):
        if not self.running or self.cap is None: return
        ret, frame = self.cap.read()
        if not ret:
            self.window.after(10, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        self._last_frame = frame.copy()
        ts = int(time.time() * 1000)
        
        # Get AI Results
        try:
            hands_landmarks, prediction = self.engine.process_frame(frame, ts)
            annotated_frame = draw_landmarks(frame, hands_landmarks)
        except Exception as e:
            annotated_frame = frame
            prediction = "Error"
            print(e)

        # Convert to GUI format
        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((self.canvas_w, self.canvas_h))
        photo = ImageTk.PhotoImage(img)
        self._photo_ref = photo
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.pred_var.set(prediction)

        self.window.after(10, self._update_frame)

    def _on_close(self):
        self.stop_camera()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.mainloop()