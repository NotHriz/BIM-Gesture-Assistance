import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from engine import GestureEngine # Import your AI logic
import time

class GestureApp:
    def __init__(self, window):
        self.window = window
        self.window.title("BIM Gesture System")
        
        # Connect to your AI Engine
        self.engine = GestureEngine()
        self.cap = cv2.VideoCapture(1)

        # UI Components
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.label_result = ttk.Label(window, text="Prediction: ...", font=("Arial", 20))
        self.label_result.pack(pady=20)

        self.update_app()

    def update_app(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            ts = int(time.time() * 1000)
            
            # CALLING THE AI ENGINE
            processed_frame, prediction = self.engine.process_frame(frame, ts)

            # Update GUI
            img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.label_result.config(text=f"Gesture: {prediction}")

        self.window.after(10, self.update_app)

root = tk.Tk()
app = GestureApp(root)
root.mainloop()