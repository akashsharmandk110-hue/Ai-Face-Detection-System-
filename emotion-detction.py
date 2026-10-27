import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from deepface import DeepFace
import threading
import time


class EmotionDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("950x680")
        self.window.configure(bg="#102A43")
        self.window.minsize(900, 620)

        self.cap = None
        self.running = False
        self.current_frame = None
        self.current_emotion = "No face"
        self.emotion_scores = {}
        self.lock = threading.Lock()
        self.thread = None

        self.setup_ui()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Top application header
        header_frame = tk.Frame(self.window, bg="#102A43", height=90)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        title = tk.Label(
            header_frame,
            text="Face Emotion Assistant",
            font=("Helvetica", 24, "bold"),
            bg="#102A43",
            fg="#F7FAFC"
        )
        title.pack(anchor="w", padx=20, pady=(16, 3))

        subtitle = tk.Label(
            header_frame,
            text="Real-time facial emotion analysis",
            font=("Helvetica", 11),
            bg="#102A43",
            fg="#BEE3F8"
        )
        subtitle.pack(anchor="w", padx=20)

        # Status ribbon below header
        self.status_frame = tk.Frame(self.window, bg="#16324F")
        self.status_frame.pack(fill=tk.X)

        self.status_label = tk.Label(
            self.status_frame,
            text="Ready: start the camera",
            font=("Helvetica", 10, "bold"),
            bg="#16324F",
            fg="#E6FFFA"
        )
        self.status_label.pack(fill=tk.X, padx=20, pady=8)

        main_frame = tk.Frame(self.window, bg="#102A43")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(main_frame, bg="#1B365C", highlightthickness=2, highlightbackground="#9BE7D6")
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        sidebar = tk.Frame(main_frame, bg="#203B59", width=280)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Detected Emotion", font=("Helvetica", 12, "bold"),
                 bg="#203B59", fg="#EAF7FF").pack(pady=(20, 5))
        self.lbl_emotion = tk.Label(sidebar, text="No face", font=("Helvetica", 16, "bold"),
                                     bg="#203B59", fg="#92E6A7")
        self.lbl_emotion.pack(pady=5)

        tk.Label(sidebar, text="Confidence Breakdown", font=("Helvetica", 11, "underline"),
                 bg="#203B59", fg="#EAF7FF").pack(pady=(15, 5))

        self.stats_frame = tk.Frame(sidebar, bg="#203B59")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        self.bars = {}
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        for emo in emotions:
            frame = tk.Frame(self.stats_frame, bg="#203B59")
            frame.pack(fill=tk.X, pady=2)

            lbl = tk.Label(frame, text=emo.capitalize(), font=("Helvetica", 9),
                           bg="#203B59", fg="#EAF7FF", width=8, anchor="w")
            lbl.pack(side=tk.LEFT)

            pbar = ttk.Progressbar(frame, orient="horizontal", length=120,
                                   mode="determinate", maximum=100)
            pbar.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self.bars[emo] = pbar

        # Instruction card
        info = tk.Label(sidebar,
                        text="Place your face in front of the camera and press Start.",
                        bg="#203B59", fg="#BEE3F8", justify=tk.LEFT,
                        wraplength=235, font=("Helvetica", 9))
        info.pack(pady=(12, 8))

        control_frame = tk.Frame(self.window, bg="#102A43")
        control_frame.pack(fill=tk.X, pady=10)

        self.btn_start = tk.Button(control_frame, text="Start Camera", command=self.start_camera,
                                    font=("Helvetica", 11), bg="#2ECC71", fg="white", width=14)
        self.btn_start.pack(side=tk.LEFT, padx=15)

        self.btn_stop = tk.Button(control_frame, text="Stop Camera", command=self.stop_camera,
                                  font=("Helvetica", 11), bg="#C0392B", fg="white", width=14,
                                  state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_snap = tk.Button(control_frame, text="Take Snapshot", command=self.take_snapshot,
                                  font=("Helvetica", 11), bg="#2980B9", fg="white", width=14,
                                  state=tk.DISABLED)
        self.btn_snap.pack(side=tk.RIGHT, padx=15)

    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_snap.config(state=tk.NORMAL)

        self.thread = threading.Thread(target=self.analyze_emotions_worker, daemon=True)
        self.thread.start()

        self.update_video_feed()

    def stop_camera(self):
        self.running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.config(image='')
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_snap.config(state=tk.DISABLED)
        self.lbl_emotion.config(text="---")
        self.current_emotion = "N/A"
        self.emotion_scores = {}

        for bar in self.bars.values():
            bar['value'] = 0

    def update_video_feed(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            with self.lock:
                self.current_frame = frame.copy()
                emotion = self.current_emotion
                scores = self.emotion_scores.copy()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.putText(rgb_frame, f"Emotion: {emotion.upper()}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.lbl_emotion.config(text=emotion.capitalize() if emotion != "N/A" else "---")

            for emo, bar in self.bars.items():
                if emo in scores:
                    score = max(0.0, min(100.0, float(scores.get(emo, 0))))
                    bar['value'] = score
                else:
                    bar['value'] = 0

        if self.running:
            self.window.after(30, self.update_video_feed)

    def normalize_result(self, result):
        """Return the first usable DeepFace emotion dict from any supported return shape."""
        if isinstance(result, dict):
            if "dominant_emotion" in result or "emotion" in result:
                return result
            return None

        if isinstance(result, list):
            if not result:
                return None

            # result can be list[dict], list[list[dict]], or list[list[list[dict]]]
            for item in result:
                if isinstance(item, dict):
                    if "dominant_emotion" in item or "emotion" in item:
                        return item
                elif isinstance(item, list):
                    for nested in item:
                        if isinstance(nested, dict):
                            if "dominant_emotion" in nested or "emotion" in nested:
                                return nested
                        elif isinstance(nested, list):
                            for inner in nested:
                                if isinstance(inner, dict):
                                    if "dominant_emotion" in inner or "emotion" in inner:
                                        return inner

        return None

    def analyze_emotions_worker(self):
        while self.running:
            with self.lock:
                frame_to_process = self.current_frame.copy() if self.current_frame is not None else None

            if frame_to_process is None:
                time.sleep(0.1)
                continue

            detected = False
            last_error = None
            backends = ['opencv', 'retinaface', 'mtcnn', 'ssd']

            for backend in backends:
                try:
                    results = DeepFace.analyze(
                        img_path=frame_to_process,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend=backend,
                        silent=True
                    )

                    result = self.normalize_result(results)
                    if result is None:
                        last_error = ValueError(f"No face data from {backend}")
                        continue

                    emotion = str(result.get("dominant_emotion", "neutral")).lower()
                    emotion_scores = result.get("emotion", {}) or {}
                    if not isinstance(emotion_scores, dict):
                        emotion_scores = {}

                    with self.lock:
                        self.current_emotion = emotion
                        self.emotion_scores = emotion_scores

                    detected = True
                    break

                except Exception as exc:
                    last_error = exc
                    print(f"DeepFace {backend} error: {exc}")

            if not detected:
                print(f"DeepFace error: {last_error}")
                with self.lock:
                    self.current_emotion = "N/A"
                    self.emotion_scores = {}

            time.sleep(0.1)

    def take_snapshot(self):
        with self.lock:
            if self.current_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_snapshot_{timestamp}.png"
                cv2.imwrite(filename, self.current_frame)
                messagebox.showinfo("Saved", f"Snapshot saved successfully as:\n{filename}")

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root, "College Project - Face Emotion Recognition")
    root.mainloop()

