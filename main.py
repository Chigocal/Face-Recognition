import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import face_recognition
import datetime
import threading
import numpy as np
import os
import time
import json
import csv
import winsound
from collections import deque
import queue

# --- Configuration ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance_logs.json")

ADMIN_USER = "admin"
ADMIN_PASS = "1234"

# --- Thresholds (Calibrated for "Hold" Logic) ---
EAR_THRESHOLD = 0.20        # Blink (Lower = Harder to trigger accidentally)
YAW_THRESHOLD_LEFT = 0.65   # Look Left (Lower = Turn head more)
YAW_THRESHOLD_RIGHT = 1.50  # Look Right (Higher = Turn head more)
MATCH_THRESHOLD = 0.45      # Strictness of face matching
REQUIRED_HOLD_FRAMES = 6    # Must hold pose for ~6 processing cycles (approx 1 sec)

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NEXUS: Advanced Biometric System")
        self.geometry("1400x900")
        
        self.cap = None
        self.is_running = False
        self.authenticated = False
        self.active_mode = "idle" 
        
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.stop_event = threading.Event()
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.enrollment_queue = deque([])
        self.current_challenge = None
        self.enrollment_captures = []
        self.challenge_hold_counter = 0 # NEW: Counter for holding pose
        
        self.verification_cooldown = {} 
        self.ui_detected_faces = []
        self.ui_instruction_text = "Initializing..."
        self.ui_instruction_color = "gray"

        self.load_encodings()
        self.show_login_screen()

    # --- 1. SYSTEM RESET ---
    def reset_system_state(self):
        self.enrollment_queue.clear()
        self.enrollment_captures = []
        self.verification_cooldown = {}
        self.ui_detected_faces = []
        self.current_challenge = None
        self.challenge_hold_counter = 0
        self.active_mode = "idle"

    def show_login_screen(self):
        self.clear_ui()
        self.login_frame = ctk.CTkFrame(self, fg_color=("#1a1a2e", "#0f0f1e"))
        self.login_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        box = ctk.CTkFrame(self.login_frame, width=400, height=500, corner_radius=20, border_width=2, border_color="#00d4ff")
        box.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(box, text="NEXUS PRIME", font=("Segoe UI", 32, "bold"), text_color="#00d4ff").pack(pady=(50, 10))
        ctk.CTkLabel(box, text="Identity Management System", font=("Segoe UI", 12), text_color="#8b8b8b").pack(pady=(0, 40))

        self.user_entry = ctk.CTkEntry(box, placeholder_text="Username", width=280, height=50)
        self.user_entry.pack(pady=10)
        self.pass_entry = ctk.CTkEntry(box, placeholder_text="Password", show="*", width=280, height=50)
        self.pass_entry.pack(pady=10)

        ctk.CTkButton(box, text="AUTHENTICATE", command=self.attempt_login, width=280, height=50, 
                      fg_color="#00d4ff", text_color="black", font=("Segoe UI", 14, "bold")).pack(pady=40)

    def attempt_login(self):
        if self.user_entry.get() == ADMIN_USER and self.pass_entry.get() == ADMIN_PASS:
            self.authenticated = True
            self.reset_system_state()
            self.login_frame.destroy()
            self.setup_main_ui()
            self.start_engine()
        else:
            ctk.CTkLabel(self.login_frame, text="Access Denied", text_color="red").pack()

    # --- 2. MAIN UI ---
    def setup_main_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=("#121216", "#0a0a0f"))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(self.sidebar, text="NEXUS", font=("Segoe UI", 36, "bold"), text_color="#00d4ff").pack(pady=(40, 5))
        ctk.CTkLabel(self.sidebar, text="v7.0 Strict Hold", font=("Segoe UI", 10), text_color="#555").pack(pady=(0, 40))

        self.nav_button("👤  New Enrollment", self.switch_to_enrollment)
        self.nav_button("👁  Live Verification", self.switch_to_verification)
        self.nav_button("📋  Audit Logs", self.switch_to_logs)
        
        ctk.CTkButton(self.sidebar, text="🚪 Logout", command=self.logout, fg_color="#ff4757", height=40).pack(side="bottom", pady=30, padx=20, fill="x")

        self.main_area = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.switch_to_enrollment()

    def nav_button(self, text, cmd):
        ctk.CTkButton(self.sidebar, text=text, command=cmd, height=50, corner_radius=10,
                     fg_color="transparent", hover_color="#1e1e24", anchor="w", 
                     font=("Segoe UI", 14), text_color="#ddd").pack(pady=5, padx=15, fill="x")

    def switch_to_enrollment(self):
        self.active_mode = "register"
        self.clear_main_area()
        ctrl_panel = ctk.CTkFrame(self.main_area, width=350, fg_color="#1e1e24", corner_radius=15)
        ctrl_panel.pack(side="right", fill="y", padx=10)
        ctrl_panel.pack_propagate(False)
        video_panel = ctk.CTkFrame(self.main_area, fg_color="black", corner_radius=15)
        video_panel.pack(side="left", fill="both", expand=True, padx=10)
        self.camera_label = ctk.CTkLabel(video_panel, text="Camera Active", text_color="gray")
        self.camera_label.pack(fill="both", expand=True)

        ctk.CTkLabel(ctrl_panel, text="ENROLL USER", font=("Segoe UI", 22, "bold"), text_color="white").pack(pady=30)
        self.en_name = ctk.CTkEntry(ctrl_panel, placeholder_text="Full Legal Name", height=45)
        self.en_name.pack(pady=10, padx=20, fill="x")
        self.en_id = ctk.CTkEntry(ctrl_panel, placeholder_text="Employee ID", height=45)
        self.en_id.pack(pady=10, padx=20, fill="x")
        self.instruction_label = ctk.CTkLabel(ctrl_panel, text="Ready to Start", font=("Segoe UI", 16, "bold"), text_color="#00d4ff", wraplength=300)
        self.instruction_label.pack(pady=30)
        self.action_btn = ctk.CTkButton(ctrl_panel, text="START SEQUENCE", command=self.start_enrollment_sequence, height=50, fg_color="#00d4ff", text_color="black", font=("Segoe UI", 14, "bold"))
        self.action_btn.pack(pady=20, padx=20, side="bottom")

    def switch_to_verification(self):
        self.active_mode = "attendance"
        self.clear_main_area()
        video_panel = ctk.CTkFrame(self.main_area, fg_color="black", corner_radius=15)
        video_panel.pack(fill="both", expand=True)
        self.camera_label = ctk.CTkLabel(video_panel, text="Camera Active", text_color="gray")
        self.camera_label.pack(fill="both", expand=True)
        self.overlay_label = ctk.CTkLabel(self.main_area, text="", font=("Segoe UI", 24, "bold"), text_color="#00d4ff", bg_color="transparent")
        self.overlay_label.place(relx=0.5, rely=0.1, anchor="center")

    def switch_to_logs(self):
        self.active_mode = "idle"
        self.clear_main_area()
        top = ctk.CTkFrame(self.main_area, fg_color="transparent")
        top.pack(fill="x", pady=20)
        ctk.CTkLabel(top, text="Audit Logs", font=("Segoe UI", 24, "bold")).pack(side="left")
        ctk.CTkButton(top, text="📥 Export CSV", command=self.export_csv, width=120, fg_color="#2ecc71").pack(side="right")
        textbox = ctk.CTkTextbox(self.main_area, font=("Consolas", 12))
        textbox.pack(fill="both", expand=True)
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                logs = json.load(f)
                header = f"{'TIMESTAMP':<25} {'NAME':<30} {'CONFIDENCE':<10}\n" + "-"*80 + "\n"
                textbox.insert("0.0", header)
                for log in reversed(logs):
                    textbox.insert("end", f"{log['time']:<25} {log['name']:<30} {log['confidence']:<10}\n")
        textbox.configure(state="disabled")

    # --- 4. ENGINE ---
    def start_engine(self):
        self.is_running = True
        self.stop_event.clear()
        threading.Thread(target=self.camera_loop, daemon=True).start()
        threading.Thread(target=self.processing_loop, daemon=True).start()
        self.update_ui_loop()

    def camera_loop(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        while self.is_running and not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.frame_lock: self.latest_frame = frame.copy()
            time.sleep(0.01)
        self.cap.release()

    def processing_loop(self):
        frame_counter = 0
        while self.is_running and not self.stop_event.is_set():
            frame_counter += 1
            if frame_counter % 2 != 0: 
                time.sleep(0.01); continue

            frame = None
            with self.frame_lock:
                if self.latest_frame is not None: frame = self.latest_frame.copy()
            
            if frame is None or self.active_mode == "idle":
                time.sleep(0.05); continue

            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            if self.active_mode == "register" and self.current_challenge:
                self.process_enrollment(rgb, frame)
            elif self.active_mode == "attendance":
                self.process_attendance(rgb)

    def process_enrollment(self, rgb_small, full_frame):
        landmarks = face_recognition.face_landmarks(rgb_small)
        passed = False
        challenge = self.current_challenge

        if landmarks:
            face = landmarks[0]
            if challenge == "blink":
                l_ear = self.get_ear(face['left_eye'])
                r_ear = self.get_ear(face['right_eye'])
                if (l_ear + r_ear) / 2 < EAR_THRESHOLD: passed = True
            elif challenge == "left":
                nose = face['nose_tip'][0][0]
                l_eye = face['left_eye'][0][0]
                r_eye = face['right_eye'][3][0]
                # Stricter: Ratio must be LESS than threshold
                if abs(nose - r_eye) > 0 and (abs(nose - l_eye) / abs(nose - r_eye)) < YAW_THRESHOLD_LEFT: passed = True
            elif challenge == "right":
                nose = face['nose_tip'][0][0]
                l_eye = face['left_eye'][0][0]
                r_eye = face['right_eye'][3][0]
                # Stricter: Ratio must be MORE than threshold
                if abs(nose - r_eye) > 0 and (abs(nose - l_eye) / abs(nose - r_eye)) > YAW_THRESHOLD_RIGHT: passed = True
            elif challenge == "center":
                passed = True

        # --- NEW: HOLD POSE LOGIC ---
        if passed:
            self.challenge_hold_counter += 1
            if self.challenge_hold_counter >= 2: # Show user they are doing it right
                 self.ui_instruction_text = f"HOLD STILL... ({self.challenge_hold_counter}/{REQUIRED_HOLD_FRAMES})"
                 self.ui_instruction_color = "#00ff00"
            
            if self.challenge_hold_counter >= REQUIRED_HOLD_FRAMES:
                # SUCCESS
                self.challenge_hold_counter = 0 # Reset for next challenge
                self.current_challenge = None
                self.ui_instruction_text = "CAPTURING..."
                winsound.Beep(1000, 100)
                time.sleep(0.3)
                self.enrollment_captures.append(full_frame)
                self.after(0, self.next_enrollment_challenge)
        else:
            # If they break the pose, reset the counter
            self.challenge_hold_counter = 0
            if self.current_challenge:
                self.ui_instruction_text = f"Action Required: {self.current_challenge.upper()}"
                if self.current_challenge == "blink": self.ui_instruction_text = "Action: BLINK EYES"
                elif self.current_challenge == "left": self.ui_instruction_text = "Action: TURN HEAD LEFT"
                elif self.current_challenge == "right": self.ui_instruction_text = "Action: TURN HEAD RIGHT"
                elif self.current_challenge == "center": self.ui_instruction_text = "Action: LOOK CENTER"
                self.ui_instruction_color = "#ffd700"

    def process_attendance(self, rgb_small):
        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)
        detected = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            top *= 2; right *= 2; bottom *= 2; left *= 2
            name = "Unknown"
            conf = 0.0
            if self.known_face_encodings:
                dists = face_recognition.face_distance(self.known_face_encodings, enc)
                if len(dists) > 0:
                    min_dist = np.min(dists)
                    if min_dist < MATCH_THRESHOLD:
                        name = self.known_face_names[np.argmin(dists)]
                        conf = 1 - min_dist
                        self.after(0, lambda n=name, c=conf: self.log_attendance(n, c))
            detected.append((top, right, bottom, left, name, conf))
        self.ui_detected_faces = detected

    def update_ui_loop(self):
        if not self.is_running: return
        if self.active_mode in ["register", "attendance"] and hasattr(self, 'camera_label') and self.camera_label.winfo_exists():
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None: frame = self.latest_frame.copy()
            if frame is not None:
                if self.active_mode == "register":
                    if hasattr(self, 'instruction_label'):
                        self.instruction_label.configure(text=self.ui_instruction_text, text_color=self.ui_instruction_color)
                elif self.active_mode == "attendance":
                    for (t, r, b, l, name, conf) in self.ui_detected_faces:
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (l, t), (r, b), color, 2)
                        cv2.putText(frame, f"{name}", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
                self.camera_label.configure(image=ctk_img, text="")
        self.after(15, self.update_ui_loop)

    # --- 5. HELPERS ---
    def get_ear(self, eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    def start_enrollment_sequence(self):
        name = self.en_name.get().strip()
        uid = self.en_id.get().strip()
        if not name or not uid: return
        self.enrollment_captures = []
        self.enrollment_queue = deque(["center", "blink", "left", "right", "center"])
        self.action_btn.configure(state="disabled", text="IN PROGRESS...")
        self.next_enrollment_challenge()

    def next_enrollment_challenge(self):
        if not self.enrollment_queue:
            self.ui_instruction_text = "PROCESSING..."
            self.ui_instruction_color = "#00d4ff"
            threading.Thread(target=self.finish_enrollment_background, daemon=True).start()
            return
        
        # DELAY so user sees the change
        time.sleep(1.0) 
        self.challenge_hold_counter = 0 # Reset counter
        self.current_challenge = self.enrollment_queue.popleft()

    def finish_enrollment_background(self):
        try:
            encodings = []
            for frm in self.enrollment_captures:
                rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                if encs: encodings.append(encs[0])
            if not encodings:
                self.after(0, lambda: self.finish_enroll_ui(False, "Face not clear"))
                return
            avg_enc = np.mean(encodings, axis=0)
            if self.known_face_encodings:
                dists = face_recognition.face_distance(self.known_face_encodings, avg_enc)
                if np.min(dists) < MATCH_THRESHOLD:
                    self.after(0, lambda: self.finish_enroll_ui(False, "User exists!"))
                    return
            new_user = {"name": self.en_name.get().strip(), "id": self.en_id.get().strip(), "encodings": [e.tolist() for e in encodings], "created": str(datetime.datetime.now())}
            users = []
            if os.path.exists(USERS_FILE):
                with open(USERS_FILE, 'r') as f: users = json.load(f)
            users.append(new_user)
            with open(USERS_FILE, 'w') as f: json.dump(users, f, indent=4)
            self.load_encodings()
            self.after(0, lambda: self.finish_enroll_ui(True, "Success!"))
        except Exception as e:
            self.after(0, lambda: self.finish_enroll_ui(False, f"Error: {str(e)}"))

    def finish_enroll_ui(self, success, message):
        self.action_btn.configure(state="normal", text="START SEQUENCE")
        self.ui_instruction_text = message
        self.ui_instruction_color = "#00ff00" if success else "#ff0000"
        self.current_challenge = None
        if success:
            self.en_name.delete(0, 'end')
            self.en_id.delete(0, 'end')

    def log_attendance(self, name, conf):
        now = time.time()
        if name in self.verification_cooldown:
            if now - self.verification_cooldown[name] < 300: # 5 Mins
                self.show_overlay_message("⏳ ALREADY VERIFIED", "#ffcc00")
                return
        self.verification_cooldown[name] = now
        winsound.Beep(1200, 300)
        self.show_overlay_message(f"✅ VERIFIED: {name}", "#00ff00")
        logs = []
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f: logs = json.load(f)
        logs.append({"name": name, "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "confidence": f"{conf:.2%}"})
        with open(ATTENDANCE_FILE, 'w') as f: json.dump(logs, f, indent=4)

    def show_overlay_message(self, text, color):
        if hasattr(self, 'overlay_label') and self.overlay_label.winfo_exists():
            self.overlay_label.configure(text=text, text_color=color)
            self.after(3000, lambda: self.overlay_label.configure(text="") if hasattr(self, 'overlay_label') else None)

    def load_encodings(self):
        self.known_face_encodings = []
        self.known_face_names = []
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                data = json.load(f)
                for user in data:
                    for enc in user['encodings']:
                        self.known_face_encodings.append(np.array(enc))
                        self.known_face_names.append(user['name'])

    def export_csv(self):
        if not os.path.exists(ATTENDANCE_FILE): return
        path = ctk.filedialog.asksaveasfilename(defaultextension=".csv")
        if path:
            with open(ATTENDANCE_FILE, 'r') as f: data = json.load(f)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    def clear_main_area(self):
        for widget in self.main_area.winfo_children(): widget.destroy()

    def clear_ui(self):
        for widget in self.winfo_children(): widget.destroy()

    def logout(self):
        self.is_running = False 
        self.stop_event.set()
        if self.cap: self.cap.release()
        self.authenticated = False
        self.show_login_screen()

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()