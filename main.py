import customtkinter as ctk
import cv2
from PIL import Image
import face_recognition
import datetime
import threading
import numpy as np
import os
import time
import json
from tkinter import messagebox
import winsound
import pygetwindow as gw
from collections import deque
import queue
from typing import Optional

# --- Configuration & Styling ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance_logs.json")

class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NEXUS Biometric Verification System")
        self.geometry("1400x850")
        
        # Core State
        self.cap = None
        self.is_running = False
        self.video_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=2)  # Limited queue to prevent lag
        self.display_queue = queue.Queue(maxsize=1)
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Liveness & Quality State
        self.blink_detected = False
        self.head_movement_detected = False
        self.eye_aspect_ratio_threshold = 0.20
        self.blink_counter = 0
        self.head_positions = deque(maxlen=15)
        self.face_quality_score = 0
        
        # Registration State
        self.registration_captures = []
        self.capture_count = 0
        
        # Performance tracking
        self.last_processing_time = 0
        self.processing_interval = 3  # Process every 3rd frame
        
        # Thread control
        self.stop_event = threading.Event()
        
        self.load_encodings()
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color=("#1a1a2e", "#0f0f1e"))
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)

        title_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        title_frame.pack(pady=30, padx=20)
        ctk.CTkLabel(title_frame, text="⬡ NEXUS", font=("Segoe UI", 32, "bold"), text_color="#00d4ff").pack()
        ctk.CTkLabel(title_frame, text="Biometric Security", font=("Segoe UI", 11), text_color="#8b8b8b").pack()

        nav_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        nav_frame.pack(pady=40, padx=15, fill="x")

        self.create_nav_button("👤  Enroll User", self.show_register_frame, nav_frame)
        self.create_nav_button("🔍  Verify Identity", self.show_attendance_frame, nav_frame)
        self.create_nav_button("📊  Dashboard", self.show_dashboard_frame, nav_frame)
        self.create_nav_button("📋  Audit Logs", self.show_logs_frame, nav_frame)
        self.create_nav_button("⚙️  Manage Users", self.show_manage_frame, nav_frame)

        status_frame = ctk.CTkFrame(self.sidebar_frame, fg_color=("#252540", "#1a1a30"), corner_radius=10)
        status_frame.pack(side="bottom", pady=20, padx=15, fill="x")
        ctk.CTkLabel(status_frame, text="SYSTEM STATUS", font=("Segoe UI", 10, "bold"), text_color="#8b8b8b").pack(pady=(10, 5))
        self.status_label = ctk.CTkLabel(status_frame, text="● Online", font=("Segoe UI", 12), text_color="#00ff88")
        self.status_label.pack(pady=(0, 10))

        # --- Main Display Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color=("#1e1e1e", "#121212"), border_width=1, border_color=("#2a2a2a", "#1a1a1a"))
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)

        self.show_register_frame()

    def create_nav_button(self, text, command, parent):
        btn = ctk.CTkButton(parent, text=text, command=command, height=50, corner_radius=12,
                           fg_color=("#2a2a40", "#1f1f35"), hover_color=("#00d4ff", "#00a8cc"),
                           text_color="#ffffff", font=("Segoe UI", 13, "bold"), anchor="w",
                           border_width=1, border_color=("#3a3a50", "#2a2a40"))
        btn.pack(pady=8, fill="x")
        return btn

    # --- Security & Core Logic ---
    def security_check(self):
        """Check for virtual cameras"""
        try:
            virtual_cams = ["OBS Virtual", "ManyCam", "Snap Camera", "XSplit VCam"]
            for window in gw.getAllTitles():
                if any(cam.lower() in window.lower() for cam in virtual_cams):
                    return False
        except:
            pass  # Continue if check fails
        return True

    def load_encodings(self):
        """Load face encodings from JSON file"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
            try:
                with open(USERS_FILE, "r") as f:
                    users = json.load(f)
                    for user in users:
                        encodings = user.get("encodings", [user.get("encoding")])
                        for enc in encodings:
                            if enc:
                                self.known_face_names.append(user["name"])
                                self.known_face_encodings.append(np.array(enc))
                                self.known_face_ids.append(user.get("id", user["name"]))
            except Exception as e:
                print(f"Load Error: {e}")

    def clear_main_frame(self):
        """Stop all camera and processing threads"""
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads to stop
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if hasattr(self, 'composer_thread') and self.composer_thread and self.composer_thread.is_alive():
            self.composer_thread.join(timeout=1.0)
        if hasattr(self, 'bio_thread') and self.bio_thread and self.bio_thread.is_alive():
            self.bio_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        time.sleep(0.1)
        
        # Clear frame queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except:
                pass
        
        # Clear UI
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        
        self.stop_event.clear()

    # --- thresholds ---
    REGISTRATION_THRESHOLD = 0.30  # Strict: Must be very similar to be considered "already registered"
    VERIFICATION_THRESHOLD = 0.35  # Strict: Prevents lookalikes from verifying
    LIVENESS_EAR_THRESHOLD = 0.003 # Variance threshold for static image detection

    def start_camera(self, label_widget, mode):
        """Start camera with optimized settings"""
        if not self.security_check():
            messagebox.showwarning("⚠️ Security Alert", "Virtual camera detected. Use physical webcam.")
            return
            
        try:
            # Try DSHOW first (faster on Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            # Fallback if DSHOW fails
            if not self.cap.isOpened():
                print("DSHOW failed, trying default backend...")
                self.cap = cv2.VideoCapture(0)
                
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Cannot open camera. Please check connection.")
                return
                
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            self.is_running = True
            self.stop_event.clear()
            
            # Reset Shared State
            self.latest_frame = None
            self.frame_lock = threading.Lock()
            self.processed_data = {
                "face_locations": [],
                "face_names": [],
                "liveness_stats": {}
            }
            
            # Anti-Spoofing State for Attendance
            self.attendance_state = {}  # {name: [ear_values]}
            
            # Start threads
            self.video_thread = threading.Thread(target=self.video_capture_loop, daemon=True)
            self.composer_thread = threading.Thread(target=self.video_composer_loop, args=(mode,), daemon=True)
            self.bio_thread = threading.Thread(target=self.biometric_loop, args=(mode,), daemon=True)
            
            self.video_thread.start()
            self.composer_thread.start()
            self.bio_thread.start()
            
            # Start display update
            self.update_camera_widget(label_widget)
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")

    def video_capture_loop(self):
        """Dedicated thread for capturing frames - PURE CAPTURE"""
        while self.is_running and self.cap and self.cap.isOpened() and not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                
                # Atomic update of latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    
                time.sleep(0.01) # Slight yield
                    
            except Exception as e:
                print(f"Capture error: {e}")
                break
                
        self.is_running = False

    def biometric_loop(self, mode):
        """Dedicated thread for Heavy AI Processing"""
        last_process_time = 0
        
        while self.is_running and not self.stop_event.is_set():
            # Get latest frame safely
            frame_to_process = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame.copy()
            
            if frame_to_process is None:
                time.sleep(0.01)
                continue

            try:
                # --- LIVENESS CHECK (Registration Mode) ---
                if mode == "register":
                    rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
                    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    new_liveness_stats = {"quality": 0, "blink": False, "movement": False}
                    
                    if face_locations:
                        quality = self.calculate_face_quality(frame_to_process, face_locations[0])
                        self.face_quality_score = quality # Update instance var for UI
                        new_liveness_stats["quality"] = quality
                        
                        top, right, bottom, left = face_locations[0]
                        center = ((left + right) // 2, (top + bottom) // 2)
                        self.head_positions.append(center)
                        
                        if len(self.head_positions) >= 10:
                            movement = np.std([p[0] for p in self.head_positions]) + np.std([p[1] for p in self.head_positions])
                            if movement > 15:
                                self.head_movement_detected = True # Update logic state
                    
                    for face_landmarks in face_landmarks_list:
                        left_ear = self.get_eye_aspect_ratio(face_landmarks['left_eye'], range(6))
                        right_ear = self.get_eye_aspect_ratio(face_landmarks['right_eye'], range(6))
                        ear = (left_ear + right_ear) / 2.0
                        
                        # Debug print (optional, helpful for tuning)
                        # print(f"EAR: {ear:.3f}")
                        
                        if ear < 0.25: # Increased threshold (was 0.20)
                            self.blink_counter += 1
                        else:
                            # Reduced requirement to 1 frame (was 2) because processing might be slow
                            if self.blink_counter >= 1:
                                self.blink_detected = True 
                            self.blink_counter = 0

                    self.processed_data["liveness_stats"] = new_liveness_stats

                # --- FACE RECOGNITION (Attendance Mode) ---
                elif mode == "attendance":
                    # Throttle recognition to prevent CPU overload, but run independently of UI
                    if time.time() - last_process_time > 0.15:  # Slightly faster for liveness tracking
                        small = cv2.resize(frame_to_process, (0, 0), fx=0.5, fy=0.5)
                        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        
                        # 1. Detection
                        locs = face_recognition.face_locations(rgb_small)
                        encs = face_recognition.face_encodings(rgb_small, locs)
                        
                        # 2. Landmarks for Liveness
                        landmarks_list = face_recognition.face_landmarks(rgb_small, locs)
                        
                        found_names = []
                        valid_names_in_frame = set()
                        
                        for idx, (loc, enc) in enumerate(zip(locs, encs)):
                            name = "Unknown"
                            if self.known_face_encodings:
                                face_distances = face_recognition.face_distance(self.known_face_encodings, enc)
                                if len(face_distances) > 0 and np.min(face_distances) <= self.VERIFICATION_THRESHOLD:
                                    best_idx = np.argmin(face_distances)
                                    name = self.known_face_names[best_idx]
                                    confidence = 1 - face_distances[best_idx]
                                    
                                    # --- PASSIVE LIVENESS CHECK ---
                                    if idx < len(landmarks_list):
                                        lm = landmarks_list[idx]
                                        left_ear = self.get_eye_aspect_ratio(lm['left_eye'], range(6))
                                        right_ear = self.get_eye_aspect_ratio(lm['right_eye'], range(6))
                                        ear = (left_ear + right_ear) / 2.0
                                        
                                        # Use a temporary key for buffering to avoid revealing identity yet
                                        temp_key = f"face_{best_idx}" 
                                        
                                        if temp_key not in self.attendance_state:
                                            self.attendance_state[temp_key] = deque(maxlen=20)
                                        self.attendance_state[temp_key].append(ear)
                                        
                                        # Check Liveness Variance
                                        if len(self.attendance_state[temp_key]) >= 10:
                                            ear_std = np.std(self.attendance_state[temp_key])
                                            if ear_std > self.LIVENESS_EAR_THRESHOLD:
                                                # verified: REVEAL NAME
                                                name = self.known_face_names[best_idx]
                                                self.after(0, lambda n=name, c=confidence: self.mark_attendance(n, c))
                                            else:
                                                # spoof: HIDE NAME, WARN USER
                                                name = "❌ SPOOF: Liveness Failed"
                                        else:
                                            name = "Analyzing Liveness..."
                                        
                                        valid_names_in_frame.add(temp_key)
                                    
                            found_names.append(name)
                        
                        # Clean up buffer for people who left the frame
                        current_keys = list(self.attendance_state.keys())
                        for k in current_keys:
                            if k not in valid_names_in_frame:
                                del self.attendance_state[k]
                        
                        # Update shared data
                        self.processed_data["face_locations"] = locs
                        self.processed_data["face_names"] = found_names
                        last_process_time = time.time()

            except Exception as e:
                print(f"Bio loop error: {e}")
            
            # Small sleep to prevent thread starvation
            time.sleep(0.01)

    def video_composer_loop(self, mode):
        """Dedicated thread for Visual Composition (30 FPS)"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 1. Get latest frame
                frame = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        frame = self.latest_frame.copy()
                
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 2. Draw Visuals
                display = frame.copy()
                h, w, _ = display.shape
                
                # Draw Overlay (Cool Corners)
                overlay = display.copy()
                corners = [(w//2-200, h//2-250), (w//2+200, h//2-250), 
                          (w//2-200, h//2+250), (w//2+200, h//2+250)]
                
                for i, (x, y) in enumerate(corners):
                    color = (0, 212, 255)
                    length = 40
                    # Draw corners logic...
                    if i == 0:
                        cv2.line(overlay, (x, y), (x+length, y), color, 3)
                        cv2.line(overlay, (x, y), (x, y+length), color, 3)
                    elif i == 1:
                        cv2.line(overlay, (x, y), (x-length, y), color, 3)
                        cv2.line(overlay, (x, y), (x, y+length), color, 3)
                    elif i == 2:
                        cv2.line(overlay, (x, y), (x+length, y), color, 3)
                        cv2.line(overlay, (x, y), (x, y-length), color, 3)
                    else:
                        cv2.line(overlay, (x, y), (x-length, y), color, 3)
                        cv2.line(overlay, (x, y), (x, y-length), color, 3)
                
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

                # 3. Draw Biometric Results (from shared state)
                if mode == "attendance":
                    locs = self.processed_data["face_locations"]
                    names = self.processed_data["face_names"]
                    
                    if locs and names and len(locs) == len(names):
                        # Scale back up since detection was on 0.5x image
                        for (top, right, bottom, left), name in zip(locs, names):
                            top *= 2
                            right *= 2
                            bottom *= 2
                            left *= 2
                            
                            color = (0, 255, 136) # Green (Verified)
                            if "Unknown" in name:
                                color = (255, 255, 255) # White
                            elif "Analyzing" in name:
                                color = (0, 212, 255) # Blue/Cyan
                            elif "SPOOF" in name:
                                color = (0, 0, 255) # Red
                            
                            cv2.rectangle(display, (left, top), (right, bottom), color, 3)
                            cv2.putText(display, name, (left, top-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 4. Prepare for UI
                pil_image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                
                try:
                    self.display_queue.put(pil_image, block=False)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait() # Remove old frame
                        self.display_queue.put(pil_image, block=False)
                    except:
                        pass
                
                # Maintain ~30 FPS for the UI
                time.sleep(0.033) 
                    
            except Exception as e:
                print(f"Composition error: {e}")
                break

    def update_camera_widget(self, label_widget):
        """Update camera widget with new frames"""
        if not self.is_running or not label_widget.winfo_exists():
            return
            
        try:
            # Get latest frame from display queue
            pil_image = None
            try:
                pil_image = self.display_queue.get_nowait()
            except queue.Empty:
                pass
            
            if pil_image:
                imgtk = ctk.CTkImage(pil_image, size=(800, 600))
                label_widget.configure(image=imgtk)
                label_widget.image = imgtk
                
        except Exception:
            pass
            
        # Schedule next update
        self.after(30, lambda: self.update_camera_widget(label_widget))

    # --- Biometric Analysis ---
    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate eye aspect ratio for blink detection"""
        pts = [landmarks[i] for i in eye_indices]
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return (v1 + v2) / (2.0 * h) if h > 0 else 0

    def calculate_face_quality(self, frame, face_location):
        """Calculate face quality score"""
        top, right, bottom, left = face_location
        face_region = frame[top:bottom, left:right]
        if face_region.size == 0:
            return 0
            
        # Size score
        face_size = (bottom - top) * (right - left)
        size_score = min(face_size / 50000, 1.0)
        
        # Brightness score
        brightness = np.mean(face_region)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # Sharpness score
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500, 1.0)
        
        return (size_score + brightness_score + blur_score) / 3.0

    # --- View Transitions ---
    def show_register_frame(self):
        """Show user registration frame"""
        self.clear_main_frame()
        self.blink_detected = False
        self.head_movement_detected = False
        self.registration_captures = []
        self.capture_count = 0
        
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.pack(pady=20, padx=30, fill="x")
        ctk.CTkLabel(header, text="🔐 Secure User Enrollment", 
                    font=("Segoe UI", 28, "bold"), text_color="#00d4ff").pack(anchor="w")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(pady=10, padx=30, fill="both", expand=True)

        left_panel = ctk.CTkFrame(content, fg_color="#252540", corner_radius=15)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        cam_label = ctk.CTkLabel(left_panel, text="", fg_color="black", corner_radius=12)
        cam_label.pack(pady=15, padx=15, fill="both", expand=True)

        right_panel = ctk.CTkFrame(content, fg_color="#252540", corner_radius=15, width=350)
        right_panel.pack(side="right", fill="y", padx=(15, 0))
        right_panel.pack_propagate(False)

        self.entry_name = ctk.CTkEntry(right_panel, placeholder_text="Full Legal Name", 
                                      width=310, height=45, font=("Segoe UI", 14))
        self.entry_name.pack(pady=10, padx=20)
        self.entry_id = ctk.CTkEntry(right_panel, placeholder_text="Employee/Student ID", 
                                    width=310, height=45, font=("Segoe UI", 14))
        self.entry_id.pack(pady=10, padx=20)

        status_frame = ctk.CTkFrame(right_panel, fg_color="#1a1a30", corner_radius=10)
        status_frame.pack(pady=20, padx=20, fill="x")
        
        self.blink_status = ctk.CTkLabel(status_frame, text="❌ Blink: Pending", 
                                        text_color="#ff6b6b", font=("Segoe UI", 12))
        self.blink_status.pack(pady=3)
        
        self.movement_status = ctk.CTkLabel(status_frame, text="❌ Movement: Pending", 
                                           text_color="#ff6b6b", font=("Segoe UI", 12))
        self.movement_status.pack(pady=3)
        
        self.quality_status = ctk.CTkLabel(status_frame, text="⚠️ Quality: Analyzing...", 
                                          text_color="#ffd93d", font=("Segoe UI", 12))
        self.quality_status.pack(pady=3)

        self.reg_btn = ctk.CTkButton(right_panel, text="🎯 Capture (0/3)", 
                                    command=self.capture_registration_sample,
                                    font=("Segoe UI", 14, "bold"), height=45)
        self.reg_btn.pack(pady=20, padx=20, side="bottom")

        self.start_camera(cam_label, "register")
        self.update_liveness_status()

    def update_liveness_status(self):
        """Update liveness detection status"""
        if not hasattr(self, 'blink_status') or not self.blink_status.winfo_exists():
            return
            
        if self.blink_detected:
            self.blink_status.configure(text="✅ Blink: Passed", text_color="#00ff88")
        if self.head_movement_detected:
            self.movement_status.configure(text="✅ Movement: Passed", text_color="#00ff88")
        if self.face_quality_score >= 0.4:
            self.quality_status.configure(text=f"✅ Quality: OK ({self.face_quality_score:.0%})", 
                                         text_color="#00ff88")
        
        self.after(500, self.update_liveness_status)

    def capture_registration_sample(self):
        """Capture face sample for registration"""
        if not self.blink_detected or not self.head_movement_detected:
            messagebox.showwarning("⚠️ Liveness Check", 
                                 "Please perform liveness actions (blink & move head).")
            return
            
        if not self.entry_name.get().strip() or not self.entry_id.get().strip():
            messagebox.showwarning("Input Required", 
                                 "Please enter both Name and ID.")
            return

        self.capture_count += 1
        
        # Get frame from shared state
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
        
        if frame is not None:
            self.registration_captures.append(frame)
            winsound.Beep(800, 100)
        else:
            messagebox.showwarning("No Frame", "Could not capture frame. Try again.")
            self.capture_count -= 1
            return
            
        self.reg_btn.configure(text=f"🎯 Capture ({self.capture_count}/3)")

        if self.capture_count >= 3:
            self.reg_btn.configure(state="disabled", text="⏳ Processing...")
            threading.Thread(target=self._threaded_registration, 
                           args=(self.entry_name.get().strip(), 
                                 self.entry_id.get().strip(), 
                                 self.registration_captures.copy()), 
                           daemon=True).start()

    # --- thresholds ---
    REGISTRATION_THRESHOLD = 0.30  # Strict: Must be very similar to be considered "already registered"
    VERIFICATION_THRESHOLD = 0.35  # Strict: Prevents lookalikes from verifying

    def _threaded_registration(self, name, user_id, frames):
        """Threaded registration processing"""
        try:
            all_encodings = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb, num_jitters=3)
                if encs:
                    all_encodings.append(encs[0])

            if len(all_encodings) < 2:
                self.after(0, lambda: self._reg_complete(False, 
                    "Could not extract enough facial features. Please try again."))
                return
            
            # Calculate average encoding for the new user
            avg_encoding = np.mean(all_encodings, axis=0)

            # Check for duplicates against existing users
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(self.known_face_encodings, avg_encoding)
                min_dist_idx = np.argmin(face_distances)
                min_dist = face_distances[min_dist_idx]
                
                if min_dist < self.REGISTRATION_THRESHOLD:
                    existing_name = self.known_face_names[min_dist_idx]
                    self.after(0, lambda: self._reg_complete(False, 
                        f"Registration Failed: Face already registered as '{existing_name}'."))
                    return

            user_data = {
                "name": name,
                "id": user_id,
                "encodings": [e.tolist() for e in all_encodings],
                "registered_at": str(datetime.datetime.now())
            }
            
            users = []
            if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
                try:
                    with open(USERS_FILE, "r") as f:
                        users = json.load(f)
                except:
                    users = []
                    
            users.append(user_data)
            
            with open(USERS_FILE, "w") as f:
                json.dump(users, f, indent=4)
            
            self.after(0, self.load_encodings)
            self.after(0, lambda: self._reg_complete(True, f"✅ Successfully enrolled {name}!"))
            
        except Exception as e:
            self.after(0, lambda: self._reg_complete(False, f"Registration error: {str(e)}"))

    def _reg_complete(self, success, msg):
        """Handle registration completion"""
        self.reg_btn.configure(state="normal", text="🎯 Capture (0/3)")
        self.capture_count = 0
        self.registration_captures = []
        
        # Reset Inputs
        self.entry_name.delete(0, 'end')
        self.entry_id.delete(0, 'end')
        
        # Reset Liveness State
        self.blink_detected = False
        self.head_movement_detected = False
        self.face_quality_score = 0
        self.blink_counter = 0
        self.head_positions.clear()
        
        # Reset UI Status Labels
        if hasattr(self, 'blink_status') and self.blink_status.winfo_exists():
            self.blink_status.configure(text="❌ Blink: Pending", text_color="#ff6b6b")
            self.movement_status.configure(text="❌ Movement: Pending", text_color="#ff6b6b")
            self.quality_status.configure(text="⚠️ Quality: Analyzing...", text_color="#ffd93d")

        if success:
            messagebox.showinfo("Success", msg)
        else:
            messagebox.showerror("Error", msg)

    def show_attendance_frame(self):
        """Show attendance/verification frame"""
        self.clear_main_frame()
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.pack(pady=20, padx=30, fill="x")
        ctk.CTkLabel(header, text="🔍 Biometric Verification", 
                    font=("Segoe UI", 28, "bold"), text_color="#00d4ff").pack(anchor="w")
        
        # Center content
        center_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        center_frame.pack(expand=True, fill="both", padx=30, pady=20)
        
        cam_label = ctk.CTkLabel(center_frame, text="", fg_color="black", corner_radius=12)
        cam_label.pack(pady=10)
        
        self.att_label = ctk.CTkLabel(center_frame, text="Awaiting recognition...", 
                                     font=("Segoe UI", 18), text_color="#8b8b8b")
        self.att_label.pack(pady=10)
        
        self.start_camera(cam_label, "attendance")

    def mark_attendance(self, name, confidence):
        """Mark attendance with throttling"""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Check recent attendance (within 2 minutes)
        logs = []
        if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
            try:
                with open(ATTENDANCE_FILE, "r") as f:
                    logs = json.load(f)
            except:
                logs = []
                
        # Prevent duplicate entries within 2 minutes
        recent_entries = [l for l in logs if l['name'] == name]
        if recent_entries:
            last_time = datetime.datetime.strptime(recent_entries[-1]['time'], "%Y-%m-%d %H:%M:%S")
            if (now - last_time).total_seconds() < 120:  # 2 minutes cooldown
                return
        
        # Add new entry
        logs.append({
            "name": name,
            "time": timestamp,
            "confidence": f"{confidence:.2%}"
        })
        
        try:
            with open(ATTENDANCE_FILE, "w") as f:
                json.dump(logs, f, indent=4)
                
            winsound.Beep(1000, 200)
            self.after(0, lambda: self._update_att_ui(name, timestamp, confidence))
            
        except Exception as e:
            print(f"Error saving attendance: {e}")

    def _update_att_ui(self, name, timestamp, confidence):
        """Update attendance UI"""
        if hasattr(self, 'att_label') and self.att_label.winfo_exists():
            self.att_label.configure(text=f"✅ VERIFIED: {name}", text_color="#00ff88")
            # Show success message but don't block UI
            self.after(100, lambda: messagebox.showinfo("✅ Verified", f"Access Granted to {name}"))

    def show_dashboard_frame(self):
        """Show dashboard frame"""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="📊 System Dashboard", 
                    font=("Segoe UI", 28, "bold"), text_color="#00d4ff").pack(pady=20, padx=30, anchor="w")
        
        stats_frame = ctk.CTkFrame(self.main_frame, fg_color="#252540", corner_radius=15)
        stats_frame.pack(pady=20, padx=30, fill="both", expand=True)
        
        users_count = len(set(self.known_face_ids))
        ctk.CTkLabel(stats_frame, text=f"Total Enrolled Users: {users_count}", 
                    font=("Segoe UI", 24)).pack(pady=40)

    def show_logs_frame(self):
        """Show audit logs frame"""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="📋 Audit Trail", 
                    font=("Segoe UI", 28, "bold"), text_color="#00d4ff").pack(pady=20, padx=30, anchor="w")
        
        box = ctk.CTkTextbox(self.main_frame, width=900, height=550, 
                           font=("Consolas", 12), corner_radius=10)
        box.pack(pady=10, padx=30, fill="both", expand=True)
        
        if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
            try:
                with open(ATTENDANCE_FILE, "r") as f:
                    logs = json.load(f)
                    
                log_str = f"{'USER':<25} {'TIMESTAMP':<25} {'CONFIDENCE':<10}\n"
                log_str += "=" * 60 + "\n"
                
                for l in reversed(logs[-50:]):  # Show last 50 entries
                    log_str += f"{l['name']:<25} {l['time']:<25} {l.get('confidence', 'N/A'):<10}\n"
                    
                box.insert("0.0", log_str)
            except Exception as e:
                box.insert("0.0", f"Error loading logs: {str(e)}")
        else:
            box.insert("0.0", "No attendance logs found.")
            
        box.configure(state="disabled")

    def show_manage_frame(self):
        """Show user management frame"""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="⚙️ User Management", 
                    font=("Segoe UI", 28, "bold"), text_color="#00d4ff").pack(pady=20, padx=30, anchor="w")
        
        users_frame = ctk.CTkScrollableFrame(self.main_frame, width=900, height=550, 
                                           fg_color="transparent")
        users_frame.pack(pady=10, padx=30, fill="both", expand=True)
        
        unique_users = {}
        if os.path.exists(USERS_FILE) and os.path.getsize(USERS_FILE) > 0:
            try:
                with open(USERS_FILE, "r") as f:
                    users = json.load(f)
                    for user in users:
                        uid = user.get("id", user["name"])
                        unique_users[uid] = user
            except Exception as e:
                ctk.CTkLabel(users_frame, text=f"Error loading users: {str(e)}", 
                           text_color="#ff6b6b").pack(pady=20)
        
        if not unique_users:
            ctk.CTkLabel(users_frame, text="No users found", 
                       text_color="#8b8b8b").pack(pady=20)
        else:
            for uid, user in unique_users.items():
                card = ctk.CTkFrame(users_frame, fg_color="#252540", corner_radius=10)
                card.pack(pady=10, padx=10, fill="x")
                
                ctk.CTkLabel(card, text=f"👤 {user['name']}", 
                           font=("Segoe UI", 16, "bold")).pack(pady=5, padx=15, anchor="w")
                ctk.CTkLabel(card, text=f"ID: {uid}", 
                           text_color="#8b8b8b").pack(padx=15, anchor="w")
                ctk.CTkLabel(card, text=f"Registered: {user.get('registered_at', 'Unknown')}", 
                           text_color="#8b8b8b").pack(padx=15, anchor="w")

    def on_closing(self):
        """Clean shutdown"""
        self.is_running = False
        self.stop_event.set()
        
        if self.cap:
            self.cap.release()
            
        time.sleep(0.2)
        self.destroy()

if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()