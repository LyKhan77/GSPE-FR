#!/usr/bin/env python3
# AI_module.py
# Dedicated AI module for background face recognition and employee tracking
# This module handles all AI processing and database updates separately from the web application

import cv2
import numpy as np
import threading
import time
import base64
import json
import os
from datetime import datetime
from db_manager import db_manager, SessionLocal, Employee
from camera_manager import camera_manager, get_all_camera_configs
from tracking_manager import tracking_manager

# Default system specifications (ditingkatkan untuk konsistensi ID tracking dan re-identification)
DEFAULT_SYSTEM_SPECS = {
    "model": "InsightFace (Buffalo_L)",
    "detection_threshold": 0.5,
    "detection_size": (320, 320),
    "recognition_cooldown": 10,  # detik
    "bbox_smoothing_factor": 0.85,  # Ditingkatkan untuk tracking yang lebih smooth
    "providers": "CUDAExecutionProvider, CPUExecutionProvider",
    "fps_target": 30,
    "frame_skip": False,
    "multi_person": True,
    "recognition_threshold": 0.5,
    # Enhanced tracking parameters (ditingkatkan untuk konsistensi ID)
    "max_distance_threshold": 150,   # Ditingkatkan dari 100 untuk toleransi posisi yang lebih baik
    "tracking_timeout": 3.0,         # Ditingkatkan dari 1.0 detik untuk mempertahankan ID lebih lama
    "movement_threshold": 5,
    # Re-identification parameters
    "embedding_similarity_threshold": 0.7  # Threshold untuk matching berbasis embedding
}

def load_system_specs():
    """Load system specifications from config file or use defaults"""
    config_file = "parameter_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Convert detection_size from list to tuple if needed
            if isinstance(config.get("detection_size"), list):
                config["detection_size"] = tuple(config["detection_size"])
            print("[INFO] Konfigurasi sistem dimuat dari file")
            return config
        except Exception as e:
            print(f"[ERROR] Gagal memuat konfigurasi: {e}")
            print("[INFO] Menggunakan konfigurasi default")
            return DEFAULT_SYSTEM_SPECS.copy()
    else:
        print("[INFO] Menggunakan konfigurasi default")
        return DEFAULT_SYSTEM_SPECS.copy()

def log_attendance(name: str):
    """Catat kehadiran karyawan ke database"""
    db_manager.log_attendance(name)

# Kelas FaceRecognitionSystem yang disederhanakan untuk keperluan AI_module.py
class FaceRecognitionSystem:
    def __init__(self):
        # Inisialisasi InsightFace dengan GPU acceleration jika tersedia
        import insightface
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Prioritaskan CUDA
        self.app = insightface.app.FaceAnalysis(providers=providers)
        # Gunakan ukuran deteksi dari konfigurasi
        SYSTEM_SPECS = load_system_specs()
        det_size = SYSTEM_SPECS['detection_size']
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # Load employees data
        self.known_employees = []
        self.known_employees_with_id = []
        self.name_to_id = {}  # Mapping for name -> employee_id
        self.load_employees()
        
    def get_embeddings(self, frame):
        """Dapatkan embeddings dari frame"""
        faces = self.app.get(frame)
        return [(face.bbox.astype(int), face.embedding) for face in faces]
    
    def load_employees(self):
        """Load data karyawan dari database"""
        self.known_employees = db_manager.get_all_employees()
        self.known_employees_with_id = db_manager.get_all_employees_with_id()
        
        # Build name-to-ID mapping for tracking_manager integration
        self.name_to_id = {}
        for employee_id, name, embedding in self.known_employees_with_id:
            self.name_to_id[name] = employee_id
        
        print(f"[INFO] Loaded {len(self.known_employees)} karyawan dari database")
        print(f"[INFO] Built name-to-ID mapping for {len(self.name_to_id)} employees")

class AIProcessingModule:
    def __init__(self):
        self.face_recognition_system = None
        self.active_streams = {}
        self.is_running = False
        self.employee_status_tracker = None
        self.processing_thread = None
        self.absence_check_thread = None
        
        # Load system specifications
        self.system_specs = load_system_specs()
        
        # Employee absence tracking
        self.employee_last_seen = {}  # {employee_name: timestamp}
        self.absence_threshold = 300  # 5 minutes in seconds
        
        # Initialize the face recognition system
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the face recognition model for background processing"""
        try:
            if self.face_recognition_system is None:
                # Import here to avoid circular imports
                from AI_module import FaceRecognitionSystem
                self.face_recognition_system = FaceRecognitionSystem()
                
                # Apply system specifications to the model
                if self.face_recognition_system and self.system_specs:
                    # Update detection size if specified
                    if 'detection_size' in self.system_specs:
                        det_size = self.system_specs['detection_size']
                        # Convert list to tuple if needed
                        if isinstance(det_size, list):
                            det_size = tuple(det_size)
            return True
        except Exception as e:
            print(f"[AI MODULE] Error initializing face recognition system: {e}")
            return False
    
    def start_processing(self):
        """Start the AI processing module"""
        if not self.is_running:
            self.is_running = True
            
            # Initialize model and load employees
            if self.initialize_model() and self.face_recognition_system:
                self.face_recognition_system.load_employees()
            
            # Start background threads
            self.processing_thread = threading.Thread(target=self._background_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.absence_check_thread = threading.Thread(target=self._absence_checking_loop)
            self.absence_check_thread.daemon = True
            self.absence_check_thread.start()
            
            return True
        return False
    
    def stop_processing(self):
        """Stop the AI processing module"""
        if self.is_running:
            self.is_running = False
            
            # Stop all active streams
            self.stop_all_streams()
            
            # Wait for threads to finish
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            if self.absence_check_thread:
                self.absence_check_thread.join(timeout=5)
                
            return True
        return False
    
    def start_stream(self, camera_id, rtsp_url, frame_callback=None):
        """Start face recognition processing for a camera stream - background processing"""
        try:
            print(f"[AI MODULE] Starting stream for camera: {camera_id}")
            
            # Stop any existing stream for this camera
            self.stop_stream(camera_id)
            
            # Start processing in a separate thread
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._process_stream, 
                args=(camera_id, rtsp_url, stop_event, frame_callback), 
                daemon=True
            )
            thread.start()
            
            self.active_streams[camera_id] = {
                'thread': thread,
                'stop_event': stop_event,
                'rtsp_url': rtsp_url,
                'frame_callback': frame_callback
            }
            
            print(f"[AI MODULE] Started background face recognition for camera {camera_id}")
            return True
        except Exception as e:
            print(f"[AI MODULE] Error starting stream for camera {camera_id}: {e}")
            import traceback
            traceback.print_exc()
            # Notify frontend about camera error
            if frame_callback:
                try:
                    frame_callback("Camera Unavailable")
                except Exception as cb_error:
                    print(f"[AI MODULE] Error sending error frame to callback: {cb_error}")
            return False
    
    def stop_stream(self, camera_id):
        """Stop face recognition processing for a camera stream"""
        if camera_id in self.active_streams:
            try:
                self.active_streams[camera_id]['stop_event'].set()
                self.active_streams[camera_id]['thread'].join(timeout=5)
                del self.active_streams[camera_id]
            except Exception as e:
                print(f"[AI MODULE] Error stopping stream for camera {camera_id}: {e}")
    
    def stop_all_streams(self):
        """Stop all active streams"""
        camera_ids = list(self.active_streams.keys())
        for camera_id in camera_ids:
            self.stop_stream(camera_id)
    
    def _process_stream(self, camera_id, rtsp_url, stop_event, frame_callback=None):
        """Process video stream and perform face recognition in background"""
        cap = None
        try:
            # Initialize face recognition system if needed
            if not self.initialize_model():
                if frame_callback:
                    try:
                        frame_callback("Camera Unavailable")
                    except Exception as e:
                        print(f"[AI MODULE] Error sending error frame: {e}")
                return
            
            system = self.face_recognition_system
            
            # Load employees if not already loaded
            if not hasattr(system, 'known_employees') or not system.known_employees:
                system.load_employees()
            
            # Ensure name_to_id mapping is available
            if not hasattr(system, 'name_to_id') or not system.name_to_id:
                system.load_employees()
            
            # Get system specifications (using the same approach as main.py)
            SYSTEM_SPECS = self.system_specs
            
            # Apply parameters from main.py implementation
            recognition_cooldown = SYSTEM_SPECS.get('recognition_cooldown', 10)
            smoothing_factor = SYSTEM_SPECS.get('bbox_smoothing_factor', 0.85)
            max_distance_threshold = SYSTEM_SPECS.get('max_distance_threshold', 150)
            tracking_timeout = SYSTEM_SPECS.get('tracking_timeout', 3.0)
            embedding_similarity_threshold = SYSTEM_SPECS.get('embedding_similarity_threshold', 0.7)
            fps_target = SYSTEM_SPECS.get('fps_target', 30)
            frame_skip = SYSTEM_SPECS.get('frame_skip', False)
            
            # Get detection parameters
            detection_threshold = SYSTEM_SPECS.get('detection_threshold', 0.5)
            multi_person = SYSTEM_SPECS.get('multi_person', True)
            
            # Calculate frame delay based on target FPS
            frame_delay = 1.0 / fps_target if fps_target > 0 else 0.033  # Default to ~30 FPS
            
            # Open video stream (RTSP or webcam)
            if rtsp_url.isdigit():
                # Webcam input
                camera_index = int(rtsp_url)
                cap = cv2.VideoCapture(camera_index)
            else:
                # RTSP stream
                cap = cv2.VideoCapture(rtsp_url)
            
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                # Send error frame to frontend
                if frame_callback:
                    try:
                        frame_callback("Camera Unavailable")
                    except Exception as cb_error:
                        print(f"[AI MODULE] Error sending error frame to callback: {cb_error}")
                
                return
            
            frame_count = 0
            # For face tracking with ID persistent and re-identification
            tracked_faces = {}  # {track_id: {'bbox', 'last_seen', 'name', 'embedding', 'confidence_history'}}
            next_track_id = 1
            bbox_history = {}  # For smoothing bounding box
            last_recognition = {}  # To prevent repeated attendance logging
            
            while not stop_event.is_set() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    # Try to reconnect
                    cap.release()
                    time.sleep(2)
                    if rtsp_url.isdigit():
                        cap = cv2.VideoCapture(int(rtsp_url))
                    else:
                        cap = cv2.VideoCapture(rtsp_url)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not cap.isOpened():
                        # Send error frame to frontend
                        if frame_callback:
                            try:
                                frame_callback("Camera Unavailable")
                            except Exception as cb_error:
                                print(f"[AI MODULE] Error sending error frame to callback: {cb_error}")
                        
                        # Continue trying to reconnect
                        time.sleep(5)  # Wait longer before next reconnect attempt
                        continue
                
                # Process frame with face recognition every few frames for performance
                process_frame = True
                if frame_skip:
                    process_frame = (frame_count % 3 == 0)  # Process every 3rd frame
                
                if process_frame:
                    try:
                        # Flip frame horizontally for mirror effect (same as main.py)
                        frame = cv2.flip(frame, 1)
                        
                        # Detect and recognize faces using the same approach as main.py
                        # Note: Don't resize frame to maintain consistent coordinate system
                        faces = system.app.get(frame)
                        
                        # Filter faces based on detection threshold if needed (same as main.py)
                        detection_threshold = SYSTEM_SPECS.get('detection_threshold', 0.5)
                        if detection_threshold > 0.5:  # Only filter if threshold is higher than default
                            faces = [face for face in faces if hasattr(face, 'det_score') and face.det_score >= detection_threshold]
                        
                        # Handle multi-person detection (same as main.py)
                        multi_person = SYSTEM_SPECS.get('multi_person', True)
                        if not multi_person and len(faces) > 0:
                            # Only process the face with highest detection score
                            faces = [max(faces, key=lambda f: f.det_score if hasattr(f, 'det_score') else 0)]
                        
                        current_time = time.time()
                        
                        # Detect current faces with coordinates (same as main.py)
                        current_faces = []
                        for i, face in enumerate(faces):
                            bbox = face.bbox.astype(int)
                            center_x = (bbox[0] + bbox[2]) // 2
                            center_y = (bbox[1] + bbox[3]) // 2
                            current_faces.append({
                                'index': i,
                                'bbox': bbox,
                                'center': (center_x, center_y),
                                'face_obj': face,
                                'embedding': face.embedding
                            })
                        
                        # Match detected faces with tracked faces using position and embedding (same as main.py)
                        matched_tracks = set()
                        matched_faces = set()
                        
                        # For each existing tracked face (same as main.py)
                        for track_id, track_data in list(tracked_faces.items()):
                            if current_faces:
                                # Find closest face for this tracked face
                                min_distance = float('inf')
                                best_match = None
                                best_similarity = 0
                                
                                for face_data in current_faces:
                                    if face_data['index'] in matched_faces:
                                        continue
                                        
                                    # Calculate distance between center points
                                    dx = track_data['center'][0] - face_data['center'][0]
                                    dy = track_data['center'][1] - face_data['center'][1]
                                    distance = (dx * dx + dy * dy) ** 0.5
                                    
                                    # Calculate similarity based on embedding
                                    similarity = self._calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                                    
                                    # Use combination of distance and similarity for matching
                                    # Prioritize high similarity, but consider distance
                                    if similarity > embedding_similarity_threshold:
                                        # If similarity is high, ignore slightly farther distance
                                        if distance < min_distance:
                                            min_distance = distance
                                            best_match = face_data
                                            best_similarity = similarity
                                    elif distance < min_distance and distance < max_distance_threshold:
                                        # If similarity is low but distance is close, also consider
                                        min_distance = distance
                                        best_match = face_data
                                        best_similarity = similarity
                                
                                # If we have a good match
                                if best_match and (best_similarity > embedding_similarity_threshold or min_distance < max_distance_threshold):
                                    matched_tracks.add(track_id)
                                    matched_faces.add(best_match['index'])
                                    
                                    # Update tracked face with new data
                                    bbox = best_match['bbox']
                                    face = best_match['face_obj']
                                    
                                    # Apply smoothing to bounding box
                                    if track_id in bbox_history:
                                        # Use weighted average for smoothing
                                        prev_bbox = bbox_history[track_id]
                                        smoothed_bbox = (smoothing_factor * prev_bbox + (1 - smoothing_factor) * bbox).astype(int)
                                    else:
                                        smoothed_bbox = bbox
                                    
                                    # Save smoothed bounding box for next frame
                                    bbox_history[track_id] = smoothed_bbox
                                    
                                    # Update tracking data
                                    center_x = (bbox[0] + bbox[2]) // 2
                                    center_y = (bbox[1] + bbox[3]) // 2
                                    tracked_faces[track_id] = {
                                        'bbox': smoothed_bbox,
                                        'center': (center_x, center_y),
                                        'last_seen': current_time,
                                        'face_obj': face,
                                        'embedding': face.embedding,  # Update embedding
                                        'confidence_history': track_data.get('confidence_history', []) + [best_similarity]
                                    }
                                    
                                    # Process recognition for this face (same as main.py)
                                    embedding = face.embedding
                                    name = "Unknown"
                                    best_score = 0
                                    
                                    # Use the same employee recognition logic as main.py
                                    for emp_name, emp_embedding in system.known_employees:
                                        # Calculate cosine similarity
                                        similarity = tracking_manager.calculate_embedding_similarity(embedding, emp_embedding)
                                        
                                        if similarity > best_score:
                                            best_score = similarity
                                            name = emp_name
                                    
                                    # Save name to tracked face
                                    tracked_faces[track_id]['name'] = name
                                    
                                    # Threshold for recognition (same as main.py)
                                    recognition_threshold = SYSTEM_SPECS.get('recognition_threshold', 
                                                                           SYSTEM_SPECS['detection_threshold'])
                                    if best_score > recognition_threshold:
                                        # Add name label
                                        label = f"{name} ({best_score:.2f}) ID:{track_id}"
                                        cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        
                                        # Log attendance if not logged within cooldown period
                                        if name not in last_recognition or \
                                           current_time - last_recognition[name] > recognition_cooldown:
                                            self.log_attendance(name)
                                            last_recognition[name] = current_time
                                        
                                        # Update last seen for employee monitoring
                                        self.employee_last_seen[name] = current_time
                                    else:
                                        # Face not recognized
                                        label = f"Unknown ID:{track_id}"
                                        cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    
                                    # Draw smoothed bounding box
                                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                    cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), 
                                                 (smoothed_bbox[2], smoothed_bbox[3]), color, 2)
                        
                        # Add new faces that are not yet tracked (same as main.py)
                        for face_data in current_faces:
                            if face_data['index'] not in matched_faces:
                                # Try to match with existing tracked faces based on embedding similarity
                                best_similarity = 0
                                best_track_id = None
                                
                                for track_id, track_data in tracked_faces.items():
                                    if track_id in matched_tracks:
                                        continue
                                        
                                    # Calculate similarity based on embedding
                                    similarity = self._calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                                    
                                    # If similarity is high and better than threshold, match
                                    if similarity > best_similarity and similarity > embedding_similarity_threshold:
                                        best_similarity = similarity
                                        best_track_id = track_id
                                
                                # If we found a match based on embedding, use the same ID
                                if best_track_id:
                                    track_id = best_track_id
                                    # Update tracked face information
                                    bbox = face_data['bbox']
                                    face = face_data['face_obj']
                                    
                                    # Apply smoothing to bounding box (no history, so use original bbox)
                                    smoothed_bbox = bbox
                                    bbox_history[track_id] = smoothed_bbox
                                    
                                    # Update tracking data
                                    center_x = (bbox[0] + bbox[2]) // 2
                                    center_y = (bbox[1] + bbox[3]) // 2
                                    tracked_faces[track_id] = {
                                        'bbox': smoothed_bbox,
                                        'center': (center_x, center_y),
                                        'last_seen': current_time,
                                        'face_obj': face,
                                        'embedding': face.embedding,
                                        'name': track_data.get('name', "Unknown"),
                                        'confidence_history': track_data.get('confidence_history', []) + [best_similarity]
                                    }
                                    
                                    # Use name from previous tracking if available
                                    name = tracked_faces[track_id]['name']
                                else:
                                    # Create new track ID
                                    track_id = next_track_id
                                    next_track_id += 1
                                    
                                    bbox = face_data['bbox']
                                    face = face_data['face_obj']
                                    
                                    # Apply smoothing to bounding box (no history, so use original bbox)
                                    smoothed_bbox = bbox
                                    bbox_history[track_id] = smoothed_bbox
                                    
                                    # Save tracking data
                                    center_x = (bbox[0] + bbox[2]) // 2
                                    center_y = (bbox[1] + bbox[3]) // 2
                                    tracked_faces[track_id] = {
                                        'bbox': smoothed_bbox,
                                        'center': (center_x, center_y),
                                        'last_seen': current_time,
                                        'face_obj': face,
                                        'embedding': face.embedding,
                                        'name': "Unknown",
                                        'confidence_history': []
                                    }
                                
                                # Process recognition for this new face (same as main.py)
                                embedding = face_data['embedding']
                                name = "Unknown"
                                best_score = 0
                                
                                # Use the same employee recognition logic as main.py
                                for emp_name, emp_embedding in system.known_employees:
                                    # Calculate cosine similarity
                                    similarity = np.dot(embedding, emp_embedding) / (
                                        np.linalg.norm(embedding) * np.linalg.norm(emp_embedding)
                                    )
                                    
                                    if similarity > best_score:
                                        best_score = similarity
                                        name = emp_name
                                
                                # Save name to tracked face
                                tracked_faces[track_id]['name'] = name
                                
                                # Threshold for recognition (same as main.py)
                                recognition_threshold = SYSTEM_SPECS.get('recognition_threshold', 
                                                                       SYSTEM_SPECS['detection_threshold'])
                                if best_score > recognition_threshold:
                                    # Add name label
                                    label = f"{name} ({best_score:.2f}) ID:{track_id}"
                                    cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    
                                    # Log attendance if not logged within cooldown period
                                    if name not in last_recognition or \
                                       current_time - last_recognition[name] > recognition_cooldown:
                                        self.log_attendance(name)
                                        last_recognition[name] = current_time
                                    
                                    # Update last seen for employee monitoring
                                    self.employee_last_seen[name] = current_time
                                else:
                                    # Face not recognized
                                    label = f"Unknown ID:{track_id}"
                                    cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                
                                # Draw bounding box
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), 
                                             (smoothed_bbox[2], smoothed_bbox[3]), color, 2)
                        
                        # Clean up tracked faces that haven't been seen for a while (same as main.py)
                        tracked_faces = {k: v for k, v in tracked_faces.items() 
                                       if current_time - v['last_seen'] < tracking_timeout}
                        
                        # Track employees and update database using tracking_manager
                        detected_employees = []
                        for track_id, track_data in tracked_faces.items():
                            if track_data['name'] != "Unknown":
                                detected_employees.append(track_data['name'])
                        
                        # Update employee status using tracking_manager for each detected employee
                        for employee_name in detected_employees:
                            employee_id = system.name_to_id.get(employee_name)
                            if employee_id:
                                # Use tracking_manager for unified tracking and DB updates
                                presence_data = tracking_manager.track_employee_presence(
                                    employee_id, employee_name, camera_id
                                )
                                print(f"[AI MODULE] Tracked {employee_name} at camera {presence_data['camera_name']}")
                            else:
                                # Fallback for employees not in mapping
                                self.update_employee_status(employee_name, camera_id)
                                db_manager.log_employee_location(employee_name, camera_id)
                                print(f"[AI MODULE] Fallback tracking for {employee_name} (no employee_id found)")
                        
                        # Add frame info (same as main.py)
                        cv2.putText(frame, f"Faces: {len(current_faces)} | Tracked: {len(tracked_faces)} | AI: Active", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                    except Exception as e:
                        print(f"[AI MODULE] Error in face recognition processing: {e}")
                        # Still add frame to stream even if processing fails (same as main.py)
                        cv2.putText(frame, "AI: Error", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Encode frame to base64 and send to callback
                if frame_callback:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode()
                        # Send frame to callback - let application.py handle error detection
                        frame_callback(frame_base64)
                    except Exception as e:
                        print(f"[AI MODULE] Error sending frame to callback: {e}")
                        # Send a simple error indicator
                        try:
                            frame_callback("Camera Unavailable")
                        except Exception as e2:
                            print(f"[AI MODULE] Error sending error frame: {e2}")
                
                frame_count += 1
                # Delay to maintain target FPS
                if fps_target > 0:
                    time.sleep(frame_delay)
                else:
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.03)  # ~30 FPS default
                
        except Exception as e:
            print(f"[AI MODULE] Error in face recognition worker: {e}")
        finally:
            if cap:
                cap.release()
    
    def update_employee_status(self, employee_name, camera_id, status='AVAILABLE'):
        """Update employee status in database"""
        # Get employee ID from name
        session = SessionLocal()
        employee = session.query(Employee).filter(Employee.name == employee_name).first()
        session.close()
        
        if employee:
            db_manager.update_employee_tracking(employee.employee_id, camera_id, status)
        else:
            print(f"[AI MODULE] Employee {employee_name} not found in database")
    
    def _background_processing_loop(self):
        """Background loop for continuous processing"""
        while self.is_running:
            try:
                time.sleep(1)  # Keep the thread alive
            except Exception as e:
                print(f"[AI MODULE] Error in background processing loop: {e}")
    
    def _absence_checking_loop(self):
        """Background loop for checking employee absence"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check for absent employees
                absent_employees = []
                for employee_name, last_seen_time in self.employee_last_seen.items():
                    if current_time - last_seen_time > self.absence_threshold:
                        absent_employees.append(employee_name)
                
                # Update status for absent employees
                if absent_employees:
                    self._mark_employees_absent(absent_employees)
                
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"[AI MODULE] Error in absence checking loop: {e}")
    
    def _calculate_embedding_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"[AI MODULE] Error calculating similarity: {e}")
            return 0.0
    
    def log_attendance(self, name: str):
        """Log employee attendance to database"""
        db_manager.log_attendance(name)
    
    def _create_error_frame(self, error_message):
        """Create a base64 encoded error frame with error message"""
        try:
            # Create a black image with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add error text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = error_message or "Camera Unavailable"
            text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] // 2
            
            cv2.putText(frame, text, (text_x, text_y), 
                       font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode()
        except Exception as e:
            print(f"[AI MODULE] Error creating error frame: {e}")
            return "Camera Unavailable"
    
    def _mark_employees_absent(self, employee_names):
        """Mark employees as absent in the database"""
        try:
            session = SessionLocal()
            
            for employee_name in employee_names:
                status_record = session.query(EmployeeStatus).filter_by(employee_name=employee_name).first()
                if status_record and status_record.status == 'available':
                    status_record.status = 'off'
                    # Don't update last_seen here as we want to keep the last seen time
                    print(f"[AI MODULE] Marked {employee_name} as absent")
            
            session.commit()
            session.close()
        except Exception as e:
            print(f"[AI MODULE] Error marking employees as absent: {e}")
    
    def get_employee_status(self, employee_name):
        """Get current status of an employee"""
        try:
            session = SessionLocal()
            status_record = session.query(EmployeeStatus).filter_by(employee_name=employee_name).first()
            session.close()
            
            if status_record:
                return {
                    'status': status_record.status,
                    'last_seen': status_record.last_seen,
                    'current_camera': status_record.current_camera
                }
            return None
        except Exception as e:
            print(f"[AI MODULE] Error getting employee status: {e}")
            return None
    
    def get_all_employee_statuses(self):
        """Get status of all employees"""
        return db_manager.get_employee_statuses()

# Global instance for background processing
ai_processing_module = AIProcessingModule()

# Function to run continuous background processing
def run_background_inference():
    """Run continuous background face recognition inference"""
    print("[AI MODULE] Running continuous background face recognition inference...")
    
    # Initialize model
    if not ai_processing_module.initialize_model():
        print("[AI MODULE] Failed to initialize face recognition model")
        return
    
    # Start processing
    ai_processing_module.start_processing()
    
    # Keep running in background
    try:
        while ai_processing_module.is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n[AI MODULE] Stopping background inference...")
        ai_processing_module.stop_processing()
        print("[AI MODULE] Background inference stopped")

if __name__ == '__main__':
    # Run background inference when executed directly
    run_background_inference()
