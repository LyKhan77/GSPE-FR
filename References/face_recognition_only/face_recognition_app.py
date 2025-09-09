#!/usr/bin/env python3
# face_recognition_app.py
# Sistem face recognition yang direvisi dengan flow:
# webcam --> register wajah ke DB --> InsightFace --> inferensi --> Logging

import cv2
import numpy as np
import os
import insightface
import sys
import time
from database import (
    get_all_employees, 
    get_all_employees_with_id, 
    save_employee, 
    delete_employee_by_name, 
    log_attendance, 
    load_system_specs,
    DEFAULT_SYSTEM_SPECS
)

# Import modul untuk enhanced tracking dan monitoring
try:
    from enhanced_tracking_config import get_enhanced_tracking_config
    ENHANCED_TRACKING_AVAILABLE = True
except ImportError:
    ENHANCED_TRACKING_AVAILABLE = False
    print("[INFO] Enhanced tracking config tidak tersedia")

# Import modul untuk employee status tracking
try:
    from employee_status_tracker import EmployeeStatusTracker
    EMPLOYEE_STATUS_TRACKING_AVAILABLE = True
except ImportError:
    EMPLOYEE_STATUS_TRACKING_AVAILABLE = False
    print("[INFO] Employee status tracker tidak tersedia")

# Import modul untuk update employee status ke database
try:
    # Modul ini tidak tersedia setelah refaktor, jadi kita abaikan
    DATABASE_UPDATE_AVAILABLE = False
    print("[INFO] Database update module tidak tersedia")
except ImportError:
    DATABASE_UPDATE_AVAILABLE = False
    print("[INFO] Database update module tidak tersedia")
    import traceback
    traceback.print_exc()

# Load system specifications
SYSTEM_SPECS = load_system_specs()

# Struktur data untuk employee tracking dan monitoring (untuk dashboard di masa depan)
class EmployeeTracking:
    """Kelas untuk tracking employee dan activity monitoring"""
    
    def __init__(self):
        self.employee_status = {}  # {employee_name: {'last_seen': timestamp, 'camera': camera_id, 'status': 'present/absent'}}
        self.employee_durations = {}  # {employee_name: {'entry_time': timestamp, 'total_duration': seconds}}
        self.alerts = []  # List of alert messages
        
    def update_employee_status(self, employee_name, camera_id, timestamp):
        """Update status employee ketika terdeteksi"""
        self.employee_status[employee_name] = {
            'last_seen': timestamp,
            'camera': camera_id,
            'status': 'present'
        }
        
        # Jika ini pertama kali employee terdeteksi, catat waktu masuk
        if employee_name not in self.employee_durations:
            self.employee_durations[employee_name] = {
                'entry_time': timestamp,
                'total_duration': 0
            }
    
    def check_activity_alerts(self, current_time, alert_threshold=300):  # 300 detik = 5 menit
        """Periksa apakah ada employee yang tidak terdeteksi selama lebih dari threshold"""
        alerts = []
        for employee_name, status in self.employee_status.items():
            time_since_last_seen = current_time - status['last_seen']
            if time_since_last_seen > alert_threshold and status['status'] == 'present':
                # Update status menjadi absent
                self.employee_status[employee_name]['status'] = 'absent'
                alert_msg = f"ALERT: {employee_name} tidak terdeteksi selama {time_since_last_seen:.0f} detik"
                alerts.append(alert_msg)
                self.alerts.append({
                    'timestamp': current_time,
                    'employee': employee_name,
                    'message': alert_msg
                })
        return alerts
    
    def get_employee_report(self, employee_name):
        """Dapatkan laporan lengkap untuk seorang employee"""
        if employee_name not in self.employee_status:
            return None
            
        status = self.employee_status[employee_name]
        duration_info = self.employee_durations.get(employee_name, {})
        
        # Hitung durasi saat ini jika masih present
        duration = 0
        if status['status'] == 'present' and 'entry_time' in duration_info:
            duration = time.time() - duration_info['entry_time']
        elif 'total_duration' in duration_info:
            duration = duration_info['total_duration']
            
        return {
            'name': employee_name,
            'last_seen': status['last_seen'],
            'camera': status['camera'],
            'status': status['status'],
            'duration_seconds': duration,
            'duration_formatted': self._format_duration(duration)
        }
    
    def _format_duration(self, seconds):
        """Format durasi dalam bentuk yang mudah dibaca"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def get_all_employees_status(self):
        """Dapatkan status semua employee"""
        return self.employee_status
    
    def get_recent_alerts(self, limit=10):
        """Dapatkan alert terbaru"""
        return self.alerts[-limit:] if len(self.alerts) > limit else self.alerts

def get_tracking_parameters():
    """Dapatkan parameter tracking yang ditingkatkan"""
    return {
        'max_distance_threshold': SYSTEM_SPECS.get('max_distance_threshold', 150),
        'tracking_timeout': SYSTEM_SPECS.get('tracking_timeout', 3.0)
    }

def show_system_specs():
    """Tampilkan spesifikasi sistem face recognition"""
    print("\n=== Spesifikasi Sistem Face Recognition ===")
    print(f"Model                    : {SYSTEM_SPECS['model']}")
    print(f"Detection Threshold      : {SYSTEM_SPECS['detection_threshold']}")
    print(f"Detection Size           : {SYSTEM_SPECS['detection_size'][0]}x{SYSTEM_SPECS['detection_size'][1]}")
    print(f"Recognition Cooldown     : {SYSTEM_SPECS['recognition_cooldown']} detik")
    print(f"BBox Smoothing Factor    : {SYSTEM_SPECS['bbox_smoothing_factor']}")
    print(f"Providers                : {SYSTEM_SPECS['providers']}")
    print(f"Target FPS               : {SYSTEM_SPECS['fps_target']}")
    print(f"Frame Skip               : {SYSTEM_SPECS['frame_skip']}")
    print(f"Multi-Person Detection   : {SYSTEM_SPECS['multi_person']}")
    print("==========================================")

def list_all_employees():
    """Tampilkan semua karyawan dalam database"""
    employees = get_all_employees()
    
    if employees:
        print("\nDaftar Karyawan:")
        print("-" * 30)
        for i, (name, _) in enumerate(employees, 1):
            print("{0}. {1}".format(i, name))
    else:
        print("Tidak ada karyawan dalam database")
    return employees

class FaceRecognitionSystem:
    def __init__(self):
        # Inisialisasi InsightFace dengan GPU acceleration jika tersedia
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Prioritaskan CUDA
        self.app = insightface.app.FaceAnalysis(providers=providers)
        # Gunakan ukuran deteksi dari konfigurasi
        det_size = SYSTEM_SPECS['detection_size']
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.load_employees()
        
        # Inisialisasi employee tracking untuk dashboard monitoring
        self.employee_tracking = EmployeeTracking()
        
        # Inisialisasi employee status tracker untuk deteksi karyawan keluar ruangan
        if EMPLOYEE_STATUS_TRACKING_AVAILABLE:
            self.employee_status_tracker = EmployeeStatusTracker(absence_threshold_seconds=60)  # 1 menit
        else:
            self.employee_status_tracker = None
            print("[INFO] Employee status tracking tidak tersedia")
        
    def load_employees(self):
        """Load data karyawan dari database"""
        self.known_employees = get_all_employees()
        self.known_employees_with_id = get_all_employees_with_id()
        print(f"[INFO] Loaded {len(self.known_employees)} karyawan dari database")
        
    def _calculate_embedding_similarity(self, embedding1, embedding2):
        """Hitung cosine similarity antara dua embedding"""
        try:
            # Normalisasi embedding
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # Hitung cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"[ERROR] Gagal menghitung similarity: {e}")
            return 0.0
        
    def register_employee(self, name):
        """Registrasi karyawan baru"""
        cap = cv2.VideoCapture(0)
        
        print(f"[INFO] Registrasi karyawan: {name}")
        print("[INFO] Tekan 'c' untuk capture wajah, 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Gagal membaca frame dari kamera")
                break
                
            # Flip frame horizontally untuk efek mirror
            frame = cv2.flip(frame, 1)
            
            # Deteksi wajah menggunakan InsightFace
            faces = self.app.get(frame)
            
            # Gambar bounding box untuk semua wajah yang terdeteksi
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            cv2.imshow("Registrasi Karyawan", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                # Gunakan wajah pertama yang terdeteksi
                face = faces[0]
                embedding = face.embedding
                bbox = face.bbox.astype(int)
                
                # Crop wajah dari frame
                face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # Simpan ke database
                employee_id = save_employee(name, embedding, face_img)
                if employee_id:
                    print(f"[INFO] Karyawan {name} memiliki ID database: {employee_id}")
                break
            elif key == ord('q'):
                print("[INFO] Registrasi dibatalkan")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.load_employees()  # Reload data karyawan
        
    def register_employee_rtsp(self, name):
        """Registrasi karyawan baru menggunakan RTSP CCTV"""
        rtsp_url = "rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1"  # URL RTSP default
        cap = cv2.VideoCapture(rtsp_url)
        
        # Periksa apakah koneksi RTSP berhasil
        if not cap.isOpened():
            print("[ERROR] Gagal membuka stream RTSP. Pastikan URL RTSP benar dan CCTV terhubung.")
            return
        
        print(f"[INFO] Registrasi karyawan: {name} melalui RTSP stream: {rtsp_url}")
        print("[INFO] Tekan 'c' untuk capture wajah, 'q' untuk keluar")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Gagal membaca frame dari stream RTSP. Mungkin koneksi terputus.")
                break
                
            # Deteksi wajah menggunakan InsightFace
            faces = self.app.get(frame)
            
            # Gambar bounding box untuk semua wajah yang terdeteksi
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            cv2.imshow("Registrasi Karyawan RTSP", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                # Gunakan wajah pertama yang terdeteksi
                face = faces[0]
                embedding = face.embedding
                bbox = face.bbox.astype(int)
                
                # Crop wajah dari frame
                face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # Simpan ke database
                employee_id = save_employee(name, embedding, face_img)
                if employee_id:
                    print(f"[INFO] Karyawan {name} memiliki ID database: {employee_id}")
                break
            elif key == ord('q'):
                print("[INFO] Registrasi dibatalkan")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        self.load_employees()  # Reload data karyawan
        
    def recognize_faces(self):
        """Inferensi face recognition dengan tracking yang lebih baik"""
        cap = cv2.VideoCapture(0)
        
        print("[INFO] Memulai face recognition. Tekan 'q' untuk keluar")
        print("[INFO] Spesifikasi sistem:")
        print(f"       - Threshold: {SYSTEM_SPECS['detection_threshold']}")
        print(f"       - Cooldown: {SYSTEM_SPECS['recognition_cooldown']} detik")
        print(f"       - Smoothing: {SYSTEM_SPECS['bbox_smoothing_factor']}")
        
        # Untuk mencegah pencatatan kehadiran berulang
        last_recognition = {}
        recognition_cooldown = SYSTEM_SPECS['recognition_cooldown']  # detik
        
        import time
        
        # Untuk monitoring FPS
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Untuk smoothing bounding box dengan tracking yang lebih baik
        bbox_history = {}
        smoothing_factor = SYSTEM_SPECS['bbox_smoothing_factor']  # Faktor smoothing (0.0-1.0, semakin tinggi semakin smooth)
        
        # Untuk face tracking dengan ID persisten dan re-identification
        tracked_faces = {}  # {track_id: {'bbox', 'last_seen', 'name', 'embedding', 'confidence_history'}}
        next_track_id = 1
        max_distance_threshold = SYSTEM_SPECS.get("max_distance_threshold", 150)  # Threshold untuk matching wajah antar frame
        tracking_timeout = SYSTEM_SPECS.get("tracking_timeout", 3.0)  # Timeout untuk tracking
        
        # Untuk activity monitoring
        employee_last_seen = {}  # {employee_name: last_seen_time}
        
        # Untuk re-identification berbasis embedding
        embedding_similarity_threshold = 0.7  # Threshold untuk matching berbasis embedding
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Gagal membaca frame dari kamera")
                break
                
            # Flip frame horizontally untuk efek mirror
            frame = cv2.flip(frame, 1)
            
            # Deteksi dan pengenalan wajah menggunakan InsightFace
            faces = self.app.get(frame)
            
            current_time = time.time()
            
            # Deteksi wajah saat ini dengan koordinat
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
            
            # Match wajah yang terdeteksi dengan tracked faces menggunakan kombinasi posisi dan embedding
            matched_tracks = set()
            matched_faces = set()
            
            # Untuk setiap tracked face yang ada
            for track_id, track_data in list(tracked_faces.items()):
                if current_faces:
                    # Cari wajah terdekat untuk tracked face ini
                    min_distance = float('inf')
                    best_match = None
                    best_similarity = 0
                    
                    for face_data in current_faces:
                        if face_data['index'] in matched_faces:
                            continue
                            
                        # Hitung jarak antara center points
                        dx = track_data['center'][0] - face_data['center'][0]
                        dy = track_data['center'][1] - face_data['center'][1]
                        distance = (dx * dx + dy * dy) ** 0.5
                        
                        # Hitung similarity berdasarkan embedding
                        similarity = self._calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                        
                        # Gunakan kombinasi jarak dan similarity untuk matching
                        # Prioritaskan similarity tinggi, tapi pertimbangkan jarak
                        if similarity > embedding_similarity_threshold:
                            # Jika similarity tinggi, abaikan sedikit jarak yang lebih jauh
                            if distance < min_distance:
                                min_distance = distance
                                best_match = face_data
                                best_similarity = similarity
                        elif distance < min_distance and distance < get_tracking_parameters()["max_distance_threshold"]:
                            # Jika similarity rendah tapi jarak dekat, pertimbangkan juga
                            min_distance = distance
                            best_match = face_data
                            best_similarity = similarity
                    
                    # Jika ada match yang baik
                    if best_match and (best_similarity > embedding_similarity_threshold or min_distance < get_tracking_parameters()["max_distance_threshold"]):
                        matched_tracks.add(track_id)
                        matched_faces.add(best_match['index'])
                        
                        # Update tracked face dengan data baru
                        bbox = best_match['bbox']
                        face = best_match['face_obj']
                        
                        # Terapkan smoothing pada bounding box
                        if track_id in bbox_history:
                            # Gunakan weighted average untuk smoothing
                            prev_bbox = bbox_history[track_id]
                            smoothed_bbox = (smoothing_factor * prev_bbox + (1 - smoothing_factor) * bbox).astype(int)
                        else:
                            smoothed_bbox = bbox
                        
                        # Simpan bounding box yang dihaluskan untuk frame berikutnya
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
                        
                        # Proses recognisi untuk wajah ini
                        embedding = face.embedding
                        name = "Unknown"
                        best_score = 0
                        
                        for emp_name, emp_embedding in self.known_employees:
                            # Hitung cosine similarity
                            similarity = np.dot(embedding, emp_embedding) / (
                                np.linalg.norm(embedding) * np.linalg.norm(emp_embedding)
                            )
                            
                            if similarity > best_score:
                                best_score = similarity
                                name = emp_name
                        
                        # Simpan nama ke tracked face
                        tracked_faces[track_id]['name'] = name
                        
                        # Threshold untuk pengenalan
                        recognition_threshold = SYSTEM_SPECS.get('recognition_threshold', 
                                                               SYSTEM_SPECS['detection_threshold'])
                        if best_score > recognition_threshold:
                            # Tambahkan label nama
                            label = f"{name} ({best_score:.2f}) ID:{track_id}"
                            cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Catat kehadiran jika belum dicatat dalam cooldown period
                            if name not in last_recognition or \
                               current_time - last_recognition[name] > recognition_cooldown:
                                log_attendance(name)
                                last_recognition[name] = current_time
                            
                            # Update last seen untuk employee monitoring
                            employee_last_seen[name] = current_time
                        else:
                            # Wajah tidak dikenali
                            label = f"Unknown ID:{track_id}"
                            cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Gambar bounding box yang dihaluskan
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), 
                                     (smoothed_bbox[2], smoothed_bbox[3]), color, 2)
            
            # Tambahkan wajah baru yang belum di-track
            for face_data in current_faces:
                if face_data['index'] not in matched_faces:
                    # Coba cocokkan dengan tracked faces yang sudah ada berdasarkan embedding similarity
                    best_similarity = 0
                    best_track_id = None
                    
                    for track_id, track_data in tracked_faces.items():
                        if track_id in matched_tracks:
                            continue
                            
                        # Hitung similarity berdasarkan embedding
                        similarity = self._calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                        
                        # Jika similarity tinggi dan lebih baik dari threshold, cocokkan
                        if similarity > best_similarity and similarity > embedding_similarity_threshold:
                            best_similarity = similarity
                            best_track_id = track_id
                    
                    # Jika ditemukan match berdasarkan embedding, gunakan ID yang sama
                    if best_track_id:
                        track_id = best_track_id
                        # Update informasi tracked face
                        bbox = face_data['bbox']
                        face = face_data['face_obj']
                        
                        # Terapkan smoothing pada bounding box (tidak ada history, jadi gunakan bbox asli)
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
                        
                        # Gunakan nama dari tracking sebelumnya jika ada
                        name = tracked_faces[track_id]['name']
                    else:
                        # Buat track ID baru
                        track_id = next_track_id
                        next_track_id += 1
                        
                        bbox = face_data['bbox']
                        face = face_data['face_obj']
                        
                        # Terapkan smoothing pada bounding box (tidak ada history, jadi gunakan bbox asli)
                        smoothed_bbox = bbox
                        bbox_history[track_id] = smoothed_bbox
                        
                        # Simpan tracking data
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
                    
                    # Proses recognisi untuk wajah baru ini
                    embedding = face_data['embedding']
                    name = "Unknown"
                    best_score = 0
                    
                    for emp_name, emp_embedding in self.known_employees:
                        # Hitung cosine similarity
                        similarity = np.dot(embedding, emp_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(emp_embedding)
                        )
                        
                        if similarity > best_score:
                            best_score = similarity
                            name = emp_name
                    
                    # Simpan nama ke tracked face
                    tracked_faces[track_id]['name'] = name
                    
                    # Threshold untuk pengenalan
                    recognition_threshold = SYSTEM_SPECS.get('recognition_threshold', 
                                                           SYSTEM_SPECS['detection_threshold'])
                    if best_score > recognition_threshold:
                        # Tambahkan label nama
                        label = f"{name} ({best_score:.2f}) ID:{track_id}"
                        cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Catat kehadiran jika belum dicatat dalam cooldown period
                        if name not in last_recognition or \
                           current_time - last_recognition[name] > recognition_cooldown:
                            log_attendance(name)
                            last_recognition[name] = current_time
                        
                        # Update last seen untuk employee monitoring
                        employee_last_seen[name] = current_time
                    else:
                        # Wajah tidak dikenali
                        label = f"Unknown ID:{track_id}"
                        cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Gambar bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), 
                                 (smoothed_bbox[2], smoothed_bbox[3]), color, 2)
            
            # Bersihkan tracked faces yang tidak terlihat dalam beberapa frame
            tracked_faces = {k: v for k, v in tracked_faces.items() 
                           if current_time - v['last_seen'] < get_tracking_parameters()["tracking_timeout"]}  # Hapus jika tidak terlihat > 1 detik
            
            # Periksa karyawan yang absent jika employee status tracker tersedia
            if self.employee_status_tracker:
                newly_absent = self.employee_status_tracker.check_absences()
                # Tampilkan notifikasi untuk karyawan yang baru absent
                for employee_name in newly_absent:
                    print(f"[STATUS] {employee_name} tidak terdeteksi lebih dari 1 menit, status: absent")
            
            # Hitung dan tampilkan FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Hitung FPS setiap 30 frame
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time if elapsed_time > 0 else 0
                start_time = time.time()
                
            # Tampilkan FPS dan jumlah wajah di frame
            cv2.putText(frame, f"Faces: {len(current_faces)} Tracked: {len(tracked_faces)} FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Face Recognition", frame)
            
            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Sistem face recognition dihentikan")
        
        # Return employee monitoring data untuk kebutuhan dashboard
        return employee_last_seen

    def recognize_faces_rtsp(self):
        """Inferensi face recognition dengan input RTSP dari CCTV dengan tracking yang lebih baik"""
        rtsp_url = "rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1"  # URL RTSP default
        cap = cv2.VideoCapture(rtsp_url)
        
        # Periksa apakah koneksi RTSP berhasil
        if not cap.isOpened():
            print("[ERROR] Gagal membuka stream RTSP. Pastikan URL RTSP benar dan CCTV terhubung.")
            return
        
        print(f"[INFO] Memulai face recognition dari RTSP stream: {rtsp_url}")
        print("[INFO] Tekan 'q' untuk keluar")
        print("[INFO] Spesifikasi sistem:")
        print(f"       - Threshold: {SYSTEM_SPECS['detection_threshold']}")
        print(f"       - Cooldown: {SYSTEM_SPECS['recognition_cooldown']} detik")
        print(f"       - Smoothing: {SYSTEM_SPECS['bbox_smoothing_factor']}")
        
        # Untuk mencegah pencatatan kehadiran berulang
        last_recognition = {}
        recognition_cooldown = SYSTEM_SPECS['recognition_cooldown']  # detik
        
        import time
        
        # Untuk monitoring FPS
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Untuk smoothing bounding box dengan tracking yang lebih baik
        bbox_history = {}
        smoothing_factor = SYSTEM_SPECS['bbox_smoothing_factor']  # Faktor smoothing (0.0-1.0, semakin tinggi semakin smooth)
        
        # Untuk face tracking dengan ID persisten dan re-identification
        tracked_faces = {}  # {track_id: {'bbox', 'last_seen', 'name', 'embedding', 'confidence_history'}}
        next_track_id = 1
        max_distance_threshold = SYSTEM_SPECS.get("max_distance_threshold", 150)  # Threshold untuk matching wajah antar frame
        tracking_timeout = SYSTEM_SPECS.get("tracking_timeout", 3.0)  # Timeout untuk tracking
        
        # Untuk activity monitoring
        employee_last_seen = {}  # {employee_name: last_seen_time}
        
        # Untuk re-identification berbasis embedding
        embedding_similarity_threshold = 0.7  # Threshold untuk matching berbasis embedding
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Gagal membaca frame dari stream RTSP. Mungkin koneksi terputus.")
                break
                
            # Deteksi dan pengenalan wajah menggunakan InsightFace
            faces = self.app.get(frame)
            
            current_time = time.time()
            
            # Deteksi wajah saat ini dengan koordinat
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
            
            # Match wajah yang terdeteksi dengan tracked faces menggunakan kombinasi posisi dan embedding
            matched_tracks = set()
            matched_faces = set()
            
            # Untuk setiap tracked face yang ada
            for track_id, track_data in list(tracked_faces.items()):
                if current_faces:
                    # Cari wajah terdekat untuk tracked face ini
                    min_distance = float('inf')
                    best_match = None
                    best_similarity = 0
                    
                    for face_data in current_faces:
                        if face_data['index'] in matched_faces:
                            continue
                            
                        # Hitung jarak antara center points
                        dx = track_data['center'][0] - face_data['center'][0]
                        dy = track_data['center'][1] - face_data['center'][1]
                        distance = (dx * dx + dy * dy) ** 0.5
                        
                        # Hitung similarity berdasarkan embedding
                        similarity = self._calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                        
                        # Gunakan kombinasi jarak dan similarity untuk matching
                        # Prioritaskan similarity tinggi, tapi pertimbangkan jarak
                        if similarity > embedding_similarity_threshold:
                            # Jika similarity tinggi, abaikan sedikit jarak yang lebih jauh
                            if distance < min_distance:
                                min_distance = distance
                                best_match = face_data
                                best_similarity = similarity
                        elif distance < min_distance and distance < get_tracking_parameters()["max_distance_threshold"]:
                            # Jika similarity rendah tapi jarak dekat, pertimbangkan juga
                            min_distance = distance
                            best_match = face_data
                            best_similarity = similarity
                    
                    # Jika ada match yang baik
                    if best_match and (best_similarity > embedding_similarity_threshold or min_distance < get_tracking_parameters()["max_distance_threshold"]):
                        matched_tracks.add(track_id)
                        matched_faces.add(best_match['index'])
                        
                        # Update tracked face dengan data baru
                        bbox = best_match['bbox']
                        face = best_match['face_obj']
                        
                        # Terapkan smoothing pada bounding box
                        if track_id in bbox_history:
                            # Gunakan weighted average untuk smoothing
                            prev_bbox = bbox_history[track_id]
                            smoothed_bbox = (smoothing_factor * prev_bbox + (1 - smoothing_factor) * bbox).astype(int)
                        else:
                            smoothed_bbox = bbox
                        
                        # Simpan bounding box yang dihaluskan untuk frame berikutnya
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
                        
                        # Proses recognisi untuk wajah ini
                        embedding = face.embedding
                        name = "Unknown"
                        best_score = 0
                        
                        for emp_name, emp_embedding in self.known_employees:
                            # Hitung cosine similarity
                            similarity = np.dot(embedding, emp_embedding) / (
                                np.linalg.norm(embedding) * np.linalg.norm(emp_embedding)
                            )
                            
                            if similarity > best_score:
                                best_score = similarity
                                name = emp_name
                        
                        # Simpan nama ke tracked face
                        tracked_faces[track_id]['name'] = name
                        
                        # Threshold untuk pengenalan
                        recognition_threshold = SYSTEM_SPECS.get('recognition_threshold', 
                                                               SYSTEM_SPECS['detection_threshold'])
                        if best_score > recognition_threshold:
                            # Tambahkan label nama
                            label = f"{name} ({best_score:.2f}) ID:{track_id}"
                            cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Catat kehadiran jika belum dicatat dalam cooldown period
                            if name not in last_recognition or \
                               current_time - last_recognition[name] > recognition_cooldown:
                                log_attendance(name)
                                last_recognition[name] = current_time
                            
                            # Update last seen untuk employee monitoring
                            employee_last_seen[name] = current_time
                        else:
                            # Wajah tidak dikenali
                            label = f"Unknown ID:{track_id}"
                            cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Gambar bounding box yang dihaluskan
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), 
                                     (smoothed_bbox[2], smoothed_bbox[3]), color, 2)
            
            # Tambahkan wajah baru yang belum di-track
            for face_data in current_faces:
                if face_data['index'] not in matched_faces:
                    # Coba cocokkan dengan tracked faces yang sudah ada berdasarkan embedding similarity
                    best_similarity = 0
                    best_track_id = None
                    
                    for track_id, track_data in tracked_faces.items():
                        if track_id in matched_tracks:
                            continue
                            
                        # Hitung similarity berdasarkan embedding
                        similarity = self._calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                        
                        # Jika similarity tinggi dan lebih baik dari threshold, cocokkan
                        if similarity > best_similarity and similarity > embedding_similarity_threshold:
                            best_similarity = similarity
                            best_track_id = track_id
                    
                    # Jika ditemukan match berdasarkan embedding, gunakan ID yang sama
                    if best_track_id:
                        track_id = best_track_id
                        # Update informasi tracked face
                        bbox = face_data['bbox']
                        face = face_data['face_obj']
                        
                        # Terapkan smoothing pada bounding box (tidak ada history, jadi gunakan bbox asli)
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
                        
                        # Gunakan nama dari tracking sebelumnya jika ada
                        name = tracked_faces[track_id]['name']
                    else:
                        # Buat track ID baru
                        track_id = next_track_id
                        next_track_id += 1
                        
                        bbox = face_data['bbox']
                        face = face_data['face_obj']
                        
                        # Terapkan smoothing pada bounding box (tidak ada history, jadi gunakan bbox asli)
                        smoothed_bbox = bbox
                        bbox_history[track_id] = smoothed_bbox
                        
                        # Simpan tracking data
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
                    
                    # Proses recognisi untuk wajah baru ini
                    embedding = face_data['embedding']
                    name = "Unknown"
                    best_score = 0
                    
                    for emp_name, emp_embedding in self.known_employees:
                        # Hitung cosine similarity
                        similarity = np.dot(embedding, emp_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(emp_embedding)
                        )
                        
                        if similarity > best_score:
                            best_score = similarity
                            name = emp_name
                    
                    # Simpan nama ke tracked face
                    tracked_faces[track_id]['name'] = name
                    
                    # Threshold untuk pengenalan
                    recognition_threshold = SYSTEM_SPECS.get('recognition_threshold', 
                                                           SYSTEM_SPECS['detection_threshold'])
                    if best_score > recognition_threshold:
                        # Tambahkan label nama
                        label = f"{name} ({best_score:.2f}) ID:{track_id}"
                        cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Catat kehadiran jika belum dicatat dalam cooldown period
                        if name not in last_recognition or \
                           current_time - last_recognition[name] > recognition_cooldown:
                            log_attendance(name)
                            last_recognition[name] = current_time
                        
                        # Update last seen untuk employee monitoring
                        employee_last_seen[name] = current_time
                    else:
                        # Wajah tidak dikenali
                        label = f"Unknown ID:{track_id}"
                        cv2.putText(frame, label, (smoothed_bbox[0], smoothed_bbox[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Gambar bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (smoothed_bbox[0], smoothed_bbox[1]), 
                                 (smoothed_bbox[2], smoothed_bbox[3]), color, 2)
            
            # Bersihkan tracked faces yang tidak terlihat dalam beberapa frame
            tracked_faces = {k: v for k, v in tracked_faces.items() 
                           if current_time - v['last_seen'] < get_tracking_parameters()["tracking_timeout"]}  # Hapus jika tidak terlihat > 1 detik
            
            # Periksa karyawan yang absent jika employee status tracker tersedia
            if self.employee_status_tracker:
                newly_absent = self.employee_status_tracker.check_absences()
                # Tampilkan notifikasi untuk karyawan yang baru absent
                for employee_name in newly_absent:
                    print(f"[STATUS] {employee_name} tidak terdeteksi lebih dari 1 menit, status: absent")
            
            # Hitung dan tampilkan FPS
            frame_count += 1
            if frame_count % 30 == 0:  # Hitung FPS setiap 30 frame
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time if elapsed_time > 0 else 0
                start_time = time.time()
                
            # Tampilkan FPS dan jumlah wajah di frame
            cv2.putText(frame, f"Faces: {len(current_faces)} Tracked: {len(tracked_faces)} FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Face Recognition RTSP", frame)
            
            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Sistem face recognition RTSP dihentikan")
        
        # Return employee monitoring data untuk kebutuhan dashboard
        return employee_last_seen

def manage_employee_menu(system):
    """Menu untuk mengelola karyawan"""
    print("=== Manajemen Karyawan ===")
    
    while True:
        # Tampilkan daftar karyawan
        employees = list_all_employees()
        
        print("\nPilih opsi:")
        print("1. Tambah karyawan (Webcam)")
        print("2. Tambah karyawan (RTSP CCTV)")
        print("3. Hapus karyawan")
        print("4. Lihat spesifikasi sistem")
        print("5. Parameter tuning")
        print("6. Kembali ke menu utama")
        
        try:
            choice = input("Masukkan pilihan (1-6): ").strip()
        except EOFError:
            print("\nKembali ke menu utama...")
            break
        
        if choice == "1":
            # Tambah karyawan menggunakan webcam
            try:
                name = input("Masukkan nama karyawan baru: ").strip()
            except EOFError:
                print("\nKembali ke menu manajemen karyawan...")
                continue
                
            if not name:
                print("[ERROR] Nama tidak boleh kosong!")
                continue
            
            # Cek apakah karyawan sudah ada
            employee_names = [emp[0] for emp in employees]
            if name in employee_names:
                print(f"[ERROR] Karyawan '{name}' sudah ada dalam database!")
                continue
                
            print(f"\nMemulai registrasi untuk '{name}'...")
            print("Arahkan wajah ke webcam dan tekan 'c' untuk capture, 'q' untuk batal")
            system.register_employee(name)
            
        elif choice == "2":
            # Tambah karyawan menggunakan RTSP CCTV
            try:
                name = input("Masukkan nama karyawan baru: ").strip()
            except EOFError:
                print("\nKembali ke menu manajemen karyawan...")
                continue
                
            if not name:
                print("[ERROR] Nama tidak boleh kosong!")
                continue
            
            # Cek apakah karyawan sudah ada
            employee_names = [emp[0] for emp in employees]
            if name in employee_names:
                print(f"[ERROR] Karyawan '{name}' sudah ada dalam database!")
                continue
                
            print(f"\nMemulai registrasi untuk '{name}' melalui RTSP...")
            print("Arahkan wajah ke kamera CCTV dan tekan 'c' untuk capture, 'q' untuk batal")
            rtsp_url = "rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1"  # URL RTSP default
            print(f"[INFO] Menggunakan RTSP stream {rtsp_url}")
            system.register_employee_rtsp(name)
            
        elif choice == "3":
            # Hapus karyawan
            if not employees:
                print("Tidak ada karyawan untuk dihapus")
                continue
                
            try:
                idx = int(input("Masukkan nomor karyawan yang ingin dihapus: ")) - 1
                if 0 <= idx < len(employees):
                    name = employees[idx][0]
                    print(f"\nApakah Anda yakin ingin menghapus karyawan '{name}'? (y/N)")
                    
                    try:
                        confirmation = input().strip().lower()
                    except EOFError:
                        print("Penghapusan dibatalkan")
                        continue
                        
                    if confirmation == 'y' or confirmation == 'yes':
                        delete_employee_by_name(name)
                        # Reload data karyawan setelah penghapusan
                        system.load_employees()
                    else:
                        print("Penghapusan dibatalkan")
                else:
                    print("[ERROR] Nomor karyawan tidak valid!")
            except ValueError:
                print("[ERROR] Masukkan nomor yang valid!")
            except EOFError:
                print("\nKembali ke menu manajemen karyawan...")
                continue
                
        elif choice == "4":
            # Lihat spesifikasi sistem
            show_system_specs()
            
        elif choice == "5":
            # Jalankan parameter tuning script
            try:
                import subprocess
                import sys
                # Jalankan parameter_tuning.py dengan inherit stdout/stdin agar interaktif
                result = subprocess.run([sys.executable, "parameter_tuning.py"])
                if result.returncode != 0:
                    print("[ERROR] Gagal menjalankan parameter tuning")
                else:
                    # Reload system specs after tuning
                    global SYSTEM_SPECS
                    SYSTEM_SPECS = load_system_specs()
                    print("[INFO] Konfigurasi sistem diperbarui")
            except Exception as e:
                print(f"[ERROR] Gagal menjalankan parameter tuning: {e}")
            
        elif choice == "6":
            print("Kembali ke menu utama...")
            break
            
        else:
            print("[ERROR] Pilihan tidak valid!")

def main():
    print("=== Sistem Face Recognition yang Direvisi ===")
    system = FaceRecognitionSystem()
    
    while True:
        print("\n1. Manage employee")
        print("2. Jalankan face recognition (Webcam)")
        print("3. Jalankan face recognition (RTSP CCTV)")
        print("4. Lihat spesifikasi sistem")
        print("5. Parameter tuning")
        print("6. Keluar")
        
        choice = input("Pilih opsi (1-6): ").strip()
        
        if choice == "1":
            manage_employee_menu(system)
                
        elif choice == "2":
            system.recognize_faces()
            
        elif choice == "3":
            # Gunakan RTSP CCTV dengan IP default
            rtsp_url = "rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1"  # URL RTSP default
            print(f"[INFO] Menggunakan RTSP stream: {rtsp_url}")
            system.recognize_faces_rtsp()
            
        elif choice == "4":
            show_system_specs()
            
        elif choice == "5":
            # Jalankan parameter tuning script
            try:
                import subprocess
                import sys
                # Jalankan parameter_tuning.py dengan inherit stdout/stdin agar interaktif
                result = subprocess.run([sys.executable, "parameter_tuning.py"])
                if result.returncode != 0:
                    print("[ERROR] Gagal menjalankan parameter tuning")
                else:
                    # Reload system specs after tuning
                    global SYSTEM_SPECS
                    SYSTEM_SPECS = load_system_specs()
                    print("[INFO] Konfigurasi sistem diperbarui")
            except Exception as e:
                print(f"[ERROR] Gagal menjalankan parameter tuning: {e}")
            
        elif choice == "6":
            print("[INFO] Terima kasih telah menggunakan sistem!")
            break
            
        else:
            print("[ERROR] Pilihan tidak valid!")

if __name__ == "__main__":
    main()