#!/usr/bin/env python3
# tracking_manager.py
# Tracking manager for the face recognition system

import time
import numpy as np
from datetime import datetime
from db_manager import db_manager, SessionLocal, Camera

class NotificationManager:
    """Simple notification manager for employee absence alerts"""
    
    def __init__(self):
        self.notifications = []
        self.alert_threshold = 120  # 2 minutes in seconds
    
    def check_and_notify_absences(self, employee_statuses):
        """Check for employee absences and generate notifications"""
        current_time = time.time()
        new_notifications = []
        
        for employee_name, status_data in employee_statuses.items():
            if status_data['status'] == 'present':
                continue
                
            time_since_seen = current_time - status_data['last_seen']
            
            # Generate alert if employee absent > threshold
            if time_since_seen > self.alert_threshold:
                minutes_ago = int(time_since_seen // 60)
                notification = {
                    'id': f"{employee_name}_{int(current_time)}",
                    'employee': employee_name,
                    'message': f"{employee_name} tidak terdeteksi selama {minutes_ago} menit",
                    'timestamp': current_time,
                    'type': 'absence_alert',
                    'camera': status_data.get('camera', 'Unknown')
                }
                new_notifications.append(notification)
        
        # Add new notifications
        self.notifications.extend(new_notifications)
        
        # Keep only recent notifications (last 50)
        self.notifications = self.notifications[-50:]
        
        return new_notifications
    
    def get_notifications(self, limit=10):
        """Get recent notifications"""
        return self.notifications[-limit:] if len(self.notifications) > limit else self.notifications
    
    def clear_notifications(self):
        """Clear all notifications"""
        self.notifications = []

# Global notification manager
notification_manager = NotificationManager()

class TrackingManager:
    """Manager for employee tracking and monitoring"""
    
    def __init__(self, absence_threshold=300):  # 5 minutes default
        self.employee_last_seen = {}  # {employee_name: timestamp}
        self.employee_durations = {}  # {employee_name: {'entry_time': timestamp, 'total_duration': seconds}}
        self.employee_status = {}  # {employee_name: {'last_seen': timestamp, 'camera': camera_id, 'status': 'present/absent'}}
        self.alerts = []  # List of alert messages
        self.absence_threshold = absence_threshold
        self.tracked_faces = {}  # {track_id: {'bbox', 'last_seen', 'name', 'embedding', 'confidence_history'}}
        self.next_track_id = 1
        self.bbox_history = {}
    
    def update_employee_status(self, employee_name, camera_id):
        """Update employee status when detected"""
        current_time = time.time()
        self.employee_status[employee_name] = {
            'last_seen': current_time,
            'camera': camera_id,
            'status': 'present'
        }
        
        # Update last seen timestamp
        self.employee_last_seen[employee_name] = current_time
        
        # If this is the first time employee is detected, record entry time
        if employee_name not in self.employee_durations:
            self.employee_durations[employee_name] = {
                'entry_time': current_time,
                'total_duration': 0
            }

    def resolve_camera_name(self, camera_id):
        """Resolve camera_name from camera_id using DB. Returns camera_name or camera_id if not found."""
        try:
            session = SessionLocal()
            cam = session.query(Camera).filter_by(camera_id=camera_id).first()
            session.close()
            return cam.camera_name if cam else camera_id
        except Exception:
            # Fail-safe: return camera_id when DB unavailable
            return camera_id

    def update_employee_status_db(self, employee_id, camera_id, status='AVAILABLE'):
        """Persist tracking and location to DB using db_manager."""
        try:
            db_manager.update_employee_tracking(employee_id, camera_id, status)
            db_manager.log_employee_location(employee_id, camera_id)
        except Exception as e:
            print(f"[TRACKING] DB update failed for {employee_id} at {camera_id}: {e}")

    def track_employee_presence(self, employee_id, employee_name, camera_id):
        """Unified helper to update in-memory status and DB, returning a rich presence dict.

        Args:
            employee_id (str): Unique employee ID from DB
            employee_name (str): Human-readable employee name
            camera_id (str): Camera identifier as stored in DB

        Returns:
            dict: {
                'employee_id', 'employee_name', 'status', 'last_seen',
                'camera_id', 'camera_name'
            }
        """
        # Update in-memory by name (kept for backward compatibility in UI logs)
        self.update_employee_status(employee_name, camera_id)

        # Persist to DB
        self.update_employee_status_db(employee_id, camera_id, status='AVAILABLE')

        # Resolve camera_name for downstream consumers
        cam_name = self.resolve_camera_name(camera_id)

        return {
            'employee_id': employee_id,
            'employee_name': employee_name,
            'status': 'AVAILABLE',
            'last_seen': datetime.fromtimestamp(self.employee_last_seen.get(employee_name, time.time())),
            'camera_id': camera_id,
            'camera_name': cam_name,
        }
    
    def check_absences(self):
        """Check for employee absences and generate alerts"""
        current_time = time.time()
        absent_employees = []
        
        # First, check database for employees not seen for >10 minutes and mark as UNAVAILABLE
        from db_manager import db_manager
        timeout_count = db_manager.check_and_update_unavailable_employees(timeout_minutes=10)
        
        for employee_name, status_info in self.employee_status.items():
            last_seen_time = status_info['last_seen']
            time_since_last_seen = current_time - last_seen_time
            
            # Check if employee has been absent for more than threshold
            if time_since_last_seen > self.absence_threshold:
                # Check if employee was previously marked as present
                if (employee_name in self.employee_status and 
                    self.employee_status[employee_name]['status'] == 'present'):
                    # Mark employee as absent
                    self.employee_status[employee_name]['status'] = 'absent'
                    absent_employees.append(employee_name)
                    alert_msg = f"ALERT: {employee_name} tidak terdeteksi selama {time_since_last_seen:.0f} detik"
                    self.alerts.append({
                        'timestamp': current_time,
                        'employee': employee_name,
                        'message': alert_msg
                    })
                    print(f"[TRACKING] {alert_msg}")
                    
                    # Also mark as UNAVAILABLE in database
                    db_manager.mark_employee_unavailable(employee_name)
                    
                    # Generate notification via notification_manager
                    notification_manager.check_and_notify_absences({
                        employee_name: {
                            'status': 'absent',
                            'last_seen': last_seen_time,
                            'camera': self.employee_status[employee_name].get('camera', 'Unknown')
                        }
                    })
        
        return absent_employees
    
    def get_employee_report(self, employee_name):
        """Get comprehensive report for an employee"""
        if employee_name not in self.employee_status:
            return None
            
        status = self.employee_status[employee_name]
        duration_info = self.employee_durations.get(employee_name, {})
        
        # Calculate current duration
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
        """Format duration in readable form"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def get_all_employees_status(self):
        """Get status of all employees"""
        return self.employee_status
    
    def get_recent_alerts(self, limit=10):
        """Get recent alerts"""
        return self.alerts[-limit:] if len(self.alerts) > limit else self.alerts
    
    def calculate_embedding_similarity(self, embedding1, embedding2):
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
            print(f"[TRACKING] Error calculating similarity: {e}")
            return 0.0
    
    def update_tracked_face(self, track_id, bbox, center, face_obj, embedding):
        """Update tracked face information"""
        current_time = time.time()
        self.tracked_faces[track_id] = {
            'bbox': bbox,
            'center': center,
            'last_seen': current_time,
            'face_obj': face_obj,
            'embedding': embedding,
            'name': self.tracked_faces.get(track_id, {}).get('name', "Unknown")
        }
    
    def create_new_track(self, bbox, center, face_obj, embedding):
        """Create new face track"""
        track_id = self.next_track_id
        self.next_track_id += 1
        current_time = time.time()
        
        self.tracked_faces[track_id] = {
            'bbox': bbox,
            'center': center,
            'last_seen': current_time,
            'face_obj': face_obj,
            'embedding': embedding,
            'name': "Unknown"
        }
        
        return track_id
    
    def cleanup_old_tracks(self, timeout=3.0):
        """Clean up old face tracks"""
        current_time = time.time()
        self.tracked_faces = {
            k: v for k, v in self.tracked_faces.items() 
            if current_time - v['last_seen'] < timeout
        }
    
    def match_faces(self, current_faces, max_distance_threshold=150, embedding_similarity_threshold=0.7):
        """Match current faces with tracked faces"""
        matched_tracks = set()
        matched_faces = set()
        updated_tracks = {}
        
        # Match tracked faces with current faces
        for track_id, track_data in list(self.tracked_faces.items()):
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
                    similarity = self.calculate_embedding_similarity(track_data['embedding'], face_data['embedding'])
                    
                    # Use combination of distance and similarity for matching
                    if similarity > embedding_similarity_threshold:
                        if distance < min_distance:
                            min_distance = distance
                            best_match = face_data
                            best_similarity = similarity
                    elif distance < min_distance and distance < max_distance_threshold:
                        min_distance = distance
                        best_match = face_data
                        best_similarity = similarity
                
                # If we found a good match
                if best_match and (best_similarity > embedding_similarity_threshold or min_distance < max_distance_threshold):
                    matched_tracks.add(track_id)
                    matched_faces.add(best_match['index'])
                    
                    # Update tracked face with new data
                    updated_tracks[track_id] = {
                        'bbox': best_match['bbox'],
                        'center': best_match['center'],
                        'last_seen': time.time(),
                        'face_obj': best_match['face_obj'],
                        'embedding': best_match['embedding'],
                        'name': track_data.get('name', "Unknown")
                    }
        
        # Add new faces that weren't matched
        for face_data in current_faces:
            if face_data['index'] not in matched_faces:
                track_id = self.create_new_track(
                    face_data['bbox'],
                    face_data['center'],
                    face_data['face_obj'],
                    face_data['embedding']
                )
                updated_tracks[track_id] = self.tracked_faces[track_id]
        
        self.tracked_faces = updated_tracks
        return matched_tracks, matched_faces

# Global instance
tracking_manager = TrackingManager()