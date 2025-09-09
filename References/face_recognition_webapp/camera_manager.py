#!/usr/bin/env python3
# camera_manager.py
# Camera manager for the face recognition system with unified Camera model

import cv2
import threading
import time
import os
import json
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Define Camera model to match db_manager.py
# We'll define it here but won't create the table since it should already exist
Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"
    
    camera_id = Column(String(50), primary_key=True)
    camera_name = Column(String(100), nullable=False)
    location = Column(String(100))
    rtsp_url = Column(Text)  # alamat stream untuk track-location
    status = Column(String(20), default='offline')  # 'online', 'offline', 'error'
    is_active = Column(Boolean, default=False)

# Setup database engine and session (reusing the same database)
engine = create_engine("sqlite:///attendance.db")
SessionLocal = sessionmaker(bind=engine)

# Camera configurations directory
CAMERA_CONFIGS_DIR = "camera_configs"
FIELD_FOLDER = "field"

def get_camera_configs_from_folders():
    """Scan camera_configs directory and return camera configurations"""
    configs = []
    
    # Ensure the directory exists
    Path(CAMERA_CONFIGS_DIR).mkdir(exist_ok=True)
    
    # Scan all subdirectories in camera_configs
    try:
        for item in Path(CAMERA_CONFIGS_DIR).iterdir():
            if item.is_dir():
                config = load_camera_config_from_folder(item)
                if config:
                    configs.append(config)
    except Exception as e:
        print(f"[ERROR] Error scanning camera configs directory: {e}")
    
    # Also scan field folder structure like Pertamina system
    try:
        if os.path.exists(FIELD_FOLDER):
            for field_item in Path(FIELD_FOLDER).iterdir():
                if field_item.is_dir() and (field_item / 'place').exists():
                    place_folder = field_item / 'place'
                    for place_item in place_folder.iterdir():
                        if place_item.is_dir():
                            json_path = place_item / 'cam_info.json'
                            if json_path.exists():
                                cameras = load_cameras_from_json(json_path, field_item.name, place_item.name)
                                configs.extend(cameras)
    except Exception as e:
        print(f"[ERROR] Error scanning field folder structure: {e}")
    
    return configs

def load_camera_config_from_folder(folder_path):
    """Load camera configuration from a folder"""
    try:
        folder_name = folder_path.name
        
        # Look for config.json in the folder
        config_file = folder_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:
            # Create default config if config.json doesn't exist
            config_data = {
                "name": folder_name,
                "rtsp_url": f"rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1",
                "status": "offline",
                "is_active": False
            }
            # Save default config
            save_camera_config_to_folder(folder_path, config_data)
        
        # Add folder-based ID
        config_data['id'] = folder_name.lower().replace(' ', '_')
        
        return config_data
    except Exception as e:
        print(f"[ERROR] Error loading config from folder {folder_path}: {e}")
        return None

def load_cameras_from_json(json_path, field_name, place_name):
    """Load cameras from JSON file in field/place structure"""
    try:
        with open(json_path, 'r') as f:
            cameras_data = json.load(f)
        
        configs = []
        for camera_data in cameras_data:
            # Create config with hierarchical ID
            config = {
                'id': f"{field_name.lower().replace(' ', '_')}_{place_name.lower().replace(' ', '_')}_{camera_data['id']}",
                'name': str(camera_data['name']),  # Ensure it's a string
                'rtsp_url': str(camera_data['rtsp']),  # Ensure it's a string
                'status': 'offline',
                'is_active': False
            }
            configs.append(config)
        
        return configs
    except Exception as e:
        print(f"[ERROR] Error loading cameras from {json_path}: {e}")
        return []

def save_camera_config_to_folder(folder_path, config_data):
    """Save camera configuration to a folder"""
    try:
        config_file = folder_path / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Error saving config to folder {folder_path}: {e}")
        return False

def create_camera_folder(folder_name, rtsp_url=None):
    """Create a new camera configuration folder"""
    try:
        folder_path = Path(CAMERA_CONFIGS_DIR) / folder_name
        folder_path.mkdir(exist_ok=True)
        
        # Create default config
        config_data = {
            "name": folder_name,
            "rtsp_url": rtsp_url or f"rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1",
            "status": "offline",
            "is_active": False
        }
        
        save_camera_config_to_folder(folder_path, config_data)
        return True
    except Exception as e:
        print(f"[ERROR] Error creating camera folder {folder_name}: {e}")
        return False

def sync_camera_configs_with_database():
    """Sync folder-based camera configs with database"""
    try:
        # Get camera configs from folders
        folder_configs = get_camera_configs_from_folders()
        
        # Get database session
        session = SessionLocal()
        
        # Get existing cameras from database
        db_cameras = session.query(Camera).all()
        db_camera_ids = {cam.camera_id: cam for cam in db_cameras}
        
        # Determine which camera should be active (first one alphabetically)
        active_camera_id = None
        if folder_configs:
            # Sort by ID and make the first one active
            sorted_configs = sorted(folder_configs, key=lambda x: x['id'])
            active_camera_id = sorted_configs[0]['id']
        
        # Sync folder configs to database
        for config in folder_configs:
            camera_id = config['id']
            is_active = (camera_id == active_camera_id) if active_camera_id else False
            
            if camera_id in db_camera_ids:
                # Update existing camera
                db_camera = db_camera_ids[camera_id]
                db_camera.camera_name = config['name']
                db_camera.rtsp_url = config.get('rtspUrl', config.get('rtsp_url', ''))
                db_camera.status = config['status']
                db_camera.is_active = is_active
            else:
                # Create new camera
                new_camera = Camera(
                    camera_id=camera_id,
                    camera_name=config['name'],
                    rtsp_url=config.get('rtspUrl', config.get('rtsp_url', '')),
                    status=config['status'],
                    is_active=is_active
                )
                session.add(new_camera)
        
        # Remove cameras from database that no longer have folders
        folder_config_ids = {config['id'] for config in folder_configs}
        for camera_id, db_camera in db_camera_ids.items():
            if camera_id not in folder_config_ids:
                session.delete(db_camera)
        
        # Commit changes
        session.commit()
        session.close()
        
        return True
    except Exception as e:
        print(f"[ERROR] Error syncing camera configs with database: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_all_camera_configs():
    """Get all camera configurations (from folders and database)"""
    # Sync first
    sync_camera_configs_with_database()
    
    # Get from database
    session = SessionLocal()
    cameras = session.query(Camera).all()
    camera_list = []
    
    for cam in cameras:
        camera_info = {
            'id': cam.camera_id,
            'name': cam.camera_name,
            'rtspUrl': cam.rtsp_url,
            'status': cam.status,
            'isActive': cam.is_active
        }
        camera_list.append(camera_info)
    
    session.close()
    
    # If no cameras found, return empty list (no defaults)
    return camera_list

# Initialize with default cameras if directory is empty
def initialize_default_cameras():
    """Initialize default camera configurations"""
    configs_dir = Path(CAMERA_CONFIGS_DIR)
    if not any(configs_dir.iterdir()):
        # Create default camera folders
        create_camera_folder("CAM1", "rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1")
        create_camera_folder("CAM2", "rtsp://admin:gspe-intercon@192.168.0.64:554/Channels/Stream1")
        create_camera_folder("CAM3", "0")  # Webcam

# Run initialization
initialize_default_cameras()

class CameraManager:
    """Manager for all camera operations"""
    
    def __init__(self):
        self.active_streams = {}
        self.stream_lock = threading.Lock()
    
    def initialize_camera_stream(self, camera_config):
        """Initialize camera stream based on configuration"""
        try:
            camera_id = camera_config['id']
            rtsp_url = camera_config.get('rtspUrl') or camera_config.get('rtsp_url')
            
            # Handle different camera types
            if rtsp_url and rtsp_url != "0":
                # RTSP camera
                cap = cv2.VideoCapture(rtsp_url)
            else:
                # Webcam (default camera)
                cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                with self.stream_lock:
                    self.active_streams[camera_id] = {
                        'capture': cap,
                        'config': camera_config,
                        'last_frame_time': time.time(),
                        'status': 'online'
                    }
                print(f"[CAMERA MANAGER] Camera {camera_id} initialized successfully")
                return True
            else:
                print(f"[CAMERA MANAGER] Failed to open camera {camera_id}")
                return False
                
        except Exception as e:
            print(f"[CAMERA MANAGER] Error initializing camera {camera_id}: {e}")
            return False
    
    def get_frame(self, camera_id):
        """Get frame from camera stream"""
        try:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    stream = self.active_streams[camera_id]
                    cap = stream['capture']
                    
                    ret, frame = cap.read()
                    if ret:
                        stream['last_frame_time'] = time.time()
                        stream['status'] = 'online'
                        return frame
                    else:
                        stream['status'] = 'error'
                        return None
            return None
        except Exception as e:
            print(f"[CAMERA MANAGER] Error getting frame from camera {camera_id}: {e}")
            return None
    
    def release_camera(self, camera_id):
        """Release camera stream"""
        try:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    stream = self.active_streams[camera_id]
                    if stream['capture'].isOpened():
                        stream['capture'].release()
                    del self.active_streams[camera_id]
                    print(f"[CAMERA MANAGER] Camera {camera_id} released")
        except Exception as e:
            print(f"[CAMERA MANAGER] Error releasing camera {camera_id}: {e}")
    
    def get_active_cameras(self):
        """Get list of active cameras"""
        try:
            with self.stream_lock:
                return list(self.active_streams.keys())
        except Exception as e:
            print(f"[CAMERA MANAGER] Error getting active cameras: {e}")
            return []
    
    def get_camera_status(self, camera_id):
        """Get camera status"""
        try:
            with self.stream_lock:
                if camera_id in self.active_streams:
                    return self.active_streams[camera_id]['status']
                return 'offline'
        except Exception as e:
            print(f"[CAMERA MANAGER] Error getting camera status: {e}")
            return 'error'
    
    def initialize_all_cameras(self):
        """Initialize all configured cameras"""
        try:
            camera_configs = get_all_camera_configs()
            initialized_cameras = []
            
            for config in camera_configs:
                if config.get('isActive', False):
                    if self.initialize_camera_stream(config):
                        initialized_cameras.append(config['id'])
            
            print(f"[CAMERA MANAGER] Initialized {len(initialized_cameras)} cameras: {initialized_cameras}")
            return initialized_cameras
        except Exception as e:
            print(f"[CAMERA MANAGER] Error initializing all cameras: {e}")
            return []
    
    def get_all_camera_configs(self):
        """Get all camera configurations (from folders and database)"""
        return get_all_camera_configs()

# Global instance
camera_manager = CameraManager()