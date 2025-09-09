#!/usr/bin/env python3
# database.py
# Database operations for the face recognition system

import pickle
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base

# Setup database
engine = create_engine("sqlite:///attendance.db")
SessionLocal = sessionmaker(bind=engine)

# Define database models
Base = declarative_base()

class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    embedding = Column(LargeBinary)  # simpan numpy array sebagai binary

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(engine)

def save_employee(name, embedding, face_image):
    """Simpan data karyawan ke database dan gambar ke folder"""
    session = SessionLocal()
    
    # Cek apakah karyawan dengan nama ini sudah ada
    existing_employee = session.query(Employee).filter(Employee.name == name).first()
    if existing_employee:
        # Update data karyawan yang sudah ada
        existing_employee.embedding = pickle.dumps(embedding)
        emp = existing_employee
        print(f"[INFO] Data karyawan {name} berhasil diperbarui")
    else:
        # Simpan embedding ke database dengan ID unik
        data = pickle.dumps(embedding)
        emp = Employee(name=name, embedding=data)
        session.add(emp)
        print(f"[INFO] Karyawan {name} berhasil didaftarkan")
    
    # Pastikan direktori ada sebelum menyimpan gambar
    if not os.path.exists("data/registered_faces"):
        os.makedirs("data/registered_faces")
    
    # Simpan gambar wajah ke folder
    img_path = f"data/registered_faces/{name}.jpg"
    import cv2
    cv2.imwrite(img_path, face_image)
    
    session.commit()
    session.close()
    return emp.id if hasattr(emp, 'id') else None

def get_all_employees():
    """Ambil semua data karyawan dari database"""
    session = SessionLocal()
    employees = session.query(Employee).all()
    data = [(emp.name, pickle.loads(emp.embedding)) for emp in employees]
    session.close()
    return data

def get_all_employees_with_id():
    """Ambil semua data karyawan dari database termasuk ID"""
    session = SessionLocal()
    employees = session.query(Employee).all()
    data = [(emp.id, emp.name, pickle.loads(emp.embedding)) for emp in employees]
    session.close()
    return data

def delete_employee_by_name(name):
    """Hapus karyawan berdasarkan nama"""
    session = SessionLocal()
    
    try:
        # Cari karyawan berdasarkan nama
        employee = session.query(Employee).filter(Employee.name == name).first()
        
        if employee:
            # Hapus karyawan dari database
            session.delete(employee)
            session.commit()
            
            # Hapus file gambar wajah karyawan jika ada
            image_path = f"data/registered_faces/{name}.jpg"
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"[SUCCESS] File gambar '{image_path}' berhasil dihapus")
            
            print(f"[SUCCESS] Karyawan '{name}' berhasil dihapus dari database")
            return True
        else:
            print(f"[ERROR] Karyawan '{name}' tidak ditemukan dalam database")
            return False
            
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Gagal menghapus karyawan: {e}")
        return False
    finally:
        session.close()

def log_attendance(name: str):
    """Catat kehadiran karyawan ke database"""
    session = SessionLocal()
    record = Attendance(employee_name=name)
    session.add(record)
    session.commit()
    session.close()
    print(f"[ATTENDANCE] {name} hadir pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Default system specifications
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