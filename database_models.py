from __future__ import annotations

import datetime
import json
import os
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Date,
    Boolean,
    LargeBinary,
    ForeignKey,
    Time,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from sqlalchemy import event

# --- Konfigurasi Dasar --- #
# SQLite file will be created under db/attendance.db
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, 'db')
os.makedirs(DB_DIR, exist_ok=True)
DB_FILE = os.path.join(DB_DIR, 'attendance.db')
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Ensure SQLite enforces foreign keys when using ondelete='CASCADE'
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


# --- Model Tabel --- #

class Employee(Base):
    """Data master karyawan dan struktur atasan-bawahan sederhana."""

    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    department = Column(String)
    position = Column(String)
    phone_number = Column(String)  # untuk notifikasi WhatsApp
    is_active = Column(Boolean, default=True, nullable=False)

    # Relasi org (atasan)
    supervisor_id = Column(Integer, ForeignKey('employees.id'))
    supervisor = relationship("Employee", remote_side=[id])

    # Backrefs
    face_templates = relationship("FaceTemplate", back_populates="employee", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="employee", cascade="all, delete-orphan")
    attendances = relationship("Attendance", back_populates="employee", cascade="all, delete-orphan")
    presence = relationship("Presence", back_populates="employee", uselist=False, cascade="all, delete-orphan")
    alert_logs = relationship("AlertLog", back_populates="employee", cascade="all, delete-orphan")


class FaceTemplate(Base):
    """Embedding wajah (mis. 512-D ArcFace) untuk identifikasi."""

    __tablename__ = 'face_templates'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id', ondelete='CASCADE'), nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)  # simpan bytes dari np.ndarray.tobytes()
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    # metadata untuk multi-angle dan kualitas
    pose_label = Column(String(16))  # contoh: 'front', 'left', 'right'
    quality_score = Column(Float)    # skor kualitas [0..1] (berdasarkan blur/brightness/size)

    employee = relationship("Employee", back_populates="face_templates")


class Camera(Base):
    """Master kamera (diselaraskan dengan folder `camera_configs/`)."""

    __tablename__ = 'cameras'

    id = Column(Integer, primary_key=True, index=True)  # selaras dengan id di config.json
    name = Column(String, nullable=False)
    location_zone = Column(String)  # contoh: 'Entrance Zone', 'Production Area'
    rtsp_url = Column(String)  # disimpan di DB (tidak diekspos ke frontend)

    # Backrefs
    events = relationship("Event", back_populates="camera")


class Event(Base):
    """Log setiap deteksi/kemunculan wajah (mendukung unknown)."""

    __tablename__ = 'events'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id', ondelete='CASCADE'), nullable=True, index=True)  # None untuk unknown
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    similarity_score = Column(Float)  # skor kemiripan jika recognized
    track_id = Column(String)  # opsional: id tracking dari tracker

    # Backrefs
    employee = relationship("Employee", back_populates="events")
    camera = relationship("Camera", back_populates="events")

    __table_args__ = (
        Index('ix_events_emp_ts', 'employee_id', 'timestamp'),
    )


class Presence(Base):
    """Status kehadiran real-time (available/off) per karyawan.
    Diperbarui oleh Presence Tracker berdasarkan Event terakhir.
    """

    __tablename__ = 'presence'

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id', ondelete='CASCADE'), nullable=False, unique=True, index=True)
    status = Column(String, default='off', nullable=False)  # 'available' atau 'off'
    last_seen_ts = Column(DateTime)  # cap waktu terakhir terlihat
    last_camera_id = Column(Integer, ForeignKey('cameras.id'))

    employee = relationship("Employee", back_populates="presence")


class Attendance(Base):
    """Rekap absensi harian (first_in_ts & last_out_ts)."""

    __tablename__ = 'attendances'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id', ondelete='CASCADE'), nullable=False, index=True)
    date = Column(Date, nullable=False, default=datetime.date.today)
    first_in_ts = Column(DateTime)  # kapan pertama terlihat hari itu
    last_out_ts = Column(DateTime)  # kapan terakhir tidak terlihat (keluar area)
    status = Column(String, default='ABSENT')  # PRESENT, ABSENT, LATE, dll

    employee = relationship("Employee", back_populates="attendances")

    __table_args__ = (
        UniqueConstraint('employee_id', 'date', name='uq_attendance_emp_date'),
        Index('ix_attendance_emp_date', 'employee_id', 'date'),
    )


class AlertLog(Base):
    """Log notifikasi/peringatan yang dikirim (WhatsApp, dll)."""

    __tablename__ = 'alert_logs'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id', ondelete='CASCADE'))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    alert_type = Column(String)  # contoh: OUT_OF_AREA
    message = Column(String)
    notified_to = Column(String)  # contoh: Supervisor John Doe / phone number

    employee = relationship("Employee", back_populates="alert_logs")


# --- Utilitas DB --- #

def init_db() -> None:
    """Buat semua tabel jika belum ada."""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized ({DB_FILE})")


def get_session() -> Session:
    return SessionLocal()


def seed_cameras_from_configs(camera_dir: str = 'camera_configs') -> int:
    """Membaca folder `camera_configs/` dan sinkronkan tabel Camera.

    Meng-upsert (insert or update) berdasarkan id kamera yang ada di config.json.
    Return jumlah kamera yang diproses.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base_dir, camera_dir)
    if not os.path.isdir(root):
        print(f"Camera config directory not found: {root}")
        return 0

    processed = 0
    with get_session() as db:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            cfg_path = os.path.join(path, 'config.json')
            if not os.path.isfile(cfg_path):
                continue
            try:
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                cam_id = int(cfg.get('id'))
                cam_name = cfg.get('name') or f"CAM {cam_id}"
                rtsp_url = cfg.get('rtsp_url', '')
                # optional location in config
                location_zone = cfg.get('location') or cfg.get('zone')

                cam = db.get(Camera, cam_id)
                if cam is None:
                    cam = Camera(id=cam_id, name=cam_name, location_zone=location_zone, rtsp_url=rtsp_url)
                    db.add(cam)
                else:
                    cam.name = cam_name
                    cam.location_zone = location_zone
                    cam.rtsp_url = rtsp_url
                processed += 1
            except Exception as e:
                print(f"Failed to process {cfg_path}: {e}")
        db.commit()
    print(f"Seeded/updated {processed} camera(s) from {root}")
    return processed


if __name__ == '__main__':
    # Jalankan modul ini untuk inisialisasi DB dan seed kamera
    init_db()
    seed_cameras_from_configs()
