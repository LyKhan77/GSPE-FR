import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Date, Boolean, LargeBinary, ForeignKey, Time
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# --- Konfigurasi Dasar --- #
DATABASE_URL = "sqlite:///attendance.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Model Tabel --- #

class Employee(Base):
    """Menyimpan data master karyawan."""
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    department = Column(String)
    position = Column(String)
    phone_number = Column(String) # Untuk notifikasi WhatsApp
    is_active = Column(Boolean, default=True)
    
    # Relasi
    supervisor_id = Column(Integer, ForeignKey('employees.id'))
    supervisor = relationship("Employee", remote_side=[id])
    face_templates = relationship("FaceTemplate", back_populates="employee")
    events = relationship("Event", back_populates="employee")
    attendances = relationship("Attendance", back_populates="employee")

class FaceTemplate(Base):
    """Menyimpan embedding wajah untuk setiap karyawan."""
    __tablename__ = 'face_templates'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    embedding = Column(LargeBinary, nullable=False) # Vektor 512-D dari ArcFace
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relasi
    employee = relationship("Employee", back_populates="face_templates")

class Camera(Base):
    """Menyimpan data master kamera dan lokasinya."""
    __tablename__ = 'cameras'

    id = Column(Integer, primary_key=True, index=True) # Sesuai dengan ID di config
    name = Column(String, nullable=False)
    location_zone = Column(String) # e.g., 'Pintu Masuk', 'Area Produksi'
    rtsp_url = Column(String)

    # Relasi
    events = relationship("Event", back_populates="camera")

class Event(Base):
    """Mencatat setiap kejadian wajah terdeteksi (log 'last seen')."""
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    similarity_score = Column(Float) # Skor kemiripan dari face matching

    # Relasi
    employee = relationship("Employee", back_populates="events")
    camera = relationship("Camera", back_populates="events")

class Attendance(Base):
    """Mencatat absensi harian setiap karyawan."""
    __tablename__ = 'attendances'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    date = Column(Date, nullable=False, default=datetime.date.today)
    first_in = Column(Time) # Jam masuk pertama kali
    last_out = Column(Time) # Jam keluar terakhir kali
    status = Column(String, default='ABSENT') # e.g., 'PRESENT', 'ABSENT', 'LATE'

    # Relasi
    employee = relationship("Employee", back_populates="attendances")

class AlertLog(Base):
    """Mencatat semua notifikasi/peringatan yang dikirim."""
    __tablename__ = 'alert_logs'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey('employees.id'))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    alert_type = Column(String) # e.g., 'OUT_OF_AREA'
    message = Column(String)
    notified_to = Column(String) # e.g., 'Supervisor John Doe'

# --- Fungsi Utilitas --- #

def init_db():
    """Membuat semua tabel di database jika belum ada."""
    Base.metadata.create_all(bind=engine)
    print("Database initialized!")

if __name__ == '__main__':
    # Jalankan file ini secara langsung untuk membuat database dan tabel
    init_db()
