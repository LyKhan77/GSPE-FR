#!/usr/bin/env python3
# db_manager.py
# Database manager for the face recognition system with updated schema for real-time tracking

import pickle
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, LargeBinary, func
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

# Setup database engine and session
engine = create_engine("sqlite:///attendance.db")
SessionLocal = sessionmaker(bind=engine)

# Define base for models
Base = declarative_base()

# Updated database models to match EmployeeTracking.txt requirements

class Employee(Base):
    __tablename__ = "employees"
    employee_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    embedding = Column(LargeBinary)  # simpan numpy array sebagai binary
    department = Column(String(50))
    role = Column(String(50))
    
    # Relationships
    tracking_records = relationship("EmployeeTracking", back_populates="employee")
    locations = relationship("EmployeeLocation", back_populates="employee")

class Camera(Base):
    __tablename__ = "cameras"
    camera_id = Column(String(50), primary_key=True)
    camera_name = Column(String(100), nullable=False)
    location = Column(String(100))
    rtsp_url = Column(Text)  # alamat stream untuk track-location
    status = Column(String(20), default='offline')  # 'online', 'offline', 'error'
    is_active = Column(Boolean, default=False)

class EmployeeTracking(Base):
    __tablename__ = "employee_tracking"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), ForeignKey('employees.employee_id'))
    camera_id = Column(String(50), ForeignKey('cameras.camera_id'))
    last_seen = Column(DateTime, nullable=False)
    status = Column(String(20), default='UNAVAILABLE')  # 'AVAILABLE' or 'UNAVAILABLE'
    
    # Relationships
    employee = relationship("Employee", back_populates="tracking_records")
    camera = relationship("Camera")

class EmployeeLocation(Base):
    __tablename__ = "employee_locations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), ForeignKey('employees.employee_id'))
    camera_id = Column(String(50), ForeignKey('cameras.camera_id'))
    timestamp = Column(DateTime, default=datetime.now)
    
    # Relationships
    employee = relationship("Employee", back_populates="locations")
    camera = relationship("Camera")

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class DatabaseManager:
    """Manager for all database operations with real-time tracking support"""
    
    @staticmethod
    def get_all_employees():
        """Get all employees from database"""
        try:
            session = SessionLocal()
            employees = session.query(Employee).all()
            # Return in the same format as before for compatibility
            data = [(emp.name, pickle.loads(emp.embedding)) for emp in employees if emp.embedding]
            session.close()
            return data
        except Exception as e:
            print("[DB MANAGER] Error getting employees: {}".format(e))
            return []
    
    @staticmethod
    def get_all_employees_with_id():
        """Get all employees from database with ID"""
        try:
            session = SessionLocal()
            employees = session.query(Employee).all()
            # Return tuples of (employee_id, employee_name, embedding)
            data = [(emp.employee_id, emp.name, pickle.loads(emp.embedding)) for emp in employees if emp.embedding]
            session.close()
            return data
        except Exception as e:
            print("[DB MANAGER] Error getting employees with ID: {}".format(e))
            return []
    
    @staticmethod
    def get_all_employees_detailed():
        """Get all employees with full details for tracking panel"""
        try:
            session = SessionLocal()
            employees = session.query(Employee).all()
            session.close()
            return employees
        except Exception as e:
            print("[DB MANAGER] Error getting detailed employees: {}".format(e))
            return []
    
    @staticmethod
    def get_all_cameras():
        """Get all cameras from database"""
        try:
            session = SessionLocal()
            cameras = session.query(Camera).all()
            session.close()
            return cameras
        except Exception as e:
            print("[DB MANAGER] Error getting cameras: {}".format(e))
            return []
    
    @staticmethod
    def log_attendance(employee_name):
        """Log employee attendance to database"""
        try:
            session = SessionLocal()
            record = Attendance(employee_name=employee_name)
            session.add(record)
            session.commit()
            session.close()
            print("[ATTENDANCE] {} hadir pada {}".format(employee_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        except Exception as e:
            print("[DB MANAGER] Error logging attendance: {}".format(e))
    
    @staticmethod
    def update_employee_tracking(employee_id, camera_id, status='AVAILABLE'):
        """Update employee tracking in database using UPSERT logic"""
        try:
            session = SessionLocal()
            
            # Use UPSERT logic: INSERT ... ON CONFLICT DO UPDATE
            # First, try to find existing record for this employee
            existing_record = session.query(EmployeeTracking).filter(
                EmployeeTracking.employee_id == employee_id
            ).order_by(EmployeeTracking.last_seen.desc()).first()
            
            if existing_record:
                # Update existing record
                existing_record.camera_id = camera_id
                existing_record.last_seen = datetime.now()
                existing_record.status = status
                print("[TRACKING] Updated existing tracking for employee {} at camera {} with status {}".format(employee_id, camera_id, status))
            else:
                # Create new tracking record
                tracking_record = EmployeeTracking(
                    employee_id=employee_id,
                    camera_id=camera_id,
                    last_seen=datetime.now(),
                    status=status
                )
                session.add(tracking_record)
                print("[TRACKING] Created new tracking for employee {} at camera {} with status {}".format(employee_id, camera_id, status))
            
            session.commit()
            session.close()
        except Exception as e:
            print("[DB MANAGER] Error updating employee tracking: {}".format(e))
    
    @staticmethod
    def mark_employee_unavailable(employee_id):
        """Mark employee as unavailable in tracking using UPSERT logic"""
        try:
            session = SessionLocal()
            
            # Find the latest tracking record for the employee
            existing_record = session.query(EmployeeTracking).filter(
                EmployeeTracking.employee_id == employee_id
            ).order_by(EmployeeTracking.last_seen.desc()).first()
            
            if existing_record:
                # Update existing record to UNAVAILABLE
                existing_record.status = 'UNAVAILABLE'
                existing_record.last_seen = datetime.now()
                print("[TRACKING] Updated employee {} to UNAVAILABLE".format(employee_id))
            else:
                # Create new tracking record with UNAVAILABLE status
                tracking_record = EmployeeTracking(
                    employee_id=employee_id,
                    camera_id='default',  # Placeholder
                    last_seen=datetime.now(),
                    status='UNAVAILABLE'
                )
                session.add(tracking_record)
                print("[TRACKING] Created new UNAVAILABLE record for employee {}".format(employee_id))
            
            session.commit()
            session.close()
        except Exception as e:
            print("[DB MANAGER] Error marking employee as unavailable: {}".format(e))
    
    @staticmethod
    def check_and_update_unavailable_employees(timeout_minutes=10):
        """Check for employees not seen for >timeout_minutes and mark as UNAVAILABLE"""
        try:
            session = SessionLocal()
            
            # Calculate timeout threshold
            timeout_threshold = datetime.now() - timedelta(minutes=timeout_minutes)
            
            # Get latest tracking record for each employee
            subquery = session.query(
                EmployeeTracking.employee_id,
                func.max(EmployeeTracking.last_seen).label('max_last_seen')
            ).group_by(EmployeeTracking.employee_id).subquery()
            
            latest_tracking = session.query(EmployeeTracking).join(
                subquery,
                (EmployeeTracking.employee_id == subquery.c.employee_id) &
                (EmployeeTracking.last_seen == subquery.c.max_last_seen)
            ).all()
            
            updated_count = 0
            for record in latest_tracking:
                # Check if employee hasn't been seen for more than timeout_minutes
                if (record.last_seen < timeout_threshold and 
                    record.status == 'AVAILABLE'):
                    
                    # Update to UNAVAILABLE
                    record.status = 'UNAVAILABLE'
                    record.last_seen = datetime.now()
                    updated_count += 1
                    print("[TIMEOUT] Employee {} marked UNAVAILABLE (not seen for >{} minutes)".format(
                        record.employee_id, timeout_minutes))
            
            if updated_count > 0:
                session.commit()
                print("[TIMEOUT] Updated {} employees to UNAVAILABLE due to timeout".format(updated_count))
            
            session.close()
            return updated_count
        except Exception as e:
            print("[DB MANAGER] Error checking unavailable employees: {}".format(e))
            return 0
    
    @staticmethod
    def get_employee_statuses():
        """Get all employee statuses from database for real-time tracking"""
        try:
            session = SessionLocal()
            
            # Get the latest tracking record for each employee
            # This is a simplified version for SQLite
            subquery = session.query(
                EmployeeTracking.employee_id,
                func.max(EmployeeTracking.last_seen).label('max_last_seen')
            ).group_by(EmployeeTracking.employee_id).subquery()
            
            latest_tracking = session.query(EmployeeTracking).join(
                subquery,
                (EmployeeTracking.employee_id == subquery.c.employee_id) &
                (EmployeeTracking.last_seen == subquery.c.max_last_seen)
            ).all()
            
            # Get employee names
            employees = {emp.employee_id: emp for emp in session.query(Employee).all()}
            cameras = {cam.camera_id: cam for cam in session.query(Camera).all()}
            
            statuses = []
            for record in latest_tracking:
                employee = employees.get(record.employee_id)
                camera = cameras.get(record.camera_id)
                
                if employee:
                    statuses.append({
                        'employee_id': record.employee_id,
                        'employee_name': employee.name,
                        'status': record.status,
                        'last_seen': record.last_seen,
                        'camera_id': record.camera_id,
                        'camera_name': camera.camera_name if camera else record.camera_id
                    })
            
            session.close()
            return statuses
        except Exception as e:
            print("[DB MANAGER] Error getting employee statuses: {}".format(e))
            return []
    
    @staticmethod
    def log_employee_location(employee_id, camera_id):
        """Log employee location to database"""
        try:
            session = SessionLocal()
            location = EmployeeLocation(
                employee_id=employee_id,
                camera_id=camera_id
            )
            session.add(location)
            session.commit()
            session.close()
        except Exception as e:
            print("[DB MANAGER] Error logging employee location: {}".format(e))

# Global instance
db_manager = DatabaseManager()