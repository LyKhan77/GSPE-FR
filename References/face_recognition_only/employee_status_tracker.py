#!/usr/bin/env python3
# employee_status_tracker.py
# Modul untuk tracking status employee (present/absent) berdasarkan deteksi wajah

import time
from datetime import datetime
from employee_monitoring import EmployeeMonitor
from db_utils import get_all_employees

class EmployeeStatusTracker:
    """Kelas untuk tracking status employee berdasarkan deteksi wajah"""
    
    def __init__(self, absence_threshold_seconds=30):
        """
        Inisialisasi EmployeeStatusTracker
        
        Args:
            absence_threshold_seconds (int): Threshold waktu dalam detik untuk menentukan 
                                           karyawan absent (default: 30 detik)
        """
        self.monitor = EmployeeMonitor()
        self.absence_threshold = absence_threshold_seconds
        self.employee_status = {}  # {employee_name: {'last_seen': timestamp, 'status': 'present'/'absent', 'camera': camera_id}}
        self.status_change_callbacks = []  # List of callback functions for status changes
        self.employee_cameras = {}  # {employee_name: [list of cameras where detected]}
        
        # Inisialisasi dengan daftar karyawan dari database
        self._initialize_employees_from_db()
        
    def _initialize_employees_from_db(self):
        """Inisialisasi status karyawan berdasarkan data di database"""
        try:
            employees = get_all_employees()
            for name, _ in employees:
                # Inisialisasi semua karyawan dengan status 'absent' sampai terdeteksi
                self.employee_status[name] = {
                    'last_seen': 0,  # Belum pernah terdeteksi
                    'status': 'off',  # off = tidak terdeteksi
                    'camera': None,
                    'cameras': []  # List of all cameras where this employee was detected
                }
            print(f"[INFO] EmployeeStatusTracker initialized with {len(employees)} employees from database")
        except Exception as e:
            print(f"[ERROR] Failed to initialize employees from database: {e}")
    
    def update_employee_status(self, employee_name, camera_id=None):
        """
        Update status karyawan ketika terdeteksi
        
        Args:
            employee_name (str): Nama karyawan
            camera_id (str): ID kamera tempat karyawan terdeteksi
        """
        current_time = time.time()
        
        # Log aktivitas karyawan
        self.monitor.log_employee_activity(
            employee_name, 
            "detected", 
            camera_id=camera_id,
            details={'status': 'available'}
        )
        
        # Update status karyawan di database
        if DATABASE_UPDATE_AVAILABLE:
            try:
                update_employee_status(employee_name, camera_id, 'available')
            except Exception as e:
                print(f"[ERROR] Failed to update employee status in database: {e}")
        
        # Update status karyawan
        if employee_name not in self.employee_status:
            # Karyawan baru yang mungkin baru saja diregistrasi
            self.employee_status[employee_name] = {
                'last_seen': current_time,
                'status': 'available',  # available = terdeteksi
                'camera': camera_id,
                'cameras': [camera_id] if camera_id else []
            }
            # Panggil callback untuk status change
            self._notify_status_change(employee_name, 'available', camera_id)
            print(f"[TRACKING] {employee_name} is now available at {camera_id or 'Unknown Camera'}")
        else:
            # Karyawan sudah ada dalam tracking
            previous_status = self.employee_status[employee_name]['status']
            self.employee_status[employee_name]['last_seen'] = current_time
            self.employee_status[employee_name]['camera'] = camera_id
            
            # Tambahkan camera ke daftar jika belum ada
            if camera_id and camera_id not in self.employee_status[employee_name]['cameras']:
                self.employee_status[employee_name]['cameras'].append(camera_id)
            
            # Jika sebelumnya off, ubah ke available
            if previous_status == 'off':
                self.employee_status[employee_name]['status'] = 'available'
                # Panggil callback untuk status change
                self._notify_status_change(employee_name, 'available', camera_id)
                cameras_str = ', '.join(self.employee_status[employee_name]['cameras'])
                print(f"[TRACKING] {employee_name} is now available at {camera_id or 'Unknown Camera'} (Cameras: {cameras_str})")
            # Jika sudah available dan berpindah camera
            elif previous_status == 'available' and self.employee_status[employee_name]['camera'] != camera_id:
                cameras_str = ', '.join(self.employee_status[employee_name]['cameras'])
                print(f"[TRACKING] {employee_name} moved to {camera_id or 'Unknown Camera'} (Cameras: {cameras_str})")
    
    def check_absences(self):
        """
        Periksa karyawan yang absent berdasarkan threshold waktu
        
        Returns:
            list: Daftar karyawan yang baru saja dianggap absent
        """
        current_time = time.time()
        newly_absent = []
        
        # Hanya periksa karyawan yang saat ini available
        for employee_name, status_info in self.employee_status.items():
            if status_info['status'] == 'available':
                time_since_last_seen = current_time - status_info['last_seen']
                
                # Jika sudah melewati threshold, tandai sebagai off
                if time_since_last_seen > self.absence_threshold:
                    self.employee_status[employee_name]['status'] = 'off'
                    newly_absent.append(employee_name)
                    
                    # Update status karyawan di database
                    if DATABASE_UPDATE_AVAILABLE:
                        try:
                            mark_employee_off(employee_name)
                        except Exception as e:
                            print(f"[ERROR] Failed to update employee status in database: {e}")
                    
                    # Log aktivitas off
                    self.monitor.log_employee_activity(
                        employee_name, 
                        "off", 
                        details={
                            'status': 'off',
                            'seconds_since_last_seen': time_since_last_seen,
                            'last_camera': status_info['camera']
                        }
                    )
                    
                    # Panggil callback untuk status change
                    self._notify_status_change(employee_name, 'off', status_info['camera'])
                    cameras_str = ', '.join(status_info['cameras']) if status_info['cameras'] else 'None'
                    print(f"[TRACKING] {employee_name} is now off (Last seen at {status_info['camera'] or 'Unknown Camera'} after {time_since_last_seen:.1f}s, Cameras: {cameras_str})")
        
        return newly_absent
    
    def get_employee_status(self, employee_name):
        """
        Dapatkan status karyawan
        
        Args:
            employee_name (str): Nama karyawan
            
        Returns:
            dict: Informasi status karyawan atau None jika tidak ditemukan
        """
        return self.employee_status.get(employee_name)
    
    def get_all_employees_status(self):
        """
        Dapatkan status semua karyawan
        
        Returns:
            dict: Dictionary status semua karyawan
        """
        return self.employee_status.copy()
    
    def get_available_employees(self):
        """
        Dapatkan daftar karyawan yang saat ini available (terdeteksi)
        
        Returns:
            list: Daftar nama karyawan yang available
        """
        return [name for name, info in self.employee_status.items() 
                if info['status'] == 'available']
    
    def get_off_employees(self):
        """
        Dapatkan daftar karyawan yang saat ini off (tidak terdeteksi)
        
        Returns:
            list: Daftar nama karyawan yang off
        """
        return [name for name, info in self.employee_status.items() 
                if info['status'] == 'off']
    
    def add_status_change_callback(self, callback):
        """
        Tambahkan callback function untuk notifikasi perubahan status
        
        Args:
            callback (function): Fungsi callback dengan parameter (employee_name, new_status, camera_id)
        """
        self.status_change_callbacks.append(callback)
    
    def _notify_status_change(self, employee_name, new_status, camera_id):
        """
        Notify semua callback tentang perubahan status karyawan
        
        Args:
            employee_name (str): Nama karyawan
            new_status (str): Status baru ('available' atau 'off')
            camera_id (str): ID kamera tempat perubahan terjadi
        """
        for callback in self.status_change_callbacks:
            try:
                callback(employee_name, new_status, camera_id)
            except Exception as e:
                print(f"[ERROR] Error in status change callback: {e}")
    
    def refresh_employee_list(self):
        """Refresh daftar karyawan dari database"""
        try:
            employees = get_all_employees()
            employee_names = [name for name, _ in employees]
            
            # Tambahkan karyawan baru yang mungkin baru saja diregistrasi
            for name in employee_names:
                if name not in self.employee_status:
                    self.employee_status[name] = {
                        'last_seen': 0,
                        'status': 'off',
                        'camera': None,
                        'cameras': []
                    }
            
            # Hapus karyawan yang mungkin dihapus dari database
            current_employees = list(self.employee_status.keys())
            for name in current_employees:
                if name not in employee_names:
                    del self.employee_status[name]
                    print(f"[TRACKING] Removed {name} from tracking (deleted from database)")
                    
            print(f"[INFO] Employee list refreshed. Total employees: {len(self.employee_status)}")
        except Exception as e:
            print(f"[ERROR] Failed to refresh employee list: {e}")
    
    def get_status_report(self):
        """
        Dapatkan laporan status karyawan
        
        Returns:
            dict: Laporan status karyawan
        """
        available = self.get_available_employees()
        off = self.get_off_employees()
        
        report = {
            'timestamp': time.time(),
            'timestamp_formatted': datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            'total_employees': len(self.employee_status),
            'available_count': len(available),
            'off_count': len(off),
            'available_employees': available,
            'off_employees': off,
            'details': self.employee_status.copy()
        }
        
        return report

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi tracker dengan threshold 30 detik
    tracker = EmployeeStatusTracker(absence_threshold_seconds=30)
    
    # Contoh callback untuk notifikasi perubahan status
    def status_change_notification(employee_name, new_status, camera_id):
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[NOTIFICATION] {timestamp} - {employee_name} is now {new_status} at {camera_id or 'Unknown Camera'}")
    
    # Tambahkan callback
    tracker.add_status_change_callback(status_change_notification)
    
    # Simulasi deteksi karyawan
    print("Simulasi deteksi karyawan...")
    tracker.update_employee_status("John Doe", "CAM1")
    tracker.update_employee_status("Jane Smith", "CAM2")
    
    print("\nStatus awal:")
    report = tracker.get_status_report()
    print(f"Available: {report['available_employees']}")
    print(f"Off: {report['off_employees']}")
    
    print("\nMenunggu 35 detik untuk simulasi absence...")
    time.sleep(35)
    
    # Periksa absence
    off_employees = tracker.check_absences()
    print(f"\nKaryawan yang baru off: {off_employees}")
    
    print("\nStatus setelah pengecekan:")
    report = tracker.get_status_report()
    print(f"Available: {report['available_employees']}")
    print(f"Off: {report['off_employees']}")