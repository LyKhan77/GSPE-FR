#!/usr/bin/env python3
# enhanced_tracking_config.py
# Konfigurasi untuk enhanced face tracking

# Konfigurasi default untuk enhanced tracking
ENHANCED_TRACKING_CONFIG = {
    # Parameter untuk face tracking
    "max_distance_threshold": 150,   # Ditingkatkan dari 100 untuk toleransi posisi yang lebih baik
    "tracking_timeout": 3.0,         # Ditingkatkan dari 1.0 detik untuk mempertahankan ID lebih lama
    "min_detection_confidence": 0.5, # Confidence minimum untuk deteksi
    
    # Parameter untuk smoothing yang lebih baik
    "bbox_smoothing_factor": 0.85,   # Faktor smoothing yang lebih tinggi untuk hasil lebih smooth
    "movement_threshold": 5,        # Threshold pergerakan minimal untuk update bounding box
    "velocity_smoothing": 0.7,      # Faktor smoothing untuk kecepatan pergerakan
    
    # Parameter untuk multi-camera support
    "camera_switch_delay": 2.0,     # Delay saat switch camera (detik)
    "cross_camera_matching": True,  # Izinkan matching wajah antar kamera
    
    # Parameter untuk activity monitoring
    "activity_log_interval": 30,    # Interval logging aktivitas (detik)
    "absent_alert_threshold": 300,  # Threshold alert untuk karyawan tidak terlihat (detik)
    
    # Parameter untuk performa
    "max_tracked_faces": 20,        # Maksimum wajah yang di-track secara bersamaan
    "cleanup_interval": 5.0,        # Interval pembersihan tracking data (detik)
}

def get_enhanced_tracking_config():
    """Dapatkan konfigurasi enhanced tracking"""
    return ENHANCED_TRACKING_CONFIG.copy()

def update_tracking_config(config_updates):
    """Update konfigurasi tracking dengan nilai baru"""
    config = ENHANCED_TRACKING_CONFIG.copy()
    config.update(config_updates)
    return config

# Rekomendasi konfigurasi berdasarkan kasus penggunaan
def get_office_environment_config():
    """Konfigurasi optimal untuk lingkungan kantor"""
    return {
        "bbox_smoothing_factor": 0.9,      # Sangat smooth untuk lingkungan stabil
        "max_distance_threshold": 150,     # Threshold lebih besar untuk toleransi pergerakan
        "movement_threshold": 3,           # Threshold pergerakan lebih sensitif
        "tracking_timeout": 3.0,           # Timeout lebih lama untuk konsistensi ID
        "absent_alert_threshold": 300,     # 5 menit untuk alert karyawan tidak terlihat
    }

def get_dynamic_environment_config():
    """Konfigurasi untuk lingkungan dinamis (seperti CCTV umum)"""
    return {
        "bbox_smoothing_factor": 0.75,     # Sedikit lebih responsif
        "max_distance_threshold": 150,     # Threshold lebih besar untuk toleransi gerakan
        "movement_threshold": 8,           # Threshold pergerakan lebih tinggi
        "tracking_timeout": 3.0,           # Timeout lebih lama untuk stabilitas
        "absent_alert_threshold": 600,     # 10 menit untuk alert
    }

def get_high_accuracy_config():
    """Konfigurasi untuk akurasi maksimal"""
    return {
        "bbox_smoothing_factor": 0.95,     # Sangat smooth
        "max_distance_threshold": 120,     # Threshold sedang untuk presisi tinggi
        "movement_threshold": 2,           # Threshold sangat sensitif
        "tracking_timeout": 3.0,           # Timeout lebih lama untuk konsistensi
        "absent_alert_threshold": 180,     # 3 menit untuk alert lebih cepat
    }

if __name__ == "__main__":
    # Contoh penggunaan
    print("Default Enhanced Tracking Config:")
    print(get_enhanced_tracking_config())
    
    print("\nOffice Environment Config:")
    print(get_office_environment_config())
    
    print("\nDynamic Environment Config:")
    print(get_dynamic_environment_config())