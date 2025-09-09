#!/usr/bin/env python3
# parameter_tuning.py
# Script untuk menyesuaikan parameter sistem face recognition

import json
import os
from main import DEFAULT_SYSTEM_SPECS, load_system_specs, show_system_specs

CONFIG_FILE = "parameter_config.json"

def load_parameter_config():
    """Load parameter configuration from file or use defaults"""
    return load_system_specs()

def save_parameter_config(config):
    """Save parameter configuration to file"""
    try:
        # Convert tuple to list for JSON serialization
        config_to_save = config.copy()
        if isinstance(config_to_save.get("detection_size"), tuple):
            config_to_save["detection_size"] = list(config_to_save["detection_size"])
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print("[SUCCESS] Konfigurasi parameter berhasil disimpan ke {}".format(CONFIG_FILE))
        return True
    except Exception as e:
        print("[ERROR] Gagal menyimpan konfigurasi: {}".format(e))
        return False

def display_current_parameters(config):
    """Display current parameter values"""
    print("\n=== Parameter Sistem Saat Ini ===")
    for key, value in config.items():
        # Format khusus untuk beberapa parameter
        if key == "detection_size":
            print("{:<25}: {}x{}".format(key, value[0], value[1]))
        elif key == "recognition_cooldown" or key == "tracking_timeout":
            print("{:<25}: {} detik".format(key, value))
        elif key == "bbox_smoothing_factor" or key == "detection_threshold" or key == "recognition_threshold":
            print("{:<25}: {:.2f}".format(key, value))
        else:
            print("{:<25}: {}".format(key, value))
    print("=" * 35)

def tune_detection_threshold(config):
    """Tune detection threshold parameter"""
    print("\n--- Tuning Detection Threshold ---")
    print("Nilai saat ini: {}".format(config["detection_threshold"]))
    print("Deskripsi: Threshold untuk deteksi wajah (0.0-1.0)")
    print("Semakin tinggi nilai, semakin ketat deteksi")
    
    try:
        new_value = float(input("Masukkan nilai baru (0.0-1.0) atau tekan Enter untuk membatalkan: "))
        if 0.0 <= new_value <= 1.0:
            config["detection_threshold"] = new_value
            print("[SUCCESS] Detection threshold diubah menjadi {}".format(new_value))
        else:
            print("[ERROR] Nilai harus antara 0.0 dan 1.0")
    except ValueError:
        print("Dibatalkan.")

def tune_recognition_threshold(config):
    """Tune recognition threshold parameter"""
    print("\n--- Tuning Recognition Threshold ---")
    print("Nilai saat ini: {}".format(config["recognition_threshold"]))
    print("Deskripsi: Threshold untuk pengenalan wajah (0.0-1.0)")
    print("Semakin tinggi nilai, semakin ketat pengenalan")
    
    try:
        new_value = float(input("Masukkan nilai baru (0.0-1.0) atau tekan Enter untuk membatalkan: "))
        if 0.0 <= new_value <= 1.0:
            config["recognition_threshold"] = new_value
            print("[SUCCESS] Recognition threshold diubah menjadi {}".format(new_value))
        else:
            print("[ERROR] Nilai harus antara 0.0 dan 1.0")
    except ValueError:
        print("Dibatalkan.")

def tune_detection_size(config):
    """Tune detection size parameter"""
    print("\n--- Tuning Detection Size ---")
    print("Nilai saat ini: {}x{}".format(config["detection_size"][0], config["detection_size"][1]))
    print("Deskripsi: Ukuran input untuk deteksi wajah")
    print("Ukuran yang lebih besar = akurasi lebih tinggi tapi kinerja lebih lambat")
    print("Ukuran umum: (320,320), (640,640), (1024,1024)")
    
    try:
        width_input = input("Masukkan lebar (angka) atau tekan Enter untuk membatalkan: ")
        if width_input:
            width = int(width_input)
            height = int(input("Masukkan tinggi (angka): "))
            if width > 0 and height > 0:
                config["detection_size"] = (width, height)
                print("[SUCCESS] Detection size diubah menjadi {}x{}".format(width, height))
            else:
                print("[ERROR] Lebar dan tinggi harus angka positif")
    except ValueError:
        print("Dibatalkan.")

def tune_recognition_cooldown(config):
    """Tune recognition cooldown parameter"""
    print("\n--- Tuning Recognition Cooldown ---")
    print("Nilai saat ini: {} detik".format(config["recognition_cooldown"]))
    print("Deskripsi: Jeda antar pencatatan kehadiran (detik)")
    print("Semakin tinggi nilai, semakin lama jeda antar pencatatan")
    
    try:
        new_value = int(input("Masukkan nilai baru (detik) atau tekan Enter untuk membatalkan: "))
        if new_value >= 0:
            config["recognition_cooldown"] = new_value
            print("[SUCCESS] Recognition cooldown diubah menjadi {} detik".format(new_value))
        else:
            print("[ERROR] Nilai harus angka positif atau nol")
    except ValueError:
        print("Dibatalkan.")

def tune_bbox_smoothing_factor(config):
    """Tune bounding box smoothing factor parameter"""
    print("\n--- Tuning BBox Smoothing Factor ---")
    print("Nilai saat ini: {}".format(config["bbox_smoothing_factor"]))
    print("Deskripsi: Faktor smoothing bounding box (0.0-1.0)")
    print("Semakin tinggi nilai, semakin halus bounding box (tapi kurang responsif)")
    
    try:
        new_value = float(input("Masukkan nilai baru (0.0-1.0) atau tekan Enter untuk membatalkan: "))
        if 0.0 <= new_value <= 1.0:
            config["bbox_smoothing_factor"] = new_value
            print("[SUCCESS] BBox smoothing factor diubah menjadi {}".format(new_value))
        else:
            print("[ERROR] Nilai harus antara 0.0 dan 1.0")
    except ValueError:
        print("Dibatalkan.")

def tune_max_distance_threshold(config):
    """Tune max distance threshold parameter"""
    print("\n--- Tuning Max Distance Threshold ---")
    print("Nilai saat ini: {} pixels".format(config["max_distance_threshold"]))
    print("Deskripsi: Threshold jarak untuk matching wajah antar frame")
    print("Semakin tinggi nilai, semakin besar toleransi perubahan posisi")
    
    try:
        new_value = int(input("Masukkan nilai baru (pixels) atau tekan Enter untuk membatalkan: "))
        if new_value > 0:
            config["max_distance_threshold"] = new_value
            print("[SUCCESS] Max distance threshold diubah menjadi {} pixels".format(new_value))
        else:
            print("[ERROR] Nilai harus angka positif")
    except ValueError:
        print("Dibatalkan.")

def tune_tracking_timeout(config):
    """Tune tracking timeout parameter"""
    print("\n--- Tuning Tracking Timeout ---")
    print("Nilai saat ini: {} detik".format(config["tracking_timeout"]))
    print("Deskripsi: Timeout untuk mempertahankan ID tracking")
    print("Semakin tinggi nilai, semakin lama ID dipertahankan saat wajah tidak terlihat")
    
    try:
        new_value = float(input("Masukkan nilai baru (detik) atau tekan Enter untuk membatalkan: "))
        if new_value >= 0:
            config["tracking_timeout"] = new_value
            print("[SUCCESS] Tracking timeout diubah menjadi {} detik".format(new_value))
        else:
            print("[ERROR] Nilai harus angka positif atau nol")
    except ValueError:
        print("Dibatalkan.")

def tune_fps_target(config):
    """Tune FPS target parameter"""
    print("\n--- Tuning Target FPS ---")
    print("Nilai saat ini: {}".format(config["fps_target"]))
    print("Deskripsi: Target frame per second")
    print("Sesuaikan dengan kemampuan hardware Anda")
    
    try:
        new_value = int(input("Masukkan nilai baru atau tekan Enter untuk membatalkan: "))
        if new_value > 0:
            config["fps_target"] = new_value
            print("[SUCCESS] Target FPS diubah menjadi {}".format(new_value))
        else:
            print("[ERROR] Nilai harus angka positif")
    except ValueError:
        print("Dibatalkan.")

def tune_embedding_similarity_threshold(config):
    """Tune embedding similarity threshold parameter"""
    print("\n--- Tuning Embedding Similarity Threshold ---")
    print("Nilai saat ini: {:.2f}".format(config["embedding_similarity_threshold"]))
    print("Deskripsi: Threshold untuk matching wajah berbasis embedding")
    print("Semakin tinggi nilai, semakin ketat matching berbasis karakteristik wajah")
    
    try:
        new_value = float(input("Masukkan nilai baru (0.0-1.0) atau tekan Enter untuk membatalkan: "))
        if 0.0 <= new_value <= 1.0:
            config["embedding_similarity_threshold"] = new_value
            print("[SUCCESS] Embedding similarity threshold diubah menjadi {:.2f}".format(new_value))
        else:
            print("[ERROR] Nilai harus antara 0.0 dan 1.0")
    except ValueError:
        print("Dibatalkan.")

def reset_to_defaults(config):
    """Reset all parameters to default values"""
    print("\n--- Reset ke Default ---")
    print("Apakah Anda yakin ingin mereset semua parameter ke nilai default? (y/N)")
    confirmation = input().strip().lower()
    
    if confirmation == 'y' or confirmation == 'yes':
        config.clear()
        config.update(DEFAULT_SYSTEM_SPECS.copy())
        print("[SUCCESS] Semua parameter telah direset ke nilai default")
    else:
        print("Reset dibatalkan")

def reset_to_defaults(config):
    """Reset all parameters to default values"""
    print("\n--- Reset ke Default ---")
    print("Apakah Anda yakin ingin mereset semua parameter ke nilai default? (y/N)")
    confirmation = input().strip().lower()
    
    if confirmation == 'y' or confirmation == 'yes':
        config.clear()
        config.update(DEFAULT_SYSTEM_SPECS.copy())
        print("[SUCCESS] Semua parameter telah direset ke nilai default")
    else:
        print("Reset dibatalkan")

def main_menu(config):
    """Display main tuning menu"""
    while True:
        print("\n=== Parameter Tuning Menu ===")
        print("1. Lihat parameter saat ini")
        print("2. Tune detection threshold")
        print("3. Tune recognition threshold")
        print("4. Tune detection size")
        print("5. Tune recognition cooldown")
        print("6. Tune bbox smoothing factor")
        print("7. Tune max distance threshold")
        print("8. Tune tracking timeout")
        print("9. Tune embedding similarity threshold")
        print("10. Tune target FPS")
        print("11. Reset ke default")
        print("12. Simpan dan keluar")
        print("13. Keluar tanpa menyimpan")
        
        try:
            choice = input("Pilih opsi (1-13): ").strip()
            
            if choice == "1":
                display_current_parameters(config)
            elif choice == "2":
                tune_detection_threshold(config)
            elif choice == "3":
                tune_recognition_threshold(config)
            elif choice == "4":
                tune_detection_size(config)
            elif choice == "5":
                tune_recognition_cooldown(config)
            elif choice == "6":
                tune_bbox_smoothing_factor(config)
            elif choice == "7":
                tune_max_distance_threshold(config)
            elif choice == "8":
                tune_tracking_timeout(config)
            elif choice == "9":
                tune_embedding_similarity_threshold(config)
            elif choice == "10":
                tune_fps_target(config)
            elif choice == "11":
                reset_to_defaults(config)
            elif choice == "12":
                if save_parameter_config(config):
                    print("[INFO] Konfigurasi disimpan. Restart aplikasi untuk menerapkan perubahan.")
                    break
            elif choice == "13":
                print("Keluar tanpa menyimpan perubahan...")
                break
            else:
                print("[ERROR] Pilihan tidak valid!")
        except KeyboardInterrupt:
            print("\n\nKeluar tanpa menyimpan perubahan...")
            break
        except EOFError:
            print("\n\nKeluar tanpa menyimpan perubahan...")
            break

def main():
    print("=== Parameter Tuning Tool ===")
    
    # Load current configuration
    config = load_parameter_config()
    
    # Display current settings
    display_current_parameters(config)
    
    # Show main menu
    main_menu(config)
    
    print("\nTerima kasih telah menggunakan Parameter Tuning Tool!")

if __name__ == "__main__":
    main()