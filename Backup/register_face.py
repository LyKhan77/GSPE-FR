import cv2
import numpy as np
import insightface
from sqlalchemy.orm import sessionmaker

from database_models import engine, Employee, FaceTemplate, SessionLocal

# --- Konfigurasi --- #
WEBCAM_INDEX = 0

def main():
    """Fungsi utama untuk menjalankan proses pendaftaran wajah."""
    # 1. Inisialisasi Model InsightFace
    try:
        face_analysis = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        print("Model InsightFace berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat model InsightFace: {e}")
        return

    # 2. Ambil Gambar dari Webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka webcam di index {WEBCAM_INDEX}.")
        return

    print("\n--- Pendaftaran Wajah Karyawan ---")
    print("Arahkan wajah ke kamera.")
    print("Tekan [Spasi] untuk mengambil gambar.")
    print("Tekan [Q] untuk keluar.")

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari webcam.")
            break

        cv2.imshow('Registrasi Wajah - Tekan Spasi untuk Ambil Gambar', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured_frame = frame
            print("Gambar berhasil diambil!")
            break
        elif key == ord('q'):
            print("Pendaftaran dibatalkan.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        return

    # 3. Deteksi Wajah dan Ekstrak Embedding
    try:
        faces = face_analysis.get(captured_frame)
        if len(faces) == 0:
            print("Error: Tidak ada wajah yang terdeteksi. Silakan coba lagi.")
            return
        if len(faces) > 1:
            print("Error: Terdeteksi lebih dari satu wajah. Pastikan hanya ada satu wajah di depan kamera.")
            return
        
        face = faces[0]
        embedding = face.normed_embedding
        print("Embedding wajah berhasil diekstrak.")

    except Exception as e:
        print(f"Error saat analisis wajah: {e}")
        return

    # 4. Input Data Karyawan
    print("\n--- Masukkan Data Karyawan ---")
    employee_code = input("Kode Karyawan (e.g., EMP001): ").strip()
    name = input("Nama Lengkap: ").strip()
    department = input("Departemen: ").strip()
    position = input("Posisi/Jabatan: ").strip()
    phone_number = input("Nomor Telepon (opsional): ").strip()

    if not all([employee_code, name, department, position]):
        print("Error: Kode, Nama, Departemen, dan Posisi tidak boleh kosong.")
        return

    # 5. Simpan ke Database
    db = SessionLocal()
    try:
        # Cek apakah karyawan sudah ada
        existing_employee = db.query(Employee).filter(Employee.employee_code == employee_code).first()
        if existing_employee:
            print(f"Karyawan dengan kode {employee_code} sudah ada. Menambahkan template wajah baru.")
            employee = existing_employee
        else:
            print(f"Membuat data karyawan baru untuk {name}.")
            employee = Employee(
                employee_code=employee_code,
                name=name,
                department=department,
                position=position,
                phone_number=phone_number
            )
            db.add(employee)
            db.flush() # flush untuk mendapatkan ID karyawan baru

        # Buat template wajah baru
        face_template = FaceTemplate(
            employee_id=employee.id,
            embedding=embedding.tobytes() # Simpan sebagai bytes
        )
        db.add(face_template)
        db.commit()
        print("\n*** Pendaftaran Berhasil! ***")
        print(f"Wajah untuk {name} ({employee_code}) telah disimpan ke database.")

    except Exception as e:
        db.rollback()
        print(f"\nError saat menyimpan ke database: {e}")
    finally:
        db.close()

if __name__ == '__main__':
    main()
