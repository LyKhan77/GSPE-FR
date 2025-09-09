import cv2
import numpy as np
import insightface
from sqlalchemy.exc import IntegrityError

from database_models import Employee, FaceTemplate, SessionLocal, init_db

# --- Konfigurasi --- #
WEBCAM_INDEX = 0  # ubah jika webcam bukan index 0


def load_insightface():
    """Load InsightFace FaceAnalysis (buffalo_l) on CPU."""
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def capture_frame(webcam_index: int = WEBCAM_INDEX):
    cap = cv2.VideoCapture(webcam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka webcam di index {webcam_index}.")
        return None

    print("\n--- Pendaftaran Wajah Karyawan ---")
    print("Arahkan wajah ke kamera.")
    print("Tekan [Spasi] untuk mengambil gambar.")
    print("Tekan [Q] untuk keluar.")

    captured = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal mengambil frame dari webcam.")
            break
        cv2.imshow('Register Face - Press [Space] to capture, [Q] to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            captured = frame.copy()
            break
        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def extract_embedding(face_app, bgr_image: np.ndarray) -> np.ndarray | None:
    """Return a 512-D embedding from the largest detected face.
    bgr_image: OpenCV BGR frame.
    """
    faces = face_app.get(bgr_image)
    if not faces:
        print("Tidak ada wajah terdeteksi. Coba lagi.")
        return None

    # pilih wajah terbesar
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    face = faces[0]

    emb = None
    if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
        emb = face.normed_embedding
    elif hasattr(face, 'embedding') and face.embedding is not None:
        emb = face.embedding
    else:
        print("Embedding tidak tersedia dari model. Pastikan model buffalo_l dimuat dengan benar.")
        return None

    emb = np.asarray(emb, dtype=np.float32).flatten()
    if emb.size == 0:
        print("Embedding kosong.")
        return None
    return emb


def register_employee():
    # Pastikan DB ada tabelnya
    init_db()

    # Input data karyawan
    print("\n=== Input Data Karyawan ===")
    employee_code = input("Kode Karyawan (unik): ").strip()
    name = input("Nama: ").strip()
    department = input("Departemen (opsional): ").strip() or None
    position = input("Jabatan (opsional): ").strip() or None
    phone_number = input("No. WhatsApp (opsional): ").strip() or None

    # Ambil foto
    frame = capture_frame(WEBCAM_INDEX)
    if frame is None:
        return

    # Load model & ekstrak embedding
    try:
        face_app = load_insightface()
    except Exception as e:
        print(f"Error saat memuat InsightFace: {e}")
        return

    emb = extract_embedding(face_app, frame)
    if emb is None:
        return

    # Simpan ke DB
    with SessionLocal() as db:
        # Cek eksistensi employee_code
        emp = db.query(Employee).filter(Employee.employee_code == employee_code).one_or_none()
        if emp is None:
            emp = Employee(
                employee_code=employee_code,
                name=name,
                department=department,
                position=position,
                phone_number=phone_number,
                is_active=True,
            )
            db.add(emp)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                print("Employee code sudah digunakan. Batalkan.")
                return
            db.refresh(emp)
        else:
            print(f"Employee dengan kode {employee_code} sudah ada. Embedding akan ditambahkan.")

        # Simpan FaceTemplate
        # Konversi embedding ke bytes untuk disimpan sebagai LargeBinary
        emb_bytes = emb.astype(np.float32).tobytes()
        tmpl = FaceTemplate(employee_id=emp.id, embedding=emb_bytes)
        db.add(tmpl)
        db.commit()
        print(f"Registrasi berhasil untuk {emp.name} (code: {emp.employee_code}). Template wajah disimpan.")


if __name__ == '__main__':
    register_employee()
