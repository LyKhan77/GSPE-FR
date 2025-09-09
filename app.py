import os
import json
import base64
import threading
import datetime as dt
import time
import signal
import sys
import subprocess
import io
import csv
from typing import Dict, Any, Optional
import numpy as np
import cv2
import shutil

from flask import Flask, jsonify, render_template, request, send_file, Response
from flask_socketio import SocketIO, emit
from database_models import SessionLocal, Employee, Camera, FaceTemplate, Attendance, Presence, Event, AlertLog  # DB models
from database_models import seed_cameras_from_configs

# AI tracking manager
try:
    from module_AI import ai_manager
except Exception as _e:
    ai_manager = None

try:
    import cv2
except ImportError:
    cv2 = None

# Inisialisasi Flask dan Socket.IO
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'change-me'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Inisialisasi variabel global
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_DIR = os.path.join(BASE_DIR, 'camera_configs')
DB_DIR = os.path.join(BASE_DIR, 'db')
os.makedirs(DB_DIR, exist_ok=True)
TRACK_STATE_PATH = os.path.join(DB_DIR, 'tracking_mode.json')
_camera_map: Dict[int, Dict[str, Any]] = {}
_workers_by_sid: Dict[str, Any] = {}
# Camera status cache (TTL)
_CAM_STATUS_TTL = 10.0  # seconds
_cam_status_cache: Dict[int, Dict[str, Any]] = {}
# Optional: Face embedding engine (lazy)
_face_app = None
# Directory to store cropped face images per template
FACE_IMG_DIR = os.path.join(BASE_DIR, 'face_images')


def _safe_name(name: Optional[str]) -> str:
    try:
        s = (name or '').strip()
        if not s:
            return 'unknown'
        # Keep alnum, space, dash, underscore; replace space with underscore
        cleaned = ''.join(ch for ch in s if ch.isalnum() or ch in (' ', '-', '_'))
        cleaned = '_'.join(part for part in cleaned.split())
        return cleaned[:64]  # limit length
    except Exception:
        return 'unknown'

def _employee_image_dir(emp: 'Employee') -> str:
    return os.path.join(FACE_IMG_DIR, _safe_name(getattr(emp, 'name', None)) or f"ID_{emp.id}")

# --- Time helpers ---
def _to_iso_utc(dtobj: Optional[dt.datetime]) -> Optional[str]:
    """Serialize a datetime as ISO 8601 with explicit UTC 'Z'.
    We store timestamps in UTC-naive; treat naive as UTC and add 'Z'.
    """
    if dtobj is None:
        return None
    try:
        if dtobj.tzinfo is None:
            return dtobj.isoformat() + 'Z'
        return dtobj.astimezone(dt.timezone.utc).isoformat().replace('+00:00', 'Z')
    except Exception:
        try:
            return dtobj.isoformat()
        except Exception:
            return None


def _get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
        # Use the same providers and detection size as module_AI (from parameter_config.json)
        det_size = (640, 640)
        providers = ['CPUExecutionProvider']
        try:
            if ai_manager is not None and hasattr(ai_manager, 'engine') and ai_manager.engine is not None:
                det_size = tuple(ai_manager.engine.det_size)
                providers = list(ai_manager.engine.providers)
        except Exception:
            pass
        # Try with configured providers first, then fallback to defaults
        try:
            _face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            _face_app.prepare(ctx_id=0, det_size=det_size)
            return _face_app
        except Exception as e1:
            print(f"FaceAnalysis init with providers {providers} failed: {e1}. Retrying with default providers...")
            _face_app = FaceAnalysis(name='buffalo_l')
            _face_app.prepare(ctx_id=0, det_size=det_size)
            return _face_app
    except Exception as e:
        print(f"Failed to init FaceAnalysis: {e}")
        return None


def load_cameras() -> Dict[int, Dict[str, Any]]:
    cams: Dict[int, Dict[str, Any]] = {}
    if not os.path.isdir(CAMERA_DIR):
        return cams
    for name in os.listdir(CAMERA_DIR):
        path = os.path.join(CAMERA_DIR, name)
        if not os.path.isdir(path):
            continue
        cfg_path = os.path.join(path, 'config.json')
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            cam_id = int(cfg.get('id'))
            cams[cam_id] = {
                'id': cam_id,
                'name': cfg.get('name', f'CAM {cam_id}'),
                'rtsp_url': cfg.get('rtsp_url', ''),
                'enabled': bool(cfg.get('enabled', False)),
            }
        except Exception:
            continue
    return cams


# --- Daily maintenance: purge events older than today ---

def purge_old_events():
    try:
        with SessionLocal() as db:
            from database_models import Event  # local import to avoid circular
            # Compute start of today in server local time converted to UTC assumption-free by comparing date only
            today = dt.date.today()
            # Delete events where date(timestamp) != today
            # SQLite lacks date() on SQLAlchemy by default; fetch and delete in chunks
            rows = db.query(Event).all()
            removed = 0
            for r in rows:
                try:
                    if not r.timestamp or r.timestamp.date() != today:
                        db.delete(r)
                        removed += 1
                except Exception:
                    continue
            db.commit()
            if removed:
                print(f"[MAINT] Purged {removed} old events (kept only today)")
    except Exception as e:
        print(f"[MAINT] purge_old_events error: {e}")


def _seconds_until_midnight_local() -> float:
    now = dt.datetime.now()
    tomorrow = now.date() + dt.timedelta(days=1)
    midnight = dt.datetime.combine(tomorrow, dt.time.min)
    return max(1.0, (midnight - now).total_seconds())


def schedule_midnight_purge():
    def _job():
        while True:
            try:
                purge_old_events()
            except Exception:
                pass
            time.sleep(_seconds_until_midnight_local())
    t = threading.Thread(target=_job, daemon=True)
    t.start()

# Initialize schedulers at import time (single-process assumption)
# Moved to after function definitions below to avoid NameError on import ordering.

# --- Tracking schedule & state (WIB) ---
def _default_tracking_state():
    return {
        'auto_schedule': True,
        'tracking_active': False,   # will be computed at startup
        'suppress_alerts': False,
        'pause_until': None,        # ISO local time string or None
        'work_hours': '08:30-17:30',
        'lunch_break': '12:00-13:00',
    }

_tracking_state = _default_tracking_state()

def _load_tracking_state():
    global _tracking_state
    try:
        if os.path.isfile(TRACK_STATE_PATH):
            with open(TRACK_STATE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _tracking_state.update(data)
    except Exception:
        pass

def _save_tracking_state():
    try:
        with open(TRACK_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(_tracking_state, f, indent=2)
    except Exception:
        pass

def _parse_range(s: str):
    try:
        a, b = [x.strip() for x in s.split('-', 1)]
        a_h, a_m = [int(x) for x in a.split(':', 1)]
        b_h, b_m = [int(x) for x in b.split(':', 1)]
        return (a_h, a_m), (b_h, b_m)
    except Exception:
        return (8,30), (17,30)

def _now_local():
    # Treat server local time as WIB if server runs in WIB; otherwise this is local server time
    return dt.datetime.now()

def _in_range(now: dt.datetime, rng: str) -> bool:
    (h1, m1), (h2, m2) = _parse_range(rng)
    start = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
    end = now.replace(hour=h2, minute=m2, second=0, microsecond=0)
    if end <= start:
        # overnight range not expected here; treat as always false
        return False
    return start <= now <= end

def _maybe_update_tracking_state():
    now = _now_local()
    # Handle manual pause
    pu = _tracking_state.get('pause_until')
    if pu:
        try:
            until = dt.datetime.fromisoformat(pu)
            if now < until:
                # During manual pause
                _tracking_state['tracking_active'] = False
                _tracking_state['suppress_alerts'] = True
                return
            else:
                # Pause expired
                _tracking_state['pause_until'] = None
        except Exception:
            _tracking_state['pause_until'] = None
    if bool(_tracking_state.get('auto_schedule', True)):
        work_ok = _in_range(now, str(_tracking_state.get('work_hours', '08:30-17:30')))
        lunch_on = _in_range(now, str(_tracking_state.get('lunch_break', '12:00-13:00')))
        _tracking_state['tracking_active'] = bool(work_ok)
        _tracking_state['suppress_alerts'] = bool(lunch_on)
    # persist periodically via caller

def schedule_tracking_manager():
    def _job():
        while True:
            try:
                _maybe_update_tracking_state()
                _save_tracking_state()
            except Exception:
                pass
            time.sleep(15)
    t = threading.Thread(target=_job, daemon=True)
    t.start()

@app.route('/api/schedule/state')
def api_schedule_state():
    _maybe_update_tracking_state()
    st = dict(_tracking_state)
    return jsonify(st)

@app.route('/api/schedule/mode', methods=['POST'])
def api_schedule_mode():
    data = request.get_json(silent=True) or {}
    auto = data.get('auto_schedule')
    if auto is not None:
        _tracking_state['auto_schedule'] = bool(auto)
    # Allow manual overrides when auto_schedule is False
    if not _tracking_state.get('auto_schedule', True):
        if 'tracking_active' in data:
            _tracking_state['tracking_active'] = bool(data.get('tracking_active'))
        if 'suppress_alerts' in data:
            _tracking_state['suppress_alerts'] = bool(data.get('suppress_alerts'))
    # Update schedule ranges if provided
    wh = data.get('work_hours')
    lb = data.get('lunch_break')
    if isinstance(wh, str) and '-' in wh:
        _tracking_state['work_hours'] = wh
    if isinstance(lb, str) and '-' in lb:
        _tracking_state['lunch_break'] = lb
    # Clear manual pause if any
    if data.get('clear_pause'):
        _tracking_state['pause_until'] = None
    _maybe_update_tracking_state()
    _save_tracking_state()
    return jsonify({'ok': True, 'state': _tracking_state})

@app.route('/api/schedule/pause', methods=['POST'])
def api_schedule_pause():
    data = request.get_json(silent=True) or {}
    minutes = data.get('minutes')
    until_s = data.get('until')  # ISO local time string
    now = _now_local()
    until = None
    if isinstance(minutes, (int, float)) and minutes > 0:
        until = now + dt.timedelta(minutes=float(minutes))
    elif isinstance(until_s, str):
        try:
            until = dt.datetime.fromisoformat(until_s)
        except Exception:
            pass
    if until is None:
        return jsonify({'error': 'invalid_pause'}), 400
    _tracking_state['pause_until'] = until.isoformat()
    _maybe_update_tracking_state()
    _save_tracking_state()
    return jsonify({'ok': True, 'state': _tracking_state})


@app.route('/')
def index():
    return render_template('index.html')


# Serve GSPE logo asset
@app.route('/assets/logo')
def asset_logo():
    try:
        p = os.path.join(os.path.dirname(__file__), 'templates', 'src', 'LOGO_GSPE_transparent.png')
        return send_file(p, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/cameras')
def api_cameras():
    global _camera_map
    _camera_map = load_cameras()
    # Do not expose rtsp_url to the client
    payload = [{'id': c['id'], 'name': c.get('name', f"CAM {c['id']}")} for c in _camera_map.values()]
    # sort by id for stable ordering
    payload.sort(key=lambda x: x['id'])
    return jsonify(payload)


@app.route('/api/cameras', methods=['POST'])
def api_add_camera():
    """Create a new camera by writing camera_configs/CAM<ID>/config.json."""
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    rtsp_url = (data.get('rtsp_url') or '').strip()
    cam_id = data.get('id')
    if not name or not rtsp_url:
        return jsonify({'error': 'name and rtsp_url are required'}), 400
    try:
        cams = load_cameras()
        # determine id
        if cam_id is None or str(cam_id).strip() == '':
            next_id = (max(cams.keys()) + 1) if cams else 1
        else:
            next_id = int(cam_id)
            if next_id in cams:
                return jsonify({'error': 'id already exists'}), 409
        # prepare folder
        folder = os.path.join(CAMERA_DIR, f'CAM{next_id}')
        cfg_path = os.path.join(folder, 'config.json')
        if os.path.exists(cfg_path):
            return jsonify({'error': 'config already exists'}), 409
        os.makedirs(folder, exist_ok=True)
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump({'id': next_id, 'name': name, 'rtsp_url': rtsp_url, 'enabled': False}, f, indent=4)
        # refresh cache
        global _camera_map
        _camera_map = load_cameras()
        return jsonify({'ok': True, 'camera': {'id': next_id, 'name': name}}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Reports: Attendance & Alert Logs ---
@app.route('/api/report/attendance')
def api_report_attendance():
    """Return attendance rows with optional filters. Supports JSON or CSV via ?format=csv."""
    args = request.args
    from_s = args.get('from') or args.get('date_from') or args.get('start')
    to_s = args.get('to') or args.get('date_to') or args.get('end')
    emp_id = args.get('employee_id')
    fmt = (args.get('format') or '').lower()
    start_d = end_d = None
    try:
        if from_s:
            y, m, d = [int(x) for x in str(from_s).split('-')]
            start_d = dt.date(y, m, d)
        if to_s:
            y, m, d = [int(x) for x in str(to_s).split('-')]
            end_d = dt.date(y, m, d)
    except Exception:
        return jsonify({'error': 'invalid_date'}), 400
    with SessionLocal() as db:
        q = db.query(Attendance, Employee).join(Employee, Attendance.employee_id == Employee.id)
        if emp_id:
            try:
                q = q.filter(Attendance.employee_id == int(emp_id))
            except Exception:
                pass
        if start_d:
            q = q.filter(Attendance.date >= start_d)
        if end_d:
            q = q.filter(Attendance.date <= end_d)
        q = q.order_by(Attendance.date.desc(), Employee.name.asc())
        rows = q.all()
        # JSON output
        if fmt != 'csv':
            data = []
            for att, emp in rows:
                data.append({
                    'employee_id': att.employee_id,
                    'employee_code': emp.employee_code,
                    'employee_name': emp.name,
                    'date': att.date.isoformat() if att.date else None,
                    'first_in_ts': _to_iso_utc(att.first_in_ts),
                    'last_out_ts': _to_iso_utc(att.last_out_ts),
                    'status': att.status,
                })
            return jsonify(data)
        # CSV output
        sio = io.StringIO()
        writer = csv.writer(sio)
        writer.writerow(['Employee Code', 'Employee Name', 'Date', 'First In', 'Last Out', 'Status'])
        for att, emp in rows:
            writer.writerow([
                emp.employee_code or '',
                emp.name or '',
                att.date.isoformat() if att.date else '',
                att.first_in_ts.isoformat(sep=' ') if att.first_in_ts else '',
                att.last_out_ts.isoformat(sep=' ') if att.last_out_ts else '',
                att.status or '',
            ])
        output = sio.getvalue()
        return Response(output, mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=attendance.csv'})


@app.route('/api/report/alerts')
def api_report_alerts():
    """Return alert logs with optional filters. Supports JSON or CSV via ?format=csv."""
    args = request.args
    from_s = args.get('from') or args.get('date_from') or args.get('start')
    to_s = args.get('to') or args.get('date_to') or args.get('end')
    emp_id = args.get('employee_id')
    fmt = (args.get('format') or '').lower()
    start_dt = end_dt = None
    try:
        if from_s:
            y, m, d = [int(x) for x in str(from_s).split('-')]
            start_dt = dt.datetime(y, m, d, 0, 0, 0)
        if to_s:
            y, m, d = [int(x) for x in str(to_s).split('-')]
            end_dt = dt.datetime(y, m, d, 23, 59, 59, 999000)
    except Exception:
        return jsonify({'error': 'invalid_date'}), 400
    with SessionLocal() as db:
        q = db.query(AlertLog, Employee).join(Employee, AlertLog.employee_id == Employee.id, isouter=True)
        if emp_id:
            try:
                q = q.filter(AlertLog.employee_id == int(emp_id))
            except Exception:
                pass
        if start_dt:
            q = q.filter(AlertLog.timestamp >= start_dt)
        if end_dt:
            q = q.filter(AlertLog.timestamp <= end_dt)
        q = q.order_by(AlertLog.timestamp.desc())
        rows = q.all()
        if fmt != 'csv':
            data = []
            for log, emp in rows:
                data.append({
                    'timestamp': _to_iso_utc(log.timestamp),
                    'employee_id': log.employee_id,
                    'employee_code': (emp.employee_code if emp else None),
                    'employee_name': (emp.name if emp else None),
                    'alert_type': log.alert_type,
                    'message': log.message,
                    'notified_to': log.notified_to,
                })
            return jsonify(data)
        # CSV
        sio = io.StringIO()
        writer = csv.writer(sio)
        writer.writerow(['Timestamp', 'Employee Code', 'Employee Name', 'Alert Type', 'Message', 'Notified To'])
        for log, emp in rows:
            writer.writerow([
                log.timestamp.isoformat(sep=' ') if log.timestamp else '',
                (emp.employee_code if emp else ''),
                (emp.name if emp else ''),
                log.alert_type or '',
                log.message or '',
                log.notified_to or '',
            ])
        output = sio.getvalue()
        return Response(output, mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=alert_logs.csv'})

@app.route('/api/employees', methods=['GET'])
def list_employees():
    """Return list of employees for Manage Employee UI."""
    with SessionLocal() as db:
        rows = db.query(Employee).order_by(Employee.id.asc()).all()
        data = []
        for e in rows:
            data.append({
                'id': e.id,
                'employee_code': e.employee_code,
                'name': e.name,
                'department': e.department,
                'position': e.position,
                'phone_number': e.phone_number,
                'is_active': e.is_active,
            })
        return jsonify(data)


@app.route('/api/employees', methods=['POST'])
def add_employee():
    """Create a new employee (without face template). Face can be added via register tool."""
    payload = request.get_json(silent=True) or {}
    required = ['employee_code', 'name']
    if any(not payload.get(k) for k in required):
        return jsonify({'error': 'employee_code and name required'}), 400
    with SessionLocal() as db:
        # Check duplicate code
        exists = db.query(Employee).filter(Employee.employee_code == payload['employee_code']).first()
        if exists:
            return jsonify({'error': 'employee_code already exists'}), 409
        e = Employee(
            employee_code=payload['employee_code'],
            name=payload.get('name'),
            department=payload.get('department'),
            position=payload.get('position'),
            phone_number=payload.get('phone_number'),
            is_active=bool(payload.get('is_active', True)),
        )
        db.add(e)
        db.commit()
        return jsonify({'id': e.id}), 201


@app.route('/api/employees/<int:eid>', methods=['PUT'])
def update_employee(eid: int):
    payload = request.get_json(silent=True) or {}
    with SessionLocal() as db:
        e = db.get(Employee, eid)
        if not e:
            return jsonify({'error': 'not found'}), 404
        # Update allowed fields
        # If updating employee_code, enforce uniqueness
        new_code = payload.get('employee_code', None)
        if new_code is not None and str(new_code) != str(e.employee_code):
            exists = db.query(Employee).filter(Employee.employee_code == new_code, Employee.id != eid).first()
            if exists:
                return jsonify({'error': 'employee_code already exists'}), 409
            e.employee_code = new_code
        for field in ['name', 'department', 'position', 'phone_number', 'is_active']:
            if field in payload:
                setattr(e, field, payload[field])
        # If is_active is toggled, update today's Attendance.status accordingly
        try:
            if 'is_active' in payload:
                today = dt.date.today()
                att = db.query(Attendance).filter(Attendance.employee_id == eid, Attendance.date == today).first()
                desired = 'PRESENT' if bool(payload.get('is_active')) else 'ABSENT'
                if att is None:
                    att = Attendance(employee_id=eid, date=today, status=desired)
                    db.add(att)
                else:
                    att.status = desired
        except Exception:
            # Do not block on attendance sync error
            pass
        db.commit()
        return jsonify({'ok': True})


@app.route('/api/employees/<int:eid>', methods=['DELETE'])
def delete_employee(eid: int):
    with SessionLocal() as db:
        e = db.get(Employee, eid)
        if not e:
            return jsonify({'error': 'not found'}), 404
        # 1) Delete dependent rows to satisfy FK constraints (explicit for safety and older DBs)
        db.query(FaceTemplate).filter(FaceTemplate.employee_id == eid).delete(synchronize_session=False)
        db.query(Attendance).filter(Attendance.employee_id == eid).delete(synchronize_session=False)
        db.query(Presence).filter(Presence.employee_id == eid).delete(synchronize_session=False)
        db.query(AlertLog).filter(AlertLog.employee_id == eid).delete(synchronize_session=False)
        # Events: remove all event rows for this employee (do not keep dangling history)
        db.query(Event).filter(Event.employee_id == eid).delete(synchronize_session=False)
        # 2) Delete employee row
        db.delete(e)
        db.commit()
        # 3) Remove face images directory from filesystem (non-fatal)
        try:
            # Remove name-based folder and legacy id-based folder
            emp_for_dir = db.get(Employee, eid)
            # emp is already deleted, so fetch name from previous object if available
            name_dir = None
            try:
                # e still holds data prior to delete
                name_dir = _employee_image_dir(e)
            except Exception:
                name_dir = None
            legacy_dir = os.path.join(FACE_IMG_DIR, str(eid))
            for path in filter(None, [name_dir, legacy_dir]):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
        return jsonify({'ok': True})


@app.route('/api/employees/<int:eid>/face_templates', methods=['POST'])
def add_face_template(eid: int):
    """Accepts a data URL image, extracts embedding with InsightFace, and stores FaceTemplate.
    Supports optional pose_label ('front'|'left'|'right'). Computes a quality_score [0..1].
    """
    payload = request.get_json(silent=True) or {}
    data_url = payload.get('image')
    pose_label = (payload.get('pose_label') or '').lower().strip() or None
    if not data_url or 'base64,' not in data_url:
        return jsonify({'error': 'image data_url required'}), 400
    app_engine = _get_face_app()
    if app_engine is None:
        return jsonify({'error': 'Face engine not available'}), 500
    # Decode data URL
    try:
        b64 = data_url.split('base64,', 1)[1]
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'invalid image'}), 400
    except Exception as e:
        return jsonify({'error': f'bad image: {e}'}), 400
    # Detect face and get embedding
    try:
        faces = app_engine.get(frame)
        if not faces:
            return jsonify({'error': 'no_face'}), 422
        def area(f):
            box = f.bbox.astype(int)
            return max(0, (box[2]-box[0])) * max(0, (box[3]-box[1]))
        face = max(faces, key=area)
        # Avoid boolean coercion on numpy arrays; check None explicitly
        emb = getattr(face, 'normed_embedding', None)
        if emb is None:
            emb = getattr(face, 'embedding', None)
        if emb is None:
            return jsonify({'error': 'no_embedding'}), 500
        emb = emb.astype('float32')
        emb_bytes = emb.tobytes()
        # Compute simple quality metrics
        try:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box.tolist()
            h, w = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop is not None and crop.size > 0 else None
            # Blur score via Variance of Laplacian (normalize roughly)
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
            blur_score = max(0.0, min(1.0, blur_var / 200.0))  # 0..1
            # Brightness score
            mean_b = float(np.mean(gray)) / 255.0 if gray is not None else 0.0
            bright_score = max(0.0, min(1.0, (mean_b - 0.2) / 0.6))  # prefer 0.2..0.8
            # Face size score relative to frame
            face_area = max(1.0, float((x2 - x1) * (y2 - y1)))
            frame_area = float(max(1, w * h))
            size_score = max(0.0, min(1.0, (face_area / frame_area) / 0.1))  # 10% area -> score 1
            # Aggregate
            quality_score = float(0.5 * blur_score + 0.3 * bright_score + 0.2 * size_score)
        except Exception:
            quality_score = None
    except Exception as e:
        return jsonify({'error': f'embed_error: {e}'}), 500
    # Save
    with SessionLocal() as db:
        emp = db.get(Employee, eid)
        if not emp:
            return jsonify({'error': 'employee_not_found'}), 404
        ft = FaceTemplate(employee_id=eid, embedding=emb_bytes, pose_label=pose_label, quality_score=quality_score)
        db.add(ft)
        db.commit()

        # After we have ft.id, save cropped face image to filesystem
        try:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box.tolist()
            # Expand a bit and clamp
            h, w = frame.shape[:2]
            pad = int(0.1 * max(x2 - x1, y2 - y1))
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                emp_dir = _employee_image_dir(emp)
                os.makedirs(emp_dir, exist_ok=True)
                out_path = os.path.join(emp_dir, f"{ft.id}.jpg")
                cv2.imwrite(out_path, crop)
        except Exception as _:
            # Non-fatal if image saving fails
            pass

        return jsonify({'ok': True, 'template_id': ft.id, 'pose_label': pose_label, 'quality_score': quality_score})


@app.route('/api/employees/<int:eid>/face_templates', methods=['GET'])
def list_face_templates(eid: int):
    """Return list of face templates for preview purposes (embedding as base64 bytes)."""
    with SessionLocal() as db:
        emp = db.get(Employee, eid)
        if not emp:
            return jsonify({'error': 'employee_not_found'}), 404
        rows = db.query(FaceTemplate).filter(FaceTemplate.employee_id == eid).order_by(FaceTemplate.id.asc()).all()
        out = []
        for r in rows:
            img_path = os.path.join(_employee_image_dir(emp), f"{r.id}.jpg")
            has_image = os.path.isfile(img_path)
            out.append({
                'id': r.id,
                'created_at': r.created_at.isoformat() if r.created_at else None,
                'pose_label': getattr(r, 'pose_label', None),
                'quality_score': getattr(r, 'quality_score', None),
                'embedding_b64': base64.b64encode(r.embedding).decode('ascii') if r.embedding else None,
                'image_url': f"/api/face_templates/{r.id}/image" if has_image else None,
            })
        return jsonify(out)


@app.route('/api/face_templates/<int:tid>/image')
def get_face_template_image(tid: int):
    """Serve the stored cropped face image for a template if available."""
    with SessionLocal() as db:
        tpl = db.get(FaceTemplate, tid)
        if not tpl:
            return jsonify({'error': 'not_found'}), 404
        emp = db.get(Employee, tpl.employee_id)
        if not emp:
            return jsonify({'error': 'employee_not_found'}), 404
        emp_dir = _employee_image_dir(emp)
    img_path = os.path.join(emp_dir, f"{tid}.jpg")
    if not os.path.isfile(img_path):
        return jsonify({'error': 'image_not_found'}), 404
    return send_file(img_path, mimetype='image/jpeg')


def _check_camera_online(rtsp_url: str) -> bool:
    if cv2 is None:
        return False
    if not rtsp_url:
        return False
    try:
        src = rtsp_url.strip()
        if src.lower().startswith('webcam:'):
            idx = int(src.split(':', 1)[1])
            cap = cv2.VideoCapture(idx, getattr(cv2, 'CAP_DSHOW', 0))
        elif src.isdigit():
            cap = cv2.VideoCapture(int(src), getattr(cv2, 'CAP_DSHOW', 0))
        else:
            cap = cv2.VideoCapture(src)
        ok = bool(cap and cap.isOpened())
    except Exception:
        ok = False
    finally:
        try:
            if 'cap' in locals() and cap:
                cap.release()
        except Exception:
            pass
    return ok


def _get_camera_online_cached(cam_id: int, rtsp_url: str) -> bool:
    now = time.time()
    item = _cam_status_cache.get(cam_id)
    if item and (now - item['ts'] <= _CAM_STATUS_TTL):
        return bool(item['online'])
    online = _check_camera_online(rtsp_url)
    _cam_status_cache[cam_id] = {'ts': now, 'online': bool(online)}
    return bool(online)


@app.route('/api/cameras/status')
def cameras_status():
    global _camera_map
    if not _camera_map:
        _camera_map = load_cameras()
    items = []
    for cam in _camera_map.values():
        items.append({
            'id': cam['id'],
            'name': cam.get('name', f"CAM {cam['id']}")
        })
    # add online flag with cached check
    for it in items:
        rtsp_url = _camera_map[it['id']].get('rtsp_url', '')
        it['online'] = _get_camera_online_cached(it['id'], rtsp_url)
    return jsonify(items)


class StreamWorker:
    def __init__(self, sid: str, cam_id: int, rtsp_url: str):
        self.sid = sid
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        # Shared state (no direct capture; frames come from ai_manager)
        self.frame_lock = threading.Lock()
        # Non-blocking annotation
        self._frame_counter = 0
        self.annotate_busy = False
        self.annotate_thread: Optional[threading.Thread] = None
        self.last_annotated_frame: Optional[np.ndarray] = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        # Nothing else to stop; AI manager owns the capture

    def run(self):
        if cv2 is None:
            socketio.emit('stream_error', {'message': 'OpenCV not installed'}, to=self.sid)
            return
        if not self.rtsp_url:
            socketio.emit('stream_error', {'message': 'Invalid stream source'}, to=self.sid)
            return
        try:
            # Read AI/stream preferences once
            prefs = {
                'max_width': 960,
                'jpeg_quality': 70,
                'annotation_stride': 3,
                'target_fps': 20,
            }
            try:
                if ai_manager is not None and hasattr(ai_manager, 'get_stream_preferences'):
                    prefs.update(ai_manager.get_stream_preferences())
            except Exception:
                pass
            target_dt = 1.0 / max(1, int(prefs.get('target_fps', 20)))
            max_w = int(prefs.get('max_width', 960))
            jpeg_q = int(prefs.get('jpeg_quality', 70))
            stride = max(1, int(prefs.get('annotation_stride', 3)))
            while not self.stop_event.is_set():
                # Get latest frame from AI manager (shared RTSP)
                frame = None
                try:
                    if ai_manager is not None and hasattr(ai_manager, 'get_latest_frame'):
                        frame = ai_manager.get_latest_frame(self.cam_id)
                except Exception:
                    frame = None
                if frame is None:
                    time.sleep(0.01)
                    continue
                # Downscale to reduce bandwidth/CPU
                h, w = frame.shape[:2]
                if w > max_w:
                    scale = max_w / float(w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                # Decide annotation without blocking
                self._frame_counter += 1
                do_annotate = (self._frame_counter % stride == 0)
                if do_annotate and not self.annotate_busy:
                    def _annotate_job(img: np.ndarray):
                        try:
                            out = img
                            if ai_manager is not None and hasattr(ai_manager, 'annotate_frame'):
                                out = ai_manager.annotate_frame(img, self.cam_id)
                            with self.frame_lock:
                                self.last_annotated_frame = out
                        except Exception:
                            with self.frame_lock:
                                self.last_annotated_frame = img
                        finally:
                            self.annotate_busy = False
                    self.annotate_busy = True
                    self.annotate_thread = threading.Thread(target=_annotate_job, args=(frame.copy(),), daemon=True)
                    self.annotate_thread.start()

                # Choose frame to send: prefer last annotated if available
                with self.frame_lock:
                    frame_to_send = self.last_annotated_frame if self.last_annotated_frame is not None else frame
                ok, buf = cv2.imencode('.jpg', frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                if ok:
                    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                    socketio.emit('frame', {'image': b64, 'cam_id': self.cam_id}, to=self.sid)
                time.sleep(target_dt)
        finally:
            socketio.emit('stream_stopped')


# Keep workers per client sid
_workers_by_sid: Dict[str, StreamWorker] = {}


@socketio.on('connect')
def on_connect():
    # No-op; frontend will request cameras and start a stream
    pass


@socketio.on('disconnect')
def on_disconnect():
    sid = getattr(request, 'sid', None)
    stop_worker_for_sid(sid)


# ---- Manage Camera: per-camera toggle & status ----
@socketio.on('toggle_camera')
def on_toggle_camera(data):
    try:
        cam_id = int(data.get('cam_id'))
        enable = bool(data.get('enable'))
    except Exception:
        return
    try:
        if enable:
            # start only this camera
            ai_manager.start([cam_id])
        else:
            # stop only this camera
            ai_manager.stop_camera(cam_id)
    except Exception as e:
        print(f"toggle_camera error: {e}")
    # persist enabled flag to config.json
    try:
        cams = load_cameras()
        if cam_id in cams:
            folder = os.path.join(CAMERA_DIR, f'CAM{cam_id}')
            cfg_path = os.path.join(folder, 'config.json')
            if os.path.isfile(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                cfg['enabled'] = bool(enable)
                with open(cfg_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=4)
    except Exception as e:
        print(f"persist enabled failed for cam {cam_id}: {e}")
    # emit back current status
    try:
        running = ai_manager.is_camera_running(cam_id)
        socketio.emit('camera_status', {'cam_id': cam_id, 'running': running})
    except Exception:
        pass


@socketio.on('get_camera_statuses')
def on_get_camera_statuses():
    try:
        statuses = []
        cams = load_cameras()
        for cam_id in sorted(cams.keys()):
            try:
                running = ai_manager.is_camera_running(cam_id)
                # auto-start if persisted enabled and not running yet
                if not running and cams[cam_id].get('enabled'):
                    try:
                        ai_manager.start([cam_id])
                        running = True
                    except Exception:
                        pass
            except Exception:
                running = False
            statuses.append({'cam_id': cam_id, 'name': cams[cam_id].get('name') or f'CAM{cam_id}', 'running': running})
        socketio.emit('camera_statuses', {'items': statuses})
    except Exception as e:
        print(f"get_camera_statuses error: {e}")


def stop_worker_for_sid(sid: Optional[str]):
    if not sid:
        return
    worker = _workers_by_sid.pop(sid, None)
    if worker:
        worker.stop()


# request already imported at top


@socketio.on('start_stream')
def start_stream(payload):
    sid = request.sid
    try:
        cam_id = int(payload.get('cam_id'))
    except Exception:
        emit('stream_error', {'message': 'Invalid cam_id'})
        return

    # Stop any existing stream for this client
    stop_worker_for_sid(sid)

    global _camera_map
    if not _camera_map:
        _camera_map = load_cameras()
    cam = _camera_map.get(cam_id)
    if not cam:
        emit('stream_error', {'message': 'Camera not found'})
        return

    # Enforce: camera must be enabled via Manage Camera toggle
    try:
        # sync camera list (non-destructive)
        try:
            seed_cameras_from_configs()
        except Exception:
            pass
        is_running = bool(ai_manager and hasattr(ai_manager, 'is_camera_running') and ai_manager.is_camera_running(cam_id))
    except Exception:
        is_running = False
    if not is_running:
        emit('stream_error', {'message': 'Camera is Non-Active'})
        return

    worker = StreamWorker(sid, cam_id, cam.get('rtsp_url', ''))
    _workers_by_sid[sid] = worker
    worker.start()


@socketio.on('stop_stream')
def stop_stream(payload=None):
    sid = request.sid
    stop_worker_for_sid(sid)


# --- AI Tracking Endpoints ---
@app.route('/api/tracking/start', methods=['POST'])
def api_tracking_start():
    if ai_manager is None:
        return jsonify({'error': 'ai_manager_not_available'}), 500
    payload = request.get_json(silent=True) or {}
    cam_ids = payload.get('cam_ids')
    if isinstance(cam_ids, list):
        try:
            cam_ids = [int(x) for x in cam_ids]
        except Exception:
            cam_ids = None
    else:
        cam_ids = None
    try:
        # sync cameras from configs (safe if configs exist)
        seed_cameras_from_configs()
    except Exception:
        pass
    try:
        ai_manager.start(cam_ids)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/stop', methods=['POST'])
def api_tracking_stop():
    if ai_manager is None:
        return jsonify({'error': 'ai_manager_not_available'}), 500
    try:
        ai_manager.stop()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tracking/state')
def api_tracking_state():
    if ai_manager is None:
        # include active_total even when AI not available
        active_total = 0
        try:
            with SessionLocal() as db:
                active_total = int(db.query(Employee).filter(Employee.is_active == True).count())
        except Exception:
            active_total = 0
        return jsonify({'running': False, 'present': 0, 'alerts': 0, 'total': 0, 'active_total': active_total, 'employees': []})
    try:
        state = ai_manager.get_state()
        # Inject active_total from DB
        active_total = 0
        try:
            with SessionLocal() as db:
                active_total = int(db.query(Employee).filter(Employee.is_active == True).count())
        except Exception:
            active_total = 0
        if isinstance(state, dict):
            state = {**state, 'active_total': active_total}
        return jsonify(state)
    except Exception as e:
        return jsonify({'running': False, 'error': str(e)}), 500


# --- Admin: Reset logs by date range ---
@app.route('/api/admin/reset_logs', methods=['POST'])
def api_admin_reset_logs():
    """Delete rows in events and/or alert_logs within an optional date range.
    Request JSON: { table: 'events'|'alert_logs'|'both', from_date: 'YYYY-MM-DD' (optional), to_date: 'YYYY-MM-DD' (optional) }
    If no dates provided, delete ALL rows in selected tables.
    Returns counts deleted.
    """
    payload = request.get_json(silent=True) or {}
    table = (payload.get('table') or 'both').lower()
    from_s = payload.get('from_date')
    to_s = payload.get('to_date')
    # Parse dates to datetimes spanning whole days
    start_dt = None
    end_dt = None
    try:
        if from_s:
            y, m, d = [int(x) for x in str(from_s).split('-')]
            start_dt = dt.datetime(y, m, d, 0, 0, 0)
        if to_s:
            y, m, d = [int(x) for x in str(to_s).split('-')]
            end_dt = dt.datetime(y, m, d, 23, 59, 59, 999000)
    except Exception:
        return jsonify({'error': 'invalid_date'}), 400
    try:
        deleted_events = 0
        deleted_alerts = 0
        with SessionLocal() as db:
            if table in ('events', 'both'):
                q = db.query(Event)
                if start_dt:
                    q = q.filter(Event.timestamp >= start_dt)
                if end_dt:
                    q = q.filter(Event.timestamp <= end_dt)
                deleted_events = q.delete(synchronize_session=False)
            if table in ('alert_logs', 'both'):
                q2 = db.query(AlertLog)
                if start_dt:
                    q2 = q2.filter(AlertLog.timestamp >= start_dt)
                if end_dt:
                    q2 = q2.filter(AlertLog.timestamp <= end_dt)
                deleted_alerts = q2.delete(synchronize_session=False)
            db.commit()
        return jsonify({'ok': True, 'deleted_events': int(deleted_events), 'deleted_alert_logs': int(deleted_alerts)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- System: Restart process ---
@app.route('/api/system/restart', methods=['POST'])
def api_system_restart():
    """Restart this Python process. Returns immediately then execs self in a background thread."""
    try:
        def _do_restart():
            try:
                # small delay to allow HTTP response to flush
                time.sleep(0.5)
                python = sys.executable or 'python'
                cmd = [python] + list(sys.argv)
                # On Windows, detach so new process isn't killed when this one exits
                creationflags = 0
                try:
                    DETACHED_PROCESS = 0x00000008
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                except Exception:
                    creationflags = 0
                try:
                    subprocess.Popen(cmd, close_fds=True, creationflags=creationflags)
                except Exception:
                    subprocess.Popen(cmd)
            finally:
                # terminate current process to free the port
                os._exit(0)
        threading.Thread(target=_do_restart, daemon=True).start()
        return jsonify({'ok': True, 'message': 'Restarting system...'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- System: Shutdown process ---
@app.route('/api/system/shutdown', methods=['POST'])
def api_system_shutdown():
    """Gracefully stop AI/camera workers and terminate the process."""
    try:
        def _do_shutdown():
            try:
                # Stop AI manager if present
                try:
                    if ai_manager is not None and hasattr(ai_manager, 'stop'):
                        ai_manager.stop()
                except Exception:
                    pass
                # Stop all stream workers
                try:
                    for sid, worker in list(_workers_by_sid.items()):
                        try:
                            if hasattr(worker, 'stop'):
                                worker.stop()
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(0.3)
            finally:
                os._exit(0)
        threading.Thread(target=_do_shutdown, daemon=True).start()
        return jsonify({'ok': True, 'message': 'Shutting down...'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Daily maintenance: purge old events and schedule next purge
        try:
            purge_old_events()
            schedule_midnight_purge()
        except Exception:
            pass
        print("Menjalankan server di http://0.0.0.0:5000")
        print("Tekan Ctrl+C untuk menghentikan")
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000,
                    debug=False,
                    use_reloader=False,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nMenghentikan server...")
        # Hentikan semua worker yang berjalan
        for sid, worker in list(_workers_by_sid.items()):
            if hasattr(worker, 'stop'):
                worker.stop()
        sys.exit(0)
