import os
import math
import json
import time
import threading
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, Counter

import numpy as np
import cv2

from database_models import (
    get_session,
    Employee,
    Camera,
    Event,
    Presence,
    Attendance,
    AlertLog,
)

# ---- Config loader ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
PARAM_PATH = os.path.join(CONFIG_DIR, 'parameter_config.json')
DB_DIR = os.path.join(BASE_DIR, 'db')
TRACK_STATE_PATH = os.path.join(DB_DIR, 'tracking_mode.json')

# Cached tracking state for alert suppression
_track_cache = {
    'ts': 0.0,
    'state': {'tracking_active': True, 'suppress_alerts': False}
}

def _read_tracking_state() -> dict:
    now = time.time()
    try:
        # refresh at most every 5 seconds
        if now - _track_cache['ts'] < 5.0:
            return dict(_track_cache['state'])
        st = {}
        try:
            with open(TRACK_STATE_PATH, 'r', encoding='utf-8') as f:
                st = json.load(f) or {}
        except Exception:
            st = {}
        _track_cache['state'] = {
            'tracking_active': bool(st.get('tracking_active', True)),
            'suppress_alerts': bool(st.get('suppress_alerts', False))
        }
        _track_cache['ts'] = now
        return dict(_track_cache['state'])
    except Exception:
        return {'tracking_active': True, 'suppress_alerts': False}

def _alerts_allowed() -> bool:
    st = _read_tracking_state()
    # Block alerts if tracking is inactive or alerts explicitly suppressed
    return bool(st.get('tracking_active', True)) and not bool(st.get('suppress_alerts', False))


def _load_config() -> dict:
    try:
        with open(PARAM_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # Reasonable defaults
        return {
            'detection_size': [640, 640],
            'recognition_threshold': 0.65,
            'embedding_similarity_threshold': 0.45,
            'tracking_timeout': 10.0,
            'fps_target': 15,
            # Prefer GPU by default; can be overridden in config/parameter_config.json
            'providers': 'TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider',
        }


# ---- Face engine wrapper ----
class FaceEngine:
    def __init__(self, det_size: Tuple[int, int], providers: List[str]):
        self.det_size = det_size
        self.providers = providers
        self.app = None
        self._init_engine()

    def _init_engine(self):
        try:
            # Encourage TensorRT engine caching and FP16 when available
            os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_ENABLE', '1')
            os.environ.setdefault('ORT_TENSORRT_FP16_ENABLE', '1')

            from insightface.app import FaceAnalysis
            # Log available ORT providers (if ORT installed)
            try:
                import onnxruntime as ort
                print(f"[AI] ORT available providers: {getattr(ort, 'get_available_providers', lambda: [])()}")
            except Exception as e:
                print(f"[AI] ORT introspection not available: {e}")

            attempted = []
            # Build provider preference tiers
            tiers = []
            # Use configured providers as first tier if provided
            if self.providers:
                tiers.append(self.providers)
            # Then try TensorRT+CUDA
            tiers.append(['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
            # Then CUDA only
            tiers.append(['CUDAExecutionProvider'])
            # Finally CPU
            tiers.append(['CPUExecutionProvider'])

            last_err = None
            for prov in tiers:
                try:
                    attempted.append(prov)
                    self.app = FaceAnalysis(name='buffalo_l', providers=prov)
                    self.app.prepare(ctx_id=0, det_size=self.det_size)
                    self.providers = prov  # record the actual providers used
                    break
                except Exception as e:
                    last_err = e
                    self.app = None
                    continue
            if self.app is None:
                # final fallback to default FaceAnalysis which may pick CPU
                self.app = FaceAnalysis(name='buffalo_l')
                self.app.prepare(ctx_id=0, det_size=self.det_size)
                self.providers = ['CPUExecutionProvider']
            # warmup to initialize kernels if any
            try:
                import numpy as _np
                _ = self.app.get((_np.zeros((self.det_size[1], self.det_size[0], 3), dtype='uint8')))
            except Exception:
                pass
            try:
                print(f"[AI] FaceAnalysis ready. Selected Providers={self.providers}, det_size={self.det_size}. Attempts={attempted}")
                if last_err:
                    print(f"[AI] Last provider init error (non-fatal): {last_err}")
            except Exception:
                pass
        except Exception as e:
            print(f"[AI] Failed to init FaceAnalysis: {e}")
            self.app = None

    def get_faces(self, frame: np.ndarray):
        if self.app is None:
            return []
        return self.app.get(frame)

    @staticmethod
    def get_embedding(face) -> Optional[np.ndarray]:
        emb = getattr(face, 'normed_embedding', None)
        if emb is None:
            emb = getattr(face, 'embedding', None)
        if emb is None:
            return None
        return emb.astype('float32')


# ---- Known embeddings store ----
class EmbeddingStore:
    def __init__(self):
        self.by_employee: Dict[int, List[np.ndarray]] = {}
        self.employee_meta: Dict[int, Dict[str, Any]] = {}
        self._last_load_ts = 0.0
        self.reload_interval = 60.0  # seconds
        self.load()

    def load(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_load_ts < self.reload_interval):
            return
        self._last_load_ts = now
        by_emp: Dict[int, List[np.ndarray]] = {}
        meta: Dict[int, Dict[str, Any]] = {}
        with get_session() as db:
            emps = db.query(Employee).all()
            for e in emps:
                meta[e.id] = {
                    'name': e.name,
                    'department': e.department,
                }
            # lazy import to avoid circular import
            from database_models import FaceTemplate
            rows = db.query(FaceTemplate).all()
            for r in rows:
                try:
                    arr = np.frombuffer(r.embedding, dtype='float32')
                    by_emp.setdefault(r.employee_id, []).append(arr)
                except Exception:
                    continue
        self.by_employee = by_emp
        self.employee_meta = meta

    def best_match(self, emb: np.ndarray) -> Tuple[Optional[int], float]:
        # cosine similarity on L2-normalized embeddings
        if emb is None or emb.size == 0:
            return None, 0.0
        best_emp = None
        best_sim = -1.0
        q = emb
        # normalize
        norm = np.linalg.norm(q) + 1e-8
        q = q / norm
        for emp_id, arrs in self.by_employee.items():
            for ref in arrs:
                r = ref
                r_norm = np.linalg.norm(r) + 1e-8
                r = r / r_norm
                sim = float(np.dot(q, r))  # cosine in [-1,1]
                if sim > best_sim:
                    best_sim = sim
                    best_emp = emp_id
        return best_emp, best_sim if best_sim > 0 else 0.0


# ---- Tracking Manager ----
class TrackingManager:
    def __init__(self):
        cfg = _load_config()
        self.cfg = cfg
        det_size = tuple(cfg.get('detection_size', [320, 320]))
        providers = [p.strip() for p in str(cfg.get('providers', 'CPUExecutionProvider')).split(',') if p.strip()]
        self.engine = FaceEngine(det_size, providers)
        self.emb_store = EmbeddingStore()
        self.recog_thresh = float(cfg.get('recognition_threshold', 0.45))
        self.sim_thresh = float(cfg.get('embedding_similarity_threshold', 0.65))
        self.track_timeout = float(cfg.get('tracking_timeout', 10.0))
        self.fps_target = max(1, int(cfg.get('fps_target', 10)))
        # streaming prefs
        self.stream_max_width = int(cfg.get('stream_max_width', 960))
        self.jpeg_quality = int(cfg.get('jpeg_quality', 70))
        self.annotation_stride = max(1, int(cfg.get('annotation_stride', 3)))
        # smoothing params
        self.smooth_window = int(cfg.get('smoothing_window', 5))  # last N votes
        self.smooth_min_votes = int(cfg.get('smoothing_min_votes', 3))  # require at least K same votes
        self.iou_match_threshold = float(cfg.get('tracker_iou_threshold', 0.3))
        self.max_track_misses = int(cfg.get('tracker_max_misses', 8))
        # event rate control (seconds between event inserts per emp+cam)
        self.event_min_interval = float(cfg.get('event_min_interval_sec', 5.0))
        # quality gating thresholds
        self.min_blur_var = float(cfg.get('quality_min_blur_var', 50.0))
        self.min_face_area_frac = float(cfg.get('quality_min_face_area_frac', 0.01))  # 1% of frame
        self.min_brightness = float(cfg.get('quality_min_brightness', 0.15))  # 0..1
        self.max_brightness = float(cfg.get('quality_max_brightness', 0.9))
        self.min_quality_score = float(cfg.get('quality_min_score', 0.3))

        # separate threads for capture and inference
        self._cap_threads: Dict[int, threading.Thread] = {}
        self._cap_stops: Dict[int, threading.Event] = {}
        self._proc_threads: Dict[int, threading.Thread] = {}
        self._proc_stops: Dict[int, threading.Event] = {}
        self._state_lock = threading.Lock()
        # state per employee
        self.last_seen: Dict[int, dt.datetime] = {}
        self.last_cam: Dict[int, int] = {}
        # latest raw frames by camera (for UI streaming to consume)
        self._frame_lock = threading.Lock()
        self._latest_frames: Dict[int, np.ndarray] = {}
        # per-camera simple trackers
        self._tracks: Dict[int, Dict[int, Any]] = {}  # cam_id -> {track_id: Track}
        self._next_track_id: Dict[int, int] = {}
        # last event timestamp per (emp_id, cam_id)
        self._last_event_ts: Dict[Tuple[int,int], dt.datetime] = {}
        # last alert timestamp per (emp_id, alert_type)
        self._last_alert_ts: Dict[Tuple[int,str], dt.datetime] = {}

    # ---- Simple Track structure ----
    class Track:
        def __init__(self, tid: int, bbox: Tuple[int,int,int,int], now: dt.datetime):
            self.id = tid
            self.bbox = bbox  # (x1,y1,x2,y2)
            self.last_ts = now
            self.hits = 1
            self.misses = 0
            self.votes = deque(maxlen=8)
            self.final_emp_id: Optional[int] = None
            self.final_since: Optional[dt.datetime] = None

        def iou(self, bbox: Tuple[int,int,int,int]) -> float:
            x1,y1,x2,y2 = self.bbox
            a1,b1,a2,b2 = bbox
            ix1 = max(x1,a1); iy1 = max(y1,b1)
            ix2 = min(x2,a2); iy2 = min(y2,b2)
            iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
            inter = iw*ih
            if inter <= 0:
                return 0.0
            area1 = max(0, x2-x1) * max(0, y2-y1)
            area2 = max(0, a2-a1) * max(0, b2-b1)
            union = area1 + area2 - inter + 1e-6
            return float(inter/union)

    # ---- Quality metrics ----
    def _compute_quality(self, frame: np.ndarray, bbox: Tuple[int,int,int,int]) -> Tuple[float, Dict[str, float]]:
        try:
            x1,y1,x2,y2 = bbox
            h, w = frame.shape[:2]
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                return 0.0, {'blur_var': 0.0, 'brightness': 0.0, 'area_frac': 0.0}
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop is not None and crop.size > 0 else None
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
            brightness = float(np.mean(gray))/255.0 if gray is not None else 0.0
            area_frac = float((x2-x1)*(y2-y1)) / float(max(1, w*h))
            # Simple normalized subscores
            blur_score = max(0.0, min(1.0, blur_var / max(self.min_blur_var, 1.0)))
            bright_score = 1.0 if (self.min_brightness <= brightness <= self.max_brightness) else 0.0
            size_score = max(0.0, min(1.0, area_frac / max(self.min_face_area_frac, 1e-6)))
            score = float(0.5 * blur_score + 0.2 * bright_score + 0.3 * size_score)
            return score, {'blur_var': blur_var, 'brightness': brightness, 'area_frac': area_frac}
        except Exception:
            return 0.0, {'blur_var': 0.0, 'brightness': 0.0, 'area_frac': 0.0}

    def start(self, cam_ids: Optional[List[int]] = None):
        with get_session() as db:
            cams = db.query(Camera).all()
        cams_map = {c.id: c for c in cams}
        targets = list(cams_map.keys()) if not cam_ids else [cid for cid in cam_ids if cid in cams_map]
        for cid in targets:
            # Capture thread
            if not (cid in self._cap_threads and self._cap_threads[cid].is_alive()):
                cap_stop = threading.Event()
                self._cap_stops[cid] = cap_stop
                th_cap = threading.Thread(target=self._run_camera, args=(cid, cams_map[cid], cap_stop), daemon=True)
                self._cap_threads[cid] = th_cap
                th_cap.start()
            # Inference thread
            if not (cid in self._proc_threads and self._proc_threads[cid].is_alive()):
                proc_stop = threading.Event()
                self._proc_stops[cid] = proc_stop
                th_proc = threading.Thread(target=self._run_inference, args=(cid, proc_stop), daemon=True)
                self._proc_threads[cid] = th_proc
                th_proc.start()
        print(f"[AI] Started for cameras: {targets}")

    def stop(self):
        for evt in list(self._cap_stops.values()):
            evt.set()
        for th in list(self._cap_threads.values()):
            if th.is_alive():
                th.join(timeout=2.0)
        for evt in list(self._proc_stops.values()):
            evt.set()
        for th in list(self._proc_threads.values()):
            if th.is_alive():
                th.join(timeout=2.0)
        self._cap_threads.clear(); self._cap_stops.clear()
        self._proc_threads.clear(); self._proc_stops.clear()
        print("[AI] Stopped all camera workers")

    def is_running(self) -> bool:
        caps = any(th.is_alive() for th in self._cap_threads.values()) if self._cap_threads else False
        procs = any(th.is_alive() for th in self._proc_threads.values()) if self._proc_threads else False
        return caps or procs

    def is_camera_running(self, cam_id: int) -> bool:
        th_cap = self._cap_threads.get(cam_id)
        th_proc = self._proc_threads.get(cam_id)
        return (th_cap.is_alive() if th_cap else False) or (th_proc.is_alive() if th_proc else False)

    def stop_camera(self, cam_id: int) -> None:
        """Stop capture and inference threads for a single camera."""
        # signal stops
        evt_cap = self._cap_stops.pop(cam_id, None)
        evt_proc = self._proc_stops.pop(cam_id, None)
        if evt_cap:
            evt_cap.set()
        if evt_proc:
            evt_proc.set()
        # join threads
        th_cap = self._cap_threads.pop(cam_id, None)
        th_proc = self._proc_threads.pop(cam_id, None)
        if th_cap and th_cap.is_alive():
            th_cap.join(timeout=2.0)
        if th_proc and th_proc.is_alive():
            th_proc.join(timeout=2.0)
        # drop latest frame to free memory
        try:
            with self._frame_lock:
                if cam_id in self._latest_frames:
                    self._latest_frames.pop(cam_id, None)
        except Exception:
            pass
        print(f"[AI] Stopped camera {cam_id}")

    def _open_capture(self, src: str):
        try:
            s = (src or '').strip()
            # Prefer TCP and smaller buffers for RTSP
            try:
                if s.lower().startswith('rtsp://'):
                    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp|stimeout;5000000|buffer_size;102400')
            except Exception:
                pass
            if s.lower().startswith('webcam:'):
                idx = int(s.split(':', 1)[1])
                cap = cv2.VideoCapture(idx, getattr(cv2, 'CAP_DSHOW', 0))
            elif s.isdigit():
                cap = cv2.VideoCapture(int(s), getattr(cv2, 'CAP_DSHOW', 0))
            else:
                cap = cv2.VideoCapture(s)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # For webcams request MJPG to reduce latency
            try:
                if s.lower().startswith('webcam:') or s.isdigit():
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            return cap
        except Exception:
            return None

    def _run_camera(self, cam_id: int, cam: Camera, stop_evt: threading.Event):
        src = cam.rtsp_url or ''
        cap = self._open_capture(src)
        if not cap or not cap.isOpened():
            print(f"[AI] Unable to open camera {cam_id}")
            return
        try:
            interval = 1.0 / float(self.fps_target)
            frame_idx = 0
            fail_count = 0
            while not stop_evt.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fail_count += 1
                    # If repeated failures, try reconnect
                    if fail_count >= 10:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        cap = self._open_capture(src)
                        fail_count = 0
                        if not cap or not cap.isOpened():
                            time.sleep(0.5)
                            continue
                    else:
                        time.sleep(0.05)
                        continue
                else:
                    fail_count = 0
                # publish latest raw frame for UI streaming
                try:
                    with self._frame_lock:
                        self._latest_frames[cam_id] = frame
                except Exception:
                    pass
                time.sleep(interval)
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _run_inference(self, cam_id: int, stop_evt: threading.Event):
        # Consume the latest available frame without blocking capture; drop frames if lagging
        interval = 1.0 / float(max(1, self.fps_target))
        frame_idx = 0
        while not stop_evt.is_set():
            # snapshot latest frame
            frame = None
            try:
                with self._frame_lock:
                    frm = self._latest_frames.get(cam_id)
                    if frm is not None:
                        frame = frm.copy()
            except Exception:
                frame = None
            if frame is not None:
                frame_idx += 1
                if (frame_idx % self.annotation_stride) == 0:
                    try:
                        self._process_frame(cam_id, frame)
                    except Exception as e:
                        print(f"[AI] Inference error cam {cam_id}: {e}")
            time.sleep(interval)

    def _process_frame(self, cam_id: int, frame: np.ndarray):
        # Refresh known embeddings periodically
        self.emb_store.load()
        faces = self.engine.get_faces(frame)
        now = dt.datetime.utcnow()
        # Build detection list
        dets: List[Tuple[Tuple[int,int,int,int], Optional[int], float, float]] = []  # (bbox, emp_id or None, sim, quality)
        for f in (faces or []):
            bbox = getattr(f, 'bbox', None)
            if bbox is None:
                continue
            try:
                x1,y1,x2,y2 = [int(v) for v in bbox]
            except Exception:
                continue
            # quality gating
            q_score, _ = self._compute_quality(frame, (x1,y1,x2,y2))
            emp_id = None
            sim = 0.0
            emb = FaceEngine.get_embedding(f)
            if emb is not None:
                e_id, s = self.emb_store.best_match(emb)
                if e_id is not None and s >= self.sim_thresh and q_score >= self.min_quality_score:
                    emp_id, sim = e_id, s
            dets.append(((x1,y1,x2,y2), emp_id, sim, q_score))
        # Update tracks with detections
        self._update_tracks_with_dets(cam_id, dets, now)
        # After processing, apply timeouts for presence DB
        self._update_timeouts(now)

    def _should_emit_alert(self, emp_id: int, alert_type: str, ts: dt.datetime, min_interval_sec: int = 60) -> bool:
        """Return True if we should emit alert for (emp, type) considering local debounce window."""
        try:
            key = (int(emp_id), str(alert_type).upper())
            last = self._last_alert_ts.get(key)
            if last is not None:
                try:
                    if (ts - last).total_seconds() < float(min_interval_sec):
                        return False
                except Exception:
                    # if time math fails, allow emit
                    pass
            # record tentative emit time; caller should keep it on success
            self._last_alert_ts[key] = ts
            return True
        except Exception:
            return True

    def _update_tracks_with_dets(self, cam_id: int, dets: List[Tuple[Tuple[int,int,int,int], Optional[int], float, float]], now: dt.datetime):
        tracks = self._tracks.setdefault(cam_id, {})
        next_id = self._next_track_id.setdefault(cam_id, 1)
        # Associate by IOU
        unmatched = set(range(len(dets)))
        # For each track, find best det above threshold
        assignments: List[Tuple[int,int]] = []  # (track_id, det_idx)
        for tid, tr in list(tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for j in list(unmatched):
                iou = tr.iou(dets[j][0])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0 and best_iou >= self.iou_match_threshold:
                assignments.append((tid, best_idx))
                unmatched.discard(best_idx)
            else:
                # miss this frame
                tr.misses += 1
        # Update matched tracks
        for tid, j in assignments:
            tr = tracks.get(tid)
            if tr is None:
                continue
            bbox, emp_id, sim, q = dets[j]
            tr.bbox = bbox
            tr.last_ts = now
            tr.hits += 1
            tr.misses = 0
            # add vote if recognized
            if emp_id is not None:
                tr.votes.append(emp_id)
                # smoothing: majority over window
                maj_id = None
                if tr.votes:
                    cnt = Counter(tr.votes)
                    maj_id, maj_c = cnt.most_common(1)[0]
                    if maj_id is not None and maj_c >= max(1, self.smooth_min_votes):
                        # update final id
                        tr.final_emp_id = maj_id
                        if tr.final_since is None:
                            tr.final_since = now
                        # Emit presence update each time to keep alive
                        self._on_employee_seen(maj_id, cam_id, now, sim)
        # Create new tracks for unmatched detections
        for j in list(unmatched):
            bbox, emp_id, sim, q = dets[j]
            tid = next_id
            next_id += 1
            tr = self.Track(tid, bbox, now)
            if emp_id is not None:
                tr.votes.append(emp_id)
            tracks[tid] = tr
        self._next_track_id[cam_id] = next_id
        # Cleanup stale tracks
        to_del = []
        for tid, tr in tracks.items():
            if tr.misses > self.max_track_misses:
                to_del.append(tid)
        for tid in to_del:
            tracks.pop(tid, None)

    def _on_employee_seen(self, emp_id: int, cam_id: int, ts: dt.datetime, sim: float):
        with get_session() as db:
            try:
                # Exclude inactive employees from tracking and attendance first/last updates
                emp_row = db.query(Employee).filter(Employee.id == emp_id).first()
                if emp_row is not None and (emp_row.is_active is False):
                    # Option C: Always set today's attendance to ABSENT and clear timestamps
                    try:
                        today = ts.date()
                        att = db.query(Attendance).filter(Attendance.employee_id == emp_id, Attendance.date == today).first()
                        if att is None:
                            att = Attendance(employee_id=emp_id, date=today)
                            db.add(att)
                        att.first_in_ts = None
                        att.last_out_ts = None
                        att.status = 'ABSENT'
                        db.commit()
                    except Exception:
                        db.rollback()
                    return  # do not log events/presence for inactive employees
                # Insert Event only if beyond min interval per emp+cam
                do_insert = True
                key = (emp_id, cam_id)
                last_ts = self._last_event_ts.get(key)
                if last_ts is not None:
                    try:
                        if (ts - last_ts).total_seconds() < self.event_min_interval:
                            do_insert = False
                    except Exception:
                        do_insert = True
                if do_insert:
                    evt = Event(employee_id=emp_id, camera_id=cam_id, timestamp=ts, similarity_score=sim)
                    db.add(evt)
                    # update memory only when we actually log an event
                    self._last_event_ts[key] = ts
                # Presence upsert
                pres = db.query(Presence).filter(Presence.employee_id == emp_id).first()
                if pres is None:
                    pres = Presence(employee_id=emp_id, status='available', last_seen_ts=ts, last_camera_id=cam_id)
                    db.add(pres)
                else:
                    # If previously off, log a RESOLVED alert with correct absence duration
                    try:
                        if (pres.status or 'off') != 'available' and _alerts_allowed():
                            # Determine absence start: prefer Attendance.last_out_ts, fallback to Presence.last_seen_ts
                            try:
                                # resolve employee name
                                name = None
                                try:
                                    if 'emp_row' not in locals() or emp_row is None:
                                        emp_row = db.query(Employee).filter(Employee.id == emp_id).first()
                                    name = (emp_row.name if emp_row else None) or f"Employee {emp_id}"
                                except Exception:
                                    name = f"Employee {emp_id}"
                                absence_start = None
                                # today's attendance record
                                today = ts.date()
                                att = db.query(Attendance).filter(Attendance.employee_id == emp_id, Attendance.date == today).first()
                                if att and att.last_out_ts:
                                    absence_start = att.last_out_ts
                                elif pres.last_seen_ts:
                                    absence_start = pres.last_seen_ts
                                mins = 0
                                if absence_start is not None:
                                    down_sec = max(0, int((ts - absence_start).total_seconds()))
                                    # round up to next minute to avoid "0 min" for short absences
                                    mins = int(math.ceil(down_sec / 60.0)) if down_sec > 0 else 0
                                msg = f"{name} back to area after {mins} min"
                            except Exception:
                                msg = f"Employee {emp_id} back to area after 0 min"
                            # Avoid duplicate ENTER logs: check DB since last seen and debounce locally
                            should_emit = self._should_emit_alert(emp_id, 'RESOLVED', ts, 60)
                            if should_emit:
                                try:
                                    exists = db.query(AlertLog).filter(
                                        AlertLog.employee_id == emp_id,
                                        AlertLog.alert_type == 'RESOLVED',
                                        AlertLog.timestamp >= (pres.last_seen_ts or ts - dt.timedelta(days=1))
                                    ).first()
                                except Exception:
                                    exists = None
                                if not exists:
                                    db.add(AlertLog(employee_id=emp_id, timestamp=ts, alert_type='RESOLVED', message=msg))
                    except Exception:
                        pass
                    pres.status = 'available'
                    pres.last_seen_ts = ts
                    pres.last_camera_id = cam_id
                # Attendance upsert (today)
                today = ts.date()
                att = db.query(Attendance).filter(Attendance.employee_id == emp_id, Attendance.date == today).first()
                if att is None:
                    att = Attendance(employee_id=emp_id, date=today, first_in_ts=ts, status='PRESENT')
                    db.add(att)
                else:
                    if att.first_in_ts is None:
                        att.first_in_ts = ts
                    att.status = att.status or 'PRESENT'
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"[AI] DB error on seen: {e}")
        # Update in-memory
        with self._state_lock:
            self.last_seen[emp_id] = ts
            self.last_cam[emp_id] = cam_id

    def _update_timeouts(self, now: dt.datetime):
        cutoff = now - dt.timedelta(seconds=self.track_timeout)
        expired: List[int] = []
        with self._state_lock:
            for emp_id, ts in list(self.last_seen.items()):
                if ts < cutoff:
                    expired.append(emp_id)
        if not expired:
            return
        with get_session() as db:
            for emp_id in expired:
                try:
                    pres = db.query(Presence).filter(Presence.employee_id == emp_id).first()
                    if pres is not None:
                        last_seen = pres.last_seen_ts
                        secs = None
                        if last_seen is not None:
                            try:
                                secs = max(0, int((now - last_seen).total_seconds()))
                            except Exception:
                                secs = None
                        # Transition to off: update once
                        if (pres.status or 'off') != 'off':
                            pres.status = 'off'
                            # For inactive employees: Option C -> set ABSENT and clear timestamps (override)
                            emp_row = db.query(Employee).filter(Employee.id == emp_id).first()
                            today = now.date()
                            if emp_row is not None and (emp_row.is_active is False):
                                att = db.query(Attendance).filter(Attendance.employee_id == emp_id, Attendance.date == today).first()
                                if att is None:
                                    att = Attendance(employee_id=emp_id, date=today)
                                    db.add(att)
                                att.first_in_ts = None
                                att.last_out_ts = None
                                att.status = 'ABSENT'
                            else:
                                # Active employee: update last_out_ts
                                att = db.query(Attendance).filter(Attendance.employee_id == emp_id, Attendance.date == today).first()
                                if att is None:
                                    att = Attendance(employee_id=emp_id, date=today, last_out_ts=now, status='PRESENT')
                                    db.add(att)
                                else:
                                    att.last_out_ts = now
                            # If already >=60s absent at this moment, log alert once
                            if secs is not None and secs >= 60 and _alerts_allowed():
                                try:
                                    exists = db.query(AlertLog).filter(
                                        AlertLog.employee_id == emp_id,
                                        AlertLog.alert_type == 'EXIT',
                                        AlertLog.timestamp >= (pres.last_seen_ts or now - dt.timedelta(days=1))
                                    ).first()
                                    if not exists:
                                        mins = secs // 60
                                        # Use employee name for EXIT messages
                                        try:
                                            emp = db.query(Employee).filter(Employee.id == emp_id).first()
                                            emp_name = (emp.name if emp else None) or f"Employee {emp_id}"
                                        except Exception:
                                            emp_name = f"Employee {emp_id}"
                                        msg = f"{emp_name} out of area since {mins} min ago"
                                        if self._should_emit_alert(emp_id, 'EXIT', now, 60):
                                            db.add(AlertLog(employee_id=emp_id, timestamp=now, alert_type='EXIT', message=msg))
                                except Exception:
                                    pass
                        else:
                            # Already off: if just crossed 60s and no alert yet for this absence, log once
                            if secs is not None and secs >= 60 and _alerts_allowed():
                                try:
                                    exists = db.query(AlertLog).filter(
                                        AlertLog.employee_id == emp_id,
                                        AlertLog.alert_type == 'EXIT',
                                        AlertLog.timestamp >= (pres.last_seen_ts or now - dt.timedelta(days=1))
                                    ).first()
                                    if not exists:
                                        mins = secs // 60
                                        # Use employee name for EXIT messages
                                        try:
                                            emp = db.query(Employee).filter(Employee.id == emp_id).first()
                                            emp_name = (emp.name if emp else None) or f"Employee {emp_id}"
                                        except Exception:
                                            emp_name = f"Employee {emp_id}"
                                        msg = f"{emp_name} out of area since {mins} min ago"
                                        if self._should_emit_alert(emp_id, 'EXIT', now, 60):
                                            db.add(AlertLog(employee_id=emp_id, timestamp=now, alert_type='EXIT', message=msg))
                                except Exception:
                                    pass
                            # Ensure inactive employees keep ABSENT status for today
                            try:
                                emp_row = db.query(Employee).filter(Employee.id == emp_id).first()
                                if emp_row is not None and (emp_row.is_active is False):
                                    today = now.date()
                                    att = db.query(Attendance).filter(Attendance.employee_id == emp_id, Attendance.date == today).first()
                                    if att is None:
                                        att = Attendance(employee_id=emp_id, date=today)
                                        db.add(att)
                                    att.first_in_ts = None
                                    att.last_out_ts = None
                                    att.status = 'ABSENT'
                            except Exception:
                                pass
                    db.commit()
                except Exception as e:
                    db.rollback()
                    print(f"[AI] DB error on timeout: {e}")
                # Keep last_seen so we can detect 60s threshold and avoid duplicate scheduling. Do not pop here.

    # ---- Public state API ----
    def get_state(self) -> Dict[str, Any]:
        # Build current state: present/alert based on last_seen within 1 minute
        with get_session() as db:
            pres_rows = db.query(Presence).all()
            cam_map = {c.id: c for c in db.query(Camera).all()}
            # Map active employees for filtering cards
            active_emp = {e.id: e for e in db.query(Employee).filter(Employee.is_active == True).all()}
        now = dt.datetime.utcnow()
        THRESH = 60  # seconds threshold for present vs alert
        items = []
        present_count = 0
        alert_count = 0
        for p in pres_rows:
            # Only show active employees that still exist
            if p.employee_id not in active_emp:
                continue
            meta = self.emb_store.employee_meta.get(p.employee_id, {})
            last_seen_ts = p.last_seen_ts
            # Serialize UTC with Z so frontend converts to WIB/local correctly
            last_seen_iso = (last_seen_ts.isoformat() + 'Z') if last_seen_ts else None
            seconds_since = None
            if last_seen_ts is not None:
                try:
                    seconds_since = max(0, int((now - last_seen_ts).total_seconds()))
                except Exception:
                    seconds_since = None
            is_present = (seconds_since is not None and seconds_since <= THRESH)
            if is_present:
                present_count += 1
            else:
                alert_count += 1
            cam_name = cam_map.get(p.last_camera_id).name if p.last_camera_id and p.last_camera_id in cam_map else None
            items.append({
                'employee_id': p.employee_id,
                'name': meta.get('name'),
                'department': meta.get('department'),
                'status': 'available' if is_present else 'off',
                'last_seen': last_seen_iso,
                'seconds_since': seconds_since,
                'is_present': is_present,
                'camera_id': p.last_camera_id,
                'camera_name': cam_name,
            })
        # total employees tracked on cards = present + alerts (not all registered)
        total_cards = present_count + alert_count
        return {
            'running': self.is_running(),
            'present': present_count,
            'alerts': alert_count,
            'total': total_cards,
            'employees': sorted(items, key=lambda x: (not x['is_present'], (x['seconds_since'] or 1e9), x.get('name') or '')),
        }

    # ---- Visualization helper (no DB writes) ----
    def annotate_frame(self, frame: np.ndarray, cam_id: Optional[int] = None) -> np.ndarray:
        try:
            img = frame.copy()
        except Exception:
            return frame
        try:
            # Refresh embeddings lazily
            self.emb_store.load()
            faces = self.engine.get_faces(img)
            if not faces:
                return img
            for f in faces:
                try:
                    bbox = getattr(f, 'bbox', None)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    # default Unknown = red
                    color = (0, 0, 255)
                    text_color = (0, 0, 255)
                    label = 'Unknown'
                    emb = FaceEngine.get_embedding(f)
                    if emb is not None:
                        emp_id, sim = self.emb_store.best_match(emb)
                        if emp_id is not None and sim >= self.sim_thresh:
                            meta = self.emb_store.employee_meta.get(emp_id, {})
                            name = meta.get('name') or f"ID {emp_id}"
                            # Show ID and Name only
                            label = f"ID {emp_id} - {name}"
                            # recognized = green
                            color = (0, 255, 0)
                            text_color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    # background for text
                    txt = label
                    (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    ty1 = max(0, y1 - th - 6)
                    cv2.rectangle(img, (x1, ty1), (x1 + tw + 6, ty1 + th + 6), (0, 0, 0), -1)
                    # draw text in color (green if recognized, red if unknown)
                    cv2.putText(img, txt, (x1 + 3, ty1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                except Exception:
                    continue
            return img
        except Exception:
            return frame

    # ---- Streaming preferences for UI pipeline ----
    def get_stream_preferences(self) -> Dict[str, int]:
        """Expose UI streaming preferences with keys expected by app.py.

        Returns:
            dict: {
                'max_width': int,
                'jpeg_quality': int,
                'annotation_stride': int,
                'target_fps': int,
            }
        """
        try:
            return {
                'max_width': int(self.stream_max_width),
                'jpeg_quality': int(self.jpeg_quality),
                'annotation_stride': int(self.annotation_stride),
                'target_fps': int(self.fps_target),
            }
        except Exception:
            # Fallback defaults
            return {
                'max_width': 960,
                'jpeg_quality': 70,
                'annotation_stride': 3,
                'target_fps': 20,
            }

    # ---- Frames API for UI streaming ----
    def get_latest_frame(self, cam_id: int) -> Optional[np.ndarray]:
        """Return a copy of the most recent raw frame from a camera if available."""
        try:
            with self._frame_lock:
                frm = self._latest_frames.get(cam_id)
                if frm is None:
                    return None
                return frm.copy()
        except Exception:
            return None


# Singleton manager
ai_manager = TrackingManager()
