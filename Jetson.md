# Deploying FR-V3 on NVIDIA Jetson Orin Nano

This guide explains how to run the Face Recognition WebApp on a Jetson Orin Nano. It focuses on Jetson-specific dependencies and configuration.

## Prerequisites

- JetPack (L4T) for Orin Nano installed (via NVIDIA SDK Manager) with:
  - CUDA, cuDNN, TensorRT
  - GStreamer and NVIDIA multimedia stack
- Internet access for first model download (or pre-seed models)
- Optional: elevated power profile for real-time inference
  - `sudo nvpmodel -m 2 && sudo jetson_clocks`

## Python version

- Recommended: Python 3.8–3.10 (use the default Python that ships with your JetPack if possible)

## Create and activate a virtual environment

```bash
python3 -m venv ~/venvs/frv3 && source ~/venvs/frv3/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

## Jetson-specific dependency guidance

On Jetson/aarch64, prefer system/Jetson-optimized builds for performance and compatibility.

- OpenCV: install from apt (GStreamer-enabled) instead of PyPI
- ONNX Runtime GPU: install the aarch64 wheel compatible with your JetPack
- InsightFace: install from PyPI; it will use the ONNX Runtime providers you supply

### Core runtime

Install system OpenCV first:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
```

Then install the Python packages (avoid installing `opencv-python` from PyPI):

```bash
pip install --no-cache-dir \
  insightface==0.7.3 \
  numpy==1.24.3 \
  SQLAlchemy==2.0.20 \
  Pillow==11.3.0 \
  scikit-learn==1.7.1 \
  scipy==1.15.3 \
  coloredlogs==15.0.1 \
  Flask==2.3.3 \
  Flask-SocketIO==5.3.5 \
  Flask-Cors==4.0.0 \
  python-socketio==5.7.2 \
  python-engineio==4.4.1 \
  eventlet==0.33.3 \
  requests==2.32.4
```

### ONNX Runtime GPU (Jetson)

You need a Jetson-compatible `onnxruntime-gpu` wheel (TensorRT/CUDA EPs). The exact version depends on your JetPack. NVIDIA provides wheels via forums and Jetson Zoo. As an example (adjust the URL/version for your JetPack):

```bash
# Example only: replace with the correct wheel for your JetPack/L4T
pip install --no-cache-dir onnxruntime-gpu==1.16.0
```

Notes:
- If a matching GPU wheel is not available, you can temporarily use CPU `onnxruntime`, but inference will be slower.
- To see provider initialization logs, set: `export ORT_LOG_SEVERITY_LEVEL=1`.

### Optional: TensorRT engine cache and FP16

```bash
export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
export ORT_TENSORRT_FP16_ENABLE=1
```

## Configure providers for Jetson

This project reads providers from `parameter_config.json` and passes them to `insightface.FaceAnalysis`.

Recommended `parameter_config.json` for Jetson:

```json
{
  "providers": "TensorrtExecutionProvider,CUDAExecutionProvider",
  "detection_size": [640, 640],
  "recognition_threshold": 0.65,
  "embedding_similarity_threshold": 0.45,
  "tracking_timeout": 10.0,
  "fps_target": 15,
  "quality_min_blur_var": 50.0,
  "quality_min_face_area_frac": 0.01,
  "quality_min_brightness": 0.15,
  "quality_max_brightness": 0.9,
  "quality_min_score": 0.3,
  "event_min_interval_sec": 5.0
}
```

If TensorRT EP fails to initialize with your ORT build, start with just CUDA:

```json
{
  "providers": "CUDAExecutionProvider"
}
```

## Camera/RTSP capture tips

- The default RTSP over TCP used by `module_AI.py` often works well.
- If you need lower latency and your OpenCV was built with GStreamer (apt install path), consider a GStreamer pipeline.
- Example RTSP H.264 pipeline (replace URL/credentials):

```text
rtspsrc location=rtsp://user:pass@CAM/stream latency=100 protocols=tcp ! \
  rtph264depay ! h264parse ! avdec_h264 ! \
  videoconvert ! appsink drop=true sync=false
```

You can add a small enhancement in `module_AI.py::_open_capture` to support a `gst:` prefix and open the pipeline when detected.

## Model cache

InsightFace downloads models to `~/.insightface/models` on first use. Ensure the Jetson user has write permission, or pre-copy models from a development machine to the same path on Jetson.

## Running the app

```bash
# From project root
source ~/venvs/frv3/bin/activate
python app.py
```

Open the web UI in a browser pointed at the Jetson’s IP and the app port (default in `app.py`).

## Verification

- Console should show something like:

```
[AI] FaceAnalysis ready. Providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'], det_size=(640, 640)
```

- If it prints a CPU-only init, check your `onnxruntime-gpu` wheel and `providers` config.

- Start a camera stream. Verify:
  - Live detections are flowing.
  - Attendance entries appear.
  - Alert logs show a single EXIT on leaving (≥60s), and a single ENTER on return (UI maps RESOLVED→ENTER; EXIT remains EXIT).

## Troubleshooting

- `FaceAnalysis` falls back to CPU
  - Install the Jetson `onnxruntime-gpu` wheel compatible with your JetPack.
  - Try `CUDAExecutionProvider` only if TensorRT EP fails.

- RTSP not opening
  - Prefer `python3-opencv` from apt (GStreamer-enabled) instead of `opencv-python` wheels.
  - Try a GStreamer pipeline; confirm codec (H.264/H.265) and `protocols=tcp`.

- Performance/FPS too low
  - Reduce `detection_size` to `[512,512]` or `[480,480]`.
  - Lower `fps_target`.
  - Enable `ORT_TENSORRT_FP16_ENABLE=1` if supported.

## Minimal dependency list (Jetson)

Install these (plus `python3-opencv` from apt):

- insightface==0.7.3
- numpy==1.24.3
- SQLAlchemy==2.0.20
- Pillow==11.3.0
- scikit-learn==1.7.1
- scipy==1.15.3
- coloredlogs==15.0.1
- Flask==2.3.3
- Flask-SocketIO==5.3.5
- Flask-Cors==4.0.0
- python-socketio==5.7.2
- python-engineio==4.4.1
- eventlet==0.33.3
- requests==2.32.4
- onnxruntime-gpu (Jetson aarch64 wheel matching your JetPack)

> Note: Do NOT install `opencv-python` from PyPI on Jetson; use `sudo apt-get install python3-opencv`.
