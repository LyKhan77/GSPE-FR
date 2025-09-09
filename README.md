# GSPE Auto-Monitoring System

A real-time employee monitoring system using CCTV cameras and face recognition technology. The system provides live video streaming, automatic employee presence tracking, alert notifications, and comprehensive attendance management.

## Features

### 🎥 Live CCTV Streaming
- Real-time video streaming from multiple cameras
- Socket.IO-based frame transmission for low latency
- Dynamic camera switching and management
- AI inference status indicator (Online/Offline)

### 👤 Face Recognition & Tracking
- Advanced face detection and recognition using InsightFace
- Employee embedding storage and matching
- Real-time presence status updates
- Tracking-by-detection with IOU association to stabilize IDs and boxes
- Temporal ID smoothing (majority vote) to reduce flicker/switching
- Configurable similarity thresholds
- Face quality gating (blur, brightness, size) before casting votes

### 🚨 Smart Alert System
- Automatic alert generation when employees are absent >60 seconds
- Persistent alert logging in database
- Real-time notification dropdown with badge counter
- Alert resolution tracking when employees return
 - Event rate control to prevent database spam for repeated sightings

### 📊 Attendance Management
- Daily attendance tracking (first-in, last-out timestamps)
- Employee presence status (Available/Off)
- Comprehensive event logging
- Daily data maintenance and cleanup

### 🖥️ Web Dashboard
- Modern responsive UI built with Tailwind CSS
- Live employee tracking panel
- Camera management interface
- Employee registration with face capture
- Settings and configuration pages
 - Report page with Reset Logs (Events/Alert Logs) by date range

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Engine    │
│   (HTML/JS)     │◄──►│   (Flask)       │◄──►│   (module_AI)   │
│                 │    │                 │    │                 │
│ • Live Stream   │    │ • REST APIs     │    │ • Face Detect   │
│ • Notifications │    │ • Socket.IO     │    │ • Recognition   │
│ • Management    │    │ • Maintenance   │    │ • Tracking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Database      │
                       │   (SQLite)      │
                       │                 │
                       │ • Employees     │
                       │ • Events        │
                       │ • Attendance    │
                       │ • Alerts        │
                       └─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- OpenCV compatible camera or RTSP stream
- Modern web browser

### Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- Flask & Flask-SocketIO
- SQLAlchemy
- OpenCV (cv2)
- InsightFace
- NumPy

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd FR-V3
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize database**
```bash
python database_models.py
```

4. **Configure cameras**
   - Create camera configs in `camera_configs/CAM{ID}/config.json`
   - Example config:
```json
{
    "id": 1,
    "name": "Main Entrance",
    "rtsp_url": "rtsp://192.168.1.100:554/stream",
    "enabled": true,
    "location": "Entrance Zone"
}
```

Additional/advanced parameters supported (optional):

- Tracking & smoothing
  - `smoothing_window` (int, default 5)
  - `smoothing_min_votes` (int, default 3)
  - `tracker_iou_threshold` (float, default 0.3)
  - `tracker_max_misses` (int, default 8)
- Event rate control
  - `event_min_interval_sec` (float seconds, default 5.0)
- Quality gating
  - `quality_min_blur_var` (float, default 50.0)
  - `quality_min_face_area_frac` (float, default 0.01)
  - `quality_min_brightness` (float 0..1, default 0.15)
  - `quality_max_brightness` (float 0..1, default 0.9)
  - `quality_min_score` (float 0..1, default 0.3)

5. **Configure AI parameters** (optional)
   - Edit `parameter_config.json` for detection thresholds and settings

6. **Run the application**
```bash
python app.py
```

7. **Access the dashboard**
   - Open browser: `http://localhost:5000`

## Configuration

### Camera Configuration
Cameras are configured via JSON files in `camera_configs/` directory:

```
camera_configs/
├── CAM1/
│   └── config.json
├── CAM2/
│   └── config.json
└── CAM3/
    └── config.json
```

### AI Parameters
Configure face recognition settings in `parameter_config.json`:

```json
{
    "face_recognition": {
        "similarity_threshold": 0.5,
        "detection_confidence": 0.8,
        "tracking_timeout": 10
    },
    "camera": {
        "frame_width": 640,
        "frame_height": 480,
        "fps": 15
    }
}
```

## Usage

### Employee Registration
1. Navigate to Settings → Manage Employee
2. Click "Add Employee"
3. Fill employee details
4. Use face capture to register employee's face
5. Save to complete registration

### Camera Management
1. Go to Settings → Manage Camera
2. Add new cameras with RTSP URLs
3. Enable/disable cameras as needed
4. Test camera connections

### Live Monitoring
1. Select camera from CCTV page
2. Monitor AI inference status (green = online, red = offline)
3. View real-time employee tracking
4. Check notifications for alerts

### Report: Reset Logs
1. Buka halaman Report pada dashboard.
2. Klik tombol "Reset Logs".
3. Pilih tabel (Both / Events only / Alert Logs only).
4. Opsional: pilih From dan To Date (YYYY-MM-DD). Kosongkan untuk hapus semua.
5. Konfirmasi. Sistem akan menampilkan jumlah baris yang terhapus.

### Alert Management
- Alerts automatically generated when employees absent >60s
- View current alerts in notification dropdown
- Alert history stored in database
- Alerts resolved when employees return

## Database Schema

### Core Tables
- **employees**: Employee master data
- **face_templates**: Face recognition embeddings
- **cameras**: Camera configuration
- **events**: Detection event logs
- **presence**: Real-time presence status
- **attendances**: Daily attendance records
- **alert_logs**: Alert notification history

### Key Relationships
```sql
Employee 1:N FaceTemplate
Employee 1:N Event
Employee 1:1 Presence
Employee 1:N Attendance
Employee 1:N AlertLog
Camera 1:N Event
```

## API Endpoints

### REST APIs
- `GET /api/cameras` - List all cameras
- `GET /api/employees` - List all employees
- `POST /api/employees` - Create new employee
- `PUT /api/employees/{id}` - Update employee
- `DELETE /api/employees/{id}` - Delete employee
- `GET /api/tracking/state` - Get current tracking state

#### Admin
- `POST /api/admin/reset_logs` - Delete events and/or alert_logs by date range
  - Request JSON:
    ```json
    { "table": "events"|"alert_logs"|"both", "from_date": "YYYY-MM-DD", "to_date": "YYYY-MM-DD" }
    ```
  - Notes:
    - If `from_date`/`to_date` omitted, deletes all rows in selected table(s).
    - Dates are inclusive; span entire days.
  - Response JSON:
    ```json
    { "ok": true, "deleted_events": 123, "deleted_alert_logs": 45 }
    ```

### Socket.IO Events
- `start_stream` - Start camera streaming
- `stop_stream` - Stop camera streaming
- `frame` - Receive video frame
- `stream_error` - Stream error notification
- `stream_stopped` - Stream stopped notification

## Maintenance

### Daily Maintenance
The system automatically performs daily maintenance:
- Purges old Event records (keeps only current day)
- Runs at startup and scheduled at midnight
- Prevents database bloat

### Manual Maintenance
```bash
# Reinitialize database
python database_models.py

# Check system status
# Monitor logs in console output
```

## Troubleshooting

### Common Issues

**Camera not connecting:**
- Verify RTSP URL and credentials
- Check network connectivity
- Ensure camera supports H.264 encoding

**Face recognition not working:**
- Check lighting conditions
- Verify face template registration
- Adjust similarity threshold in config

**Performance issues:**
- Reduce camera resolution/FPS
- Check CPU/memory usage
- Consider hardware acceleration

**Database errors:**
- Check file permissions for `attendance.db`
- Ensure SQLite is properly installed
- Backup and reinitialize if corrupted

### Logs
Monitor console output for:
- Camera connection status
- Face recognition events
- Database operations
- Error messages

## Development

### Project Structure
```
FR-V3/
├── app.py                 # Main Flask application
├── module_AI.py          # AI/Face recognition engine
├── database_models.py    # Database models and ORM
├── parameter_config.json # AI configuration
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Frontend UI
├── camera_configs/      # Camera configurations
├── face_images/         # Employee face images
└── attendance.db        # SQLite database
```

### Adding New Features
1. Update database models in `database_models.py`
2. Add API endpoints in `app.py`
3. Implement AI logic in `module_AI.py`
4. Update frontend in `templates/index.html`

### Testing
- Test camera connections via Settings page
- Verify face recognition with known employees
- Check alert generation and resolution
- Monitor database operations

## Security Considerations

- RTSP credentials stored in database (not exposed to frontend)
- No authentication implemented (add as needed)
- SQLite file permissions should be restricted
- Consider HTTPS for production deployment

## Performance Optimization

- Use hardware acceleration for video processing
- Implement frame skipping for high FPS cameras
- Consider Redis for session management in multi-instance setup
- Database indexing on frequently queried columns

## License

[Specify your license here]

## Support

For technical support or questions:
- Check troubleshooting section
- Review console logs
- Contact development team

## Changelog

### Version 3.0
- Real-time alert logging and resolution
- Daily database maintenance
- Improved AI status indicator
- Enhanced notification system
- Modern responsive UI

### Version 3.1
- Added IOU-based tracking-by-detection and temporal ID smoothing
- Implemented face quality gating (blur/brightness/size) before voting
- Added event rate control to reduce Event table spam
- New admin API and UI to reset Events/Alert Logs by date range

---

**GSPE Auto-Monitoring System** - Intelligent employee monitoring with face recognition technology.
