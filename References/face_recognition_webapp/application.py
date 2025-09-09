#!/usr/bin/env python3
# application.py
# Main Application - Web Interface that displays everything to UI

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import time
from datetime import datetime
import threading
import os
from db_manager import db_manager, SessionLocal, Employee, Camera, EmployeeTracking, EmployeeLocation, Attendance
from camera_manager import camera_manager, get_all_camera_configs
from tracking_manager import tracking_manager, notification_manager
from AI_module import ai_processing_module, load_system_specs

# Create Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'your-secret_key_here'
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
CORS(app)

# Active streams tracking
active_streams = {}

# Background thread for monitoring absences and notifications
def background_monitoring():
    """Background thread to check for employee absences and send notifications"""
    while True:
        try:
            # Check for absent employees
            absent_employees = tracking_manager.check_absences()
            
            # Get recent notifications
            notifications = notification_manager.get_notifications()
            
            # Emit notifications to frontend if any
            if notifications:
                socketio.emit('notifications_update', {
                    'notifications': notifications,
                    'count': len(notifications)
                })
            
            # Sleep for 30 seconds before next check
            time.sleep(30)
        except Exception as e:
            print(f"[MONITORING] Error in background monitoring: {e}")
            time.sleep(60)  # Wait longer on error

# Start background monitoring thread
monitoring_thread = threading.Thread(target=background_monitoring, daemon=True)
monitoring_thread.start()
print("[APPLICATION] Background monitoring started")

# Initialize AI processing module
print("[APPLICATION] Initializing AI processing module...")
ai_processing_module.initialize_model()
print("[APPLICATION] AI processing module initialized")

def get_db_session():
    """Get database session"""
    return SessionLocal()

def format_last_seen(last_seen_time):
    """Format last seen time in a human-readable way"""
    if not last_seen_time:
        return "Never"
    
    # Handle different time formats
    if isinstance(last_seen_time, str):
        # If it's already a formatted string, return as is
        return last_seen_time
    
    # Calculate time difference
    if isinstance(last_seen_time, datetime):
        diff = datetime.now() - last_seen_time
    else:
        # Assume it's a timestamp
        diff = datetime.now() - datetime.fromtimestamp(last_seen_time if isinstance(last_seen_time, (int, float)) else time.time())
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} min ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"

def update_employee_status_in_background():
    """Background thread to update employee status and emit to clients"""
    while True:
        try:
            time.sleep(10)  # Update every 10 seconds
            
            # Get all employee statuses from the new tracking system
            status_records = db_manager.get_employee_statuses()
            
            # Emit updates to all connected clients
            status_updates = []
            for record in status_records:
                status_update = {
                    'employeeId': record['employee_id'],  # Use database employee_id for consistency
                    'employeeName': record['employee_name'],
                    'status': record['status'],
                    'lastSeen': format_last_seen(record['last_seen']) if record['last_seen'] else 'Never',
                    'cameraId': record['camera_id'],
                    'location': record['camera_name'] or 'Unknown'
                }
                status_updates.append(status_update)
            
            # Emit to all connected clients
            socketio.emit('employee_status_update', status_updates)
            
        except Exception as e:
            print(f"[APPLICATION] Error in background status update: {e}")

# Serve the main page
@app.route('/')
def index():
    """Serve the main dashboard page"""
    return send_from_directory(app.static_folder, 'index.html')

# Serve static files
@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

# WebSocket events for camera streaming
@socketio.on('connect', namespace='/camera')
def handle_camera_connect():
    """Handle camera client connection"""
    print('[APPLICATION] Camera client connected')

@socketio.on('start_stream', namespace='/camera')
def start_stream(data):
    """Start camera stream - delegate to AI module for background processing"""
    rtsp_url = data.get('rtsp_url')
    if not rtsp_url:
        # Send error frame to client
        socketio.emit('camera_frame', {
            'frame': 'Camera Unavailable',
            'rtsp_url': ''
        }, namespace='/camera')
        return
    
    # Stop any existing streams
    for url in list(active_streams.keys()):
        stop_stream({'rtsp_url': url})
    
    # Define frame callback to emit frames to client
    def frame_callback(frame_base64):
        # Always send frame to client, let frontend handle error detection
        socketio.emit('camera_frame', {
            'frame': frame_base64, 
            'rtsp_url': rtsp_url
        }, namespace='/camera')
    
    # Start face recognition processing in background (delegate to AI module)
    # Use rtsp_url as both camera_id and rtsp_url for consistency
    success = ai_processing_module.start_stream(rtsp_url, rtsp_url, frame_callback)
    
    if success:
        # Update AI status for all clients
        socketio.emit('ai_status_update', {'active': True})
        active_streams[rtsp_url] = True
    else:
        # Send error frame to client
        socketio.emit('camera_frame', {
            'frame': 'Camera Unavailable',
            'rtsp_url': rtsp_url
        }, namespace='/camera')

@socketio.on('stop_stream', namespace='/camera')
def stop_stream(data):
    """Stop camera stream - delegate to AI module"""
    rtsp_url = data.get('rtsp_url')
    print(f"[APPLICATION] Received stop_stream request for RTSP URL: {rtsp_url}")
    if not rtsp_url:
        return
    
    if rtsp_url in active_streams:
        print(f"[APPLICATION] Stopping stream: {rtsp_url}")
        # Stop face recognition processing (delegate to AI module)
        ai_processing_module.stop_stream(rtsp_url)
        
        # Update AI status for all clients
        socketio.emit('ai_status_update', {'active': False})
        del active_streams[rtsp_url]
        print(f"[APPLICATION] Stream stopped: {rtsp_url}")

# API Routes
@app.route('/api/employees')
def get_employees():
    """Get all employees with their current status"""
    try:
        # Get all employees with detailed info
        employees = db_manager.get_all_employees_detailed()
        
        # Get status for each employee
        employees_with_status = []
        for employee in employees:
            # Use database employee_id consistently instead of generating from name
            employee_id = employee.employee_id  # Use database employee_id for consistency
            
            # Get employee status from tracking table
            status_records = db_manager.get_employee_statuses()
            employee_status = next((s for s in status_records if s['employee_id'] == employee.employee_id), None)
            
            employee_info = {
                'id': employee_id,  # Use database employee_id for consistency
                'employeeId': employee.employee_id,
                'name': employee.name,
                'status': employee_status['status'] if employee_status else 'UNAVAILABLE',
                'lastSeen': format_last_seen(employee_status['last_seen']) if employee_status and employee_status['last_seen'] else 'Never',
                'cameraId': employee_status['camera_id'] if employee_status else None,
                'location': employee_status['camera_name'] if employee_status else 'Unknown',
                'department': employee.department or 'Not specified',
                'role': employee.role or 'Not specified'
            }
            employees_with_status.append(employee_info)
        
        return jsonify(employees_with_status)
    
    except Exception as e:
        print(f"[APPLICATION] Error getting employees: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees/status')
def get_employee_status():
    """Get all employee statuses"""
    try:
        statuses = db_manager.get_employee_statuses()
        return jsonify({
            'success': True,
            'data': statuses
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications')
def get_notifications():
    """Get recent notifications"""
    try:
        limit = request.args.get('limit', 10, type=int)
        notifications = notification_manager.get_notifications(limit)
        return jsonify({
            'success': True,
            'data': notifications,
            'count': len(notifications)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/notifications/clear', methods=['POST'])
def clear_notifications():
    """Clear all notifications"""
    try:
        notification_manager.clear_notifications()
        return jsonify({
            'success': True,
            'message': 'Notifications cleared'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cameras')
def get_cameras():
    """Get all camera configurations"""
    try:
        # Use the camera manager instance to get all cameras
        camera_configs = camera_manager.get_all_camera_configs()
        return jsonify(camera_configs)
    
    except Exception as e:
        print(f"[APPLICATION] Error getting cameras: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee/<employee_id>/location')
def get_employee_location(employee_id):
    """Get employee location history"""
    try:
        session = get_db_session()
        
        # Get employee locations from the new table
        locations = session.query(EmployeeLocation).filter_by(employee_id=employee_id).order_by(EmployeeLocation.timestamp.desc()).limit(10).all()
        
        location_history = []
        for loc in locations:
            camera = session.query(Camera).filter_by(camera_id=loc.camera_id).first()
            location_info = {
                'cameraId': loc.camera_id,
                'cameraName': camera.camera_name if camera else loc.camera_id,
                'timestamp': loc.timestamp.isoformat(),
                'formattedTime': format_last_seen(loc.timestamp)
            }
            location_history.append(location_info)
        
        session.close()
        
        # Get employee name
        employee = session.query(Employee).filter_by(employee_id=employee_id).first()
        employee_name = employee.name if employee else employee_id
        
        if location_history:
            return jsonify({
                'employeeId': employee_id,
                'employeeName': employee_name,
                'location': location_history[0]['cameraName'],
                'history': location_history
            })
        else:
            return jsonify({
                'employeeId': employee_id,
                'employeeName': employee_name,
                'location': 'Unknown',
                'history': []
            })
    
    except Exception as e:
        print(f"[APPLICATION] Error getting employee location: {e}")
        return jsonify({'error': str(e)}), 500

# API routes for AI control
@app.route('/api/ai/start', methods=['POST'])
def start_ai_processing():
    """Start AI processing module"""
    try:
        if ai_processing_module.start_processing():
            return jsonify({'status': 'success', 'message': 'AI processing started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start AI processing'}), 500
    except Exception as e:
        print(f"[APPLICATION] Error starting AI processing: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ai/stop', methods=['POST'])
def stop_ai_processing():
    """Stop AI processing module"""
    try:
        if ai_processing_module.stop_processing():
            return jsonify({'status': 'success', 'message': 'AI processing stopped'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to stop AI processing'}), 500
    except Exception as e:
        print(f"[APPLICATION] Error stopping AI processing: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/ai/status', methods=['GET'])
def get_ai_status():
    """Get AI processing status"""
    try:
        status = {
            'active': ai_processing_module.is_running,
            'active_streams': list(ai_processing_module.active_streams.keys())
        }
        return jsonify(status)
    except Exception as e:
        print(f"[APPLICATION] Error getting AI status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# WebSocket events for dashboard
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('[APPLICATION] Dashboard client connected')
    emit('connection_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('[APPLICATION] Dashboard client disconnected')

@socketio.on('ai_status_update', namespace='/camera')
def handle_ai_status_update(data):
    """Handle AI status updates and broadcast to all clients"""
    print(f"[APPLICATION] AI status update received: {data}")
    # Broadcast to all connected clients
    socketio.emit('ai_status_update', data)
    print(f"[APPLICATION] AI status update broadcasted to clients")

@socketio.on('request_employee_status')
def handle_employee_status_request():
    """Handle request for employee status"""
    try:
        session = get_db_session()
        
        # Get all employee statuses from the new tracking system
        status_records = db_manager.get_employee_statuses()
        
        status_updates = []
        for record in status_records:
            status_update = {
                'employeeId': record['employee_id'],  # Use database employee_id for consistency
                'employeeName': record['employee_name'],
                'status': record['status'],
                'lastSeen': format_last_seen(record['last_seen']) if record['last_seen'] else 'Never',
                'cameraId': record['camera_id'],
                'location': record['camera_name'] or 'Unknown'
            }
            status_updates.append(status_update)
        
        session.close()
        emit('employee_status_update', status_updates)
        
    except Exception as e:
        print(f"[APPLICATION] Error handling employee status request: {e}")

def start_application():
    """Start the main application"""
    try:
        # Start background thread for status updates
        status_thread = threading.Thread(target=update_employee_status_in_background)
        status_thread.daemon = True
        status_thread.start()
        
        # Start AI processing module
        if not ai_processing_module.start_processing():
            print("[APPLICATION] Warning: Failed to start AI processing module")
        
        # Run the Flask app
        socketio.run(app,
                 host="127.0.0.1",
                 port=8000,
                 debug=True,
                 use_reloader=False,
                 allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"[APPLICATION] Error starting application: {e}")

if __name__ == '__main__':
    # Start the main application
    start_application()