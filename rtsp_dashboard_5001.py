"""
Modern Dashboard for RTSP Head Counter
Displays real-time statistics with visual banners
"""

import sys
import os
import warnings

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict
from scipy.spatial import distance as dist
import time
import threading
from datetime import datetime, timedelta
import uuid

# Import database handler (SQLite - no server required)
try:
    from database_handler_sqlite import DatabaseHandler
    DB_AVAILABLE = True
    print("✓ SQLite database enabled (data saved to head_counter.db)")
except ImportError:
    print("⚠️  Database handler not available - running without database")
    DB_AVAILABLE = False


class CentroidTracker:
    """Track objects across frames using centroids and bounding boxes"""
    
    def __init__(self, max_disappeared=80, max_distance=120):
        self.next_object_id = 0
        self.objects = {}  # Centroids
        self.bboxes = {}  # Bounding boxes for better tracking
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1
        
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        
    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
            
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes = []
        for (i, (x1, y1, x2, y2)) in enumerate(detections):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_bboxes.append((x1, y1, x2, y2))
            
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(centroid, input_bboxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                    
                if D[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
                
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col])
                
        return self.objects


class RTSPStreamProcessor:
    """Process RTSP stream with head counting"""
    
    def __init__(self, rtsp_url, model_path='best.pt', conf_threshold=0.10, box_shrink=0.2, db_path='head_counter.db'):
        self.rtsp_url = rtsp_url
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.box_shrink = box_shrink
        self.iou_threshold = 0.70  # Match training IoU (best.pt trained with 0.7)
        
        # Database handler (SQLite - no config needed)
        self.db_handler = None
        if DB_AVAILABLE:
            try:
                self.db_handler = DatabaseHandler(db_path=db_path)  # SQLite with custom db path
                print("✓ Database handler initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize database: {e}")
        
        # Counters
        # Restore counts from database if available
        self.in_count = 0
        self.out_count = 0
        self.pool_count = 0
        self.current_heads = 0
        self.peak_pool_count = 0
        self.missed_in_count = 0  # Track missed IN detections
        if self.db_handler:
            summary = self.db_handler.get_today_summary()
            if summary:
                self.in_count = summary.get('total_in', 0)
                self.out_count = summary.get('total_out', 0)
                self.peak_pool_count = summary.get('peak_pool_count', 0)
                self.pool_count = self.in_count - self.out_count
                print(f"✓ Restored counts from DB: IN={self.in_count}, OUT={self.out_count}, PEAK={self.peak_pool_count}")
        
        # Line crossing tracking optimized for top-angle head detection
        self.tracker = CentroidTracker(max_disappeared=120, max_distance=100)
        self.object_zones = {}  # Track which zone each object was last seen in
        self.counted_ids = set()
        
        # Temporal smoothing for stable counts
        self.count_history = []  # Store last 5 count changes
        self.history_size = 5
        
        # Stream state
        self.cap = None
        self.frame = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Session tracking (simple shutdown detection)
        self.session_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now()
        self.heartbeat_file = 'system_heartbeat_5001.txt'
        self.uptime_seconds = 0
        self.last_shutdown_time = None
        self.shutdown_duration = 0
        self.restart_count_today = 0
        self.detect_previous_shutdown()
        
        # Auto-reset tracking
        self.last_reset_date = time.strftime('%Y-%m-%d')
        
        # Logging
        self.log_file = 'logs_5001.html'
        self.init_log_file()
        
        # Load zones
        self.load_zones()
    
    def detect_previous_shutdown(self):
        """Detect if system was shut down by checking heartbeat file"""
        try:
            if os.path.exists(self.heartbeat_file):
                with open(self.heartbeat_file, 'r') as f:
                    last_heartbeat_str = f.read().strip()
                    if last_heartbeat_str:
                        last_heartbeat = datetime.fromisoformat(last_heartbeat_str)
                        current_time = datetime.now()
                        downtime = (current_time - last_heartbeat).total_seconds()
                        
                        # If gap > 30 seconds, consider it a shutdown
                        if downtime > 30:
                            self.last_shutdown_time = last_heartbeat.strftime('%Y-%m-%d %H:%M:%S')
                            self.shutdown_duration = int(downtime)
                            self.restart_count_today = self.get_restart_count_today() + 1
                            print(f"")
                            print(f"⚠️  PREVIOUS SHUTDOWN DETECTED")
                            print(f"   Last seen: {self.last_shutdown_time}")
                            print(f"   Downtime: {self.format_duration(downtime)}")
                            print(f"   Restarts today: {self.restart_count_today}")
                            print(f"   Session ID: {self.session_id}")
                            print(f"")
                            
                            # Log to database if available
                            if self.db_handler:
                                try:
                                    self.db_handler.log_event('RESTART', self.session_id, 
                                                            self.in_count, self.out_count, self.pool_count)
                                except:
                                    pass
        except Exception as e:
            print(f"Heartbeat check error: {e}")
    
    def get_restart_count_today(self):
        """Get restart count for today from file"""
        restart_file = 'restart_count_5001.txt'
        try:
            if os.path.exists(restart_file):
                with open(restart_file, 'r') as f:
                    data = f.read().strip().split('|')
                    if len(data) == 2:
                        date_str, count = data
                        if date_str == time.strftime('%Y-%m-%d'):
                            return int(count)
        except:
            pass
        return 0
    
    def update_restart_count(self):
        """Update restart count for today"""
        restart_file = 'restart_count_5001.txt'
        try:
            with open(restart_file, 'w') as f:
                f.write(f"{time.strftime('%Y-%m-%d')}|{self.restart_count_today}")
        except:
            pass
    
    def format_duration(self, seconds):
        """Format duration in human readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def update_heartbeat(self):
        """Write current timestamp to heartbeat file"""
        try:
            with open(self.heartbeat_file, 'w') as f:
                f.write(datetime.now().isoformat())
        except:
            pass
        
    def load_zones(self):
        """Load zone configuration"""
        config_file = 'head_counter_config.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                config_type = config.get('type', 'zones')
                
                if config_type == 'two_lines':
                    # Two-line configuration - use middle point between lines
                    in_line = config.get('in_line_y', 500)
                    out_line = config.get('out_line_y', 300)
                    self.partition_y = (in_line + out_line) // 2
                    self.config_type = 'two_lines'
                else:
                    # Legacy zone configuration
                    self.upper_zone = config.get('upper_zone', [0, 0, 640, 120])
                    self.lower_zone = config.get('lower_zone', [0, 120, 640, 288])
                    self.partition_y = self.upper_zone[3]  # Bottom of upper zone
                    self.config_type = 'zones'
        else:
            # Default - use middle of frame (will be set dynamically)
            self.partition_y = 360  # Default for 720p
            self.config_type = 'partition'
    
    def init_log_file(self):
        """Initialize HTML log file"""
        html_header = '''<!DOCTYPE html>
<html>
<head>
    <title>Head Counter Logs</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        th { background: #667eea; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        .in { color: #28a745; font-weight: bold; }
        .out { color: #dc3545; font-weight: bold; }
        tr:hover { background: #f8f9fa; }
    </style>
</head>
<body>
    <h1>Head Counter Event Logs</h1>
    <p>Started: ''' + time.strftime('%Y-%m-%d %H:%M:%S') + '''</p>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Event</th>
            <th>Object ID</th>
            <th>Total IN</th>
            <th>Total OUT</th>
            <th>Net Count</th>
        </tr>
'''
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(html_header)
    
    def log_event(self, event_type, object_id):
        """Log counting event to HTML file and database"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        event_class = 'in' if event_type == 'IN' else 'out'
        net_count = self.in_count - self.out_count
        
        # HTML logging
        log_entry = f'''        <tr>
            <td>{timestamp}</td>
            <td class="{event_class}">{event_type}</td>
            <td>#{object_id}</td>
            <td>{self.in_count}</td>
            <td>{self.out_count}</td>
            <td>{net_count}</td>
        </tr>
'''
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # Database logging
        if self.db_handler:
            self.db_handler.log_event(event_type, object_id, self.in_count, self.out_count, net_count)
    
    def connect_stream(self):
        """Connect to RTSP stream with maximum error recovery"""
        # FFmpeg options optimized to handle packet loss and corruption
        if sys.platform == 'win32':
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"  # TCP for reliable delivery
                "buffer_size;16777216|"  # 16MB buffer (maximum)
                "max_delay;3000000|"  # 3 second tolerance
                "reorder_queue_size;1000|"  # Large reorder buffer
                "stimeout;10000000|"  # 10 second socket timeout
                "analyzeduration;10000000|"  # Analyze for 10 seconds
                "probesize;10000000|"  # Probe 10MB of stream
                "err_detect;ignore_err|"  # Ignore decoding errors
                "fflags;discardcorrupt+nobuffer|"  # Discard corrupted frames
                "loglevel;fatal"  # Only show fatal errors
            )
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;16777216|max_delay;3000000|"
                "reorder_queue_size;1000|stimeout;10000000|err_detect;ignore_err|"
                "fflags;discardcorrupt+nobuffer|loglevel;fatal"
            )
        
        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
        cv2.setLogLevel(0)  # Silent
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        # Enhanced error recovery settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Large buffer
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Match camera FPS
        
        return self.cap.isOpened()
    
    def process_frame(self, frame):
        """Process a single frame with error handling"""
        try:
            # Run detection with custom YOLOv11l head model - trained on pool camera footage
            results = self.model.track(
                frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=640,  # Match training image size (best.pt trained on 640)
                device='cuda',  # RTX 3060 GPU acceleration enabled
                half=True,  # FP16 for faster inference on GPU
                persist=True,
                tracker='bytetrack.yaml',
                max_det=300,  # Match training max_det (best.pt trained with 300)
                classes=[0]  # Detect heads (class 0 in custom model)
            )
            
            # Extract detections
            detections = []
            min_box_area = 400  # Minimum 20x20 pixels to filter noise
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Filter by minimum size (reject tiny noise)
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < min_box_area:
                        continue
                    
                    # Shrink bounding box
                    if self.box_shrink > 0:
                        w = x2 - x1
                        h = y2 - y1
                        shrink_w = w * self.box_shrink / 2
                        shrink_h = h * self.box_shrink / 2
                        x1, y1 = x1 + shrink_w, y1 + shrink_h
                        x2, y2 = x2 - shrink_w, y2 - shrink_h
                    
                    detections.append([x1, y1, x2, y2])
                    
                    # Draw detection
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)
            
            # Update tracker
            objects = self.tracker.update(detections)
            self.current_heads = len(objects)
            
            # Get frame height for partition
            frame_height = frame.shape[0]
            if not hasattr(self, 'partition_y') or self.partition_y > frame_height:
                self.partition_y = frame_height // 2
            
            # Check zone transitions for counting
            for object_id, centroid in objects.items():
                cx, cy = centroid
                
                # Determine current zone (upper or lower)
                current_zone = 'upper' if cy < self.partition_y else 'lower'
                
                # Get previous zone
                previous_zone = self.object_zones.get(object_id)
                
                # Detect zone transition and count
                if previous_zone is not None and previous_zone != current_zone:
                    if object_id not in self.counted_ids:
                        # Upper -> Lower = IN
                        if previous_zone == 'upper' and current_zone == 'lower':
                            self.in_count += 1
                            self.counted_ids.add(object_id)
                            self.log_event('IN', object_id)
                        # Lower -> Upper = OUT
                        elif previous_zone == 'lower' and current_zone == 'upper':
                            self.out_count += 1
                            self.counted_ids.add(object_id)
                            self.log_event('OUT', object_id)
                
                # Update zone tracking
                self.object_zones[object_id] = current_zone
                
                # Draw centroid and ID
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{object_id}", (cx - 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Clean up old zones for disappeared objects
            tracked_ids = set(objects.keys())
            disappeared_ids = set(self.object_zones.keys()) - tracked_ids
            for obj_id in disappeared_ids:
                del self.object_zones[obj_id]
                self.counted_ids.discard(obj_id)
            
            # Draw partition line
            cv2.line(frame, (0, self.partition_y), (frame.shape[1], self.partition_y), (255, 255, 0), 3)
            cv2.putText(frame, "UPPER ZONE", (10, self.partition_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "LOWER ZONE", (10, self.partition_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Calculate pool count (ensure it doesn't go negative)
            self.pool_count = max(0, self.in_count - self.out_count)
            
            # Update peak pool count
            if self.pool_count > self.peak_pool_count:
                self.peak_pool_count = self.pool_count
            
            # Update database statistics periodically (every 10 seconds)
            if hasattr(self, 'last_db_update'):
                if time.time() - self.last_db_update > 10:
                    self.update_database_stats()
                    self.update_heartbeat()  # Update heartbeat every 10 seconds
            else:
                self.last_db_update = time.time()
                self.update_heartbeat()  # Initial heartbeat
            
            return frame
            
        except Exception as e:
            # Return original frame if processing fails
            print(f"Frame processing error (skipping): {str(e)[:50]}")
            return frame
    
    def update_database_stats(self):
        """Update database with current statistics"""
        if self.db_handler:
            try:
                # Update daily summary
                self.db_handler.update_daily_summary(
                    self.in_count, self.out_count, self.pool_count, self.peak_pool_count
                )
                # Update hourly statistics
                self.db_handler.update_hourly_statistics(
                    self.in_count, self.out_count, self.pool_count
                )
                # Log system health
                self.db_handler.log_system_health(
                    self.fps, self.current_heads, 'active'
                )
                self.last_db_update = time.time()
            except Exception as e:
                print(f"⚠️  Database update error: {e}")
    
    def start(self):
        """Start processing stream"""
        self.is_running = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
    
    def _process_loop(self):
        """Main processing loop with robust error handling"""
        reconnect_attempts = 0
        max_reconnects = 10
        last_valid_frame = None
        consecutive_errors = 0
        
        if not self.connect_stream():
            print("Failed to connect to RTSP stream")
            return
        
        print(f"✓ Connected to RTSP stream")
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    
                    # Only print every 5 errors to avoid spam
                    if consecutive_errors % 5 == 1:
                        print(f"Stream error (attempt {reconnect_attempts + 1}/{max_reconnects})")
                    
                    # Use last valid frame while reconnecting
                    if last_valid_frame is not None:
                        with self.lock:
                            self.frame = last_valid_frame
                    
                    # Try to recover after multiple errors
                    if consecutive_errors > 10:
                        self.cap.release()
                        time.sleep(0.5)  # Brief wait
                        
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnects:
                            print("Max reconnect attempts reached")
                            break
                        
                        if not self.connect_stream():
                            continue
                        else:
                            reconnect_attempts = 0
                            consecutive_errors = 0
                    continue
                
                # Valid frame received - reset error counter
                consecutive_errors = 0
                reconnect_attempts = 0
                
                # Check for midnight reset
                current_date = time.strftime('%Y-%m-%d')
                if current_date != self.last_reset_date:
                    print(f"\n🔄 Midnight auto-reset triggered: {current_date}")
                    self.in_count = 0
                    self.out_count = 0
                    self.pool_count = 0
                    self.counted_ids.clear()
                    self.last_reset_date = current_date
                    self.log_event('RESET', 'AUTO')
                
                # Flip frame vertically
                frame = cv2.flip(frame, 0)
                self.frame_count += 1
                
                # Print FPS every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    self.uptime_seconds = int(elapsed)
                    print(f"Processing... FPS: {current_fps:.1f} | Heads: {self.current_heads} | IN: {self.in_count} | OUT: {self.out_count} | Uptime: {self.format_duration(elapsed)}")
                
                last_valid_frame = frame.copy()  # Keep backup
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                # Store frame
                with self.lock:
                    self.frame = processed_frame
            except Exception as e:
                print(f"Loop error: {str(e)[:50]}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_stats(self):
        """Get current statistics with missed detection tracking"""
        in_count = self.in_count
        out_count = self.out_count
        pool_count = in_count - out_count
        
        # Calculate current uptime
        self.uptime_seconds = int(time.time() - self.start_time)
        
        # Track missed IN detections when pool count goes negative
        if pool_count < 0:
            missed_entries = abs(pool_count)
            print(f"⚠️ Detection discrepancy: IN={in_count}, OUT={out_count}, Missed IN detections: {missed_entries}")
            
            # Track cumulative missed detections
            self.missed_in_count += missed_entries
            
            # Auto-correct: adjust IN count to match OUT count
            self.in_count = out_count
            in_count = out_count
            pool_count = 0
            
            # Log correction event
            if self.db_handler:
                try:
                    self.db_handler.log_event('CORRECTION', 0, self.in_count, self.out_count, pool_count)
                except Exception as e:
                    print(f"⚠️ Could not log correction: {e}")
        
        # Calculate detection accuracy
        total_expected_in = in_count + self.missed_in_count
        detection_accuracy = (in_count / total_expected_in * 100) if total_expected_in > 0 else 100
        
        return {
            'in_count': max(0, in_count),
            'out_count': max(0, out_count),
            'pool_count': max(0, pool_count),
            'current_heads': max(0, self.current_heads),
            'fps': round(self.fps, 1),
            'missed_in_count': self.missed_in_count,
            'detection_accuracy': round(detection_accuracy, 1),
            'uptime_seconds': self.uptime_seconds,
            'uptime_formatted': self.format_duration(self.uptime_seconds),
            'session_id': self.session_id,
            'restart_count_today': self.restart_count_today,
            'last_shutdown': self.last_shutdown_time,
            'shutdown_duration': self.shutdown_duration,
            'timestamp': time.time()
        }
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        # Update restart count
        if self.restart_count_today > 0:
            self.update_restart_count()
        
        # Final heartbeat update
        self.update_heartbeat()
        
        if self.db_handler:
            # Final database update before closing
            self.update_database_stats()
            # Log clean shutdown
            try:
                self.db_handler.log_event('SHUTDOWN_CLEAN', self.session_id, 
                                         self.in_count, self.out_count, self.pool_count)
            except:
                pass
            self.db_handler.close()
        
        print(f"\n✓ Session {self.session_id} ended cleanly")
        print(f"  Uptime: {self.format_duration(self.uptime_seconds)}")
        print(f"  Total frames: {self.frame_count}")


# Flask application
app = Flask(__name__)

# Global processor
processor = None


def generate_frames():
    """Generate video frames for streaming"""
    global processor
    
    while True:
        if processor is None:
            time.sleep(0.1)
            continue
        
        frame = processor.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('rtsp_dashboard.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get current statistics"""
    global processor
    if processor is None:
        return jsonify({
            'in_count': 0,
            'out_count': 0,
            'pool_count': 0,
            'current_heads': 0,
            'fps': 0,
            'timestamp': time.time()
        })
    return jsonify(processor.get_stats())


@app.route('/reset', methods=['POST'])
def reset_counts():
    """Reset all counters"""
    global processor
    if processor:
        processor.in_count = 0
        processor.out_count = 0
        processor.pool_count = 0
        processor.missed_in_count = 0
        processor.counted_ids.clear()
        return jsonify({'success': True, 'message': 'Counters reset successfully'})
    return jsonify({'success': False, 'error': 'No active processor'})


@app.route('/health')
def health():
    """Health check endpoint"""
    global processor
    return jsonify({
        'status': 'running',
        'processor_active': processor is not None and processor.is_running,
        'timestamp': time.time()
    })


def main():
    # Configuration optimized for top-angle head detection with RTX 3060 (PORT 5001)
    rtsp_url = "rtsp://Testing:Test%401234%23@10.196.211.60:554/cam/realmonitor?chanel=1subtype=0"
    model_path = 'best.pt'  # Custom trained head detection model (YOLOv11l)
    conf_threshold = 0.10  # Low threshold for head-only views
    box_shrink = 0.2  # Less aggressive for small heads
    host = '0.0.0.0'
    port = 5001  # Port 5001 for best.pt model comparison
    
    # Initialize processor
    global processor
    processor = RTSPStreamProcessor(
        rtsp_url=rtsp_url,
        model_path=model_path,
        conf_threshold=conf_threshold,
        box_shrink=box_shrink,
        db_path='head_counter_5001.db'
    )
    
    # Start processing
    processor.start()
    
    # Run Flask app
    print(f"\n{'='*60}")
    print(f"🎯 RTSP Head Counter Dashboard (PORT 5001 - best.pt)")
    print(f"{'='*60}")
    print(f"📺 Stream: {rtsp_url}")
    print(f"🤖 Model: best.pt (Custom YOLOv11l - Head Detection)")
    print(f"🌐 Dashboard: http://{host}:{port}")
    print(f"📊 Database: head_counter_5001.db")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
