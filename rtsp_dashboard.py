"""
Modern Dashboard for RTSP Head Counter
Dual-pool support with real-time statistics
"""

import sys
import os
import warnings
import smtplib
import traceback
from datetime import datetime, timedelta
from email.message import EmailMessage
import atexit

from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict
from scipy.spatial import distance as dist
import time
import threading
import pandas as pd

# Import database handler (SQLite - no server required)
try:
    from database_handler_sqlite import DatabaseHandler
    DB_AVAILABLE = True
    print("✓ SQLite database enabled (data saved to head_counter.db)")
except ImportError:
    print("⚠️  Database handler not available - running without database")
    DB_AVAILABLE = False


REPORT_TYPES = {
    'daily_peak': 'Daily Peak Report',
    'daily_guest_entry': 'Daily Guest Entry Report',
    'hourly_guest_usage': 'Hourly Guest Usage Report',
}


class ReportManager:
    """Generate, store, email, and schedule analytics reports."""

    def __init__(self, db_path='head_counter.db', report_dir='reports'):
        self.db_handler = DatabaseHandler(db_path=db_path)
        self.report_dir = report_dir
        self.generated_files = {}
        self.scheduler_state = {
            'daily_peak': None,
            'daily_guest_entry': None,
            'hourly_guest_usage': None,
        }
        self.scheduler_running = False
        self.scheduler_thread = None
        self.lock = threading.Lock()
        os.makedirs(self.report_dir, exist_ok=True)

    def _build_filename(self, report_type, date_str, hour=None):
        if hour is None:
            return f"{report_type}_{date_str}.xlsx"
        return f"{report_type}_{date_str}_{hour:02d}.xlsx"

    def _save_excel(self, rows, columns, output_path):
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=columns)
        df.to_excel(output_path, index=False)

    def _build_chart_data(self, report_type, rows):
        if not rows:
            return {'labels': [], 'datasets': []}

        if report_type in ('daily_peak', 'daily_guest_entry'):
            # X-axis = pools, one line per metric
            labels = [row.get('pool_id', 'unknown').replace('pool', 'Pool ').upper() for row in rows]
            if report_type == 'daily_peak':
                datasets = [
                    {'label': 'Entered', 'color': '#1a3ab5',
                     'data': [row.get('total_entered', 0) for row in rows]},
                    {'label': 'Exited',  'color': '#f5c518',
                     'data': [row.get('total_exited', 0) for row in rows]},
                    {'label': 'Peak',    'color': '#22a06b',
                     'data': [row.get('peak_pool_count', 0) for row in rows]},
                ]
            else:
                datasets = [
                    {'label': 'Entered', 'color': '#1a3ab5',
                     'data': [row.get('total_entered', 0) for row in rows]},
                ]
            return {'labels': labels, 'datasets': datasets}

        # Hourly: X-axis = hours, one line per pool×metric combination
        pools = sorted({row.get('pool_id', 'pool1') for row in rows})
        hours = sorted({row.get('hour', 0) for row in rows})
        labels = [f"{h:02d}:00" for h in hours]
        lookup = {(r.get('pool_id'), r.get('hour')): r for r in rows}

        pool_palette = {
            'pool1': {'entered': '#1a3ab5', 'exited': '#6b8de3'},
            'pool2': {'entered': '#f5a623', 'exited': '#f5c518'},
        }
        datasets = []
        for pool in pools:
            display = pool.replace('pool', 'Pool ')
            colors = pool_palette.get(pool, {'entered': '#22a06b', 'exited': '#15754c'})
            datasets.append({
                'label': f'{display} – Entered',
                'color': colors['entered'],
                'data': [lookup.get((pool, h), {}).get('total_entered', 0) for h in hours],
            })
            datasets.append({
                'label': f'{display} – Exited',
                'color': colors['exited'],
                'data': [lookup.get((pool, h), {}).get('total_exited', 0) for h in hours],
            })
        return {'labels': labels, 'datasets': datasets}

    def _email_report(self, report_type, file_path, date_str):
        smtp_host = os.getenv('REPORT_SMTP_HOST')
        smtp_port = int(os.getenv('REPORT_SMTP_PORT', '587'))
        smtp_user = os.getenv('REPORT_SMTP_USER')
        smtp_password = os.getenv('REPORT_SMTP_PASSWORD')
        smtp_to = os.getenv('REPORT_MAIL_TO', '')
        smtp_from = os.getenv('REPORT_MAIL_FROM', smtp_user or '')
        use_tls = os.getenv('REPORT_SMTP_TLS', 'true').lower() == 'true'

        recipients = [email.strip() for email in smtp_to.split(',') if email.strip()]
        if not (smtp_host and smtp_user and smtp_password and recipients and smtp_from):
            return False, 'Skipped: SMTP env is incomplete'

        subject = f"{REPORT_TYPES.get(report_type, report_type)} - {date_str}"
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = smtp_from
        msg['To'] = ', '.join(recipients)
        msg.set_content(
            f"Please find attached {REPORT_TYPES.get(report_type, report_type)} for {date_str}."
        )

        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        msg.add_attachment(
            file_bytes,
            maintype='application',
            subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=os.path.basename(file_path),
        )

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
            if use_tls:
                smtp.starttls()
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
        return True, 'Email sent'

    def generate_report(self, report_type, date_str=None, send_email=False, hour=None):
        if report_type not in REPORT_TYPES:
            raise ValueError(f"Unsupported report type: {report_type}")

        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        if report_type == 'daily_peak':
            rows = self.db_handler.get_daily_peak_report(date_str)
            columns = ['date', 'pool_id', 'total_entered', 'total_exited', 'net_count', 'peak_pool_count']
        elif report_type == 'daily_guest_entry':
            rows = self.db_handler.get_daily_guest_entry_report(date_str)
            columns = ['date', 'pool_id', 'total_entered']
        else:
            rows = self.db_handler.get_hourly_guest_usage_report(date_str)
            columns = ['date', 'hour', 'pool_id', 'total_entered', 'total_exited', 'net_count']

        filename = self._build_filename(report_type, date_str, hour=hour)
        file_path = os.path.join(self.report_dir, filename)
        self._save_excel(rows, columns, file_path)

        with self.lock:
            self.generated_files[report_type] = filename

        email_status = None
        if send_email:
            try:
                ok, msg = self._email_report(report_type, file_path, date_str)
                email_status = {'success': ok, 'message': msg}
            except Exception as exc:
                email_status = {'success': False, 'message': str(exc)}

        return {
            'report_type': report_type,
            'report_name': REPORT_TYPES[report_type],
            'date': date_str,
            'filename': filename,
            'rows': rows,
            'chart': self._build_chart_data(report_type, rows),
            'email': email_status,
        }

    def get_latest_filename(self, report_type):
        with self.lock:
            return self.generated_files.get(report_type)

    def start_scheduler(self):
        if self.scheduler_running:
            return
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        print('✓ Report scheduler started')

    def stop_scheduler(self):
        self.scheduler_running = False

    def _scheduler_loop(self):
        while self.scheduler_running:
            try:
                now = datetime.now()

                if now.hour == 11 and now.minute == 0:
                    run_key = now.strftime('%Y-%m-%d')
                    report_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
                    if self.scheduler_state['daily_peak'] != run_key:
                        self.generate_report('daily_peak', date_str=report_date, send_email=True)
                        self.scheduler_state['daily_peak'] = run_key
                    if self.scheduler_state['daily_guest_entry'] != run_key:
                        self.generate_report('daily_guest_entry', date_str=report_date, send_email=True)
                        self.scheduler_state['daily_guest_entry'] = run_key

                if now.minute == 0:
                    hour_key = now.strftime('%Y-%m-%d_%H')
                    if self.scheduler_state['hourly_guest_usage'] != hour_key:
                        self.generate_report(
                            'hourly_guest_usage',
                            date_str=now.strftime('%Y-%m-%d'),
                            send_email=True,
                            hour=now.hour,
                        )
                        self.scheduler_state['hourly_guest_usage'] = hour_key
            except Exception as exc:
                print(f"⚠️  Report scheduler error: {exc}")
                traceback.print_exc()

            time.sleep(30)


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
    
    def __init__(self, rtsp_url, pool_id='pool1', model_path='yolo11x.pt', conf_threshold=0.10, box_shrink=0.2, capacity=500):
        self.rtsp_url = rtsp_url
        self.pool_id = pool_id
        self.capacity = capacity
        self.device = 'cpu'
        if os.environ.get('FORCE_CPU', '0') != '1':
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.device = 'cuda'
            except Exception:
                self.device = 'cpu'
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.box_shrink = box_shrink
        self.iou_threshold = 0.60
        
        # Database handler
        self.db_handler = None
        if DB_AVAILABLE:
            try:
                self.db_handler = DatabaseHandler()
                print(f"✓ Database handler initialized for {pool_id}")
            except Exception as e:
                print(f"⚠️  Could not initialize database for {pool_id}: {e}")
        
        # Counters - restore from database
        self.in_count = 0
        self.out_count = 0
        self.pool_count = 0
        self.current_heads = 0
        self.peak_pool_count = 0
        self.missed_in_count = 0
        if self.db_handler:
            summary = self.db_handler.get_today_summary(pool_id=self.pool_id)
            if summary:
                self.in_count = summary.get('total_in', 0)
                self.out_count = summary.get('total_out', 0)
                self.peak_pool_count = summary.get('peak_pool_count', 0)
                self.pool_count = self.in_count - self.out_count
                print(f"✓ [{pool_id}] Restored: IN={self.in_count}, OUT={self.out_count}, PEAK={self.peak_pool_count}")
        
        # Tracking
        self.tracker = CentroidTracker(max_disappeared=120, max_distance=100)
        self.object_zones = {}
        self.counted_ids = set()
        self.count_history = []
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
        self.last_reset_date = time.strftime('%Y-%m-%d')
        
        # Logging
        self.log_file = f'logs_{pool_id}.html'
        self.init_log_file()
        self.load_zones()
        
    def load_zones(self):
        """Load zone configuration"""
        config_file = f'head_counter_config_{self.pool_id}.json'
        if not os.path.exists(config_file):
            config_file = 'head_counter_config.json'

        self.invert_direction = False

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                config_type = config.get('type', 'zones')
                self.invert_direction = config.get('invert_direction', False)

                if config_type == 'two_lines':
                    in_line = config.get('in_line_y', 500)
                    out_line = config.get('out_line_y', 300)
                    self.partition_y = (in_line + out_line) // 2
                    self.config_type = 'two_lines'
                else:
                    self.upper_zone = config.get('upper_zone', [0, 0, 640, 120])
                    self.lower_zone = config.get('lower_zone', [0, 120, 640, 288])
                    self.partition_y = config.get('partition_y', self.upper_zone[3])
                    self.config_type = 'zones'
        else:
            self.partition_y = 360
            self.config_type = 'partition'
    
    def init_log_file(self):
        """Initialize HTML log file"""
        html_header = '''<!DOCTYPE html>
<html>
<head>
    <title>Head Counter Logs - ''' + self.pool_id + '''</title>
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
    <h1>Head Counter Event Logs - ''' + self.pool_id + '''</h1>
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
        
        if self.db_handler:
            self.db_handler.log_event(event_type, object_id, self.in_count, self.out_count, net_count, pool_id=self.pool_id)
    
    def connect_stream(self):
        """Connect to RTSP stream with maximum error recovery"""
        if sys.platform == 'win32':
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "buffer_size;16777216|"
                "max_delay;3000000|"
                "reorder_queue_size;1000|"
                "stimeout;10000000|"
                "analyzeduration;10000000|"
                "probesize;10000000|"
                "err_detect;ignore_err|"
                "fflags;discardcorrupt+nobuffer|"
                "loglevel;fatal"
            )
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;16777216|max_delay;3000000|"
                "reorder_queue_size;1000|stimeout;10000000|err_detect;ignore_err|"
                "fflags;discardcorrupt+nobuffer|loglevel;fatal"
            )
        
        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
        try:
            cv2.setLogLevel(0)
        except AttributeError:
            pass
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        return self.cap.isOpened()
    
    def process_frame(self, frame):
        """Process a single frame with error handling"""
        try:
            results = self.model.track(
                frame, 
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                imgsz=1280,
                device=self.device,
                half=(self.device == 'cuda'),
                persist=True,
                tracker='bytetrack.yaml',
                max_det=100,
                classes=[0]
            )
            
            detections = []
            min_box_area = 400
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < min_box_area:
                        continue
                    
                    if self.box_shrink > 0:
                        w = x2 - x1
                        h = y2 - y1
                        shrink_w = w * self.box_shrink / 2
                        shrink_h = h * self.box_shrink / 2
                        x1, y1 = x1 + shrink_w, y1 + shrink_h
                        x2, y2 = x2 - shrink_w, y2 - shrink_h
                    
                    detections.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            objects = self.tracker.update(detections)
            self.current_heads = len(objects)
            
            frame_height = frame.shape[0]
            if not hasattr(self, 'partition_y') or self.partition_y > frame_height:
                self.partition_y = frame_height // 2
            
            for object_id, centroid in objects.items():
                cx, cy = centroid
                current_zone = 'upper' if cy < self.partition_y else 'lower'
                previous_zone = self.object_zones.get(object_id)

                if previous_zone is not None and previous_zone != current_zone:
                    if object_id not in self.counted_ids:
                        upper_to_lower = (previous_zone == 'upper' and current_zone == 'lower')
                        lower_to_upper = (previous_zone == 'lower' and current_zone == 'upper')
                        is_in = (lower_to_upper if self.invert_direction else upper_to_lower)
                        is_out = (upper_to_lower if self.invert_direction else lower_to_upper)
                        if is_in:
                            self.in_count += 1
                            self.counted_ids.add(object_id)
                            self.log_event('IN', object_id)
                        elif is_out:
                            self.out_count += 1
                            self.counted_ids.add(object_id)
                            self.log_event('OUT', object_id)

                self.object_zones[object_id] = current_zone
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{object_id}", (cx - 10, cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            tracked_ids = set(objects.keys())
            disappeared_ids = set(self.object_zones.keys()) - tracked_ids
            for obj_id in disappeared_ids:
                del self.object_zones[obj_id]
                self.counted_ids.discard(obj_id)
            
            cv2.line(frame, (0, self.partition_y), (frame.shape[1], self.partition_y), (255, 255, 0), 3)
            
            self.pool_count = max(0, self.in_count - self.out_count)
            if self.pool_count > self.peak_pool_count:
                self.peak_pool_count = self.pool_count
            
            if hasattr(self, 'last_db_update'):
                if time.time() - self.last_db_update > 10:
                    self.update_database_stats()
            else:
                self.last_db_update = time.time()
            
            return frame
            
        except Exception as e:
            print(f"[{self.pool_id}] Frame processing error: {str(e)[:50]}")
            return frame
    
    def update_database_stats(self):
        """Update database with current statistics"""
        if self.db_handler:
            try:
                self.db_handler.update_daily_summary(
                    self.in_count, self.out_count, self.pool_count, self.peak_pool_count, pool_id=self.pool_id
                )
                self.db_handler.update_hourly_statistics(
                    self.in_count, self.out_count, self.pool_count, pool_id=self.pool_id
                )
                self.db_handler.log_system_health(
                    self.fps, self.current_heads, 'active', pool_id=self.pool_id
                )
                self.last_db_update = time.time()
            except Exception as e:
                print(f"⚠️  [{self.pool_id}] Database update error: {e}")
    
    def log_heartbeat(self):
        """Log system heartbeat for downtime detection"""
        if self.db_handler:
            try:
                self.db_handler.log_system_health(self.fps, self.current_heads, 'active', pool_id=self.pool_id)
            except Exception as e:
                print(f"⚠️  [{self.pool_id}] Heartbeat error: {e}")
    
    def get_downtime_periods(self):
        """Detect downtime periods from system health logs"""
        if not self.db_handler:
            return []
        try:
            return self.db_handler.detect_downtime_gaps(gap_threshold_minutes=5, pool_id=self.pool_id)
        except Exception as e:
            print(f"⚠️  [{self.pool_id}] Downtime detection error: {e}")
            return []
    
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
            print(f"[{self.pool_id}] Failed to connect to RTSP stream")
            return
        
        print(f"✓ [{self.pool_id}] Connected to RTSP stream")
        
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors % 5 == 1:
                        print(f"[{self.pool_id}] Stream error (attempt {reconnect_attempts + 1}/{max_reconnects})")
                    
                    if last_valid_frame is not None:
                        with self.lock:
                            self.frame = last_valid_frame
                    
                    if consecutive_errors > 10:
                        self.cap.release()
                        time.sleep(0.5)
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnects:
                            print(f"[{self.pool_id}] Max reconnect attempts reached")
                            break
                        if not self.connect_stream():
                            continue
                        else:
                            reconnect_attempts = 0
                            consecutive_errors = 0
                    continue
                
                consecutive_errors = 0
                reconnect_attempts = 0
                
                current_date = time.strftime('%Y-%m-%d')
                if current_date != self.last_reset_date:
                    print(f"\n🔄 [{self.pool_id}] Midnight auto-reset: {current_date}")
                    self.in_count = 0
                    self.out_count = 0
                    self.pool_count = 0
                    self.peak_pool_count = 0
                    self.counted_ids.clear()
                    self.last_reset_date = current_date
                    self.log_event('RESET', 'AUTO')
                
                frame = cv2.flip(frame, 0)
                self.frame_count += 1
                
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[{self.pool_id}] FPS: {current_fps:.1f} | Heads: {self.current_heads} | IN: {self.in_count} | OUT: {self.out_count}")
                    self.log_heartbeat()
                
                last_valid_frame = frame.copy()
                processed_frame = self.process_frame(frame)
                
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                with self.lock:
                    self.frame = processed_frame
            except Exception as e:
                print(f"[{self.pool_id}] Loop error: {str(e)[:50]}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_stats(self):
        """Get current statistics"""
        in_count = self.in_count
        out_count = self.out_count
        pool_count = in_count - out_count
        
        if pool_count < 0:
            missed_entries = abs(pool_count)
            self.missed_in_count += missed_entries
            self.in_count = out_count
            in_count = out_count
            pool_count = 0
            
            if self.db_handler:
                try:
                    self.db_handler.log_event('CORRECTION', 0, self.in_count, self.out_count, pool_count, pool_id=self.pool_id)
                except Exception:
                    pass
        
        total_expected_in = in_count + self.missed_in_count
        detection_accuracy = (in_count / total_expected_in * 100) if total_expected_in > 0 else 100
        
        downtime_periods = self.get_downtime_periods()
        total_downtime_minutes = sum([d['duration_minutes'] for d in downtime_periods])
        
        return {
            'in_count': max(0, in_count),
            'out_count': max(0, out_count),
            'pool_count': max(0, pool_count),
            'current_heads': max(0, self.current_heads),
            'fps': round(self.fps, 1),
            'missed_in_count': self.missed_in_count,
            'peak_pool_count': self.peak_pool_count,
            'detection_accuracy': round(detection_accuracy, 1),
            'timestamp': time.time(),
            'downtime_periods': downtime_periods,
            'total_downtime_minutes': round(total_downtime_minutes, 1),
            'has_downtime': len(downtime_periods) > 0,
            'capacity': self.capacity
        }
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.db_handler:
            self.update_database_stats()
            self.db_handler.close()

    def reset_counters(self):
        """Reset runtime counters to zero."""
        with self.lock:
            self.in_count = 0
            self.out_count = 0
            self.pool_count = 0
            self.current_heads = 0
            self.peak_pool_count = 0
            self.missed_in_count = 0
            self.counted_ids.clear()
            self.object_zones.clear()

        try:
            self.log_event('RESET', 'MANUAL')
        except Exception:
            pass

        if self.db_handler:
            try:
                self.update_database_stats()
            except Exception:
                pass


# Flask application
app = Flask(__name__)

# Global processors for both pools
processors = {}
report_manager = ReportManager() if DB_AVAILABLE else None


@app.route('/api/reports/<report_type>', methods=['GET'])
def get_report_data(report_type):
    if not report_manager:
        return jsonify({'success': False, 'message': 'Database/reporting is disabled'}), 500
    if report_type not in REPORT_TYPES:
        return jsonify({'success': False, 'message': 'Invalid report type'}), 400

    date_str = request.args.get('date') or datetime.now().strftime('%Y-%m-%d')
    result = report_manager.generate_report(report_type, date_str=date_str, send_email=False)
    return jsonify({'success': True, **result})


@app.route('/api/reports/generate/<report_type>', methods=['POST'])
def generate_report(report_type):
    if not report_manager:
        return jsonify({'success': False, 'message': 'Database/reporting is disabled'}), 500
    if report_type not in REPORT_TYPES:
        return jsonify({'success': False, 'message': 'Invalid report type'}), 400

    payload = request.get_json(silent=True) or {}
    date_str = payload.get('date') or datetime.now().strftime('%Y-%m-%d')
    send_email_now = bool(payload.get('send_email', False))

    result = report_manager.generate_report(
        report_type,
        date_str=date_str,
        send_email=send_email_now,
    )
    return jsonify({'success': True, **result})


@app.route('/api/reports/download/<report_type>', methods=['GET'])
def download_report(report_type):
    if not report_manager:
        return jsonify({'success': False, 'message': 'Database/reporting is disabled'}), 500
    if report_type not in REPORT_TYPES:
        return jsonify({'success': False, 'message': 'Invalid report type'}), 400

    filename = request.args.get('filename') or report_manager.get_latest_filename(report_type)
    if not filename:
        return jsonify({'success': False, 'message': 'No report generated yet'}), 404

    file_path = os.path.abspath(os.path.join(report_manager.report_dir, filename))
    report_dir_abs = os.path.abspath(report_manager.report_dir)
    if not file_path.startswith(report_dir_abs):
        return jsonify({'success': False, 'message': 'Invalid file path'}), 400
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': 'Report file not found'}), 404

    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/')
def index():
    return render_template('rtsp_dashboard.html')


def generate_frames(pool_id):
    while True:
        processor = processors.get(pool_id)
        if processor is None:
            time.sleep(0.1)
            continue
        frame = processor.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<pool_id>')
def video_feed(pool_id):
    if pool_id not in processors:
        return 'Pool not found', 404
    return Response(generate_frames(pool_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Get statistics for all pools"""
    result = {}
    for pool_id, processor in processors.items():
        if processor is not None:
            result[pool_id] = processor.get_stats()
        else:
            result[pool_id] = {
                'in_count': 0, 'out_count': 0, 'pool_count': 0,
                'current_heads': 0, 'fps': 0, 'missed_in_count': 0,
                'peak_pool_count': 0, 'timestamp': time.time()
            }
    return jsonify(result)


@app.route('/reset', methods=['POST'])
def reset_counts():
    """Reset all counters for all pools"""
    for pool_id, processor in processors.items():
        if processor:
            processor.reset_counters()
    return jsonify({'success': True, 'message': 'All counters reset successfully'})


@app.route('/health')
def health():
    """Health check endpoint"""
    pool_status = {}
    for pool_id, processor in processors.items():
        pool_status[pool_id] = processor is not None and processor.is_running
    return jsonify({
        'status': 'running',
        'pools': pool_status,
        'timestamp': time.time()
    })


def main():
    # Configuration for both pools
    # Pool 1 RTSP URL
    rtsp_url_1 = "rtsp://Testing:Test%401234%23@10.196.211.60:554/cam/realmonitor?chanel=1subtype=0"
    rtsp_url_2 = "rtsp://admin:Ele%23%23%23313@10.196.211.59:554/"
    
    model_path = 'yolo11x.pt'
    conf_threshold = 0.10
    box_shrink = 0.2
    host = '0.0.0.0'
    port = 5000
    
    global processors
    
    # Initialize Pool 1
    print(f"\n{'='*60}")
    print(f"🏊 Initializing Pool 01...")
    processors['pool1'] = RTSPStreamProcessor(
        rtsp_url=rtsp_url_1,
        pool_id='pool1',
        model_path=model_path,
        conf_threshold=conf_threshold,
        box_shrink=box_shrink,
        capacity=800
    )

    # Initialize Pool 2
    print(f"🏊 Initializing Pool 02...")
    processors['pool2'] = RTSPStreamProcessor(
        rtsp_url=rtsp_url_2,
        pool_id='pool2',
        model_path=model_path,
        conf_threshold=conf_threshold,
        box_shrink=box_shrink,
        capacity=400
    )
    
    # Start both processors
    processors['pool1'].start()
    processors['pool2'].start()

    if report_manager:
        report_manager.start_scheduler()
    
    print(f"\n{'='*60}")
    print(f"🎯 Wonderla Dual Pool Monitoring Dashboard")
    print(f"{'='*60}")
    print(f"🏊 Pool 01: {rtsp_url_1[:60]}...")
    print(f"🏊 Pool 02: {rtsp_url_2[:60]}...")
    print(f"🌐 Dashboard: http://{host}:{port}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    if report_manager:
        atexit.register(report_manager.stop_scheduler)
    main()
