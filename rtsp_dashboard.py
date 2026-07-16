"""
Modern Dashboard for RTSP Head Counter
Dual-pool support with real-time statistics
"""

import sys
import os
import warnings

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
from collections import defaultdict, deque
from scipy.spatial import distance as dist
import time
import threading

# Cap CPU thread pools to avoid oversubscription. The RTSP reader threads decode
# in parallel while the inference thread runs YOLO pre/post-processing (letterbox
# resize + NMS). Left unbounded, OpenCV's and torch's thread pools each try to use
# all cores and fight the decoder threads for CPU, inflating inference wall-time
# ~3x (the GPU then sits idle waiting on contended CPU work). Bounding them keeps
# inference fast and processing FPS stable.
cv2.setNumThreads(2)
torch.set_num_threads(4)

# Telegram low-FPS alerting
from fps_alert_monitor import start_fps_monitor

# Import database handler (SQLite - no server required)
try:
    from database_handler_sqlite import DatabaseHandler
    DB_AVAILABLE = True
    print("✓ SQLite database enabled (data saved to head_counter.db)")
except ImportError:
    print("⚠️  Database handler not available - running without database")
    DB_AVAILABLE = False

# Import analytics/reporting routes (reports + reset)
try:
    from dashboard_analytics import ReportManager, register_analytics_routes
    ANALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️  Analytics module not available - running without reports")
    ANALYTICS_AVAILABLE = False

# Import per-hour model confidence monitor
try:
    from model_confidence_monitor import ModelConfidenceMonitor
    CONF_MONITOR_AVAILABLE = True
except ImportError:
    print("⚠️  Model confidence monitor not available")
    CONF_MONITOR_AVAILABLE = False


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
    
    def __init__(self, rtsp_url, pool_id='pool1', model_path='yolo11x.pt', conf_threshold=0.10, box_shrink=0.2):
        self.rtsp_url = rtsp_url
        self.pool_id = pool_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Inference is run centrally by BatchInferenceCoordinator using a single
        # shared model, so this processor no longer owns a YOLO model. Running
        # one batched inference call for all pools avoids the GIL contention of
        # multiple per-pool inference threads (which left the GPU ~85% idle and
        # capped processing FPS at ~5-6). This class keeps only the tracking,
        # counting and drawing logic.
        print(f"✓ [{pool_id}] Using device: {self.device}")
        self.conf_threshold = conf_threshold
        self.box_shrink = box_shrink
        self.iou_threshold = 0.60

        # Per-hour model confidence monitor (logs avg/min/max confidence)
        self.conf_monitor = None
        if CONF_MONITOR_AVAILABLE:
            self.conf_monitor = ModelConfidenceMonitor(pool_id, conf_threshold)
        
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
        self.frame = None            # latest PROCESSED frame (for the video feed)
        self.is_running = False
        self.lock = threading.Lock()

        # Decoupled capture: a dedicated reader thread continuously drains the
        # RTSP stream (~15 fps) and keeps only the most recent raw frame here.
        # The inference thread consumes the latest frame back-to-back instead of
        # blocking on cap.read() after every inference. This stops the GPU from
        # idling while waiting for a fresh frame (the nobuffer flag drops frames
        # that arrive during inference), which was capping processing FPS.
        self._latest_raw = None
        self._raw_seq = 0            # bumped on every new decoded frame
        self._raw_lock = threading.Lock()
        
        # Stats
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Rolling FPS tracking (recent window) for live display + low-FPS alerts
        self.fps_window = 5.0          # seconds
        self.fps_lock = threading.Lock()
        self._read_times = deque()     # timestamps of successful stream reads (input FPS)
        self._proc_times = deque()     # timestamps of completed frame processing (processing FPS)
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
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                config_type = config.get('type', 'zones')
                if 'flip_vertical' in config:
                    self.flip_code = 0 if config['flip_vertical'] else None
                else:
                    self.flip_code = config.get('flip', None)

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
            self.flip_code = 0
    
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
        # Low-latency capture: keep FFmpeg's input queues tiny so the frame we
        # process is always near-live. A large buffer_size/reorder_queue/
        # max_delay builds a FIFO backlog that never drains (arrival rate ~=
        # read rate), so cap.read() keeps returning frames that are 10-15s old
        # even though the reader keeps only the latest one. Shrinking these
        # (and BUFFERSIZE=1 below) keeps end-to-end latency ~1s.
        if sys.platform == 'win32':
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "buffer_size;102400|"
                "max_delay;500000|"
                "reorder_queue_size;0|"
                "stimeout;10000000|"
                "analyzeduration;1000000|"
                "probesize;1000000|"
                "err_detect;ignore_err|"
                "fflags;discardcorrupt+nobuffer|"
                "flags;low_delay|"
                "loglevel;fatal"
            )
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;102400|max_delay;500000|"
                "reorder_queue_size;0|stimeout;10000000|err_detect;ignore_err|"
                "fflags;discardcorrupt+nobuffer|flags;low_delay|loglevel;fatal"
            )
        
        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
        if hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        
        return self.cap.isOpened()
    
    def process_detections(self, frame, result):
        """Turn one frame's detection result into counts + an annotated frame.

        Inference (model.predict) is done centrally in batch by the
        BatchInferenceCoordinator; `result` is the ultralytics Results object
        for this frame. Object identity/counting is handled entirely by the
        custom CentroidTracker below (the model's own track IDs were never
        used), so counting behaviour is identical to before.
        """
        try:
            detections = []
            frame_confidences = []
            min_box_area = 400
            boxes = result.boxes if result is not None else None
            if boxes is not None and len(boxes) > 0:
                # Pull all boxes/confidences to CPU in ONE transfer instead of a
                # per-box GPU->CPU sync (each sync stalls while holding the GIL).
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < min_box_area:
                        continue

                    if confs is not None:
                        frame_confidences.append(float(confs[i]))

                    if self.box_shrink > 0:
                        w = x2 - x1
                        h = y2 - y1
                        shrink_w = w * self.box_shrink / 2
                        shrink_h = h * self.box_shrink / 2
                        x1, y1 = x1 + shrink_w, y1 + shrink_h
                        x2, y2 = x2 - shrink_w, y2 - shrink_h

                    detections.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if self.conf_monitor:
                self.conf_monitor.record(frame_confidences)

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
                        if previous_zone == 'upper' and current_zone == 'lower':
                            self.in_count += 1
                            self.counted_ids.add(object_id)
                            self.log_event('IN', object_id)
                        elif previous_zone == 'lower' and current_zone == 'upper':
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
        """Detect downtime periods from system health logs (cached).

        detect_downtime_gaps() is an expensive SQLite scan (~200ms). The /stats
        endpoint is polled ~2x/sec by the dashboard, and running this query on
        every poll (for every pool) held the GIL for hundreds of ms/sec in the
        Flask threads, starving the inference thread and cutting processing FPS
        by ~3x. Downtime changes slowly, so we refresh at most once per 60s and
        serve a cached result the rest of the time.
        """
        if not self.db_handler:
            return []
        now = time.time()
        cache = getattr(self, '_downtime_cache', None)
        cache_ts = getattr(self, '_downtime_cache_ts', 0)
        if cache is not None and (now - cache_ts) < 60:
            return cache
        try:
            periods = self.db_handler.detect_downtime_gaps(gap_threshold_minutes=5, pool_id=self.pool_id)
        except Exception as e:
            print(f"⚠️  [{self.pool_id}] Downtime detection error: {e}")
            periods = cache if cache is not None else []
        self._downtime_cache = periods
        self._downtime_cache_ts = now
        return periods
    
    def start(self):
        """Start the capture (reader) thread for this pool.

        Only frame capture runs per-pool now. Inference is performed centrally
        for all pools by the BatchInferenceCoordinator (one shared model, one
        batched predict call) so the GPU is fed efficiently without per-pool
        inference threads competing for the GIL.
        """
        self.is_running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def get_latest_frame(self):
        """Return (seq, frame) for the newest decoded frame, or (seq, None)."""
        with self._raw_lock:
            return self._raw_seq, self._latest_raw

    def apply_result(self, frame, result):
        """Run tracking/counting/drawing for one inference result and publish
        the annotated frame + processing-FPS tick. Called by the coordinator."""
        processed_frame = self.process_detections(frame, result)
        self._record_fps_event(self._proc_times)  # processing FPS: a frame finished processing
        _, self.fps = self.get_recent_fps()
        with self.lock:
            self.frame = processed_frame

    def _reader_loop(self):
        """Continuously drain the RTSP stream and keep only the latest frame.

        Reconnects indefinitely with capped exponential backoff so that a
        reachable, streaming camera never stays stuck at 0 fps after a
        transient outage. This thread never permanently gives up on its own;
        it only exits when is_running is cleared (shutdown/reset).

        By pulling frames as fast as they arrive (~15 fps) this keeps the
        decode pipeline drained so the processing thread always has a fresh
        frame ready without paying a per-frame read-wait.
        """
        reconnect_attempts = 0
        consecutive_errors = 0

        def _backoff(attempt):
            # 0.5s, 1s, 2s ... capped at 30s so retries continue forever
            return min(30.0, 0.5 * (2 ** min(attempt, 6)))

        # Initial connect - keep retrying instead of giving up if the camera
        # is momentarily unreachable when the app starts.
        while self.is_running and not self.connect_stream():
            reconnect_attempts += 1
            wait = _backoff(reconnect_attempts)
            if reconnect_attempts % 5 == 1:
                print(f"[{self.pool_id}] Waiting for RTSP stream (attempt {reconnect_attempts}, retry in {wait:.0f}s)")
            time.sleep(wait)
        if not self.is_running:
            return
        reconnect_attempts = 0
        print(f"✓ [{self.pool_id}] Connected to RTSP stream")

        while self.is_running:
            try:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors % 5 == 1:
                        print(f"[{self.pool_id}] Stream read error (reconnect attempt {reconnect_attempts})")

                    if consecutive_errors > 10:
                        # Stream dropped - reconnect with capped backoff and
                        # keep retrying forever (never break out of the loop).
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        reconnect_attempts += 1
                        time.sleep(_backoff(reconnect_attempts))
                        if self.connect_stream():
                            print(f"✓ [{self.pool_id}] Reconnected after {reconnect_attempts} attempt(s)")
                            reconnect_attempts = 0
                            consecutive_errors = 0
                    continue

                consecutive_errors = 0
                reconnect_attempts = 0
                self._record_fps_event(self._read_times)  # input FPS: a frame arrived from the stream

                current_date = time.strftime('%Y-%m-%d')
                if current_date != self.last_reset_date:
                    print(f"\n🔄 [{self.pool_id}] Midnight auto-reset: {current_date}")
                    self.in_count = 0
                    self.out_count = 0
                    self.pool_count = 0
                    self.peak_pool_count = 0
                    self.missed_in_count = 0
                    self.counted_ids.clear()
                    self.last_reset_date = current_date
                    self.log_event('RESET', 'AUTO')

                if self.flip_code is not None:
                    frame = cv2.flip(frame, self.flip_code)
                self.frame_count += 1

                if self.frame_count % 30 == 0:
                    input_fps, proc_fps = self.get_recent_fps()
                    print(f"[{self.pool_id}] Input FPS: {input_fps:.1f} | Proc FPS: {proc_fps:.1f} | Heads: {self.current_heads} | IN: {self.in_count} | OUT: {self.out_count}")
                    self.log_heartbeat()

                # Publish the latest raw frame for the processing thread. Only
                # the newest frame is kept; frames that arrive while inference
                # is busy are intentionally dropped so we always process the
                # freshest frame (low latency).
                with self._raw_lock:
                    self._latest_raw = frame
                    self._raw_seq += 1
            except Exception as e:
                print(f"[{self.pool_id}] Reader error: {str(e)[:50]}")
                time.sleep(0.1)

    def _record_fps_event(self, times):
        """Append a timestamp and drop entries older than the rolling window."""
        now = time.time()
        with self.fps_lock:
            times.append(now)
            cutoff = now - self.fps_window
            while times and times[0] < cutoff:
                times.popleft()

    def get_recent_fps(self):
        """Return (input_fps, processing_fps) over the recent rolling window.

        Computed relative to *now*, so a stalled stream correctly reads as ~0
        even if no new frames have arrived.
        """
        now = time.time()
        cutoff = now - self.fps_window
        with self.fps_lock:
            while self._read_times and self._read_times[0] < cutoff:
                self._read_times.popleft()
            while self._proc_times and self._proc_times[0] < cutoff:
                self._proc_times.popleft()
            input_fps = len(self._read_times) / self.fps_window
            proc_fps = len(self._proc_times) / self.fps_window
        return input_fps, proc_fps

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

        input_fps, proc_fps = self.get_recent_fps()

        return {
            'in_count': max(0, in_count),
            'out_count': max(0, out_count),
            'pool_count': max(0, pool_count),
            'current_heads': max(0, self.current_heads),
            'fps': round(proc_fps, 1),
            'input_fps': round(input_fps, 1),
            'missed_in_count': self.missed_in_count,
            'peak_pool_count': self.peak_pool_count,
            'detection_accuracy': round(detection_accuracy, 1),
            'timestamp': time.time(),
            'downtime_periods': downtime_periods,
            'total_downtime_minutes': round(total_downtime_minutes, 1),
            'has_downtime': len(downtime_periods) > 0
        }
    
    def reset_counters(self):
        """Reset all live counters for this pool (manual reset)."""
        self.in_count = 0
        self.out_count = 0
        self.pool_count = 0
        self.peak_pool_count = 0
        self.missed_in_count = 0
        self.counted_ids.clear()

    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.conf_monitor:
            self.conf_monitor.flush()
        if self.db_handler:
            self.update_database_stats()
            self.db_handler.close()


class BatchInferenceCoordinator:
    """Run YOLO inference for all pools from a single thread, in one batch.

    Each pool's reader thread keeps only its latest decoded frame. This
    coordinator collects the newest frame from every pool that has produced a
    new one and runs them through a single shared model in ONE batched
    predict() call, then hands each result back to its pool for tracking,
    counting and drawing.

    Why: running one inference pipeline per pool meant several Python threads
    contending for the GIL around ultralytics' per-call pre/post-processing.
    The GPU sat ~85% idle and processing FPS was capped at ~5-6/pool even
    though a single pipeline can do ~26 FPS. Batching removes the contention:
    one thread, one model, one predict() call feeding the GPU efficiently.
    """

    def __init__(self, processors, model_path='yolo11x.pt', conf=0.10,
                 iou=0.60, imgsz=960, max_det=100):
        self.processors = processors
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_running = False
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"✓ Shared inference model loaded on {self.device} (batched over {len(processors)} pool(s))")

    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last_seq = {pid: 0 for pid in self.processors}
        while self.is_running:
            try:
                batch_frames = []
                batch_targets = []  # parallel list of (processor, frame)
                for pid, proc in self.processors.items():
                    if proc is None:
                        continue
                    seq, frame = proc.get_latest_frame()
                    if frame is None or seq == last_seq[pid]:
                        continue
                    last_seq[pid] = seq
                    batch_frames.append(frame)
                    batch_targets.append((proc, frame))

                if not batch_frames:
                    # No new frames from any pool yet - wait briefly.
                    time.sleep(0.003)
                    continue

                results = self.model.predict(
                    batch_frames,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    imgsz=self.imgsz,
                    device=self.device,
                    half=(self.device == 'cuda'),
                    max_det=self.max_det,
                    classes=[0]
                )

                for (proc, frame), result in zip(batch_targets, results):
                    try:
                        proc.apply_result(frame, result)
                    except Exception as e:
                        print(f"[{proc.pool_id}] apply_result error: {str(e)[:50]}")
            except Exception as e:
                print(f"[coordinator] inference error: {str(e)[:60]}")
                time.sleep(0.1)

    def stop(self):
        self.is_running = False


# Flask application
app = Flask(__name__)

# Global processors for both pools
processors = {}

# Shared batched-inference coordinator (created in main())
inference_coordinator = None


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('rtsp_dashboard.html')


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
                'current_heads': 0, 'fps': 0, 'input_fps': 0, 'missed_in_count': 0,
                'peak_pool_count': 0, 'timestamp': time.time()
            }
    return jsonify(result)


def generate_frames(pool_id):
    processor = processors.get(pool_id)
    if not processor:
        return
    while True:
        frame = processor.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (960, int(h * scale)))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<pool_id>')
def video_feed(pool_id):
    if pool_id not in processors:
        return "Pool not found", 404
    return Response(generate_frames(pool_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# NOTE: The /reset route and analytics report routes are registered via
# register_analytics_routes() in main(). The reset behaviour is unchanged -
# it now calls each processor's reset_counters() method.


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
    rtsp_url_1 = "rtsp://admin:Ele%23%23%23313@10.196.211.60:554/"
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
        box_shrink=box_shrink
    )
    
    # Initialize Pool 2
    print(f"🏊 Initializing Pool 02...")
    processors['pool2'] = RTSPStreamProcessor(
        rtsp_url=rtsp_url_2,
        pool_id='pool2',
        model_path=model_path,
        conf_threshold=conf_threshold,
        box_shrink=box_shrink
    )
    
    # Start both processors (reader threads only - capture frames)
    processors['pool1'].start()
    processors['pool2'].start()

    # Start the shared batched-inference coordinator (does YOLO for all pools
    # in one thread / one batch, so the GPU is fed without GIL contention).
    global inference_coordinator
    inference_coordinator = BatchInferenceCoordinator(
        processors,
        model_path=model_path,
        conf=conf_threshold,
        iou=0.60,
        imgsz=960,
        max_det=100
    )
    inference_coordinator.start()

    # Start Telegram low-FPS alert monitor (alerts when input or processing FPS < 7)
    start_fps_monitor(processors)

    # Register analytics report routes + reset route
    if ANALYTICS_AVAILABLE:
        report_manager = None
        if DB_AVAILABLE:
            try:
                report_manager = ReportManager()
                report_manager.start_scheduler()
                print("✓ Analytics reports enabled (scheduled emails active)")
            except Exception as e:
                print(f"⚠️  Could not initialize report manager: {e}")
        register_analytics_routes(app, processors, report_manager=report_manager)
    else:
        # Fallback: keep the reset endpoint available without the analytics module
        @app.route('/reset', methods=['POST'])
        def reset_counts():
            for processor in processors.values():
                if processor:
                    processor.reset_counters()
            return jsonify({'success': True, 'message': 'All counters reset successfully'})

    print(f"\n{'='*60}")
    print(f"🎯 Wonderla Dual Pool Monitoring Dashboard")
    print(f"{'='*60}")
    print(f"🏊 Pool 01: {rtsp_url_1[:60]}...")
    print(f"🏊 Pool 02: {rtsp_url_2[:60]}...")
    print(f"🌐 Dashboard: http://{host}:{port}")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
