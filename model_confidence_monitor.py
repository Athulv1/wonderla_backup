"""
Model Confidence Monitor

Tracks the YOLO detection-confidence distribution per hour and appends one
summary line per hour to a per-pool log file.

Why confidence and not "accuracy": true accuracy needs ground-truth labels
(the real number of people in frame), which a live RTSP stream does not
provide. The confidence distribution the model produces is the standard
production proxy for model health - a drop in average confidence or a spike
in detections near the threshold flags degrading conditions (glare, night,
fog, camera issues).
"""

from datetime import datetime
from threading import Lock


class ModelConfidenceMonitor:
    """Accumulate detection confidences in memory and flush an hourly summary."""

    def __init__(self, pool_id, conf_threshold, log_path=None):
        self.pool_id = pool_id
        self.conf_threshold = conf_threshold
        self.log_path = log_path or f'model_confidence_{pool_id}.log'
        self._lock = Lock()
        self._reset_hour(datetime.now())

    def _reset_hour(self, now):
        self.date = now.strftime('%Y-%m-%d')
        self.hour = now.hour
        self.sum_conf = 0.0
        self.count = 0          # total detections this hour
        self.min_conf = None
        self.max_conf = None
        self.frames = 0         # frames processed this hour

    def record(self, confidences):
        """Record the detection confidences from one processed frame.

        `confidences` is an iterable of float scores for the detections kept
        in that frame (may be empty when nothing was detected).
        """
        now = datetime.now()
        with self._lock:
            if now.hour != self.hour or now.strftime('%Y-%m-%d') != self.date:
                self._flush_locked()
                self._reset_hour(now)
            self.frames += 1
            for c in confidences:
                c = float(c)
                self.sum_conf += c
                self.count += 1
                if self.min_conf is None or c < self.min_conf:
                    self.min_conf = c
                if self.max_conf is None or c > self.max_conf:
                    self.max_conf = c

    def _flush_locked(self):
        if self.frames == 0:
            return
        avg = (self.sum_conf / self.count) if self.count else 0.0
        min_c = self.min_conf if self.min_conf is not None else 0.0
        max_c = self.max_conf if self.max_conf is not None else 0.0
        line = (
            f"{self.date} {self.hour:02d}:00 | pool={self.pool_id} | "
            f"threshold={self.conf_threshold:.2f} | "
            f"frames={self.frames} | detections={self.count} | "
            f"avg_conf={avg:.3f} | min_conf={min_c:.3f} | max_conf={max_c:.3f}\n"
        )
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
            print(f"⚠️  [{self.pool_id}] Confidence log write error: {e}")

    def flush(self):
        """Write the current hour's summary (call on shutdown)."""
        with self._lock:
            self._flush_locked()
