"""
Gunicorn configuration for RTSP Dashboard
"""

import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Single worker since we have video processing
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5 minutes timeout for long video processing
keepalive = 5

# Restart workers after this many requests (helps with memory leaks)
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/home/user/Documents/THINKNEURALAI/wonderla_backup/logs/gunicorn_access.log"
errorlog = "/home/user/Documents/THINKNEURALAI/wonderla_backup/logs/gunicorn_error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "rtsp_dashboard"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (disabled by default)
keyfile = None
certfile = None
