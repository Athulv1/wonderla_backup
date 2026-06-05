"""
SQLite Database Handler for Head Counter
Stores counting events and statistics without requiring a database server
Supports multiple pools via pool_id
"""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class DatabaseHandler:
    """Handle SQLite database operations for head counting"""

    def __init__(self, db_path='head_counter.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    object_id INTEGER,
                    total_in INTEGER,
                    total_out INTEGER,
                    net_count INTEGER,
                    pool_id TEXT DEFAULT 'pool1'
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_in INTEGER DEFAULT 0,
                    total_out INTEGER DEFAULT 0,
                    net_count INTEGER DEFAULT 0,
                    peak_pool_count INTEGER DEFAULT 0,
                    last_updated TEXT,
                    pool_id TEXT DEFAULT 'pool1',
                    UNIQUE(date, pool_id)
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS hourly_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    total_in INTEGER DEFAULT 0,
                    total_out INTEGER DEFAULT 0,
                    net_count INTEGER DEFAULT 0,
                    last_updated TEXT,
                    pool_id TEXT DEFAULT 'pool1',
                    UNIQUE(date, hour, pool_id)
                )
            ''')
            cur.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    fps REAL,
                    current_heads INTEGER,
                    status TEXT,
                    pool_id TEXT DEFAULT 'pool1'
                )
            ''')
            self.conn.commit()
        self._migrate_tables()

    def _migrate_tables(self):
        migrations = {
            'daily_summary': ['last_updated', 'pool_id'],
            'hourly_stats':  ['last_updated', 'pool_id'],
            'system_health': ['status', 'pool_id'],
            'events':        ['pool_id'],
        }
        with self._lock:
            cur = self.conn.cursor()
            for table, columns in migrations.items():
                try:
                    cur.execute(f"PRAGMA table_info({table})")
                    existing_cols = [col[1] for col in cur.fetchall()]
                    for col in columns:
                        if col not in existing_cols:
                            default = "'pool1'" if col == 'pool_id' else 'NULL'
                            print(f"Migrating {table}: adding {col}...")
                            cur.execute(f'ALTER TABLE {table} ADD COLUMN {col} TEXT DEFAULT {default}')
                            self.conn.commit()
                except Exception:
                    pass
        self._ensure_unique_constraints()

    def _ensure_unique_constraints(self):
        """Rebuild tables that are missing required UNIQUE constraints (no data loss)."""
        checks = [
            (
                'daily_summary',
                'UNIQUE(date, pool_id)',
                '''CREATE TABLE daily_summary_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_in INTEGER DEFAULT 0,
                    total_out INTEGER DEFAULT 0,
                    net_count INTEGER DEFAULT 0,
                    peak_pool_count INTEGER DEFAULT 0,
                    last_updated TEXT,
                    pool_id TEXT DEFAULT 'pool1',
                    UNIQUE(date, pool_id)
                )''',
            ),
            (
                'hourly_stats',
                'UNIQUE(date, hour, pool_id)',
                '''CREATE TABLE hourly_stats_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    total_in INTEGER DEFAULT 0,
                    total_out INTEGER DEFAULT 0,
                    net_count INTEGER DEFAULT 0,
                    last_updated TEXT,
                    pool_id TEXT DEFAULT 'pool1',
                    UNIQUE(date, hour, pool_id)
                )''',
            ),
        ]
        with self._lock:
            cur = self.conn.cursor()
            for table, unique_sig, rebuild_sql in checks:
                try:
                    cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
                    row = cur.fetchone()
                    if row and unique_sig.upper() not in row[0].upper():
                        print(f"Rebuilding {table} to fix UNIQUE constraint...")
                        cur.execute(f'DROP TABLE IF EXISTS {table}_new')
                        cur.execute(rebuild_sql)
                        cur.execute(f"PRAGMA table_info({table})")
                        cols = ','.join(c[1] for c in cur.fetchall())
                        cur.execute(f'INSERT OR IGNORE INTO {table}_new ({cols}) SELECT {cols} FROM {table}')
                        cur.execute(f'DROP TABLE {table}')
                        cur.execute(f'ALTER TABLE {table}_new RENAME TO {table}')
                        self.conn.commit()
                        print(f"✓ {table} rebuilt successfully")
                except Exception as e:
                    print(f"⚠ Constraint fix {table}: {e}")

    def log_event(self, event_type, object_id, total_in, total_out, net_count, pool_id='pool1'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                INSERT INTO events (timestamp, event_type, object_id, total_in, total_out, net_count, pool_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, event_type, object_id, total_in, total_out, net_count, pool_id))
            self.conn.commit()

    def update_daily_summary(self, total_in, total_out, net_count, peak_pool_count, pool_id='pool1'):
        date = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                INSERT INTO daily_summary (date, total_in, total_out, net_count, peak_pool_count, last_updated, pool_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, pool_id) DO UPDATE SET
                    total_in = ?,
                    total_out = ?,
                    net_count = ?,
                    peak_pool_count = MAX(peak_pool_count, ?),
                    last_updated = ?
            ''', (date, total_in, total_out, net_count, peak_pool_count, timestamp, pool_id,
                  total_in, total_out, net_count, peak_pool_count, timestamp))
            self.conn.commit()

    def update_hourly_statistics(self, total_in, total_out, net_count, pool_id='pool1'):
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        hour = now.hour
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                INSERT INTO hourly_stats (date, hour, total_in, total_out, net_count, last_updated, pool_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, hour, pool_id) DO UPDATE SET
                    total_in = ?,
                    total_out = ?,
                    net_count = ?,
                    last_updated = ?
            ''', (date, hour, total_in, total_out, net_count, timestamp, pool_id,
                  total_in, total_out, net_count, timestamp))
            self.conn.commit()

    def log_system_health(self, fps, current_heads, status, pool_id='pool1'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                INSERT INTO system_health (timestamp, fps, current_heads, status, pool_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, fps, current_heads, status, pool_id))
            self.conn.commit()

    def get_today_summary(self, pool_id='pool1'):
        date = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                SELECT total_in, total_out, net_count, peak_pool_count
                FROM daily_summary WHERE date = ? AND pool_id = ?
            ''', (date, pool_id))
            row = cur.fetchone()
        if row:
            return {'total_in': row[0], 'total_out': row[1], 'net_count': row[2], 'peak_pool_count': row[3]}
        return None

    def get_daily_peak_report(self, date_str):
        """Per-pool daily totals and peak occupancy for a given date (YYYY-MM-DD)."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                SELECT date, pool_id, total_in, total_out, net_count, peak_pool_count
                FROM daily_summary
                WHERE date = ?
                ORDER BY pool_id
            ''', (date_str,))
            rows = cur.fetchall()
        return [
            {
                'date': row[0],
                'pool_id': row[1],
                'total_entered': row[2] or 0,
                'total_exited': row[3] or 0,
                'net_count': row[4] or 0,
                'peak_pool_count': row[5] or 0,
            }
            for row in rows
        ]

    def get_daily_guest_entry_report(self, date_str):
        """Per-pool guest entry totals for a given date (YYYY-MM-DD)."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                SELECT date, pool_id, total_in
                FROM daily_summary
                WHERE date = ?
                ORDER BY pool_id
            ''', (date_str,))
            rows = cur.fetchall()
        return [
            {
                'date': row[0],
                'pool_id': row[1],
                'total_entered': row[2] or 0,
            }
            for row in rows
        ]

    def get_hourly_guest_usage_report(self, date_str, hour=None):
        """Per-pool hourly occupancy (people inside the pool) for a given date.

        The hourly_stats.net_count column stores the live pool occupancy
        (max(0, in - out)) recorded for that hour, so it represents the number
        of people inside the pool - not the cumulative entered/exited totals.
        """
        with self._lock:
            cur = self.conn.cursor()
            if hour is None:
                cur.execute('''
                    SELECT date, hour, pool_id, net_count
                    FROM hourly_stats
                    WHERE date = ?
                    ORDER BY hour, pool_id
                ''', (date_str,))
            else:
                cur.execute('''
                    SELECT date, hour, pool_id, net_count
                    FROM hourly_stats
                    WHERE date = ? AND hour = ?
                    ORDER BY hour, pool_id
                ''', (date_str, hour))
            rows = cur.fetchall()
        return [
            {
                'date': row[0],
                'hour': row[1],
                'pool_id': row[2],
                'pool_count': row[3] or 0,
            }
            for row in rows
        ]

    def detect_downtime_gaps(self, gap_threshold_minutes=5, pool_id='pool1'):
        today = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            cur = self.conn.cursor()
            cur.execute('''
                SELECT timestamp FROM system_health
                WHERE date(timestamp) = ? AND pool_id = ?
                ORDER BY timestamp ASC
            ''', (today, pool_id))
            logs = cur.fetchall()

        if len(logs) < 2:
            return []

        downtime_periods = []
        gap_threshold_seconds = gap_threshold_minutes * 60
        for i in range(1, len(logs)):
            prev_time = datetime.strptime(logs[i-1][0], '%Y-%m-%d %H:%M:%S')
            curr_time = datetime.strptime(logs[i][0], '%Y-%m-%d %H:%M:%S')
            gap_seconds = (curr_time - prev_time).total_seconds()
            if gap_seconds > gap_threshold_seconds:
                downtime_periods.append({
                    'start': logs[i-1][0],
                    'end': logs[i][0],
                    'start_display': prev_time.strftime('%I:%M %p'),
                    'end_display': curr_time.strftime('%I:%M %p'),
                    'duration_minutes': round(gap_seconds / 60, 1),
                    'type': 'shutdown' if gap_seconds > 1800 else 'stream_issue'
                })
        return downtime_periods

    def close(self):
        with self._lock:
            self.conn.close()
