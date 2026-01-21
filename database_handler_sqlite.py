"""
SQLite Database Handler for Head Counter
Stores counting events and statistics without requiring a database server
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path


class DatabaseHandler:
    """Handle SQLite database operations for head counting"""
    
    def __init__(self, db_path='head_counter.db'):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables"""
        # Events table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                object_id INTEGER,
                total_in INTEGER,
                total_out INTEGER,
                net_count INTEGER
            )
        ''')
        
        # Daily summary table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date TEXT PRIMARY KEY,
                total_in INTEGER DEFAULT 0,
                total_out INTEGER DEFAULT 0,
                net_count INTEGER DEFAULT 0,
                peak_pool_count INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')
        
        # Migrate existing table if needed
        self._migrate_tables()
        
        # Hourly statistics table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                hour INTEGER NOT NULL,
                total_in INTEGER DEFAULT 0,
                total_out INTEGER DEFAULT 0,
                net_count INTEGER DEFAULT 0,
                last_updated TEXT,
                UNIQUE(date, hour)
            )
        ''')
        
        # System health table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                fps REAL,
                current_heads INTEGER,
                status TEXT
            )
        ''')
        
        self.conn.commit()
    
    def _migrate_tables(self):
        """Migrate existing tables to add missing columns"""
        try:
            # Check if last_updated column exists in daily_summary
            self.cursor.execute("PRAGMA table_info(daily_summary)")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if 'last_updated' not in columns:
                print("📦 Migrating daily_summary: adding last_updated column...")
                self.cursor.execute('''
                    ALTER TABLE daily_summary ADD COLUMN last_updated TEXT
                ''')
                self.conn.commit()
                print("✓ daily_summary migration complete")
            
            # Check if last_updated column exists in hourly_stats
            self.cursor.execute("PRAGMA table_info(hourly_stats)")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if columns and 'last_updated' not in columns:
                print("📦 Migrating hourly_stats: adding last_updated column...")
                self.cursor.execute('''
                    ALTER TABLE hourly_stats ADD COLUMN last_updated TEXT
                ''')
                self.conn.commit()
                print("✓ hourly_stats migration complete")
            
            # Check if status column exists in system_health
            self.cursor.execute("PRAGMA table_info(system_health)")
            columns = [col[1] for col in self.cursor.fetchall()]
            
            if columns and 'status' not in columns:
                print("📦 Migrating system_health: adding status column...")
                self.cursor.execute('''
                    ALTER TABLE system_health ADD COLUMN status TEXT
                ''')
                self.conn.commit()
                print("✓ system_health migration complete")
        except Exception as e:
            # Table might not exist yet, that's okay
            pass
    
    def log_event(self, event_type, object_id, total_in, total_out, net_count):
        """Log a counting event"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('''
            INSERT INTO events (timestamp, event_type, object_id, total_in, total_out, net_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, event_type, object_id, total_in, total_out, net_count))
        self.conn.commit()
    
    def update_daily_summary(self, total_in, total_out, net_count, peak_pool_count):
        """Update or insert daily summary"""
        date = datetime.now().strftime('%Y-%m-%d')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.cursor.execute('''
            INSERT INTO daily_summary (date, total_in, total_out, net_count, peak_pool_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_in = ?,
                total_out = ?,
                net_count = ?,
                peak_pool_count = MAX(peak_pool_count, ?),
                last_updated = ?
        ''', (date, total_in, total_out, net_count, peak_pool_count, timestamp,
              total_in, total_out, net_count, peak_pool_count, timestamp))
        self.conn.commit()
    
    def update_hourly_statistics(self, total_in, total_out, net_count):
        """Update or insert hourly statistics"""
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        hour = now.hour
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
        
        self.cursor.execute('''
            INSERT INTO hourly_stats (date, hour, total_in, total_out, net_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, hour) DO UPDATE SET
                total_in = ?,
                total_out = ?,
                net_count = ?,
                last_updated = ?
        ''', (date, hour, total_in, total_out, net_count, timestamp,
              total_in, total_out, net_count, timestamp))
        self.conn.commit()
    
    def log_system_health(self, fps, current_heads, status):
        """Log system health metrics"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute('''
            INSERT INTO system_health (timestamp, fps, current_heads, status)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, fps, current_heads, status))
        self.conn.commit()
    
    def get_today_summary(self):
        """Get today's summary statistics"""
        date = datetime.now().strftime('%Y-%m-%d')
        self.cursor.execute('''
            SELECT total_in, total_out, net_count, peak_pool_count
            FROM daily_summary
            WHERE date = ?
        ''', (date,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'total_in': row[0],
                'total_out': row[1],
                'net_count': row[2],
                'peak_pool_count': row[3]
            }
        return None
    
    def detect_downtime_gaps(self, gap_threshold_minutes=5):
        """
        Detect downtime periods by finding gaps in system_health logs
        Returns list of downtime periods with start, end, and duration
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get all health logs for today, ordered by time
        self.cursor.execute('''
            SELECT timestamp 
            FROM system_health 
            WHERE date(timestamp) = ?
            ORDER BY timestamp ASC
        ''', (today,))
        
        logs = self.cursor.fetchall()
        
        if len(logs) < 2:
            return []
        
        downtime_periods = []
        gap_threshold_seconds = gap_threshold_minutes * 60
        
        for i in range(1, len(logs)):
            prev_time = datetime.strptime(logs[i-1][0], '%Y-%m-%d %H:%M:%S')
            curr_time = datetime.strptime(logs[i][0], '%Y-%m-%d %H:%M:%S')
            
            gap_seconds = (curr_time - prev_time).total_seconds()
            
            # If gap is larger than threshold, it's a downtime period
            if gap_seconds > gap_threshold_seconds:
                downtime_periods.append({
                    'start': logs[i-1][0],
                    'end': logs[i][0],
                    'start_display': prev_time.strftime('%I:%M %p'),
                    'end_display': curr_time.strftime('%I:%M %p'),
                    'duration_minutes': round(gap_seconds / 60, 1),
                    'type': 'shutdown' if gap_seconds > 1800 else 'stream_issue'  # 30 min = shutdown
                })
        
        return downtime_periods
    
    def close(self):
        """Close database connection"""
        self.conn.close()
