"""
SQLite Database Handler for Head Counter
Thread-safe, supports multiple pools via pool_id
"""

import sqlite3
import threading
from datetime import datetime


class DatabaseHandler:
    def __init__(self, db_path='head_counter.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._migrate_tables()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _execute(self, query, params=None):
        with self.lock:
            cur = self.conn.cursor()
            try:
                cur.execute(query, params or ())
                self.conn.commit()
                return cur
            except Exception:
                cur.close()
                raise

    def _fetchall(self, query, params=None):
        with self.lock:
            cur = self.conn.cursor()
            try:
                cur.execute(query, params or ())
                return cur.fetchall()
            finally:
                cur.close()

    def _fetchone(self, query, params=None):
        with self.lock:
            cur = self.conn.cursor()
            try:
                cur.execute(query, params or ())
                return cur.fetchone()
            finally:
                cur.close()

    # ------------------------------------------------------------------ #
    # Schema                                                               #
    # ------------------------------------------------------------------ #

    def _create_tables(self):
        stmts = [
            '''CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                object_id INTEGER,
                total_in INTEGER,
                total_out INTEGER,
                net_count INTEGER,
                pool_id TEXT DEFAULT 'pool1'
            )''',
            '''CREATE TABLE IF NOT EXISTS daily_summary (
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
            '''CREATE TABLE IF NOT EXISTS hourly_stats (
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
            '''CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                fps REAL,
                current_heads INTEGER,
                status TEXT,
                pool_id TEXT DEFAULT 'pool1'
            )''',
        ]
        with self.lock:
            cur = self.conn.cursor()
            for stmt in stmts:
                cur.execute(stmt)
            self.conn.commit()
            cur.close()

    def _migrate_tables(self):
        """
        Ensure pool_id column exists on all tables, and that daily_summary /
        hourly_stats have the correct composite UNIQUE constraints.
        SQLite cannot ALTER a constraint, so we rebuild those tables if needed.
        """
        simple_adds = {
            'events':        ['pool_id'],
            'system_health': ['status', 'pool_id'],
        }
        for table, cols in simple_adds.items():
            self._add_columns_if_missing(table, cols)

        self._rebuild_if_missing_unique(
            table='daily_summary',
            unique_cols=['date', 'pool_id'],
            ddl='''CREATE TABLE daily_summary (
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
            copy_cols='date, total_in, total_out, net_count, peak_pool_count, last_updated, pool_id',
        )
        self._rebuild_if_missing_unique(
            table='hourly_stats',
            unique_cols=['date', 'hour', 'pool_id'],
            ddl='''CREATE TABLE hourly_stats (
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
            copy_cols='date, hour, total_in, total_out, net_count, last_updated, pool_id',
        )

    def _add_columns_if_missing(self, table, cols):
        with self.lock:
            cur = self.conn.cursor()
            try:
                cur.execute(f'PRAGMA table_info({table})')
                existing = {row[1] for row in cur.fetchall()}
                for col in cols:
                    if col not in existing:
                        default = "'pool1'" if col == 'pool_id' else 'NULL'
                        cur.execute(f'ALTER TABLE {table} ADD COLUMN {col} TEXT DEFAULT {default}')
                        print(f'✓ Migrated {table}: added {col}')
                self.conn.commit()
            except Exception as e:
                print(f'⚠️  Migration {table}: {e}')
            finally:
                cur.close()

    def _rebuild_if_missing_unique(self, table, unique_cols, ddl, copy_cols):
        """Rebuild table if the required UNIQUE constraint is absent."""
        with self.lock:
            cur = self.conn.cursor()
            try:
                # Check existing indexes for this table
                cur.execute(f"PRAGMA index_list({table})")
                indexes = cur.fetchall()
                # Also check if any index covers the required columns
                has_unique = False
                for idx in indexes:
                    if idx[2]:  # unique flag
                        cur.execute(f"PRAGMA index_info({idx[1]})")
                        idx_cols = [r[2] for r in cur.fetchall()]
                        if sorted(idx_cols) == sorted(unique_cols):
                            has_unique = True
                            break

                # Also accept the old PRIMARY KEY on just 'date' if pool_id not in use
                if not has_unique:
                    # Ensure pool_id column exists before rebuilding
                    cur.execute(f'PRAGMA table_info({table})')
                    existing = {row[1] for row in cur.fetchall()}
                    if 'pool_id' not in existing:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN pool_id TEXT DEFAULT 'pool1'")
                        self.conn.commit()

                    print(f'📦 Rebuilding {table} to add UNIQUE({", ".join(unique_cols)})...')
                    cur.execute(f'ALTER TABLE {table} RENAME TO _{table}_old')
                    cur.execute(ddl)
                    cur.execute(f'INSERT OR IGNORE INTO {table} ({copy_cols}) SELECT {copy_cols} FROM _{table}_old')
                    cur.execute(f'DROP TABLE _{table}_old')
                    self.conn.commit()
                    print(f'✓ {table} rebuilt successfully')
            except Exception as e:
                print(f'⚠️  Rebuild {table}: {e}')
            finally:
                cur.close()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def log_event(self, event_type, object_id, total_in, total_out, net_count, pool_id='pool1'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur = self._execute(
            'INSERT INTO events (timestamp, event_type, object_id, total_in, total_out, net_count, pool_id) VALUES (?,?,?,?,?,?,?)',
            (timestamp, event_type, object_id, total_in, total_out, net_count, pool_id)
        )
        cur.close()

    def update_daily_summary(self, total_in, total_out, net_count, peak_pool_count, pool_id='pool1'):
        date = datetime.now().strftime('%Y-%m-%d')
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur = self._execute('''
            INSERT INTO daily_summary (date, total_in, total_out, net_count, peak_pool_count, last_updated, pool_id)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(date, pool_id) DO UPDATE SET
                total_in=excluded.total_in,
                total_out=excluded.total_out,
                net_count=excluded.net_count,
                peak_pool_count=MAX(peak_pool_count, excluded.peak_pool_count),
                last_updated=excluded.last_updated
        ''', (date, total_in, total_out, net_count, peak_pool_count, ts, pool_id))
        cur.close()

    def update_hourly_statistics(self, total_in, total_out, net_count, pool_id='pool1'):
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        cur = self._execute('''
            INSERT INTO hourly_stats (date, hour, total_in, total_out, net_count, last_updated, pool_id)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(date, hour, pool_id) DO UPDATE SET
                total_in=excluded.total_in,
                total_out=excluded.total_out,
                net_count=excluded.net_count,
                last_updated=excluded.last_updated
        ''', (date, now.hour, total_in, total_out, net_count, ts, pool_id))
        cur.close()

    def log_system_health(self, fps, current_heads, status, pool_id='pool1'):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cur = self._execute(
            'INSERT INTO system_health (timestamp, fps, current_heads, status, pool_id) VALUES (?,?,?,?,?)',
            (ts, fps, current_heads, status, pool_id)
        )
        cur.close()

    def get_today_summary(self, pool_id='pool1'):
        date = datetime.now().strftime('%Y-%m-%d')
        row = self._fetchone(
            'SELECT total_in, total_out, net_count, peak_pool_count FROM daily_summary WHERE date=? AND pool_id=?',
            (date, pool_id)
        )
        if row:
            return {'total_in': row[0], 'total_out': row[1], 'net_count': row[2], 'peak_pool_count': row[3]}
        return None

    def detect_downtime_gaps(self, gap_threshold_minutes=5, pool_id='pool1'):
        today = datetime.now().strftime('%Y-%m-%d')
        rows = self._fetchall(
            "SELECT timestamp FROM system_health WHERE date(timestamp)=? AND pool_id=? ORDER BY timestamp ASC",
            (today, pool_id)
        )
        if len(rows) < 2:
            return []
        periods = []
        threshold = gap_threshold_minutes * 60
        for i in range(1, len(rows)):
            t0 = datetime.strptime(rows[i-1][0], '%Y-%m-%d %H:%M:%S')
            t1 = datetime.strptime(rows[i][0], '%Y-%m-%d %H:%M:%S')
            gap = (t1 - t0).total_seconds()
            if gap > threshold:
                periods.append({
                    'start': rows[i-1][0],
                    'end': rows[i][0],
                    'start_display': t0.strftime('%I:%M %p'),
                    'end_display': t1.strftime('%I:%M %p'),
                    'duration_minutes': round(gap / 60, 1),
                    'type': 'shutdown' if gap > 1800 else 'stream_issue',
                })
        return periods

    def close(self):
        with self.lock:
            self.conn.close()
