"""Analytics and reset orchestration for the RTSP dashboard."""

import os
import smtplib
import traceback
from datetime import datetime, timedelta
from email.message import EmailMessage
import threading

from flask import jsonify, request, send_file
import pandas as pd

try:
    from database_handler_sqlite import DatabaseHandler
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

REPORT_TYPES = {
    'daily_peak': 'Daily Peak Report',
    'daily_guest_entry': 'Daily Guest Entry Report',
    'hourly_guest_usage': 'Hourly Guest Usage Report',
}


class ReportManager:
    """Generate, store, email, and schedule analytics reports."""

    def __init__(self, db_path='head_counter.db', report_dir='reports'):
        if not DB_AVAILABLE:
            raise RuntimeError('Database handler is not available')

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
            labels = [row.get('pool_id', 'unknown').replace('pool', 'Pool ').upper() for row in rows]
            if report_type == 'daily_peak':
                datasets = [
                    {'label': 'Entered', 'color': '#1a3ab5', 'data': [row.get('total_entered', 0) for row in rows]},
                    {'label': 'Exited', 'color': '#f5c518', 'data': [row.get('total_exited', 0) for row in rows]},
                    {'label': 'Peak', 'color': '#22a06b', 'data': [row.get('peak_pool_count', 0) for row in rows]},
                ]
            else:
                datasets = [
                    {'label': 'Entered', 'color': '#1a3ab5', 'data': [row.get('total_entered', 0) for row in rows]},
                ]
            return {'labels': labels, 'datasets': datasets}

        pools = sorted({row.get('pool_id', 'pool1') for row in rows})
        hours = sorted({row.get('hour', 0) for row in rows})
        labels = [f"{h:02d}:00" for h in hours]
        lookup = {(r.get('pool_id'), r.get('hour')): r for r in rows}

        pool_palette = {
            'pool1': '#1a3ab5',
            'pool2': '#f5a623',
        }
        datasets = []
        for pool in pools:
            display = pool.replace('pool', 'Pool ')
            color = pool_palette.get(pool, '#22a06b')
            datasets.append({
                'label': f'{display} – In Pool',
                'color': color,
                'data': [lookup.get((pool, h), {}).get('pool_count', 0) for h in hours],
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
        msg.set_content(f"Please find attached {REPORT_TYPES.get(report_type, report_type)} for {date_str}.")

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
            raise ValueError(f'Unsupported report type: {report_type}')

        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')

        if report_type == 'daily_peak':
            rows = self.db_handler.get_daily_peak_report(date_str)
            columns = ['date', 'pool_id', 'total_entered', 'total_exited', 'net_count', 'peak_pool_count']
        elif report_type == 'daily_guest_entry':
            rows = self.db_handler.get_daily_guest_entry_report(date_str)
            columns = ['date', 'pool_id', 'total_entered']
        else:
            rows = self.db_handler.get_hourly_guest_usage_report(date_str, hour=hour)
            columns = ['date', 'hour', 'pool_id', 'pool_count']

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
                traceback.print_exc()
            threading.Event().wait(30)


def register_analytics_routes(app, processors, report_manager=None):
    """Register report and reset routes on a Flask app."""

    @app.route('/api/reports/<report_type>', methods=['GET'])
    def get_report_data(report_type):
        if report_manager is None:
            return jsonify({'success': False, 'message': 'Database/reporting is disabled'}), 500
        if report_type not in REPORT_TYPES:
            return jsonify({'success': False, 'message': 'Invalid report type'}), 400

        date_str = request.args.get('date') or datetime.now().strftime('%Y-%m-%d')
        try:
            result = report_manager.generate_report(report_type, date_str=date_str, send_email=False)
            return jsonify({'success': True, **result})
        except Exception as exc:
            return jsonify({'success': False, 'message': str(exc)}), 500

    @app.route('/api/reports/generate/<report_type>', methods=['POST'])
    def generate_report(report_type):
        if report_manager is None:
            return jsonify({'success': False, 'message': 'Database/reporting is disabled'}), 500
        if report_type not in REPORT_TYPES:
            return jsonify({'success': False, 'message': 'Invalid report type'}), 400

        payload = request.get_json(silent=True) or {}
        date_str = payload.get('date') or datetime.now().strftime('%Y-%m-%d')
        send_email_now = bool(payload.get('send_email', False))

        try:
            result = report_manager.generate_report(
                report_type,
                date_str=date_str,
                send_email=send_email_now,
            )
            return jsonify({'success': True, **result})
        except Exception as exc:
            return jsonify({'success': False, 'message': str(exc)}), 500

    @app.route('/api/reports/download/<report_type>', methods=['GET'])
    def download_report(report_type):
        if report_manager is None:
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

    @app.route('/reset', methods=['POST'])
    def reset_counts():
        for processor in processors.values():
            if processor:
                processor.reset_counters()
        return jsonify({'success': True, 'message': 'All counters reset successfully'})

    return app
