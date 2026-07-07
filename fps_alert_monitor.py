"""
Background monitor that sends a Telegram alert when a pool's FPS drops too low.

Fires when EITHER the rolling processing FPS OR the rolling input (stream read)
FPS falls below LOW_THRESHOLD. While a pool stays low it re-sends the low alert
once per check cycle (every CHECK_INTERVAL seconds); when both climb back above
RECOVER_THRESHOLD it sends a single recovery message (per pool).

Alerts are only generated inside the active window (ALERT_START..ALERT_END,
local time). Outside those hours the monitor stays quiet.
"""

import time
import threading

import telegram_alerts

LOW_THRESHOLD = 7.0       # alert when fps < this
RECOVER_THRESHOLD = 7.5   # consider recovered when fps >= this (hysteresis avoids flapping at the boundary)
CHECK_INTERVAL = 120      # seconds between checks (2 minutes)
WARMUP_SECONDS = 20       # give a freshly started stream time to ramp up before evaluating

# Active alert window (local time, 24h). Alerts only fire between these times.
ALERT_START = (9, 0)      # 9:00 AM
ALERT_END = (18, 30)      # 6:30 PM


def _within_alert_window():
    """True if the current local time is inside [ALERT_START, ALERT_END]."""
    lt = time.localtime()
    now = lt.tm_hour * 60 + lt.tm_min
    start = ALERT_START[0] * 60 + ALERT_START[1]
    end = ALERT_END[0] * 60 + ALERT_END[1]
    return start <= now <= end


def start_fps_monitor(processors):
    """Launch the monitor on a daemon thread. `processors` is the pool_id -> processor dict."""
    thread = threading.Thread(target=_monitor_loop, args=(processors,), daemon=True)
    thread.start()
    return thread


def _monitor_loop(processors):
    if not telegram_alerts.is_configured():
        print("[fps-monitor] Telegram not configured - FPS alerts DISABLED. "
              "Set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID env vars or telegram_config.json.")
        return

    print(f"[fps-monitor] Active (alert < {LOW_THRESHOLD:.0f} fps, check every {CHECK_INTERVAL}s, "
          f"window {ALERT_START[0]:02d}:{ALERT_START[1]:02d}-{ALERT_END[0]:02d}:{ALERT_END[1]:02d})")
    alerting = {}  # pool_id -> bool (currently in an alerted/low state)

    while True:
        time.sleep(CHECK_INTERVAL)

        # Only generate alerts inside the active window; stay quiet otherwise.
        if not _within_alert_window():
            continue

        for pool_id, processor in processors.items():
            if processor is None or not getattr(processor, 'is_running', False):
                continue
            if time.time() - processor.start_time < WARMUP_SECONDS:
                continue

            try:
                input_fps, proc_fps = processor.get_recent_fps()
            except Exception as e:
                print(f"[fps-monitor] [{pool_id}] error: {e}")
                continue

            was_alerting = alerting.get(pool_id, False)
            is_low = input_fps < LOW_THRESHOLD or proc_fps < LOW_THRESHOLD
            ts = time.strftime('%Y-%m-%d %H:%M:%S')

            if is_low:
                alerting[pool_id] = True
                msg = (f"🚨 LOW FPS ALERT — {pool_id.upper()}\n"
                       f"Processing FPS: {proc_fps:.1f}\n"
                       f"Input FPS: {input_fps:.1f}\n"
                       f"Threshold: {LOW_THRESHOLD:.0f} fps\n"
                       f"Time: {ts}")
                telegram_alerts.send_async(msg)
                print(f"[fps-monitor] [{pool_id}] LOW FPS alert sent (proc={proc_fps:.1f}, input={input_fps:.1f})")

            elif was_alerting and input_fps >= RECOVER_THRESHOLD and proc_fps >= RECOVER_THRESHOLD:
                alerting[pool_id] = False
                msg = (f"✅ FPS RECOVERED — {pool_id.upper()}\n"
                       f"Processing FPS: {proc_fps:.1f}\n"
                       f"Input FPS: {input_fps:.1f}\n"
                       f"Time: {ts}")
                telegram_alerts.send_async(msg)
                print(f"[fps-monitor] [{pool_id}] FPS recovered (proc={proc_fps:.1f}, input={input_fps:.1f})")
