"""
Telegram alerting helper.

Credentials are read (in priority order) from:
  1. Environment variables  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
  2. telegram_config.json    {"bot_token": "...", "chat_id": "..."}

Uses only the Python standard library (urllib) - no extra dependencies.
"""

import os
import json
import threading
import urllib.parse
import urllib.request

_CONFIG_FILE = 'telegram_config.json'
_config_cache = None


def _load_config():
    """Load (bot_token, chat_id), caching the result."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()

    if (not token or not chat_id) and os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            token = token or str(data.get('bot_token', '')).strip()
            chat_id = chat_id or str(data.get('chat_id', '')).strip()
        except Exception as e:
            print(f"[telegram] Failed to read {_CONFIG_FILE}: {e}")

    _config_cache = (token, chat_id)
    return _config_cache


def is_configured():
    """True if both a bot token and chat id are available."""
    token, chat_id = _load_config()
    return bool(token and chat_id)


def send_message(text):
    """Send a Telegram message (blocking). Returns True on success."""
    token, chat_id = _load_config()
    if not token or not chat_id:
        print("[telegram] Not configured - skipping alert "
              "(set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID or telegram_config.json)")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({'chat_id': chat_id, 'text': text}).encode('utf-8')
    try:
        req = urllib.request.Request(url, data=payload)
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"[telegram] Send failed: {e}")
        return False


def send_async(text):
    """Fire-and-forget send on a background thread so it never blocks the caller."""
    threading.Thread(target=send_message, args=(text,), daemon=True).start()
