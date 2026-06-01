"""
Web-based ROI Line Drawer for Wonderla Head Counter
----------------------------------------------------
Starts a small Flask server, opens the browser, lets you drag the
counting line on a live frame, then saves partition_y to the config.

Usage:
    python draw_roi.py            # both pools
    python draw_roi.py pool1
    python draw_roi.py pool2
"""

import sys, json, os, base64, threading, webbrowser
import cv2, numpy as np
from flask import Flask, jsonify, request, render_template_string

POOLS = {
    'pool1': {
        'url':          'rtsp://admin:Ele%23%23%23313@10.196.211.60:554/',
        'config':       'head_counter_config_pool1.json',
        'flip_vertical': True,
    },
    'pool2': {
        'url':          'rtsp://admin:Ele%23%23%23313@10.196.211.59:554/',
        'config':       'head_counter_config_pool2.json',
        'flip_vertical': False,
    },
}

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ROI Drawer – {{ pool_id }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 16px; }
  h2 { margin-bottom: 8px; color: #0ff; }
  p  { margin-bottom: 12px; color: #aaa; font-size: 14px; }
  #wrap { position: relative; display: inline-block; cursor: ns-resize; }
  #frame { display: block; max-width: 100%; border: 2px solid #333; }
  #overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
              pointer-events: none; }
  .btn { margin: 10px 6px 0; padding: 10px 28px; border: none; border-radius: 6px;
         font-size: 16px; cursor: pointer; font-weight: bold; }
  #btnSave  { background: #0a0; color: #fff; }
  #btnReset { background: #555; color: #fff; }
  #info { margin-top: 10px; font-size: 15px; color: #0ff; min-height: 20px; }
</style>
</head>
<body>
<h2>ROI Line Drawer — {{ pool_id }}</h2>
<p>Click or drag on the image to position the counting line, then click <b>Save</b>.</p>

<div id="wrap">
  <img id="frame" src="data:image/jpeg;base64,{{ frame_b64 }}"
       draggable="false">
  <canvas id="overlay"></canvas>
</div>

<div>
  <button class="btn" id="btnSave">Save</button>
  <button class="btn" id="btnReset">Reset to middle</button>
</div>
<div id="info">partition_y = <span id="yval">{{ current_y }}</span></div>

<script>
const img    = document.getElementById('frame');
const canvas = document.getElementById('overlay');
const ctx    = canvas.getContext('2d');
const yval   = document.getElementById('yval');
const wrap   = document.getElementById('wrap');

let lineY    = {{ current_y }};   // in original image pixels
let natH     = {{ frame_h }};
let natW     = {{ frame_w }};
let dragging = false;

function scaleY(clientY) {
  const rect = img.getBoundingClientRect();
  const relY  = clientY - rect.top;
  return Math.round(Math.max(1, Math.min(relY * natH / rect.height, natH - 1)));
}

function drawLine() {
  const rect = img.getBoundingClientRect();
  canvas.width  = rect.width;
  canvas.height = rect.height;
  canvas.style.width  = rect.width  + 'px';
  canvas.style.height = rect.height + 'px';

  const dispY = lineY * rect.height / natH;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // shadow for contrast
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth   = 6;
  ctx.beginPath(); ctx.moveTo(0, dispY); ctx.lineTo(canvas.width, dispY); ctx.stroke();

  // main cyan line
  ctx.strokeStyle = '#00ffff';
  ctx.lineWidth   = 3;
  ctx.beginPath(); ctx.moveTo(0, dispY); ctx.lineTo(canvas.width, dispY); ctx.stroke();

  // label
  ctx.fillStyle = '#00ffff';
  ctx.font      = 'bold 15px sans-serif';
  ctx.fillText('partition_y = ' + lineY, 10, dispY - 6);

  yval.textContent = lineY;
}

img.addEventListener('load', drawLine);
if (img.complete) drawLine();
window.addEventListener('resize', drawLine);

wrap.addEventListener('mousedown', e => { dragging = true; lineY = scaleY(e.clientY); drawLine(); });
wrap.addEventListener('mousemove', e => { if (dragging) { lineY = scaleY(e.clientY); drawLine(); } });
window.addEventListener('mouseup', () => { dragging = false; });

// touch support
wrap.addEventListener('touchstart', e => { e.preventDefault(); lineY = scaleY(e.touches[0].clientY); drawLine(); }, {passive:false});
wrap.addEventListener('touchmove',  e => { e.preventDefault(); lineY = scaleY(e.touches[0].clientY); drawLine(); }, {passive:false});

document.getElementById('btnReset').onclick = () => {
  lineY = Math.round(natH / 2); drawLine();
};

document.getElementById('btnSave').onclick = () => {
  fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pool_id: '{{ pool_id }}', partition_y: lineY})
  })
  .then(r => r.json())
  .then(d => {
    document.getElementById('info').textContent = d.message;
    if (d.next_url) {
      setTimeout(() => { window.location.href = d.next_url; }, 800);
    } else {
      setTimeout(() => { window.close(); }, 1200);
    }
  });
};
</script>
</body>
</html>
"""

app   = Flask(__name__)
state = {}   # pool_id -> frame_b64 etc.

def grab_frame(url, flip_vertical):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    frame = None
    for _ in range(10):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
    cap.release()
    if frame is None:
        return None
    if flip_vertical:
        frame = cv2.flip(frame, 0)
    return frame

def load_config(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'type': 'zones', 'invert_direction': False, 'flip_vertical': False}

def save_config(path, cfg, new_y):
    cfg['partition_y'] = new_y
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved partition_y={new_y} → {path}")

@app.route('/<pool_id>')
def roi_page(pool_id):
    if pool_id not in state:
        return f"Unknown pool: {pool_id}", 404
    s = state[pool_id]
    return render_template_string(HTML,
        pool_id=pool_id,
        frame_b64=s['frame_b64'],
        current_y=s['current_y'],
        frame_h=s['frame_h'],
        frame_w=s['frame_w'],
    )

@app.route('/save', methods=['POST'])
def save():
    data    = request.json
    pool_id = data['pool_id']
    new_y   = int(data['partition_y'])
    s       = state[pool_id]
    save_config(POOLS[pool_id]['config'], s['cfg'], new_y)
    s['current_y'] = new_y

    pools  = state['_order']
    idx    = pools.index(pool_id)
    if idx + 1 < len(pools):
        next_pool = pools[idx + 1]
        msg = f"Saved! Moving to {next_pool}…"
        return jsonify({'message': msg, 'next_url': f'/{next_pool}'})
    else:
        msg = "All done! You can close this tab."
        threading.Timer(1.5, lambda: os.kill(os.getpid(), 9)).start()
        return jsonify({'message': msg, 'next_url': None})

def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else 'all'
    pools  = [target] if target in POOLS else list(POOLS.keys())
    state['_order'] = pools

    print("=" * 55)
    print("  Wonderla Web ROI Drawer")
    print("=" * 55)

    for pid in pools:
        pcfg = POOLS[pid]
        print(f"\n  [{pid}] Grabbing frame from stream…")
        frame = grab_frame(pcfg['url'], pcfg['flip_vertical'])
        if frame is None:
            print(f"  WARNING: no frame — using blank canvas")
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(frame, f"No frame from {pid}",
                        (30, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        h, w = frame.shape[:2]
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()
        cfg = load_config(pcfg['config'])
        print(f"  [{pid}] Frame: {w}x{h}, current partition_y={cfg.get('partition_y', h//2)}")

        state[pid] = {
            'frame_b64': b64,
            'frame_h':   h,
            'frame_w':   w,
            'cfg':       cfg,
            'current_y': cfg.get('partition_y', h // 2),
        }

    first = pools[0]
    url   = f"http://127.0.0.1:5050/{first}"
    print(f"\n  Open your browser at: {url}")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=5050, debug=False)

if __name__ == '__main__':
    main()
