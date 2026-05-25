# ============================================================
# Predictive Road Risk Assessment and Driver Assistance System
# Magarpatta, Pune — Real GeoJSON Route Simulation (FIXED)
# ============================================================
#
# FIX: Replaced speed (points-per-frame) with frame_hold (frames-per-point)
#      so GPS moves at a realistic pace through the map.
#      Route is now curated: Low → Medium → High → Medium → Low
#      giving only 4 clean transitions instead of 197 chaotic ones.
#      Risk display uses a smoothing window to prevent boundary flicker.

import cv2
# from networkx import display
import torch
import numpy as np
import json
import math
import serial
import time
import argparse
from collections import Counter
from model import TrafficSignNet
from transform import get_test_transforms, CLAHE_GRAY
from shapely.geometry import Point, shape

# ── 43 GTSRB Class Names ─────────────────────────────────────
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t", "Right-of-way",
    "Priority road", "Yield", "Stop", "No vehicles", "No vehicles >3.5t",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End restrictions", "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End no passing", "End no passing >3.5t"
]

# ── Sign → hazard category ────────────────────────────────────
def get_sign_category(sign_name):
    s = sign_name.lower()
    if 'stop' in s:                              return 'STOP'
    if 'speed limit' in s or 'end speed' in s:  return 'SPEED'
    if 'yield' in s:                             return 'YIELD'
    if 'no entry' in s:                          return 'WRONG_WAY'
    if 'no passing' in s:                        return 'NO_PASS'
    if 'road work' in s:                         return 'ROADWORK'
    if 'pedestrian' in s or 'children' in s:     return 'PEDESTRIAN'
    if 'slippery' in s or 'ice' in s:            return 'SLIPPERY'
    if 'caution' in s or 'curve' in s or 'bumpy' in s: return 'CAUTION'
    if 'right-of-way' in s or 'priority' in s:  return 'PRIORITY'
    return 'GENERAL'

# ── Combined assistance logic ─────────────────────────────────
def get_assistance(sign_detected, sign_name, risk_level, risk_score):
    cat  = get_sign_category(sign_name) if sign_detected else None
    tier = 'HIGH' if risk_score >= 6 else ('MEDIUM' if risk_score >= 3 else 'LOW')

    if not sign_detected:
        if tier == 'HIGH':
            return ('ALERT_ROAD',
                    f'HIGH RISK ZONE  |  Score: {risk_score:.1f}',
                    'Reduce speed. High accident probability on this road.',
                    'RED')
        elif tier == 'MEDIUM':
            return ('CAUTION_ROAD',
                    f'MODERATE RISK  |  Score: {risk_score:.1f}',
                    'Stay alert. Moderate risk road conditions ahead.',
                    'YELLOW')
        else:
            return ('SAFE',
                    f'SAFE ZONE  |  Score: {risk_score:.1f}',
                    'Road conditions are safe. Drive normally.',
                    'GREEN')

    messages = {
        ('STOP',      'HIGH'):   ('STOP sign in HIGH RISK zone!',
                                  'Come to a complete stop. High accident intersection.'),
        ('STOP',      'MEDIUM'): ('STOP sign ahead.',
                                  'Slow down and come to a complete stop.'),
        ('STOP',      'LOW'):    ('STOP sign detected.',
                                  'Stop at the intersection.'),
        ('SPEED',     'HIGH'):   ('Speed limit in HIGH RISK zone!',
                                  f'Strictly follow {sign_name}. High crash probability here.'),
        ('SPEED',     'MEDIUM'): (f'{sign_name} detected.',
                                  'Maintain the speed limit on this road.'),
        ('SPEED',     'LOW'):    (f'{sign_name} detected.',
                                  'Observe the posted speed limit.'),
        ('YIELD',     'HIGH'):   ('YIELD in HIGH RISK zone!',
                                  'Yield to all traffic. Dangerous road ahead.'),
        ('YIELD',     'MEDIUM'): ('Yield sign ahead.',
                                  'Give way to oncoming traffic.'),
        ('YIELD',     'LOW'):    ('Yield sign detected.',
                                  'Yield to traffic at junction.'),
        ('WRONG_WAY', 'HIGH'):   ('NO ENTRY — WRONG WAY! HIGH RISK!',
                                  'Do not enter. Extremely dangerous road.'),
        ('WRONG_WAY', 'MEDIUM'): ('NO ENTRY — Wrong way!',
                                  'Turn around. Do not proceed.'),
        ('WRONG_WAY', 'LOW'):    ('No entry sign detected.',
                                  'Do not enter this road.'),
        ('ROADWORK',  'HIGH'):   ('Road work in HIGH RISK zone!',
                                  'Slow down. Construction and high crash risk.'),
        ('ROADWORK',  'MEDIUM'): ('Road work ahead.',
                                  'Reduce speed near construction zone.'),
        ('ROADWORK',  'LOW'):    ('Road work detected.',
                                  'Drive carefully through construction area.'),
        ('PEDESTRIAN','HIGH'):   ('Pedestrian zone in HIGH RISK area!',
                                  'Watch for pedestrians. High danger zone.'),
        ('PEDESTRIAN','MEDIUM'): ('Pedestrian crossing ahead.',
                                  'Slow down. Watch for crossing pedestrians.'),
        ('PEDESTRIAN','LOW'):    ('Pedestrian area detected.',
                                  'Watch for pedestrians crossing.'),
        ('SLIPPERY',  'HIGH'):   ('Slippery road in HIGH RISK zone!',
                                  'Reduce speed drastically. High skid risk.'),
        ('SLIPPERY',  'MEDIUM'): ('Slippery road ahead.',
                                  'Reduce speed. Road may be slippery.'),
        ('SLIPPERY',  'LOW'):    ('Slippery road sign.',
                                  'Drive carefully. Road may be wet.'),
    }
    key = (cat, tier)
    if key in messages:
        title, detail = messages[key]
    else:
        title  = f'{sign_name} detected. Risk score: {risk_score:.1f}'
        detail = 'Stay alert and drive carefully.'

    led = 'ALERT' if tier == 'HIGH' else 'YELLOW'
    return ('SIGN_RISK', title, detail, led)

# ── Haversine distance (meters) ───────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((lat2 - lat1) * math.pi / 360) ** 2 +
         math.cos(phi1) * math.cos(phi2) *
         math.sin((lon2 - lon1) * math.pi / 360) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Nearest-neighbour sort within a group ─────────────────────
def nn_sort(segs):
    if not segs:
        return segs
    visited = [False] * len(segs)
    order = [0]
    visited[0] = True
    for _ in range(len(segs) - 1):
        last = segs[order[-1]]['geometry']['coordinates'][-1]  # [lon, lat]
        best_d, best_j = float('inf'), -1
        for j, s in enumerate(segs):
            if visited[j]:
                continue
            first = s['geometry']['coordinates'][0]
            d = haversine(last[1], last[0], first[1], first[0])
            if d < best_d:
                best_d, best_j = d, j
        order.append(best_j)
        visited[best_j] = True
    return [segs[i] for i in order]

# ── Load GeoJSON and build curated demo route ─────────────────
def load_geojson(path):
    with open(path) as f:
        data = json.load(f)
    feats = data['features']

    # Split by risk level
    low_segs  = [f for f in feats if f['properties']['RiskLevel'] == 'Low']
    med_segs  = [f for f in feats if f['properties']['RiskLevel'] == 'Medium']
    high_segs = [f for f in feats if f['properties']['RiskLevel'] == 'High']

    print(f"[GeoJSON] Loaded {len(feats)} segments — "
          f"Low: {len(low_segs)}, Medium: {len(med_segs)}, High: {len(high_segs)}")

    # Sort each group geographically (nearest-neighbour)
    print("[GeoJSON] Sorting segments geographically...")
    low_sorted  = nn_sort(low_segs)
    med_sorted  = nn_sort(med_segs)
    high_sorted = nn_sort(high_segs)

    # Curated demo route: Low → Med → High → Med(rev) → Low(rev)
    # This gives exactly 4 clean risk transitions and a coherent narrative arc
    demo_segs = (low_sorted + med_sorted + high_sorted +
                 med_sorted[::-1] + low_sorted[::-1])

    # Build flat (lat, lon, risk, score, osmid) waypoint list
    route = []
    for feat in demo_segs:
        props = feat['properties']
        risk  = props['RiskLevel']
        score = float(props.get('RiskScore', 0))
        osmid = str(props.get('osmid', ''))
        for lon, lat in feat['geometry']['coordinates']:
            route.append((lat, lon, risk, score, osmid))

    # Deduplicate consecutive identical coords
    unique = [route[0]]
    for pt in route[1:]:
        if pt[:2] != unique[-1][:2]:
            unique.append(pt)

    print(f"[Route] {len(unique)} waypoints — "
          f"Low → Medium → High → Medium → Low (4 transitions)")
    print(f"[Route] At default speed: ~4 min loop. +/- keys adjust speed.")

    # Build shapely segments for fast risk lookup
    segments = []
    for feat in feats:
        props = feat['properties']
        segments.append({
            'geom':       shape(feat['geometry']),
            'risk_level': props['RiskLevel'],
            'risk_score': float(props.get('RiskScore', 0)),
            'osmid':      str(props.get('osmid', '')),
        })

    return segments, unique

def get_road_info(segments, lat, lon):
    pt = Point(lon, lat)
    min_d, nearest = float('inf'), segments[0]
    for seg in segments:
        d = pt.distance(seg['geom'])
        if d < min_d:
            min_d = d
            nearest = seg
    return nearest['risk_level'], nearest['risk_score'], nearest['osmid']

# ── Frame preprocessing ───────────────────────────────────────
def preprocess_frame(frame):
    clahe     = CLAHE_GRAY()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(frame_rgb, (32, 32))
    gray      = clahe(resized)[:, :, 0:1]
    tensor    = get_test_transforms()(gray).unsqueeze(0)
    return tensor

# ── Serial to ESP32 ───────────────────────────────────────────
def send_command(ser, cmd):
    if ser:
        try:
            ser.write((cmd + '\n').encode())
        except Exception:
            pass

# ── Overlay drawing ───────────────────────────────────────────
STATUS_COLORS = {
    'GREEN':  (0, 200, 80),
    'YELLOW': (30, 200, 220),
    'RED':    (60, 60, 220),
    'ALERT':  (30, 30, 200),
}

def draw_overlay(frame, state):
    h, w   = frame.shape[:2]
    led    = state['led']
    color  = STATUS_COLORS.get(led, (255, 255, 255))

    # Top status bar
    cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.putText(frame, state['title'], (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Alert border flash
    if led in ('RED', 'ALERT'):
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), color, 4)

    # Bottom panel
    cv2.rectangle(frame, (0, h - 115), (w, h), (15, 15, 15), -1)

    cv2.putText(frame, state['detail'], (10, h - 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    sign_color = color if state['confidence'] > 0 else (120, 120, 120)
    cv2.putText(frame,
                f"Sign: {state['sign_name']}  ({state['confidence']:.0f}%)",
                (10, h - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sign_color, 1)

    risk_color = (60, 60, 220) if state['risk_level'] == 'High' else (
        (30, 200, 220) if state['risk_level'] == 'Medium' else (0, 200, 80))
    cv2.putText(frame,
                f"Road Risk: {state['risk_level']}  |  "
                f"Score: {state['risk_score']:.2f}  |  OSM: {state['osmid']}",
                (10, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_color, 1)

    # GPS + speed + controls
    spd_kmh = state['speed_kmh']
    cv2.putText(frame,
                f"GPS: {state['lat']:.5f}, {state['lon']:.5f}   "
                f"Sim speed: {spd_kmh:.0f} km/h   Route: {state['route_pct']:.1f}%   "
                f"[SPACE]=pause  [+/-]=speed  [Q]=quit",
                (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    # Progress bar
    bar_w = int((w - 20) * state['route_pct'] / 100)
    cv2.rectangle(frame, (10, h - 5), (w - 10, h - 2), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 5), (10 + bar_w, h - 2), color, -1)

    return frame

# ── Main ──────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Model] Loading on {device}")
    model = TrafficSignNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("[Model] Ready")

    segments, route = load_geojson(args.geojson)

    # Serial
    ser = None
    if args.port:
        try:
            ser = serial.Serial(args.port, 9600, timeout=1)
            time.sleep(2)
            print(f"[Serial] Connected on {args.port}")
        except Exception as e:
            print(f"[Serial] Skipping hardware: {e}")

    # Camera
    source = 0 if args.webcam else args.stream_url
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[Camera] Cannot open source")
        return

    # ── State ──────────────────────────────────────
    route_idx     = 0
    frame_counter = 0       # counts frames held on current waypoint
    frame_hold    = 3       # frames per waypoint (lower = faster GPS movement)
                            # at frame_hold=3, 30fps, 63m/waypoint → ~75 km/h simulated
    paused        = False
    last_alert    = ''

    # Sign detection rolling window
    CONF_THRESHOLD  = 60
    WINDOW_SIZE     = 20
    AGREE_THRESHOLD = 15
    pred_window     = []

    # Risk smoothing — look at last N waypoints, take majority
    # This prevents boundary flicker between adjacent segments of different risk
    RISK_SMOOTH     = 5     # waypoints to smooth over
    risk_window     = []    # list of (risk_level, risk_score, osmid)

    # Cache current road info (updated only when waypoint advances)
    cur_risk_level = route[0][2]
    cur_risk_score = route[0][3]
    cur_osmid      = route[0][4]

    AVG_DIST_M = 63.0       # average distance between waypoints (meters)
    FPS_EST    = 30.0       # estimated fps

    print("\n[System] Running.")
    print("         SPACE=pause  +=faster  -=slower  Q=quit\n")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Lost feed.")
                break
            frame = cv2.resize(frame, (960, 720), interpolation=cv2.INTER_LINEAR)
            

            # ── Sign recognition with temporal smoothing ──────
            tensor = preprocess_frame(frame).to(device)
            output = model(tensor)
            probs  = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            conf_pct = conf.item() * 100

            pred_window.append((pred.item(), conf_pct))
            if len(pred_window) > WINDOW_SIZE:
                pred_window.pop(0)

            high_conf = [(p, c) for p, c in pred_window if c >= CONF_THRESHOLD]
            if len(high_conf) >= AGREE_THRESHOLD:
                votes = Counter(p for p, c in high_conf)
                top_sign, top_count = votes.most_common(1)[0]
                if top_count >= AGREE_THRESHOLD:
                    avg_conf      = np.mean([c for p, c in high_conf if p == top_sign])
                    sign_name     = CLASS_NAMES[top_sign]
                    sign_conf     = avg_conf
                    sign_detected = True
                else:
                    sign_name, sign_conf, sign_detected = "Scanning...", 0, False
            else:
                sign_name, sign_conf, sign_detected = "Scanning...", 0, False

            # ── GPS simulation: advance after frame_hold frames ──
            if not paused:
                frame_counter += 1
                if frame_counter >= frame_hold:
                    frame_counter = 0
                    route_idx += 1
                    if route_idx >= len(route):
                        route_idx = 0   # loop the route

                    # Update road info from GeoJSON
                    lat, lon = route[route_idx][0], route[route_idx][1]
                    rl, rs, oid = get_road_info(segments, lat, lon)

                    # Add to smoothing window
                    risk_window.append((rl, rs, oid))
                    if len(risk_window) > RISK_SMOOTH:
                        risk_window.pop(0)

                    # Majority-vote risk for display
                    votes_risk  = Counter(r[0] for r in risk_window)
                    cur_risk_level = votes_risk.most_common(1)[0][0]
                    # Average score of the majority risk
                    majority_entries = [r for r in risk_window if r[0] == cur_risk_level]
                    cur_risk_score = sum(r[1] for r in majority_entries) / len(majority_entries)
                    # osmid from latest waypoint
                    cur_osmid = risk_window[-1][2]

            # Current position (may not change every frame — that's intentional)
            lat  = route[route_idx][0]
            lon  = route[route_idx][1]
            pct  = route_idx / max(len(route) - 1, 1) * 100

            # ── Assistance decision ───────────────────────────
            _, title, detail, led_cmd = get_assistance(
                sign_detected, sign_name, cur_risk_level, cur_risk_score)

            if led_cmd != last_alert:
                send_command(ser, led_cmd)
                last_alert = led_cmd

            # ── Simulated speed in km/h ───────────────────────
            speed_kmh = (AVG_DIST_M / (frame_hold / FPS_EST)) * 3.6

            # ── Draw overlay ──────────────────────────────────
            display = frame.copy()
            display = draw_overlay(display, {
                'led':        led_cmd,
                'title':      title,
                'detail':     detail,
                'sign_name':  sign_name,
                'confidence': sign_conf,
                'risk_level': cur_risk_level,
                'risk_score': cur_risk_score,
                'osmid':      cur_osmid,
                'lat':        lat,
                'lon':        lon,
                'route_pct':  pct,
                'speed_kmh':  speed_kmh,
            })
            
            cv2.imshow("Driver Assistance System — Magarpatta, Pune", display)

            # ── Key controls ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"[Sim] {'Paused' if paused else 'Resumed'}")
            elif key in (ord('+'), ord('=')):
                frame_hold = max(frame_hold - 1, 1)
                speed_kmh  = (AVG_DIST_M / (frame_hold / FPS_EST)) * 3.6
                print(f"[Sim] frame_hold={frame_hold} → {speed_kmh:.0f} km/h")
            elif key == ord('-'):
                frame_hold = min(frame_hold + 1, 30)
                speed_kmh  = (AVG_DIST_M / (frame_hold / FPS_EST)) * 3.6
                print(f"[Sim] frame_hold={frame_hold} → {speed_kmh:.0f} km/h")

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Assistance System')
    parser.add_argument('--model',      default='model.pt',
                        help='Trained model .pt file')
    parser.add_argument('--geojson',    required=True,
                        help='Path to Predictions.geojson')
    parser.add_argument('--port',       default=None,
                        help='COM port for ESP32 e.g. COM5')
    parser.add_argument('--stream-url', default='http://192.168.1.100/stream',
                        help='ESP32-CAM MJPEG URL')
    parser.add_argument('--webcam',     action='store_true',
                        help='Use laptop webcam instead of ESP32-CAM')
    args = parser.parse_args()
    main(args)
