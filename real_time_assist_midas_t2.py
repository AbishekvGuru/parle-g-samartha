"""
real_time_assist_midas_final.py

- YOLOv8 detection (Ultralytics)
- MiDaS monocular depth via torch.hub
- Distance estimation in meters (smoothed)
- Announce everything (T2 logic)
- Big labels with white background and black text (T1)
- Auto-calibrate on 'c' (largest person at DEFAULT_KNOWN_DISTANCE)
"""

import time, threading, queue
from collections import deque, defaultdict
import numpy as np
import cv2
import pyttsx3
from ultralytics import YOLO
import torch

# ---------------- MiDaS Setup ----------------
MIDAS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.default_transform
    midas.to(MIDAS_DEVICE).eval()
    MIDAS_AVAILABLE = True
    print("MiDaS loaded on", MIDAS_DEVICE)
except Exception as e:
    print("MiDaS load failed:", e)
    MIDAS_AVAILABLE = False

# ---------------- Parameters ----------------
CAM_ID = 0
FRAME_W, FRAME_H = 640, 360
MODEL_NAME = "yolov8n.pt"
CONF_THRESH = 0.35

# Announcement / approach / area thresholds
ANNOUNCE_INTERVAL = 3.0            # seconds between announcements for same key
APPROACH_RATIO = 1.12              # area growth ratio to indicate approaching
MIN_AREA = 2000                    # min bbox area to consider
DIST_CHANGE_THRESHOLD = 0.5        # meters change required to trigger T2 announce
MIN_DISTANCE_FOR_ANNOUNCE = 0.2    # ignore tiny noisy meters
SMOOTH_ALPHA = 0.5                 # smoothing for per-object distances

# CARE_CLASSES chosen from COCO
CARE_CLASSES = {
    "person","bicycle","car","motorcycle","bus","train","truck",
    "bench","cat","dog","chair","couch","bed","dining table",
    "bottle","cup","fork","knife","spoon","bowl","cell phone",
    "book","backpack","umbrella"
}

# Auto-calibration default (meters)
DEFAULT_KNOWN_DISTANCE = 1.0

# Depth scale (calibrated)
depth_scale = 1.0
calibrated = False
last_calibration_info = ""

# Status overlay
STATUS_DISPLAY_DURATION = 3.0
status_text = ""
status_expire = 0.0
status_lock = threading.Lock()
def set_status(text):
    global status_text, status_expire
    with status_lock:
        status_text = text
        status_expire = time.time() + STATUS_DISPLAY_DURATION
def get_status():
    with status_lock:
        if time.time() > status_expire:
            return ""
        return status_text

# ---------------- Centroid Tracker ----------------
class CentroidTracker:
    def __init__(self, max_lost=8):
        self.next_object_id = 0
        self.objects = dict()
        self.bboxes = dict()
        self.lost = dict()
        self.max_lost = max_lost
        self.history = defaultdict(lambda: deque(maxlen=8))

    def update(self, detections):
        if len(detections) == 0:
            remove_ids = []
            for oid in list(self.objects.keys()):
                self.lost[oid] = self.lost.get(oid,0)+1
                if self.lost[oid] > self.max_lost:
                    remove_ids.append(oid)
            for oid in remove_ids:
                self._deregister(oid)
            return list(self.objects.items())

        det_centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2,_,_) in detections]

        if len(self.objects) == 0:
            for i,c in enumerate(det_centroids):
                oid = self.next_object_id; self.next_object_id += 1
                self.objects[oid] = c
                self.bboxes[oid] = detections[i][:4]
                self.lost[oid] = 0
                self.history[oid].append(self._area(self.bboxes[oid]))
            return list(self.objects.items())

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]
        O = np.array(object_centroids)
        D = np.array(det_centroids)
        if O.size == 0 or D.size == 0:
            return list(self.objects.items())

        distances = np.linalg.norm(O[:, None, :] - D[None, :, :], axis=2)
        assigned_dets = set()
        assignments = {}
        rows = distances.min(axis=1).argsort()
        for r in rows:
            c = distances[r].argmin()
            if c in assigned_dets: continue
            assigned_dets.add(c)
            assignments[object_ids[r]] = c

        new_objects = {}
        new_bboxes = {}
        for oid, det_idx in assignments.items():
            cx,cy = det_centroids[det_idx]
            new_objects[oid] = (cx,cy)
            new_bboxes[oid] = detections[det_idx][:4]
            self.lost[oid] = 0
            self.history[oid].append(self._area(new_bboxes[oid]))

        for i,det in enumerate(detections):
            if i not in assigned_dets:
                oid = self.next_object_id; self.next_object_id += 1
                cx,cy = det_centroids[i]
                new_objects[oid] = (cx,cy)
                new_bboxes[oid] = det[:4]
                self.lost[oid] = 0
                self.history[oid].append(self._area(new_bboxes[oid]))

        for oid in object_ids:
            if oid not in assignments:
                self.lost[oid] = self.lost.get(oid,0) + 1
                if self.lost[oid] <= self.max_lost:
                    new_objects[oid] = self.objects[oid]
                    new_bboxes[oid] = self.bboxes.get(oid, new_bboxes.get(oid, None))
                else:
                    self._deregister(oid)

        self.objects = new_objects
        self.bboxes = new_bboxes
        return list(self.objects.items())

    def _area(self,bbox):
        x1,y1,x2,y2 = bbox
        return max(0, x2-x1) * max(0, y2-y1)

    def _deregister(self, oid):
        self.objects.pop(oid, None)
        self.bboxes.pop(oid, None)
        self.lost.pop(oid, None)
        self.history.pop(oid, None)

# ---------------- TTS Announcer ----------------
class Announcer:
    def __init__(self, interval=ANNOUNCE_INTERVAL):
        self.q = queue.Queue()
        self.last_announcements = {}
        self.interval = interval
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        engine = pyttsx3.init()
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)
    def announce(self, key, text):
        now = time.time()
        if now - self.last_announcements.get(key, 0) < self.interval:
            return
        self.last_announcements[key] = now
        self.q.put(text)
        set_status(text)
    def stop(self):
        self._stop.set()
        self.q.put(None)
        self.thread.join(timeout=1.0)

# ---------------- Depth helpers ----------------
def run_midas(frame_rgb):
    if not MIDAS_AVAILABLE: return None
    input_batch = midas_transform(frame_rgb).to(MIDAS_DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        return prediction.cpu().numpy()

def median_depth(depth_map, bbox):
    x1,y1,x2,y2 = bbox
    h,w = depth_map.shape
    x1c,x2c = max(0,int(x1)), min(w,int(x2))
    y1c,y2c = max(0,int(y1)), min(h,int(y2))
    if x2c <= x1c or y2c <= y1c: return None
    patch = depth_map[y1c:y2c, x1c:x2c]
    if patch.size == 0: return None
    return float(np.median(patch))

def calibrate_known(observed_midas_val, known_distance_m):
    if observed_midas_val is None or observed_midas_val <= 0: return None
    return known_distance_m * observed_midas_val

# ---------------- Main loop ----------------
def run():
    global depth_scale, calibrated, last_calibration_info
    model = YOLO(MODEL_NAME)
    coco_names = model.names if hasattr(model, "names") else {}
    supported = set(coco_names.values())
    effective_care = set([c for c in CARE_CLASSES if c in supported])
    if len(effective_care) == 0:
        print("Warning: No CARE_CLASSES available in model.names. Exiting.")
        return

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    tracker = CentroidTracker()
    announcer = Announcer()
    frame_idx = 0
    depth_map = None
    smoothed_distance = {}
    last_announced_distance = {}
    last_announce_time = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.")
                break
            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_idx += 1

            # MiDaS every few frames
            if MIDAS_AVAILABLE and (frame_idx % 3 == 0):
                try:
                    depth_map = run_midas(frame_rgb)
                except Exception as e:
                    print("MiDaS error:", e)
                    depth_map = None

            # YOLO detection
            try:
                results = model.predict(frame_resized, imgsz=640, conf=CONF_THRESH, verbose=False)
            except Exception as e:
                results = []
            detections = []
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes:
                    for box in boxes:
                        try:
                            xyxy = box.xyxy[0].tolist()
                        except Exception:
                            xyxy = box.xyxy.tolist()
                        x1,y1,x2,y2 = map(int, xyxy)
                        try:
                            score = float(getattr(box, "conf", [0.0])[0])
                        except Exception:
                            score = 0.0
                        try:
                            cls_idx = int(getattr(box, "cls", [0])[0])
                        except Exception:
                            cls_idx = 0
                        label = coco_names.get(cls_idx, str(cls_idx))
                        detections.append((x1,y1,x2,y2,label,score))

            # filter relevant detections
            filtered = []
            for d in detections:
                x1,y1,x2,y2,label,score = d
                area = max(0, (x2-x1)*(y2-y1))
                if area < MIN_AREA: continue
                if label in effective_care:
                    filtered.append(d)

            tracker.update(filtered)

            # analyze tracked objects
            for oid, centroid in list(tracker.objects.items()):
                bbox = tracker.bboxes.get(oid)
                if bbox is None: continue
                x1,y1,x2,y2 = bbox
                cx,cy = centroid

                # match label from filtered detections (nearest centroid)
                matched = None
                md = 1e9
                for det in filtered:
                    dx1,dy1,dx2,dy2,label,score = det
                    dcx = int((dx1+dx2)/2); dcy = int((dy1+dy2)/2)
                    d = abs(dcx - cx) + abs(dcy - cy)
                    if d < md:
                        md = d; matched = det
                if matched is None: continue
                mx1,my1,mx2,my2,label,score = matched

                # compute distance (inverse-depth -> meters)
                approx_m = None
                if depth_map is not None:
                    midas_val = median_depth(depth_map, (x1,y1,x2,y2))
                    if midas_val is not None and midas_val > 0:
                        eps = 1e-6
                        raw_dist = depth_scale / (midas_val + eps)
                        prev = smoothed_distance.get(oid, None)
                        smooth = raw_dist if prev is None else SMOOTH_ALPHA * raw_dist + (1 - SMOOTH_ALPHA) * prev
                        smoothed_distance[oid] = smooth
                        approx_m = smooth

                # decide announcements
                will_announce = False
                announce_text = None

                # approaching detection
                areas = tracker.history.get(oid, deque())
                if len(areas) >= 3:
                    last_area = areas[-1]
                    prev_mean = np.mean(list(areas)[:-1])
                    if prev_mean > 0 and (last_area / prev_mean) > APPROACH_RATIO:
                        pos = "center"
                        if cx < FRAME_W * 0.33: pos = "left"
                        elif cx > FRAME_W * 0.66: pos = "right"
                        if approx_m is not None:
                            announce_text = f"A {label} is {approx_m:.1f} meters away, approaching from the {pos}."
                        else:
                            announce_text = f"A {label} is approaching from the {pos}."
                        will_announce = True

                # T2: distance change
                if not will_announce and approx_m is not None:
                    last = last_announced_distance.get(oid, None)
                    last_time = last_announce_time.get(oid, 0)
                    now = time.time()
                    if (last is None or abs(approx_m - last) >= DIST_CHANGE_THRESHOLD) and (now - last_time >= ANNOUNCE_INTERVAL):
                        if approx_m >= MIN_DISTANCE_FOR_ANNOUNCE:
                            pos = "center"
                            if cx < FRAME_W * 0.33: pos = "left"
                            elif cx > FRAME_W * 0.66: pos = "right"
                            announce_text = f"A {label} is {approx_m:.1f} meters away, slightly to the {pos}."
                            will_announce = True
                            last_announced_distance[oid] = approx_m
                            last_announce_time[oid] = now

                if will_announce and announce_text:
                    announcer.announce(f"obj_{oid}", announce_text)

                # draw bbox and label with white box + black text (T1)
                cv2.rectangle(frame_resized, (x1,y1),(x2,y2), (0,255,0), 2)
                if approx_m is not None:
                    text_to_draw = f"{label} {approx_m:.1f}m"
                else:
                    text_to_draw = f"{label}"
                # compute text size and draw white filled rectangle then black text
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0
                thickness = 3
                (tw, th), _ = cv2.getTextSize(text_to_draw, font, scale, thickness)
                box_x1 = x1
                box_y1 = max(y1 - th - 12, 0)
                box_x2 = x1 + tw + 12
                box_y2 = box_y1 + th + 8
                cv2.rectangle(frame_resized, (box_x1, box_y1), (box_x2, box_y2), (255,255,255), -1)
                cv2.putText(frame_resized, text_to_draw, (box_x1 + 6, box_y2 - 6),
                            font, scale, (0,0,0), thickness, cv2.LINE_AA)

            # Calibration (press 'c' to auto-calibrate using largest person)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                selected = None
                max_area = 0
                for det in filtered:
                    x1,y1,x2,y2,label,score = det
                    if label != "person": continue
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        selected = det
                if selected is not None and depth_map is not None:
                    bx1,by1,bx2,by2,label,score = selected
                    observed = median_depth(depth_map, (bx1,by1,bx2,by2))
                    if observed is not None and observed > 0:
                        new_scale = calibrate_known(observed, DEFAULT_KNOWN_DISTANCE)
                        if new_scale is not None:
                            depth_scale = new_scale
                            calibrated = True
                            last_calibration_info = f"{DEFAULT_KNOWN_DISTANCE:.2f}m (auto)"
                            set_status(f"Auto-calibrated for {DEFAULT_KNOWN_DISTANCE:.2f} m")
                            print("Auto calibration OK: depth_scale =", depth_scale)
                        else:
                            set_status("Auto calibration failed")
                    else:
                        set_status("Auto calibration failed (no depth)")
                else:
                    set_status("Auto calibration failed (no person)")

            # overlay status (top)
            st = get_status()
            if st:
                overlay = frame_resized.copy()
                cv2.rectangle(overlay, (0,0),(FRAME_W,40), (255,255,255), -1)
                frame_resized = cv2.addWeighted(overlay, 0.5, frame_resized, 0.5, 0)
                cv2.putText(frame_resized, st, (8,26), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)

            # calibration footer
            if calibrated:
                cv2.putText(frame_resized, f"Calibrated: scale={depth_scale:.3f} ({last_calibration_info})",
                            (8, FRAME_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            else:
                cv2.putText(frame_resized, f"Not calibrated. Press 'c' to auto-calibrate (~{DEFAULT_KNOWN_DISTANCE} m).",
                            (8, FRAME_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            cv2.imshow("assist_debug", frame_resized)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        announcer.stop()
        print("Exit.")

if __name__ == "__main__":
    run()
