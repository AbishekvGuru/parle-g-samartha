"""
real_time_assist_final.py
-------------------------
Purpose: Real-time assistive descriptions for blind users.
Selected options:
 - S1 phrasing (Simple left/center/right)
 - Tone T1 (calm helpful): "A person is approaching from the left."
 - Announce interval: 3 seconds (reduced spam)
 - On-screen status label shows the same final sentence for a few seconds

Install:
 pip install ultralytics opencv-python pyttsx3 numpy

Run:
 python real_time_assist_final.py
"""

import time
import threading
import queue
from collections import deque, defaultdict
import numpy as np
import cv2
import pyttsx3
from ultralytics import YOLO
import signal
import sys

# ---------------- Parameters ----------------
CAM_ID = 0
MODEL_NAME = "yolov8n.pt"   # change to your local model if needed
FRAME_W = 640
FRAME_H = 360
CONF_THRESH = 0.35
ANNOUNCE_INTERVAL = 3.0     # seconds between similar announcements (user-chosen)
APPROACH_AREA_INCREASE_RATIO = 1.12  # area increase ratio to consider approaching
MIN_AREA_TO_CONSIDER = 2000  # ignore tiny detections
DEBUG_WINDOW = True         # set False if running headless
STATUS_DISPLAY_DURATION = 3.0  # seconds to show on-screen status
# classes we care about (COCO names will fill this)
CARE_CLASSES = {"person","bed","chair","couch","sofa","bench","dog","cat","car","bicycle","motorbike"}

# Shared on-screen status (updated from main thread)
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

# -------------- Centroid Tracker (robust) --------------
class CentroidTracker:
    def __init__(self, max_lost=8):
        self.next_object_id = 0
        self.objects = dict()       # id -> centroid tuple (cx,cy)
        self.bboxes = dict()        # id -> bbox (x1,y1,x2,y2)
        self.lost = dict()          # id -> frames lost
        self.max_lost = max_lost
        self.history = defaultdict(lambda: deque(maxlen=8))  # id -> recent areas

    def update(self, detections):
        """
        detections: list of tuples (x1,y1,x2,y2,label,score)
        Returns list of (id, centroid)
        """
        # If no detections, mark objects as lost and possibly remove
        if len(detections) == 0:
            remove_ids = []
            for oid in list(self.objects.keys()):
                self.lost[oid] = self.lost.get(oid, 0) + 1
                if self.lost[oid] > self.max_lost:
                    remove_ids.append(oid)
            for oid in remove_ids:
                self._deregister(oid)
            return list(self.objects.items())

        # Build centroids for detections
        det_centroids = []
        for (x1,y1,x2,y2,_,_) in detections:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            det_centroids.append((cx, cy))

        # If no current tracked objects -> register all detections
        if len(self.objects) == 0:
            for i, c in enumerate(det_centroids):
                oid = self.next_object_id
                self.next_object_id += 1
                self.objects[oid] = c
                self.bboxes[oid] = detections[i][:4]
                self.lost[oid] = 0
                area = self._area(self.bboxes[oid])
                self.history[oid].append(area)
            return list(self.objects.items())

        # If there are detections and objects, do nearest-neighbor matching
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]

        # convert to numpy arrays
        O = np.array(object_centroids)  # shape (M,2)
        D = np.array(det_centroids)     # shape (N,2)
        if O.size == 0 or D.size == 0:
            return list(self.objects.items())

        distances = np.linalg.norm(O[:, None, :] - D[None, :, :], axis=2)  # shape (M,N)

        assigned_dets = set()
        assignments = dict()  # obj_id -> det_index

        # Greedy assignment by smallest distance
        rows = distances.min(axis=1).argsort()
        for r in rows:
            c = distances[r].argmin()
            if c in assigned_dets:
                continue
            assigned_dets.add(c)
            assignments[object_ids[r]] = c

        # Prepare new state maps
        new_objects = {}
        new_bboxes = {}

        # Update assigned objects
        for oid, det_idx in assignments.items():
            cx, cy = det_centroids[det_idx]
            new_objects[oid] = (cx, cy)
            new_bboxes[oid] = detections[det_idx][:4]
            self.lost[oid] = 0
            area = self._area(new_bboxes[oid])
            self.history[oid].append(area)

        # Register unassigned detections as new objects
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                oid = self.next_object_id
                self.next_object_id += 1
                cx, cy = det_centroids[i]
                new_objects[oid] = (cx, cy)
                new_bboxes[oid] = det[:4]
                self.lost[oid] = 0
                area = self._area(new_bboxes[oid])
                self.history[oid].append(area)

        # Carry forward unassigned previous objects (but increment lost)
        for oid in object_ids:
            if oid not in assignments:
                self.lost[oid] = self.lost.get(oid, 0) + 1
                if self.lost[oid] <= self.max_lost:
                    new_objects[oid] = self.objects[oid]
                    new_bboxes[oid] = self.bboxes.get(oid, new_bboxes.get(oid, None))
                else:
                    self._deregister(oid)

        # Commit state
        self.objects = new_objects
        self.bboxes = new_bboxes
        return list(self.objects.items())

    def _area(self, bbox):
        x1,y1,x2,y2 = bbox
        return max(0, x2-x1) * max(0, y2-y1)

    def _deregister(self, oid):
        self.objects.pop(oid, None)
        self.bboxes.pop(oid, None)
        self.lost.pop(oid, None)
        self.history.pop(oid, None)

# -------------- Announcer with TTS queue --------------
class Announcer:
    def __init__(self, announce_interval=ANNOUNCE_INTERVAL):
        self.q = queue.Queue()
        self.engine_ready = threading.Event()
        self.last_announcements = {}  # key -> last_time
        self.announce_interval = announce_interval
        self._stop = threading.Event()
        # Start TTS thread (daemon)
        self.thread = threading.Thread(target=self._tts_loop, daemon=True)
        self.thread.start()
        # wait for engine to be ready
        self.engine_ready.wait(timeout=5.0)

    def _tts_loop(self):
        # Initialize engine inside thread to avoid cross-thread issues
        engine = pyttsx3.init()
        try:
            rate = engine.getProperty('rate')
            engine.setProperty('rate', max(120, rate - 15))
        except Exception:
            pass
        self.engine = engine
        self.engine_ready.set()
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            text = item
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("TTS engine error:", e)

    def announce(self, key, text):
        now = time.time()
        last = self.last_announcements.get(key, 0)
        if now - last < self.announce_interval:
            return
        self.last_announcements[key] = now
        # Put the text in queue
        self.q.put(text)
        # Also update on-screen status (main-thread-safe since announce is called from main loop)
        set_status(text)

    def stop(self):
        self._stop.set()
        # send sentinel to unblock queue
        self.q.put(None)
        # wait briefly
        self.thread.join(timeout=1.0)

# -------------- Main pipeline --------------
def run():
    # Load model
    print("Loading model...")
    model = YOLO(MODEL_NAME)
    coco_names = model.names if hasattr(model, "names") else {}
    print("Model loaded. Names count:", len(coco_names))

    # camera
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    tracker = CentroidTracker()
    announcer = Announcer()

    try:
        prev_time = time.time()
        last_print = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; exiting.")
                break

            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))

            # Run detection (single frame)
            try:
                results = model.predict(frame_resized, imgsz=640, conf=CONF_THRESH, verbose=False)
            except Exception as e:
                print("Detection error:", e)
                results = []

            detections = []
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        try:
                            xyxy = box.xyxy[0].tolist()
                        except Exception:
                            xyxy = box.xyxy.tolist()
                        x1,y1,x2,y2 = map(int, xyxy)
                        # conf and cls extraction robust
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

            # Filter relevant detections
            filtered = []
            for det in detections:
                x1,y1,x2,y2,label,score = det
                area = max(0,(x2-x1)*(y2-y1))
                if area < MIN_AREA_TO_CONSIDER:
                    continue
                if label in CARE_CLASSES:
                    filtered.append(det)

            # Update tracker (safe when filtered is empty)
            tracker.update(filtered)

            # Analyze tracked objects
            for oid, centroid in list(tracker.objects.items()):
                bbox = tracker.bboxes.get(oid)
                if bbox is None:
                    continue
                x1,y1,x2,y2 = bbox
                cx,cy = centroid

                # Find best matching detection to get label/score
                matched = None
                match_dist = 1e9
                for det in filtered:
                    dx1,dy1,dx2,dy2,label,score = det
                    dcx = int((dx1+dx2)/2); dcy = int((dy1+dy2)/2)
                    d = abs(dcx-cx) + abs(dcy-cy)
                    if d < match_dist:
                        match_dist = d
                        matched = det
                if matched is None:
                    continue
                mx1,my1,mx2,my2,label,score = matched
                # compute area history for approaching detection
                areas = tracker.history.get(oid, deque())
                if len(areas) >= 3:
                    last_area = areas[-1]
                    prev_area_mean = np.mean(list(areas)[:-1]) if len(areas) > 1 else areas[-2]
                    if prev_area_mean > 0 and (last_area / prev_area_mean) > APPROACH_AREA_INCREASE_RATIO:
                        # approaching detected (S1 phrasing, T1 tone)
                        pos = "center"
                        if cx < FRAME_W * 0.33:
                            pos = "left"
                        elif cx > FRAME_W * 0.66:
                            pos = "right"
                        text = f"A {label} is approaching from the {pos}."
                        key = f"approach_{label}_{pos}"
                        announcer.announce(key, text)

                # Example of static detection: bed ahead
                height = (y2 - y1)
                rel = height / FRAME_H
                if label == "bed" and rel > 0.35:
                    announcer.announce("bed_ahead", "There is a bed ahead.")

                # Draw debug boxes
                if DEBUG_WINDOW:
                    cv2.rectangle(frame_resized, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(frame_resized, f"{label}", (x1, max(y1-8,0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # On-screen status (top-left)
            if DEBUG_WINDOW:
                st = get_status()
                if st:
                    # draw semi-transparent rectangle for readability
                    overlay = frame_resized.copy()
                    cv2.rectangle(overlay, (0,0),(FRAME_W,40),(0,0,0), -1)
                    alpha = 0.5
                    frame_resized = cv2.addWeighted(overlay, alpha, frame_resized, 1-alpha, 0)
                    cv2.putText(frame_resized, st, (8,26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imshow("assist_debug", frame_resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # occasional FPS print
            tnow = time.time()
            if tnow - last_print > 2.0:
                fps = 1.0 / max(1e-6, (tnow - prev_time))
                print(f"FPS ~ {fps:.2f}, tracked {len(tracker.objects)}, detections {len(filtered)}")
                last_print = tnow
            prev_time = tnow

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up...")
        cap.release()
        if DEBUG_WINDOW:
            cv2.destroyAllWindows()
        announcer.stop()
        print("Exit.")

# Graceful kill on signals
def _sigterm_handler(signum, frame):
    print("Signal received, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    run()
