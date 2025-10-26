import time
import threading
import queue
from collections import deque, defaultdict
import numpy as np
import cv2
import pyttsx3
from ultralytics import YOLO
import torch

# ----------------- MiDaS Setup -----------------
MIDAS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = midas_transforms.default_transform
    midas.to(MIDAS_DEVICE).eval()
    MIDAS_AVAILABLE = True
except Exception as e:
    print("MiDaS load failed:", e)
    MIDAS_AVAILABLE = False

# ----------------- Parameters -----------------
CAM_ID = 0
FRAME_W, FRAME_H = 640, 360
MODEL_NAME = "yolov8n.pt"
CONF_THRESH = 0.35
ANNOUNCE_INTERVAL = 3.0
APPROACH_RATIO = 1.12
MIN_AREA = 2000
CARE_CLASSES = {"person","bed","chair","couch","sofa","bench","dog","cat","car","bicycle","motorbike"}

depth_scale = 1.0
calibrated = False
last_calibration_info = ""
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

# ----------------- Centroid Tracker -----------------
class CentroidTracker:
    def __init__(self, max_lost=8):
        self.next_object_id = 0
        self.objects = dict()
        self.bboxes = dict()
        self.lost = dict()
        self.max_lost = max_lost
        self.history = defaultdict(lambda: deque(maxlen=8))
    def update(self, detections):
        if len(detections)==0:
            remove_ids = []
            for oid in list(self.objects.keys()):
                self.lost[oid] = self.lost.get(oid,0)+1
                if self.lost[oid]>self.max_lost:
                    remove_ids.append(oid)
            for oid in remove_ids:
                self._deregister(oid)
            return list(self.objects.items())
        det_centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for (x1,y1,x2,y2,_,_) in detections]
        if len(self.objects)==0:
            for i,c in enumerate(det_centroids):
                oid = self.next_object_id; self.next_object_id+=1
                self.objects[oid] = c
                self.bboxes[oid] = detections[i][:4]
                self.lost[oid]=0
                self.history[oid].append(self._area(self.bboxes[oid]))
            return list(self.objects.items())
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid] for oid in object_ids]
        O = np.array(object_centroids)
        D = np.array(det_centroids)
        distances = np.linalg.norm(O[:,None,:] - D[None,:,:], axis=2)
        assigned_dets = set(); assignments = dict()
        rows = distances.min(axis=1).argsort()
        for r in rows:
            c = distances[r].argmin()
            if c in assigned_dets: continue
            assigned_dets.add(c)
            assignments[object_ids[r]]=c
        new_objects = {}; new_bboxes = {}
        for oid, det_idx in assignments.items():
            cx,cy = det_centroids[det_idx]
            new_objects[oid] = (cx,cy)
            new_bboxes[oid] = detections[det_idx][:4]
            self.lost[oid]=0
            self.history[oid].append(self._area(new_bboxes[oid]))
        for i,det in enumerate(detections):
            if i not in assigned_dets:
                oid = self.next_object_id; self.next_object_id+=1
                cx,cy = det_centroids[i]
                new_objects[oid]=(cx,cy)
                new_bboxes[oid]=det[:4]
                self.lost[oid]=0
                self.history[oid].append(self._area(new_bboxes[oid]))
        for oid in object_ids:
            if oid not in assignments:
                self.lost[oid]=self.lost.get(oid,0)+1
                if self.lost[oid]<=self.max_lost:
                    new_objects[oid]=self.objects[oid]
                    new_bboxes[oid]=self.bboxes.get(oid, new_bboxes.get(oid,None))
                else:
                    self._deregister(oid)
        self.objects = new_objects; self.bboxes = new_bboxes
        return list(self.objects.items())
    def _area(self,bbox):
        x1,y1,x2,y2 = bbox; return max(0,x2-x1)*max(0,y2-y1)
    def _deregister(self,oid):
        self.objects.pop(oid,None); self.bboxes.pop(oid,None); self.lost.pop(oid,None); self.history.pop(oid,None)

# ----------------- TTS -----------------
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
        self.engine = engine
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None: break
            try: engine.say(text); engine.runAndWait()
            except Exception as e: print("TTS error:",e)
    def announce(self, key, text):
        now = time.time()
        if now - self.last_announcements.get(key,0) < self.interval: return
        self.last_announcements[key]=now
        self.q.put(text)
        set_status(text)
    def stop(self):
        self._stop.set(); self.q.put(None); self.thread.join(timeout=1.0)

# ----------------- Depth Helpers -----------------
def run_midas(frame):
    if not MIDAS_AVAILABLE: return None
    input_batch = midas_transform(frame).to(MIDAS_DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
        return prediction.cpu().numpy()

def median_depth(depth_map, bbox):
    x1,y1,x2,y2 = bbox
    h,w = depth_map.shape
    x1c,x2c = max(0,int(x1)), min(w,int(x2))
    y1c,y2c = max(0,int(y1)), min(h,int(y2))
    if x2c<=x1c or y2c<=y1c: return None
    patch = depth_map[y1c:y2c, x1c:x2c]
    if patch.size==0: return None
    return float(np.median(patch))

def calibrate_known(observed, known_m):
    if observed is None or observed<=0: return None
    return known_m / observed

# ----------------- Main -----------------
def run():
    global depth_scale, calibrated, last_calibration_info
    model = YOLO(MODEL_NAME)
    coco_names = model.names if hasattr(model,"names") else {}
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    tracker = CentroidTracker()
    announcer = Announcer()
    frame_idx = 0
    depth_map = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_resized = cv2.resize(frame,(FRAME_W,FRAME_H))
            frame_idx+=1

            # MiDaS depth every 3 frames
            if MIDAS_AVAILABLE and frame_idx%3==0:
                rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                depth_map = run_midas(rgb)

            # YOLO detection
            try:
                results = model.predict(frame_resized, imgsz=640, conf=CONF_THRESH, verbose=False)
            except: results=[]
            detections=[]
            if len(results)>0:
                r = results[0]
                boxes = getattr(r,"boxes", None)
                if boxes:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0],'tolist') else box.xyxy.tolist()
                        x1,y1,x2,y2 = map(int, xyxy)
                        score = float(getattr(box,"conf",[0])[0]) if hasattr(box,"conf") else 0.0
                        cls_idx = int(getattr(box,"cls",[0])[0]) if hasattr(box,"cls") else 0
                        label = coco_names.get(cls_idx,str(cls_idx))
                        detections.append((x1,y1,x2,y2,label,score))
            # Filter
            filtered = [d for d in detections if d[4] in CARE_CLASSES and (d[2]-d[0])*(d[3]-d[1])>MIN_AREA]
            tracker.update(filtered)

            # Track + announce
            for oid, centroid in tracker.objects.items():
                bbox = tracker.bboxes.get(oid)
                if bbox is None: continue
                x1,y1,x2,y2 = bbox
                approx_m = None
                if depth_map is not None:
                    val = median_depth(depth_map, bbox)
                    if val is not None: approx_m = val*depth_scale
                areas = tracker.history.get(oid, deque())
                if len(areas)>=3:
                    last_area = areas[-1]
                    prev_mean = np.mean(list(areas)[:-1])
                    if prev_mean>0 and last_area/prev_mean>APPROACH_RATIO:
                        pos="center"
                        cx,_=centroid
                        if cx<FRAME_W*0.33: pos="left"
                        elif cx>FRAME_W*0.66: pos="right"
                        lbl="person"
                        if approx_m: text=f"A {lbl} is {approx_m:.1f}m away, approaching from {pos}."
                        else: text=f"A {lbl} is approaching from {pos}."
                        announcer.announce(f"approach_{oid}", text)
                # Draw
                cv2.rectangle(frame_resized,(x1,y1),(x2,y2),(0,255,0),2)
                lbltxt = f"person {approx_m:.1f}m" if approx_m else "person"
                cv2.putText(frame_resized,lbltxt,(x1,max(y1-8,0)), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)

            # Calibration
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            if key==ord('c'):
                selected=None
                center_x = FRAME_W/2
                best_d=1e9
                for det in filtered:
                    x1,y1,x2,y2,label,_ = det
                    if label!="person": continue
                    dcx = (x1+x2)/2
                    d = abs(dcx - center_x)
                    if d<best_d:
                        best_d=d; selected=det
                if selected is None and filtered:
                    max_area=0
                    for det in filtered:
                        x1,y1,x2,y2,label,_=det
                        if label!="person": continue
                        area=(x2-x1)*(y2-y1)
                        if area>max_area: max_area=area; selected=det
                if selected and depth_map is not None:
                    x1,y1,x2,y2,label,_=selected
                    val = median_depth(depth_map, (x1,y1,x2,y2))
                    if val:
                        s=input("Enter known distance in meters (e.g., 1.5): ")
                        try:
                            known=float(s)
                            new_scale = calibrate_known(val,known)
                            if new_scale:
                                depth_scale=new_scale; calibrated=True
                                last_calibration_info=f"{known:.2f}m"
                                set_status(f"Calibrated scale for {known:.2f}m")
                                print("Calibration successful. scale=",depth_scale)
                        except: print("Calibration failed.")

            # Overlay status
            st=get_status()
            if st:
                overlay = frame_resized.copy()
                cv2.rectangle(overlay,(0,0),(FRAME_W,30),(0,0,0),-1)
                frame_resized=cv2.addWeighted(overlay,0.5,frame_resized,0.5,0)
                cv2.putText(frame_resized,st,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            if calibrated:
                cv2.putText(frame_resized,f"Calibrated scale={depth_scale:.3f} ({last_calibration_info})",
                            (8,FRAME_H-8), cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
            cv2.imshow("Assist", frame_resized)
    finally:
        cap.release(); cv2.destroyAllWindows(); announcer.stop()

if __name__=="__main__":
    run()
