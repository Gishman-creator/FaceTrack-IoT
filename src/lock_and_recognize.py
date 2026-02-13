"""
Combined Face Locking and Multi-Face Recognition.
References: src/facelock.py and src/recognize.py

Features:
- Locks onto a specific target identity (logging actions like blinks/smiles).
- Simultaneously recognizes and labels all other faces in the frame.
- Uses the multi-face robust pipeline from recognize.py.

Run:
  python -m src.lock_and_recognize
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except ImportError:
    mp = None

from . import config

# ---------------------------------------------------------------------
# Math & Geometry Helpers
# ---------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _bbox_from_5pt(kps: np.ndarray, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min, x_max = float(np.min(k[:, 0])), float(np.max(k[:, 0]))
    y_min, y_max = float(np.min(k[:, 1])), float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist): return False
    if not (lm[1] > no[1] and rm[1] > no[1]): return False
    return True

# ---------------------------------------------------------------------
# Alignment & FaceDet
# ---------------------------------------------------------------------

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray # (5,2)

def _estimate_norm_5pt(kps_5x2: np.ndarray, out_size=(112, 112)) -> np.ndarray:
    dst = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    out_w, out_h = out_size
    if (out_w, out_h) != (112, 112):
        dst = dst * np.array([out_w/112.0, out_h/112.0], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(kps_5x2.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(kps_5x2[:3].astype(np.float32), dst[:3].astype(np.float32))
    return M.astype(np.float32)

def align_face_5pt(frame_bgr: np.ndarray, kps_5x2: np.ndarray, out_size=(112, 112)) -> Tuple[np.ndarray, np.ndarray]:
    M = _estimate_norm_5pt(kps_5x2, out_size)
    wrapped = cv2.warpAffine(frame_bgr, M, out_size, flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    return wrapped, M

# ---------------------------------------------------------------------
# Detector (Multi-face)
# ---------------------------------------------------------------------

class HaarFaceMesh5pt:
    def __init__(self, haar_xml=None, min_size=(70, 70), debug=False):
        self.debug = debug
        self.min_size = min_size
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        
        if mp is None:
            raise RuntimeError("MediaPipe not found.")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.idx_map = [33, 263, 1, 61, 291] # LeftEye, RightEye, Nose, LeftMouth, RightMouth

    def detect(self, frame, max_faces=5) -> List[FaceDet]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=self.min_size
        )
        if len(faces) == 0:
            return []
        
        # Sort by area
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]
        
        out = []
        for x, y, w, h in faces:
            # Expand ROI for FaceMesh
            mx, my = 0.25*w, 0.35*h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x-mx, y-my, x+w+mx, y+h+my, W, H)
            roi = frame[ry1:ry2, rx1:rx2]
            
            if roi.shape[0] < 20 or roi.shape[1] < 20: continue
            
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(roi_rgb)
            if not res.multi_face_landmarks: continue
            
            lm = res.multi_face_landmarks[0].landmark
            pts = []
            for i in self.idx_map:
                p = lm[i]
                pts.append([p.x * (rx2-rx1) + rx1, p.y * (ry2-ry1) + ry1])
            kps = np.array(pts, dtype=np.float32)
            
            # Corrections
            if kps[0,0] > kps[1,0]: kps[[0,1]] = kps[[1,0]]
            if kps[3,0] > kps[4,0]: kps[[3,4]] = kps[[4,3]]
            
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18*w)): continue
            
            # Re-box
            bb = _bbox_from_5pt(kps)
            bx1, by1, bx2, by2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            out.append(FaceDet(bx1, by1, bx2, by2, 1.0, kps))
            
        return out

# ---------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------

class ArcFaceEmbedderONNX:
    def __init__(self, model_path, input_size=(112, 112)):
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.w, self.h = input_size

    def embed(self, img_bgr):
        if img_bgr.shape[1] != self.w or img_bgr.shape[0] != self.h:
            img_bgr = cv2.resize(img_bgr, (self.w, self.h))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.reshape(-1)
        return v / (np.linalg.norm(v) + 1e-12)

# ---------------------------------------------------------------------
# Action Detection (from facelock.py)
# ---------------------------------------------------------------------

def _ear_from_landmarks(landmarks_list, indices, W, H):
    pts = [np.array([landmarks_list[i].x * W, landmarks_list[i].y * H]) for i in indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.5

def _mouth_width(landmarks_list, i1, i2, W, H):
    l = landmarks_list[i1]
    r = landmarks_list[i2]
    return np.hypot((r.x - l.x)*W, (r.y - l.y)*H)

def detect_actions(frame, mesh, prev_center_x, center_x, prev_ear, prev_mouth, baseline_mouth, frame_idx, last_actions):
    actions = []
    if mesh is None: return actions, None, None
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks: return actions, None, None
    lms = res.multi_face_landmarks[0].landmark
    
    cooldown = config.LOCK_ACTION_COOLDOWN_FRAMES
    
    # Movement
    if prev_center_x is not None:
        dx = center_x - prev_center_x
        if dx <= -config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_actions.get("move_left", -999) > cooldown:
                actions.append(("face_moved_left", "Left"))
                last_actions["move_left"] = frame_idx
        elif dx >= config.LOCK_MOVEMENT_THRESHOLD_PX:
            if frame_idx - last_actions.get("move_right", -999) > cooldown:
                actions.append(("face_moved_right", "Right"))
                last_actions["move_right"] = frame_idx
                
    # Blink
    ear = (_ear_from_landmarks(lms, config.LOCK_EAR_LEFT_INDICES, W, H) + 
           _ear_from_landmarks(lms, config.LOCK_EAR_RIGHT_INDICES, W, H)) / 2.0
    if ear < config.LOCK_EAR_BLINK_THRESHOLD:
        if frame_idx - last_actions.get("blink", -999) > cooldown:
            actions.append(("eye_blink", "Blink"))
            last_actions["blink"] = frame_idx
            
    # Smile
    mw = _mouth_width(lms, config.LOCK_MOUTH_LEFT_INDEX, config.LOCK_MOUTH_RIGHT_INDEX, W, H)
    if baseline_mouth and baseline_mouth > 1.0:
        if mw >= baseline_mouth * config.LOCK_SMILE_MOUTH_RATIO:
            if frame_idx - last_actions.get("smile", -999) > cooldown:
                actions.append(("smile", "Smile"))
                last_actions["smile"] = frame_idx
                
    return actions, ear, mw

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def load_database():
    if not config.DB_NPZ_PATH.exists(): return {}
    data = np.load(str(config.DB_NPZ_PATH), allow_pickle=True)
    return {k: data[k].astype(np.float32) for k in data.files}

def main():
    db = load_database()
    if not db:
        print("No database found. Run enroll.py first.")
        return
    
    names = sorted(db.keys())
    print("\nAvailable Identities:")
    for i, n in enumerate(names, 1):
        print(f" {i}. {n}")
        
    print("\nEnter name to LOCK onto (others will be recognized): ", end="")
    target = input().strip()
    if not target: target = names[0] if names else ""
    if target not in db:
        print(f"'{target}' not found. Defaulting to first." if names else "Error")
        if names: target = names[0]
        
    print(f"Locking onto: {target}")
    
    # Init
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX(config.ARCFACE_MODEL_PATH)
    
    # Action Mesh (for the locked face actions)
    action_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )
    
    # Prepared Embeddings
    db_matrix = np.stack([db[n].reshape(-1) for n in names])
    target_idx = names.index(target)
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    # State
    history_file = None
    if config.HISTORY_DIR:
        config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d%H%M%S")
        safe_name = target.replace(" ", "_").lower()
        hpath = config.HISTORY_DIR / f"{safe_name}_lock_{ts}.txt"
        history_file = open(hpath, "w", encoding="utf-8")
        history_file.write(f"# Lock History for {target}\n# Time, Action, Desc\n")
        print(f"Recording history to {hpath}")

    prev_cx = None
    baseline_mw = None
    mw_samples = []
    last_actions = {}
    frame_idx = 0
    t0 = time.time()
    frames = 0
    fps = 0.0
    
    # Thresholds
    DIST_THRESH = config.DEFAULT_DISTANCE_THRESHOLD
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_idx += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            fps = frames / elapsed
            frames = 0
            t0 = time.time()
        frames += 1
        
        vis = frame.copy()
        
        # 1. Detect All Faces
        detections = det.detect(frame, max_faces=5)
        
        locked_face = None
        
        # 2. Recognize All
        for face in detections:
            aligned, _ = align_face_5pt(frame, face.kps)
            emb = embedder.embed(aligned)
            
            dists = np.array([cosine_distance(emb, db_matrix[i]) for i in range(len(names))])
            best_i = int(np.argmin(dists))
            best_dist = dists[best_i]
            
            name = names[best_i] if best_dist <= DIST_THRESH else "Unknown"
            
            # Check if this is our target
            is_target = (name == target)
            
            if is_target:
                # Logic to handle if multiple faces match target? Pick best dist.
                if locked_face is None or best_dist < locked_face['dist']:
                    locked_face = {
                        'face': face,
                        'dist': best_dist,
                        'cx': (face.x1 + face.x2)/2
                    }
            else:
                # Draw Info for others immediately
                color = (0, 255, 255) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
                cv2.putText(vis, f"{name} {best_dist:.2f}", (face.x1, face.y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 3. Handle Locked Face
        if locked_face:
            f = locked_face['face']
            cx = locked_face['cx']
            
            # Draw Locked Box
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 3)
            cv2.putText(vis, f"LOCKED: {target}", (f.x1, f.y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Action Detection
            # We pass full frame; action_mesh usually picks the largest face.
            # Assuming locked face is one of the main ones.
            # Ideally we crop? But facelock.py uses full frame.
            acts, _, mw = detect_actions(frame, action_mesh, prev_cx, cx, 0, 0, baseline_mw, frame_idx, last_actions)
            
            if mw is not None:
                mw_samples.append(mw)
                if len(mw_samples) > 20: mw_samples.pop(0)
                if baseline_mw is None and len(mw_samples) >= 10:
                    baseline_mw = float(np.median(mw_samples))
            
            prev_cx = cx
            
            for atype, adesc in acts:
                ts_str = time.strftime("%H:%M:%S")
                cv2.putText(vis, f"ACTION: {adesc}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                if history_file:
                    history_file.write(f"{time.time():.2f}, {atype}, {adesc}\n")
                    history_file.flush()
        else:
            cv2.putText(vis, f"Searching for {target}...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            prev_cx = None

        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Smart Lock & Recognize", vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    if history_file: history_file.close()

if __name__ == "__main__":
    main()
