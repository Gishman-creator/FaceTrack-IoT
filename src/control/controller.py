"""
FaceTrack IoT Controller
Combines Face Locking, Recognition, MQTT Control, and Web Dashboard.
"""
import sys
import time
import threading
import io
import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
from flask import Flask, Response, send_from_directory
from flask_socketio import SocketIO
from pathlib import Path

# Fix path to allow importing from src (if run as script)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from src import config
    from src.haar_5pt import align_face_5pt, HaarFaceMesh5pt
    from src.embed import ArcFaceEmbedderONNX
    from src.lock_and_recognize import cosine_distance, load_database, detect_actions, _mouth_width
    import mediapipe as mp
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running from the project root: python -m src.control.controller")
    sys.exit(1)

# --- Configuration ---
MQTT_BROKER = "157.173.101.159"
MQTT_PORT = 1883
MQTT_TOPIC_MOVE = "y3d/team7/movement"
MQTT_TOPIC_HEARTBEAT = "y3d/team7/heartbeat"
MQTT_CLIENT_ID = "pc_vision_team7"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global State
# Global State
lock = threading.Lock()
mqtt_client = None

def setup_mqtt():
    global mqtt_client
    try:
        # Use VERSION2 to avoid deprecation warnings
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=MQTT_CLIENT_ID)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
    except Exception as e:
        print(f"Failed to connect to MQTT Broker ({MQTT_BROKER}): {e}")
        print("-> Servo control will NOT work. Check your VPN or Broker IP.")

def publish_move(command):
    if mqtt_client:
        mqtt_client.publish(MQTT_TOPIC_MOVE, command)
        # print(f"Published: {command}")

def cv_loop(initial_target):
    
    # Load DB
    db = load_database()
    names = sorted(db.keys())
    print(f"Loaded {len(names)} identities.")
    
    # Init Models
    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX(config.ARCFACE_MODEL_PATH)
    action_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )
    
    # Prepare Embeddings
    db_matrix = None
    if names:
        db_matrix = np.stack([db[n].reshape(-1) for n in names])
        
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Camera failed to open.")
        return

    # Tracking State
    target_identity = initial_target
    print(f"Targeting: {target_identity}")
    
    locked_face = None
    prev_cx = None
    baseline_mw = None
    mw_samples = []
    last_actions = {}
    cached_names = {} # Map track_id or index -> name
    frame_idx = 0
    
    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from camera.")
            time.sleep(1)
            continue
        
        frame_idx += 1
        H, W = frame.shape[:2]
        center_x_frame = W // 2
        
        # FPS Calc
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()
            socketio.emit('status_update', {'fps': f"{fps:.1f}"})
        frames += 1
        
        # Detection
        detections = det.detect(frame, max_faces=5)
        vis = frame.copy()
        
        # Match / Update Lock
        best_match = None
        min_dist = 1.0
        
        # If we have a lock, try to track spatially first (optimization)
        # But for robustness, we do full recognition every N frames or if lost.
        # For simplicity in this merged script, we'll do recognition each frame (CPU bound, but safe).
        # To optimize, you could implement the spatial tracking from facelock.py here.
        
        for face in detections:
            aligned, _ = align_face_5pt(frame, face.kps)
            # Optimization: Only embed every 3rd frame
            # We need a way to track faces to reuse names. 
            # Since we don't have a tracker ID, we'll just re-run for all faces every Nth frame 
            # and for interim frames we can skip or just accept "Unknown" (or use previous simple logic).
            # But "Unknown" flickering is bad.
            
            # Simple heuristic: If we have cached names and faces are in similar positions, reuse?
            # Too complex for this snippet.
            # Let's just run embedding every frame but perhaps lower resolution or skip FaceMesh on background?
            
            # Actually, the user asked to increase FPS. 
            # Let's try to SKIP detection entirely on some frames?
            # No, that makes tracking laggy.
            
            # Use 'frame_idx % 2 == 0' to speed up embedding (50% work)?
            do_recognition = (frame_idx % 3 == 0)
            
            name = "Unknown"
            dist = 1.0
            
            # We use a simple index-based caching (assuming detection order is stable-ish for 1-2 frames)
            # This is flawed if faces swap places, but fine for simple single-user usage.
            # Or better: don't cache, just run it. The sleep(0.015) should be enough.
            # But if CPU is pegged, we MUST skip.
            
            if do_recognition:
                 res = embedder.embed(aligned)
                 emb = res.embedding
                 if db_matrix is not None:
                     dists = np.array([cosine_distance(emb, db_matrix[i]) for i in range(len(names))])
                     best_i = int(np.argmin(dists))
                     dist = dists[best_i]
                     if dist <= config.DEFAULT_DISTANCE_THRESHOLD:
                         name = names[best_i]
                 cached_names[tuple(face.kps[2])] = (name, dist) # Cache by nose tip? No, moving.
                 # Let's just use strict ordering for cache (dangerous but fast)
                 # cached_names[i] = ...
            else:
                 # Just use Unknown or "Tracking..."
                 # Better: Do full run. The user's machine might be slow.
                 # Reverting to full run every frame unless it's too slow.
                 res = embedder.embed(aligned)
                 emb = res.embedding
                 if db_matrix is not None:
                     dists = np.array([cosine_distance(emb, db_matrix[i]) for i in range(len(names))])
                     best_i = int(np.argmin(dists))
                     dist = dists[best_i]
                     if dist <= config.DEFAULT_DISTANCE_THRESHOLD:
                         name = names[best_i]
            
            # Label
            color = (0, 255, 0) if name == target_identity else ((0, 0, 255) if name != "Unknown" else (200, 200, 200))
            cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), color, 2)
            cv2.putText(vis, f"{name} {dist:.2f}", (face.x1, face.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if name == target_identity:
                if best_match is None or dist < min_dist:
                    best_match = face
                    min_dist = dist

        # Logic
        action_status = "Idle"
        is_locked = False
        
        if best_match:
            is_locked = True
            face = best_match
            cx = (face.x1 + face.x2) / 2
            
            # Servo Control Logic (Simple Proportional / Threshold)
            # Deadzone of 50px
            err = cx - center_x_frame
            if err < -50: # Face is to the LEFT of center
                publish_move("MOVE_LEFT") # Move camera LEFT to center it?
                # Actually, if face is LEFT (pixel < center), we need to pan servo LEFT (increase/decrease angle depends on mount).
                # Assuming standard: Left side of image = Left side of world. Servo needs to turn Left.
                action_status = "Moving Left"
            elif err > 50: # Face is to the RIGHT
                publish_move("MOVE_RIGHT")
                action_status = "Moving Right"
            else:
                 pass # Centered
                 # publish_move("CENTERED") # Optional
            
            # Action Detection (Blink/Smile)
            # We need to run FaceMesh on the full frame or ROI for actions
            # detect_actions wants the mesh object
            acts, _, mw = detect_actions(frame, action_mesh, prev_cx, cx, 0, 0, baseline_mw, frame_idx, last_actions)
            for atype, adesc in acts:
                socketio.emit('log', {'msg': f"Action: {adesc}"})
                cv2.putText(vis, f"ACTION: {adesc}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                
            if mw is not None:
                mw_samples.append(mw)
                if len(mw_samples) > 20: mw_samples.pop(0)
                if baseline_mw is None and len(mw_samples) >= 10:
                    baseline_mw = float(np.median(mw_samples))
            
            prev_cx = cx
            socketio.emit('status_update', {'locked': True, 'target': target_identity, 'action': action_status})
        else:
             socketio.emit('status_update', {'locked': False, 'target': target_identity, 'action': "Searching"})

        # Update Video Stream
        # ret, buffer = cv2.imencode('.jpg', vis)
        # if ret:
        #     with lock:
        #         # current_frame_jpeg = buffer.tobytes()
        #         pass
                
        # Show Local Window
        cv2.imshow("FaceTrack Controller", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping CV Loop...")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    # Signal Flask to stop? Hard to kill flask from thread.
    # We can just let the thread die.

@app.route('/')
def index():
    return send_from_directory('../ui', 'index.html')

@app.route('/video_feed')
def video_feed():
    return "Video feed disabled in local mode."

def run_server():
    setup_mqtt()
    
    # Prompt for Target
    db = load_database()
    names = sorted(db.keys())
    target = "Unknown"
    
    if not names:
        print("No faces in database! Running in recognition-only mode.")
    else:
        print("\nAvailable Identities:")
        for i, n in enumerate(names, 1):
            print(f" {i}. {n}")
        
        print("\nEnter number or name to LOCK onto (others will be ignored): ", end="")
        try:
            choice = input().strip()
            if choice.isdigit() and 1 <= int(choice) <= len(names):
                target = names[int(choice)-1]
            elif choice in names:
                target = choice
            else:
                print(f"Invalid choice. Defaulting to first: {names[0]}")
                target = names[0]
        except Exception:
            target = names[0]
            
    print(f"Selected Lock Target: {target}")

    # Start CV Thread
    t = threading.Thread(target=cv_loop, args=(target,))
    t.daemon = True
    t.start()
    
    print("Starting Web Server on 0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    run_server()
