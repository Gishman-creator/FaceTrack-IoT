# FaceTrack IoT System

**FaceTrack IoT System** is an intelligent tracking solution that connects computer vision with hardware control.

The system uses a PC webcam to detect and recognize faces in real-time using ArcFace embeddings. When a specific "target" face is identified, the software calculates its position and sends control commands via MQTT to an ESP8266 microcontroller. The ESP8266 then drives a servo motor to physically track the face. A real-time web dashboard allows users to monitor the video feed, tracking status, and system logs.

---

## 📥 Installation

### 1. Setup Environment
Open your terminal and run:

```bash
# Clone the repository
git clone https://github.com/Gishman-creator/FaceTrack-IoT
cd FaceTrack-IoT

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install ArcFace Model

**Option A: Automatic (Windows/Linux/Mac)**
We have provided a script to handle the download and setup for you.
```bash
python download_model.py
```

**Option B: Manual Setup**
1.  Download `buffalo_l.zip` from [InsightFace Model Zoo](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip).
2.  Extract the zip file.
3.  Rename the file `w600k_r50.onnx` to `embedder_arcface.onnx`.
4.  Create a folder named `models` in the project root.
5.  Move `embedder_arcface.onnx` into `models/`.

---

## 🚀 Usage Guide

### 1. Verification
Ensure your hardware and software are ready:
```bash
python -m src.camera      # Check webcam feed
python -m src.landmarks   # Check facial mapping
```

### 2. Enroll Faces (Build Database)
This step saves face embeddings to the `data/` folder.
```bash
python -m src.enroll
```
*   **Enter Name**: Type a person's name in the terminal.
*   **Space**: Capture a photo manually.
*   **'a'**: Toggle auto-capture mode (recommended).
*   **'s'**: Save and finish enrollment (aim for 15+ samples).
*   **'q'**: Quit without saving.

### 3. Flash ESP8266 (Microcontroller)
1. Open `esp8266/esp8266_facetrack/esp8266_facetrack.ino` in Arduino IDE.
2. Install the **PubSubClient** and **Servo** libraries.
3. Configure your WiFi credentials and MQTT broker IP in the code.
4. Upload to your ESP8266 board.

### 4. Start Controller (PC)
Run the main controller script to start face tracking, MQTT communication, and the web server:
```bash
python -m src.control.controller
```
*   Select the target identity to lock onto when prompted in the terminal.
*   A local window will open showing the camera feed with tracking info.

### 5. Access Dashboard
Open your web browser and go to:
[http://localhost:5000](http://localhost:5000)

Live Dashboard on:
[http://157.173.101.159:9369](http://157.173.101.159:9369/)

*   View system status (Locked Target, Servo Action).
*   See real-time logs of actions (Blink, Smile).

---

## 🔧 Troubleshooting

**"No module named mediapipe"**
Reinstall the specific compatible version:
```bash
pip install -r requirements.txt --force-reinstall
```

**Camera not opening**
*   Check if another app is using the camera.
*   Try running `python -m src.camera` to debug.

**Low Accuracy?**
*   Ensure good lighting during enrollment.
*   Capture at least 15 photos per person at different angles (left, right, up, down).
