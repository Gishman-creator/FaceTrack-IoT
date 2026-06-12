#include <WiFi.h>          // Updated for ESP32
#include <PubSubClient.h>
#include <ESP32Servo.h>    // Updated for ESP32 (Requires "ESP32Servo" library in Library Manager)

// --- Configuration ---
const char* ssid = "Peace And Love Ploclaimers"; // Spelled with an 'l' as scanned
const char* password = "loveisthekey";
const char* mqtt_server = "157.173.101.159"; 

const int mqtt_port = 1883;
const int scan_step_angle = 20;        // Angle to step during scanning
const int scan_pause_ms = 1000;        // Time to pause at each step (milliseconds)
const char* client_id = "esp32_team7_client"; // Updated identifier name for clarity
const char* topic_movement = "y3d/team7/movement";
const char* topic_heartbeat = "y3d/team7/heartbeat";
const char* topic_ack = "y3d/team7/ack";

// Servo Configuration
Servo myServo;
// Note: Ensure Pin 14 (GPIO14) on your ESP32 board is a free PWM pin.
const int servoPin = 5; 
int targetAngle = 90;
int actualAngle = 90;
const int moveStep = 10; // Step size reduced for smoother movement
int searchDirection = 1; // Used to track direction during continuous search

WiFiClient espClient;
PubSubClient client(espClient);

// --- Functions ---

// Helper function to scan and print nearby networks
void scanAndPrintNetworks() {
  Serial.println("\n--- Scanning for available Wi-Fi networks... ---");
  int n = WiFi.scanNetworks();
  
  if (n == 0) {
    Serial.println("[!] No networks found. Ensure a 2.4GHz access point is nearby.");
  } else {
    Serial.print("[!] "); Serial.print(n); Serial.println(" networks found:");
    for (int i = 0; i < n; ++i) {
      Serial.print("  -> ");
      Serial.print(i + 1);
      Serial.print(": ");
      Serial.print(WiFi.SSID(i));
      Serial.print(" (");
      Serial.print(WiFi.RSSI(i));
      Serial.println("dBm)");
      delay(10);
    }
  }
  Serial.println("-----------------------------------------------\n");
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long startAttemptTime = millis();
  unsigned long lastLogTime = 0;
  int secondsCounter = 0;

  // Loop non-blockingly for initial connection setup
  while (WiFi.status() != WL_CONNECTED) {
    unsigned long currentMillis = millis();

    // Log the failure message exactly every 1 second (1000ms)
    if (currentMillis - lastLogTime >= 1000) {
      lastLogTime = currentMillis;
      secondsCounter++;
      Serial.print("["); Serial.print(secondsCounter); Serial.print("s] ");
      Serial.println("Couldn't connect to WiFi...");
    }

    // Every 10 seconds, print available Wi-Fi options and force a retry
    if (currentMillis - startAttemptTime >= 10000) {
      Serial.println("\n[!] 10 seconds reached without connection.");
      
      // Run the network diagnostic scan
      scanAndPrintNetworks();
      
      Serial.print("Retrying connection to: "); Serial.println(ssid);
      WiFi.disconnect(); // Clear old instance
      WiFi.begin(ssid, password);
      
      startAttemptTime = currentMillis; // Reset the 10-second anchor
      secondsCounter = 0;               // Reset visual log counter
    }
    
    yield(); // Keeps watchdog timer happy
  }

  Serial.println("");
  Serial.println("WiFi connected successfully!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void moveServo(int delta) {
  targetAngle += delta;
  
  // Constrain to physical servo limits (0-180) and reverse search direction if searching
  if (targetAngle <= 0) {
    targetAngle = 0;
    searchDirection = 1;
  }
  if (targetAngle >= 180) {
    targetAngle = 180;
    searchDirection = -1;
  }
  
  Serial.print("Target Angle set: "); Serial.println(targetAngle);
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  Serial.println(message);

  // Execute control commands
  if (message.startsWith("STEP:")) {
    int step = message.substring(5).toInt();
    moveServo(step);
  } else if (message.startsWith("SEARCH")) {
    static unsigned long lastSearchMove = 0;
    if (millis() - lastSearchMove >= scan_pause_ms) {
      moveServo(scan_step_angle * searchDirection);
      lastSearchMove = millis();
    }
  } else if (message.indexOf("MOVE_LEFT") >= 0) {
    moveServo(moveStep); 
  } else if (message.indexOf("MOVE_RIGHT") >= 0) {
    moveServo(-moveStep);
  } else if (message.indexOf("CENTERED") >= 0) {
    targetAngle = 90;
    actualAngle = 90;
    myServo.write(actualAngle);
    Serial.println("Servo Recentered");
  }

  // Publish acknowledgment to PC controller
  client.publish(topic_ack, "ACK");
}

void reconnect() {
  // Only attempt MQTT if WiFi is up
  if (WiFi.status() != WL_CONNECTED) return;

  Serial.print("Attempting MQTT connection...");
  if (client.connect(client_id)) {
    Serial.println("connected");
    client.subscribe(topic_movement);
    // Notify controller that we are ready
    client.publish(topic_ack, "READY");
  } else {
    Serial.print("failed, rc=");
    Serial.print(client.state());
    Serial.println(" try again in 5 seconds");
    // No long delay here to keep loop() responsive
  }
}

void setup() {
  Serial.begin(115200);
  delay(500); 
  
  // ESP32Servo setup: Allow allocation of all timer channels
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  // Standard servo frequency is 50Hz
  myServo.setPeriodHertz(50); 
  
  // Initialize hardware
  myServo.attach(servoPin, 500, 2400); // Standard min/max pulse widths for SG90/MG90S servos
  myServo.write(actualAngle); 

  // 1. Establish WiFi first
  setup_wifi();
  
  // 2. Setup MQTT parameters
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  // 1. Always keep the MQTT client processing
  client.loop(); 

  // 2. Handle WiFi Reconnection if it drops mid-operation
  if (WiFi.status() != WL_CONNECTED) {
    static unsigned long lastWifiRetry = 0;
    static unsigned long lastWifiLog = 0;

    if (millis() - lastWifiLog > 1000) {
      Serial.println("WiFi connection lost. Awaiting reconnect...");
      lastWifiLog = millis();
    }

    if (millis() - lastWifiRetry > 10000) { 
      Serial.println("Retrying WiFi background hook...");
      scanAndPrintNetworks(); // Diagnostic scan inside the main loop as well
      WiFi.begin(ssid, password);
      lastWifiRetry = millis();
    }
    return; 
  }

  // 3. Handle MQTT Reconnection (Non-blocking)
  if (!client.connected()) {
    static unsigned long lastMqttRetry = 0;
    if (millis() - lastMqttRetry > 5000) {
      reconnect();
      lastMqttRetry = millis();
    }
  }

  // 4. Heartbeat logic
  static unsigned long lastMsg = 0;
  if (millis() - lastMsg > 5000) {
    lastMsg = millis();
    if (client.connected()) {
      client.publish(topic_heartbeat, "{\"status\": \"ONLINE\"}");
    }
  }

  // 5. Smooth Servo Easing Logic
  static unsigned long lastServoUpdate = 0;
  const unsigned long servoSpeedDelay = 15; // Milliseconds per degree step (lower is faster/less smooth, higher is slower/smoother)
  if (millis() - lastServoUpdate >= servoSpeedDelay) {
    if (actualAngle < targetAngle) {
      actualAngle++;
      myServo.write(actualAngle);
    } else if (actualAngle > targetAngle) {
      actualAngle--;
      myServo.write(actualAngle);
    }
    lastServoUpdate = millis();
  }

  // 6. Serial Input Command Processing (e.g. type 'center' to recenter servo)
  while (Serial.available() > 0) {
    char c = Serial.read();
    static String inputBuffer = "";
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        inputBuffer.trim();
        if (inputBuffer.equalsIgnoreCase("center")) {
          targetAngle = 90;
          Serial.println("Serial Command: Centering Servo");
        }
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
      if (inputBuffer.length() > 50) {
        inputBuffer = "";
      }
    }
  }
}