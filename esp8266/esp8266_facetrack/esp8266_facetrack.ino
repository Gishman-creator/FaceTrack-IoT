// Face-Locked Servo Controller
// Board: ESP8266 (NodeMCU, Wemos D1 mini, ...)

#include <ESP8266WiFi.h>    
#include <PubSubClient.h>
#include <Servo.h>          

// --- Configuration ---
const char* ssid = "Peace And Love Ploclaimers"; 
const char* password = "loveisthekey";
const char* mqtt_server = "157.173.101.159"; 

const int mqtt_port = 1883;
const int scan_step_angle = 20;        // Angle to step during scanning
const int scan_pause_ms = 1000;        // Time to pause at each step (milliseconds)

// Unique client ID buffer to prevent public broker collisions
char client_id[40]; 

const char* topic_movement = "y3d/team7/movement";
const char* topic_heartbeat = "y3d/team7/heartbeat";
const char* topic_ack = "y3d/team7/ack";

// Servo Configuration
Servo myServo;
// Pin 5 maps to D1 on most NodeMCU/Wemos D1 Mini boards
const int servoPin = 5; 
int targetAngle = 90;
int actualAngle = 90;
const int moveStep = 10; 
int searchDirection = 1; // Used to track direction during continuous search

WiFiClient espClient;
PubSubClient client(espClient);

// --- Functions ---

// Helper function to scan and print nearby networks safely on ESP8266
void scanAndPrintNetworks() {
  Serial.println("\n--- Scanning for available Wi-Fi networks... ---");
  
  // false, false handles blocking scan and skips hidden SSIDs for memory safety
  int n = WiFi.scanNetworks(false, false); 
  
  if (n == 0) {
    Serial.println("[!] No networks found. Check power supply or 2.4GHz availability.");
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
      yield(); // Feeds the ESP8266 hardware watchdog to prevent reset crashes
    }
  }
  
  WiFi.scanDelete(); // Free memory resources allocated for the scan
  Serial.println("-----------------------------------------------\n");
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.persistent(false); // Protects internal flash memory from wear
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  
  // CRITICAL FIX: Drops RF power output slightly to prevent USB brownouts during transmission
  WiFi.setOutputPower(15); 
  Serial.println("[INFO] RF Output Power capped at 15dBm to control power spikes.");

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
      WiFi.disconnect(); 
      WiFi.begin(ssid, password);
      
      startAttemptTime = currentMillis; // Reset the 10-second anchor
      secondsCounter = 0;               // Reset visual log counter
    }
    
    yield(); // Keeps background system processes alive
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
  if (WiFi.status() != WL_CONNECTED) return;

  Serial.print("Attempting MQTT connection to Custom Broker... ");
  if (client.connect(client_id)) {
    Serial.println("connected");
    client.subscribe(topic_movement);
    client.publish(topic_ack, "READY");
  } else {
    Serial.print("failed, rc=");
    Serial.print(client.state());
    Serial.println(" try again in 5 seconds");
  }
}

void setup() {
  Serial.begin(115200);
  delay(500); 
  
  // Dynamic generation of unique Client ID using ESP8266 hardware chip ID
  snprintf(client_id, sizeof(client_id), "esp8266_team7_%06X", ESP.getChipId());
  Serial.print("Generated Client ID: ");
  Serial.println(client_id);
  
  // Initialize hardware servo
  myServo.attach(servoPin); 
  myServo.write(actualAngle); 

  // 1. Establish WiFi first (Runs setOutputPower inside)
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
      scanAndPrintNetworks(); 
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
  const unsigned long servoSpeedDelay = 15; 
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

  // 6. Serial Input Command Processing
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