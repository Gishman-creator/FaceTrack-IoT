#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// --- Configuration ---
const char* ssid = "Main Hall";
const char* password = "Meeting@2024";
const char* mqtt_server = "157.173.101.159"; 

const int mqtt_port = 1883;
const char* client_id = "esp8266_team7_client";
const char* topic_movement = "y3d/team7/movement";
const char* topic_heartbeat = "y3d/team7/heartbeat";

// Servo Configuration
Servo myServo;
const int servoPin = 14; // GPIO14 is D5 on NodeMCU
int currentAngle = 90;   
const int moveStep = 30; // Step size as requested
int searchDirection = 1; // Used to track direction during continuous search

WiFiClient espClient;
PubSubClient client(espClient);

// --- Functions ---

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  // Wait until WiFi is officially connected
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void moveServo(int delta) {
  currentAngle += delta;
  
  // Constrain to physical servo limits (0-180) and reverse search direction if searching
  if (currentAngle <= 0) {
    currentAngle = 0;
    searchDirection = 1;
  }
  if (currentAngle >= 180) {
    currentAngle = 180;
    searchDirection = -1;
  }
  
  myServo.write(currentAngle);
  Serial.print("Moving by: "); Serial.print(delta);
  Serial.print(" | New Angle: "); Serial.println(currentAngle);
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
    moveServo(2 * searchDirection); // Continuously pan to find face
  } else if (message.indexOf("MOVE_LEFT") >= 0) {
    moveServo(moveStep); 
  } else if (message.indexOf("MOVE_RIGHT") >= 0) {
    moveServo(-moveStep);
  } else if (message.indexOf("CENTERED") >= 0) {
    currentAngle = 90;
    myServo.write(currentAngle);
    Serial.println("Servo Recentered");
  }
}

void reconnect() {
  // Only attempt MQTT if WiFi is up
  if (WiFi.status() != WL_CONNECTED) return;

  Serial.print("Attempting MQTT connection...");
  if (client.connect(client_id)) {
    Serial.println("connected");
    client.subscribe(topic_movement);
  } else {
    Serial.print("failed, rc=");
    Serial.print(client.state());
    Serial.println(" try again in 5 seconds");
    // No long delay here to keep loop() responsive
  }
}

void setup() {
  // Use 115200 to match your ESP8266's default speed
  Serial.begin(115200);
  
  // Initialize hardware
  myServo.attach(servoPin);
  myServo.write(currentAngle); 

  // 1. Establish WiFi first
  setup_wifi();
  
  // 2. Setup MQTT parameters
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  // 1. Always keep the MQTT client processing
  client.loop(); 

  // 2. Handle WiFi Reconnection
  if (WiFi.status() != WL_CONNECTED) {
    static unsigned long lastWifiRetry = 0;
    if (millis() - lastWifiRetry > 5000) {
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
}