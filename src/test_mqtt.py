import time
import paho.mqtt.client as mqtt

# Get broker config from controller
try:
    from src.control.controller import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC_HEARTBEAT
except ImportError:
    MQTT_BROKER = "157.173.101.159"
    MQTT_PORT = 1883
    MQTT_TOPIC_HEARTBEAT = "y3d/team7/heartbeat"

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"[OK] Successfully connected to MQTT Broker at {MQTT_BROKER}!")
        client.subscribe(MQTT_TOPIC_HEARTBEAT)
        print(f"[OK] Subscribed to topic: {MQTT_TOPIC_HEARTBEAT}")
    else:
        print(f"[ERROR] Failed to connect, return code {rc}")

def on_subscribe(client, userdata, mid, *args, **kwargs):
    # Publish a test message now that the subscription is confirmed active
    test_msg = "PING_TEST_CONNECTION"
    client.publish(MQTT_TOPIC_HEARTBEAT, test_msg)
    print(f"[OK] Published test message '{test_msg}' to {MQTT_TOPIC_HEARTBEAT}")

def on_message(client, userdata, msg):
    content = msg.payload.decode('utf-8')
    print(f"[OK] Received message back: {msg.topic} -> {content}")
    if content == "PING_TEST_CONNECTION":
        print("\n=======================================================")
        print(" SUCCESS: MQTT Broker is working and responding perfectly!")
        print("=======================================================")
        client.disconnect()

def main():
    print("=======================================================")
    print(f"Diagnosing MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print("=======================================================\n")
    
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=10)
    except Exception as e:
        print(f"[ERROR] Could not establish connection to {MQTT_BROKER}: {e}")
        print("\nPossible causes:")
        print(" 1. The broker is down or the IP is incorrect.")
        print(" 2. You are not on the required VPN or local network to access it.")
        print(" 3. Port 1883 is blocked by your firewall.")
        return

    # Loop briefly to process messages
    client.loop_start()
    
    t_end = time.time() + 5.0
    while client.is_connected() and time.time() < t_end:
        time.sleep(0.1)
        
    client.loop_stop()
    
    if not client.is_connected() and time.time() >= t_end:
        print("\n[TIMEOUT] Connected, but did not receive the test message back. Check topic permissions.")
    elif client.is_connected():
        client.disconnect()
        print("\nTest completed.")

if __name__ == "__main__":
    main()
