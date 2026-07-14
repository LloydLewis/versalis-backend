import paho.mqtt.client as mqtt
import random
import time

print("Sensor script started")

try:
    # Initializing MQTT Client using Callback API Version 2 (paho-mqtt >= 2.0)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    print("Connecting to broker...")
    client.connect("localhost", 1883)
    print("Connected!")
    client.loop_start()

    while True:
        # Simulate EEG / heart rate biofeedback reading
        heart_rate = random.randint(60, 100)
        
        # Publish to the 'sensor/heartrate' topic
        result = client.publish("sensor/heartrate", heart_rate)
        print(f"Sent: {heart_rate} bpm")
        
        time.sleep(1)

except Exception as e:
    print(f"ERROR: {e}")
    input("Press Enter to close...")
