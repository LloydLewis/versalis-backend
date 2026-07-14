# MQTT Biofeedback Data Pipeline — Proof of Concept

This folder contains a proof-of-concept local telemetry pipeline designed 
to verify the bioneuro/EEG data ingestion flow before hardware integration.

Data flows from a simulated sensor through a local MQTT broker directly 
into a Realm database on the Android/XR device — no intermediate storage, 
no cloud dependency during the session.

## Prerequisites

- **Python 3.x** installed
- **Mosquitto MQTT Broker** installed and configured
- **Android Studio** with the Kotlin app open and running on an emulator or device

## Architecture

sensor.py  →  Mosquitto broker  →  Android/Kotlin app  →  Realm database
(publishes)       (routes)            (subscribes)          (stores)

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the MQTT Broker**:
   Ensure your local MQTT broker is running on port 1883.
   Mosquitto must be started with a config file that allows connections from the Android emulator. Create a file called `mosquitto.conf` containing:
   listener 1883 0.0.0.0
   allow_anonymous true

   Then start Mosquitto pointing to that file:

   ```bash
   mosquitto -v -c path\to\mosquitto.conf
   ```
   You should not see a "starting in local only mode". If you do, the config file was not loaded correctly

3. **Launch the Sensor Simulator**:
   Open a *second* terminal and start the sensor script:
   ```bash
   python -u sensor.py
   ```
   You should see:
   ```
   Sensor script started
   Connecting to broker...
   Connected!
   Sent: 85 bpm
   Sent: 70 bpm
   ...
   ```

4. **Run the Android app**

Run the app from Android Studio on the emulator. The app will:

- Connect to Mosquitto automatically on launch
- Subscribe to `sensor/heartrate`
- Display the latest BPM reading live on screen, updated every second
- Write every reading directly to the local Realm database

Check Logcat (filter by `com.example.myapplication`) to verify:

Connected to Mosquitto
Subscribed to sensor/heartrate
Received: 85 bpm
Written to Realm - BPM: 85

5. **Verify Data Storage**:
Use the **Read All BPM Readings** button in the app to print all records 
stored in Realm to Logcat. Each record contains the BPM value and the 
timestamp (Unix milliseconds) of when it was received.
