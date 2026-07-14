# Versalis — Backend & Data Layer

Internship project | Cognitology / PsycReality | Cybersecurity & Backend Team

Versalis is an Adaptive VR Therapy Engine designed for clinical mental health settings. This repository contains the backend architecture, local database layer, and data ingestion pipeline built during Phase 1 and Phase 2 of the internship.

---

## Project Overview

Versalis runs entirely offline during live therapy sessions. A patient wears a Quest 3 VR headset while a Mental Health Professional (MHP) monitors the session from a clinician desktop. Biofeedback signals (heart rate, EEG, GSR) are captured in real time, processed by an adaptive engine, and stored locally in an encrypted database. After the session ends, a summary is synced to the Prosper AI cloud platform (MHMS).

This repository covers the data and backend layer of that system.

---

## Repository Structure

```
MyApplication/
├── app/                        # Android/Kotlin application
│   └── src/main/
│       └── com.example.myapplication/
│           ├── database/       # Realm database initialisation
│           ├── models/         # Realm schema (BioData, ButtonPress)
│           ├── mqtt/           # MQTT subscriber (HiveMQ client)
│           ├── repository/     # Database read/write operations
│           └── MainActivity.kt # Entry point — wires MQTT and Realm together
└── mqtt/                       # Python MQTT pipeline (proof of concept)
    ├── sensor.py               # Simulates biofeedback sensor publishing to Mosquitto
    ├── mosquitto.conf          # Mosquitto broker config (allows external connections)
    ├── requirements.txt        # Python dependencies
    └── README.md               # Pipeline setup instructions
```

---

## What Has Been Built

### Local Database (Realm)
- Encrypted local Realm database running on the Android/Quest device
- Schema covers biometric readings (BPM, EEG, GSR, blood pressure, voice stress), patient baseline values, and adaptive engine intensity recommendations
- Designed for offline-first operation — no network required during a live session
- AES-256 encryption at rest via Realm's built-in encryption layer

### MQTT Data Ingestion Pipeline
- HiveMQ Kotlin client subscribes to Mosquitto broker on `sensor/heartrate`
- Incoming biofeedback readings are parsed and written directly to Realm
- No intermediate SQLite storage — data flows from sensor to Realm in one step
- Python simulator (`sensor.py`) publishes randomised BPM readings every second to simulate hardware before physical sensors are available

### Proof of Concept Demo
The Android app demonstrates two flows simultaneously:
1. **MQTT → Realm** — live BPM display updated every second from incoming MQTT messages
2. **Manual write → Realm** — button press flow to verify direct database writes and reads

---

## Tech Stack

| Layer | Technology |
|---|---|
| Device database | Realm (Kotlin SDK) |
| MQTT client (Android) | HiveMQ MQTT Client 1.3.3 |
| MQTT broker | Mosquitto 2.x |
| Sensor simulator | Python 3 + paho-mqtt |
| Android UI | Jetpack Compose |
| Language (Android) | Kotlin |
| Language (pipeline) | Python 3 |

---

## Running the Pipeline

See [`mqtt/README.md`](mqtt/README.md) for full setup instructions.

Quick start:

```bash
# Terminal 1 — start broker
mosquitto -v -c mqtt/mosquitto.conf

# Terminal 2 — start sensor simulator
cd mqtt
python sensor.py

# Android Studio — run the app on the emulator
```

---

## Security Notes

- `local.properties` is excluded from version control (contains local SDK path)
- No patient data, encryption keys, or credentials are committed to this repository
- The `.db` files are excluded — Realm database files remain on device only
- Production deployment requires key management via Android Keystore (not yet implemented — Phase 2)

---

## Team

**Lloyd Lewis** — Cybersecurity & Backend  
University of Wollongong in Dubai | Cognitology Internship  
Supervised by Basma | Technical mentor: Dr. Amit
