# Versalis — Backend & Data Layer

Internship project | Cognitology / PsycReality | Cybersecurity & Backend Team

Versalis is an Adaptive VR Therapy Engine designed for clinical mental health settings. This repository contains the backend architecture, local database layer, and data ingestion pipeline built during Phase 1 and Phase 2 of the internship.

---

## Project Overview

Versalis runs entirely offline during live therapy sessions. A patient wears a Quest 3 VR headset while a Mental Health Professional (MHP) monitors the session from a clinician desktop. Biofeedback signals (heart rate, EEG, GSR, blood pressure, voice stress, and more) are captured in real time, processed by an adaptive engine, and stored locally in an AES-256 encrypted Realm database. After the session ends, a summary is synced to the Prosper AI cloud platform (MHMS).

This repository covers the data and backend layer of that system.

---

## Repository Structure

```
MyApplication/
├── app/                        # Android/Kotlin application
│   └── src/main/
│       └── com.example.myapplication/
│           ├── database/
│           │   ├── RealmDatabase.kt    # Realm initialisation with encryption
│           │   └── KeystoreManager.kt  # Android Keystore key generation and retrieval
│           ├── models/
│           │   ├── BiometricReading.kt # Full biometric schema (44 fields)
│           │   └── ButtonPress.kt      # Demo write model
│           ├── mqtt/
│           │   └── BioDataSubscriber.kt # HiveMQ MQTT subscriber
│           ├── repository/
│           │   └── BioDataRepository.kt # Realm read/write operations
│           └── MainActivity.kt          # Entry point — wires encryption, MQTT, and Realm
└── mqtt/                       # Python MQTT pipeline (proof of concept)
    ├── sensor.py               # Publishes real biometric CSV data to Mosquitto
    ├── mosquitto.conf          # Broker config (allows connections from Android emulator)
    ├── requirements.txt        # Python dependencies
    └── README.md               # Pipeline setup instructions
```

---

## What Has Been Built

### Encryption (Android Keystore)
- 64-byte AES-256 Realm encryption key generated using `SecureRandom` on first launch
- Key is encrypted using a hardware-backed AES-GCM key stored in Android Keystore — never exists in plaintext on disk
- Encrypted key stored as `realm_key.enc` in app-private storage; retrieved and decrypted on every subsequent launch
- Key generation and Realm initialisation run on a background thread (`Dispatchers.IO`) to avoid blocking the main thread
- Key is zeroed from memory immediately after Realm opens

### Local Database (Realm)
- Encrypted local Realm database running on the Android/XR device
- `BiometricReading` schema covers 44 fields across seven signal groups: EEG bands, EEG derived indices, heart rate and HRV, respiratory, blood pressure and oxygen, skin conductance, and voice-derived signals
- Also stores adaptive engine output (distress score, intensity direction, threshold flags) and clinical safety flags (adverse event, clinician intervention, AI escalation)
- `PatientBaseline` schema captures resting values at session start for deviation calculation
- `IntensityRecommendation` schema logs engine recommendations and therapist responses separately from raw readings
- Offline-first — no network dependency during a live session
- `deleteRealmIfMigrationNeeded()` enabled for development; to be removed before clinical deployment

### MQTT Data Ingestion Pipeline
- HiveMQ Kotlin client (1.3.3) subscribes to Mosquitto broker on `sensor/biometric`
- Full JSON biometric payload parsed on receipt and written directly to Realm — no intermediate SQLite layer
- `sensor.py` reads real multimodal biometric CSV data provided by Cognitology and publishes each row as a structured JSON message at a configurable interval
- 44 fields published per message covering all signal types in the `BiometricReading` schema
- Topic changed from `sensor/heartrate` (single integer) to `sensor/biometric` (full JSON payload)

### Proof of Concept Demo
The Android app demonstrates two flows simultaneously:
1. **MQTT → Realm** — live summary line (HR, anxiety score, session phase) updating every second from incoming MQTT messages; full record query button displays all 44 fields of the latest reading on screen
2. **Manual write → Realm** — button press flow verifying direct database writes and record count queries

---

## Tech Stack

| Layer | Technology |
|---|---|
| Device database | Realm (Kotlin SDK 1.16.0) |
| Database encryption | AES-256 via Realm + Android Keystore (AES-GCM, hardware-backed) |
| MQTT client (Android) | HiveMQ MQTT Client 1.3.3 |
| MQTT broker | Mosquitto 2.x |
| Sensor simulator | Python 3 + paho-mqtt + pandas |
| Android UI | Jetpack Compose |
| Language (Android) | Kotlin |
| Language (pipeline) | Python 3 |

---

## Running the Pipeline

See [`mqtt/README.md`](mqtt/README.md) for full setup instructions.

Quick start:

```bash
# Terminal 1 — start broker with external access enabled
mosquitto -v -c mqtt/mosquitto.conf

# Terminal 2 — start CSV publisher
cd mqtt
python sensor.py

# Android Studio — run the app on the emulator or XR device
```

The app will show a loading screen ("Initialising secure database...") while the Keystore key is retrieved and Realm is opened. Once ready, the full UI renders and MQTT data begins flowing automatically.

---

## Security Notes

- `local.properties` is excluded from version control (contains local SDK path)
- No patient data, encryption keys, or credentials are committed to this repository
- `.realm` and `.enc` files are excluded — database and key files remain on device only
- Android Keystore key is hardware-backed — cannot be exported or read by any other app
- Realm encryption key is zeroed from memory immediately after database opens
- `deleteRealmIfMigrationNeeded()` is active — database wipes on schema change during development; must be replaced with a proper migration strategy before production
- Production deployment requires pseudonymisation enforcement at schema level, key rotation policy, and removal of all development flags

