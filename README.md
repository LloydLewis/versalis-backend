# Versalis — Adaptive VR Therapy Engine

Internship project | Cognitology / PsycReality

Versalis is an Adaptive VR Therapy Engine. A patient wears a Quest 3 VR headset (Unreal Engine) while a
clinician monitors the session from a desktop dashboard. Biofeedback signals are captured, run through an
in-conversation LLM therapist with safety guardrails, stored locally in an encrypted Realm database on the
Android/XR device, and mirrored to the clinician dashboard in real time. The system is designed to run
**fully offline** during a live session.

This is a monorepo of otherwise-independent services that only talk to each other over local HTTP/MQTT/WebSocket —
there is no shared build system. Each top-level folder has its own dependency set and is started separately
(or all at once via `start.py`).

---

## Repository Structure

| Folder | What it is |
|---|---|
| `database/` | The Android Studio Gradle project (package `com.example.myapplication`) — Kotlin/Jetpack Compose app, encrypted Realm DB, MQTT subscriber, and a local Ktor server that receives LLM interactions from the bridge |
| `database/mqtt/` | Python publisher (`sensor.py`) that reads the biometric CSV and replays it onto a Mosquitto broker |
| `unreal/LLM/` | FastAPI service (`server.py`, port 8008) that Unreal's ACE ASR calls as an OpenAI-compatible chat endpoint |
| `llmsegment/` | `aisegment.py` — the conversational-AI logic (Ollama calls, Llama Guard safety check, self-harm override, conversation history) |
| `bridge/` | FastAPI service (`main.py`, port 8002) — the hub connecting Unreal, the Android app, and the dashboard: VR frame mirror, a command WebSocket to Unreal, and biometric/LLM-interaction relay |
| `dashboard/` | Streamlit clinician dashboard — voice analysis, biofeedback display, optional Ollama clinical summaries, session intensity recommendations |
| `data/` | The synthetic multimodal biometric CSV consumed by `sensor.py` and the dashboard's ML pipeline |
| `start.py` | Orchestrator that launches the whole local stack in one go (Windows) |
| `Handover/` | A self-contained, not-yet-applied patch set that turns the Unreal conversation flow from push-to-talk into continuous conversation |

---

## Architecture

### Biometrics
```
data/*.csv → sensor.py → Mosquitto (sensor/biometric) ─┬→ dashboard (direct MQTT subscribe)
                                                          └→ Android BioDataSubscriber → Realm
```

### LLM conversation
```
Patient speech → Unreal ACE ASR
                    → POST unreal/LLM/server.py:8008 (/chat/completions or /v1/chat/completions)
                        → llmsegment/aisegment.py
                            1. regex self-harm check → clinician override message if hit
                            2. Ollama chat completion (single sentence)
                            3. Llama Guard pass over the finished exchange → safe fallback if unsafe
                        ← reply
                    → POST bridge:8002/llm/interaction
                        → bridge stores latest in memory (dashboard does not currently display it)
                        → bridge forwards to the Android app's Ktor server (10.0.2.2:8080/realm/llm-interaction)
                          → written to Realm
```

Both `/chat/completions` and `/v1/chat/completions` are accepted by the LLM service — some clients (NVIDIA
ACE's `BP_ASR_Debug`, for one) default to the OpenAI-standard `/v1/` prefix.

---

## Prerequisites

Install once, per machine:

- **Python 3** with per-folder dependencies:
  ```bash
  pip install -r bridge/requirements.txt -r unreal/LLM/requirements.txt \
              -r llmsegment/requirements.txt -r database/mqtt/requirements.txt \
              -r dashboard/requirements.txt
  ```
- **[Ollama](https://ollama.com)**, on `PATH`, with the three models `aisegment.py` calls by name pulled:
  ```bash
  ollama pull llama3.2:1b
  ollama pull llama3.1:8b-instruct-q4_k_m
  ollama pull llama-guard3:1b
  ```
- **[Mosquitto](https://mosquitto.org/download/)**, on `PATH` (or already running as a Windows service)
- **Android Studio** matching the version this project was built against (AGP 9.2.1, Kotlin 2.0.21,
  `compileSdk`/`targetSdk` 35, Jetpack XR alpha libraries) — check the exact build under
  Help → About, and grab the matching build from the
  [release archive](https://developer.android.com/studio/archive) if your install auto-updates past it.
  Android SDK Platform 35 needs to be installed via the SDK Manager.
- **Unreal Engine** project with the NVIDIA ACE ASR/TTS plugins — this repo only contains the Python service
  that project talks to (`unreal/LLM/server.py`); the `.uproject` itself lives outside this repo.

---

## Running the full stack

```bash
python start.py
```

This launches, each in its own console window: Ollama, an Ollama warmup request, the LLM FastAPI service
(port 8008), the bridge (port 8002), Mosquitto, the sensor simulator, and the Streamlit dashboard (port 8501).
It only pre-flight-checks that the required files exist — it does not install dependencies.

Two steps it can't do for you:
1. **Run the Android app from Android Studio.** Open the `database/` folder specifically (not the repo root) —
   that's the actual Gradle project root. On launch it opens Realm, connects to MQTT, and starts the Ktor
   server on port 8080 that receives LLM interactions from the bridge.
2. **Press Play in the Unreal project.** `10.0.2.2` is hardcoded in a few places (`BioDataSubscriber.kt`,
   the bridge's forward to the Ktor server) — that address only resolves to "the host machine" from inside
   the **Android Emulator**, so the Android app needs to run in the emulator, not a physical device, unless
   those addresses are changed to the host's real IP.

Once everything is up:

| Service | Address |
|---|---|
| Ollama | http://localhost:11434 |
| LLM server | http://127.0.0.1:8008 |
| Bridge | http://localhost:8002 (docs at `/docs`) |
| Mosquitto | localhost:1883 |
| Dashboard | http://localhost:8501 |
| Bridge health check | http://localhost:8002/health |
| Latest biometric | http://localhost:8002/biometric/latest |
| Latest LLM interaction | http://localhost:8002/llm/latest |

## Running services individually

```bash
# Ollama — must be running before the LLM service or dashboard AI summary
ollama serve

# LLM service — Unreal ASR/TTS backend, port 8008
cd unreal/LLM
uvicorn server:app --host 127.0.0.1 --port 8008

# Bridge — Unreal <-> dashboard <-> Android hub, port 8002
cd bridge
uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# MQTT: broker, then publisher
mosquitto -v -c database/mqtt/mosquitto.conf
cd database/mqtt
python sensor.py

# Clinician dashboard, port 8501
cd dashboard
streamlit run integrated_avrte_dashboard.py
```

If running the dashboard on its own (not via `start.py`), start Mosquitto **first** — the dashboard's MQTT
subscriber connects on a background thread with no reconnect loop, so it needs the broker up before it
starts, or it'll throw `ConnectionRefusedError` once and never receive live data for that session.

---

## Troubleshooting

- **`ConnectionRefusedError` on port 1883 (dashboard or Android)** — Mosquitto isn't running yet. Start it
  before the dashboard/app, or just use `start.py` which handles ordering.
- **`WinError 10048` binding port 8002 (or any service port)** — something's already bound to that port,
  usually a leftover console window from a previous `start.py` run that didn't fully close. Find it with
  `netstat -ano | findstr :8002` and `taskkill /PID <pid> /F`, or just close the stray window.
  `start.py` opens each service in its own new console window and doesn't auto-close old ones.
  Uvicorn on Windows: `[Errno 10048] error while attempting to bind on address` is `netstat` + `taskkill`,
  no code fix needed.
- **`sounddevice.PortAudioError: Error querying device -1`** — no default microphone is available on that
  machine (no mic connected, no default input device set in Windows Sound settings, or a VM/RDP session
  without audio passthrough). Not a code bug — check Windows Sound settings and microphone privacy
  permissions first.
- **Unreal: `POST http://127.0.0.1:8008/... ConnectionError` / "Connection refused"** — nothing is listening
  on port 8008. Confirm the LLM server console window is actually up (not crashed) at the moment you test —
  a quick sanity check is opening `http://127.0.0.1:8008/health` in a browser right before testing in Unreal.
- **Unreal: `HTTP 401` from `api.openai.com`** — this is unrelated to the local pipeline. NVIDIA ACE's
  `BP_ASR_Debug` sample blueprint ships with its own independent LLM connector defaulted to OpenAI's real
  API with no key configured. It's a separate call from the one your virtual-human character makes to
  `127.0.0.1:8008` — safe to ignore, or repoint its Base URL at `127.0.0.1:8008` to quiet the log noise.
- **`ModuleNotFoundError: No module named 'aisegment'`** — fixed as of this revision (the `sys.path` hack in
  `unreal/LLM/server.py` previously pointed one directory too shallow). If you still see this, you're
  running a stale copy of `server.py`.
- **Two copies of `server.py` drifting out of sync** — if your Unreal project keeps its own copy of
  `unreal/LLM/server.py` rather than running this repo's copy directly, keep them in sync manually (or
  better, point the Unreal-side setup at running this repo's copy instead of a duplicate).

---

## Security & Privacy

- Encrypted local Realm database on-device: a 64-byte AES-256 key, itself encrypted at rest via a
  hardware-backed Android Keystore AES-GCM key. The plaintext key is zeroed from memory immediately after
  Realm opens.
- `local.properties` is excluded from version control (contains the local Android SDK path).
- No patient data, encryption keys, or credentials are committed to this repository.
- `.realm` and `.enc` files are excluded from version control — database and key files remain on-device only.
- `deleteRealmIfMigrationNeeded()` is currently active for development — it wipes the database on schema
  change and must be replaced with a proper migration strategy before any clinical deployment.
- The whole system is designed to run fully offline during a live session — the only outbound calls are to
  `localhost`/`10.0.2.2` between the services listed above.
