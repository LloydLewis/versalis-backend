import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def start_service(name, cmd, cwd=None, delay=2):
    resolved_cwd = os.path.abspath(cwd) if cwd else None
    try:
        subprocess.Popen(
            cmd,
            cwd=resolved_cwd,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print(f"  [{name}] started")
        time.sleep(delay)
    except FileNotFoundError as e:
        print(f"  [{name}] ERROR — {e}")
        print(f"           Make sure the tool is installed and on your PATH")


def check_file(path, label):
    if not os.path.exists(path):
        print(f"  WARNING: {label} not found at {path}")
        return False
    return True


print()
print("=" * 55)
print("  Versalis — Full System Startup")
print("=" * 55)
print()

# ── Paths ───────────────────────────────────────────────────
bridge_dir     = os.path.join(ROOT, "bridge")
mqtt_dir       = os.path.join(ROOT, "database", "mqtt")
llm_dir        = os.path.join(ROOT, "unreal", "server")
dashboard_dir  = os.path.join(ROOT, "dashboard")

# ── Pre-flight checks ───────────────────────────────────────
print("Pre-flight checks...")

all_ok = True
all_ok &= check_file(os.path.join(bridge_dir,    "main.py"),           "bridge/main.py")
all_ok &= check_file(os.path.join(llm_dir,       "server.py"),         "unreal/server/server.py")
all_ok &= check_file(os.path.join(mqtt_dir,      "sensor.py"),         "database/mqtt/sensor.py")
all_ok &= check_file(os.path.join(mqtt_dir,      "mosquitto.conf"),    "database/mqtt/mosquitto.conf")
all_ok &= check_file(
    os.path.join(ROOT, "data", "synthetic_multimodal_part_01.csv"),
    "data/synthetic_multimodal_part_01.csv"
)
all_ok &= check_file(
    os.path.join(dashboard_dir, "integrated_avrte_dashboard.py"),
    "dashboard/integrated_avrte_dashboard.py"
)

if not all_ok:
    print()
    print("Fix the missing files above before starting.")
    input("Press Enter to exit...")
    sys.exit(1)

print("  All files found")
print()

# ── Services ────────────────────────────────────────────────
print("Starting services...")
print()

# 1. Ollama LLM server
start_service(
    name  = "Ollama",
    cmd   = ["ollama", "serve"],
    delay = 5
)

# 2. Ollama model warmup — pre-loads llama3.2:1b into memory
warmup_cmd = (
    'Invoke-WebRequest -UseBasicParsing '
    '-Uri "http://127.0.0.1:11434/api/chat" '
    '-Method Post -ContentType "application/json" '
    '-Body \'{"model":"llama3.2:1b","messages":[{"role":"user","content":"hi"}],'
    '"stream":false,"keep_alive":"60m","options":{"num_ctx":1024}}\''
)
start_service(
    name  = "Ollama warmup (llama3.2:1b)",
    cmd   = ["powershell", "-NoExit", "-Command", warmup_cmd],
    delay = 8
)

# 3. LLM FastAPI server (Unreal ASR/TTS backend — port 8008)
start_service(
    name  = "LLM server (port 8008)",
    cmd   = [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", "8008"],
    cwd   = llm_dir,
    delay = 3
)

# 4. Bridge service (Unreal mirror + dashboard commands — port 8002)
start_service(
    name  = "Bridge service (port 8002)",
    cmd   = ["cmd", "/k", sys.executable, "-m", "uvicorn", "main:app",
             "--host", "0.0.0.0", "--port", "8002"],
    cwd   = bridge_dir,
    delay = 2
)

# 5. Mosquitto MQTT broker
start_service(
    name  = "Mosquitto broker",
    cmd   = ["mosquitto", "-v", "-c", "mosquitto.conf"],
    cwd   = mqtt_dir,
    delay = 2
)

# 6. Biofeedback sensor simulator
# Reads the PsycReality CSV and publishes biometric readings to Mosquitto
start_service(
    name  = "Sensor simulator",
    cmd   = [sys.executable, "sensor.py"],
    cwd   = mqtt_dir,
    delay = 2
)

# 7. Clinician dashboard
start_service(
    name  = "Clinician dashboard",
    cmd   = [sys.executable, "-m", "streamlit", "run",
             "integrated_avrte_dashboard.py",
             "--server.port", "8501"],
    cwd   = dashboard_dir,
    delay = 3
)

# ── Done ────────────────────────────────────────────────────
print()
print("=" * 55)
print("  All services running")
print()
print("  Addresses:")
print("  - Ollama:          http://localhost:11434")
print("  - LLM server:      http://127.0.0.1:8008")
print("  - Bridge:          http://localhost:8002")
print("  - Bridge docs:     http://localhost:8002/docs")
print("  - Mosquitto:       localhost:1883")
print("  - Dashboard:       http://localhost:8501")
print()
print("  Verification links:")
print("  - Bridge health:   http://localhost:8002/health")
print("  - Mirror status:   http://localhost:8002/mirror/status")
print("  - Latest biometric:http://localhost:8002/biometric/latest")
print("  - Latest LLM:      http://localhost:8002/llm/latest")
print()
print("  Manual steps still required:")
print("  1. Run Android app from Android Studio")
print("  2. Press Play in Unreal Editor")
print("=" * 55)
print()
input("Press Enter to close this window (services keep running)...")