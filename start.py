import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def start_service(name, cmd, cwd=None, delay=2, shell=False):
    """Launch a service in its own terminal window."""
    resolved_cwd = os.path.abspath(cwd) if cwd else None
    try:
        subprocess.Popen(
            cmd,
            cwd=resolved_cwd,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            shell=shell
        )
        print(f"  [{name}] started")
        time.sleep(delay)
    except FileNotFoundError as e:
        print(f"  [{name}] ERROR — {e}")
        print(f"           Command: {cmd}")


def check_file(path, label):
    if not os.path.exists(path):
        print(f"  WARNING: {label} not found at: {path}")
        return False
    return True


# ── Banner ─────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  Versalis — Unreal + Dashboard Connection Startup")
print("=" * 55)
print()

# ── Pre-flight checks ───────────────────────────────────────────────────────
print("Pre-flight checks...")

bridge_main = os.path.join(ROOT, "bridge", "main.py")
llm_server  = r"C:\UnrealProjects\UE\LLM\server.py"

all_ok = True
all_ok &= check_file(bridge_main, "bridge/main.py")
all_ok &= check_file(llm_server,  r"C:\UnrealProjects\UE\LLM\server.py")

if not all_ok:
    print()
    print("One or more required files are missing. Fix the above before starting.")
    input("Press Enter to exit...")
    sys.exit(1)

print("  All files found")
print()

# ── Service startup ─────────────────────────────────────────────────────────
print("Starting services...")
print()

# 1. Ollama LLM server
start_service(
    name  = "Ollama",
    cmd   = ["ollama", "serve"],
    delay = 5
)

# 2. Warm up llama3.2:1b — pre-loads the model into memory
# Uses PowerShell to run the Invoke-WebRequest command you specified
warmup_cmd = (
    'Invoke-WebRequest '
    '-UseBasicParsing '
    '-Uri "http://127.0.0.1:11434/api/chat" '
    '-Method Post '
    '-ContentType "application/json" '
    '-Body \'{"model":"llama3.2:1b","messages":[{"role":"user","content":"hi"}],"stream":false,"keep_alive":"60m","options":{"num_ctx":1024}}\''
)

start_service(
    name  = "Ollama model warmup (llama3.2:1b)",
    cmd   = ["powershell", "-NoExit", "-Command", warmup_cmd],
    delay = 8
)

# 3. LLM FastAPI server (Unreal ASR/TTS backend)
start_service(
    name  = "LLM server (port 8008)",
    cmd   = [
        sys.executable, "-m", "uvicorn",
        "server:app",
        "--host", "127.0.0.1",
        "--port", "8008"
    ],
    cwd   = r"C:\UnrealProjects\UE\LLM",
    delay = 3
)

# 4. Bridge service (Unreal <-> Dashboard mirror + encouragement commands)
bridge_dir = os.path.join(ROOT, "bridge")
print(f"  [Bridge] Looking in: {bridge_dir}")
print(f"  [Bridge] main.py exists: {os.path.exists(os.path.join(bridge_dir, 'main.py'))}")

start_service(
    name  = "Bridge service (port 8002)",
    cmd   = [
        "cmd", "/k",
        sys.executable, "-m", "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", "8002",
        "--reload"
    ],
    cwd   = bridge_dir,
    delay = 2
)

# ── Done ────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  All services running")
print()
print("  Service addresses:")
print("  - Ollama:          http://localhost:11434")
print("  - LLM server:      http://127.0.0.1:8008")
print("  - Bridge service:  http://localhost:8002")
print("  - Bridge docs:     http://localhost:8002/docs")
print()
print("  Next steps:")
print("  1. Press Play in Unreal Editor")
print("  2. Run the clinician dashboard:")
print("     streamlit run integrated_avrte_dashboard.py")
print("=" * 55)
print()
input("Press Enter to close this window (services keep running)...")
