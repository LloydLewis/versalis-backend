import paho.mqtt.client as mqtt
import pandas as pd
import json
import time
from pathlib import Path
print("Sensor script started")

# ── Load CSV ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
CSV_PATH = REPO_ROOT / "data" / "synthetic_multimodal_part_01.csv"

# Optional check to ensure the file exists
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find CSV file at: {CSV_PATH}")

# If using open() or pandas, pass CSV_PATH directly (or str(CSV_PATH))
print(f"Loading CSV from: {CSV_PATH}")

# Only load the columns that map to BiometricReading
REQUIRED_COLUMNS = [
    "participant_id",
    "session_id",
    "timestamp_utc",
    "session_phase",
    "sampling_rate_hz",
    "eeg_channel",
    "eeg_alpha_uv2",
    "eeg_beta_uv2",
    "eeg_theta_uv2",
    "eeg_delta_uv2",
    "eeg_gamma_uv2",
    "frontal_alpha_asymmetry",
    "attention_index",
    "meditation_index",
    "cognitive_load",
    "mental_fatigue",
    "heart_rate_bpm",
    "hrv_rmssd_ms",
    "hrv_sdnn_ms",
    "hrv_pnn50_pct",
    "hrv_lf_hf_ratio",
    "respiratory_rate_rpm",
    "tidal_volume_ml",
    "respiratory_variability",
    "spo2_pct",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg",
    "gsr_us",
    "skin_conductance_level_us",
    "skin_conductance_response_us",
    "skin_temp_c",
    "speech_rate_wpm",
    "voice_energy_norm",
    "pause_duration_s",
    "voice_f0_hz",
    "speech_sentiment",
    "voice_stress_score",
    "anxiety_score",
    "stress_score",
    "depression_index",
    "data_quality_score",
    "adverse_event_flag",
    "clinician_intervention_flag",
    "ai_escalation_flag",
]

print(f"Loading CSV: {CSV_PATH}")
try:
    df = pd.read_csv(CSV_PATH, usecols=REQUIRED_COLUMNS)
    print(f"Loaded {len(df)} rows")
except FileNotFoundError:
    print(f"ERROR: Could not find {CSV_PATH}")
    print("Make sure the CSV file is in the same folder as sensor.py")
    input("Press Enter to close...")
    exit(1)
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    input("Press Enter to close...")
    exit(1)

# ── MQTT setup ────────────────────────────────────────────────────────────────
TOPIC = "sensor/biometric"   # updated topic — now sends full biometric payload

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    print("Connecting to broker...")
    client.connect("localhost", 1883)
    print("Connected!")
    client.loop_start()

    # ── Publish each row ──────────────────────────────────────────────────────
    for index, row in df.iterrows():
        payload = {
            # Identity
            "participantId":               str(row["participant_id"]),
            "sessionId":                   str(row["session_id"]),
            "timestampUtc":                str(row["timestamp_utc"]),
            "sessionPhase":                str(row["session_phase"]).upper(),
            "samplingRateHz":              int(row["sampling_rate_hz"]),
            "dataQualityScore":            float(row["data_quality_score"]),

            # EEG bands
            "eegChannel":                  str(row["eeg_channel"]),
            "eegAlphaUv2":                 float(row["eeg_alpha_uv2"]),
            "eegBetaUv2":                  float(row["eeg_beta_uv2"]),
            "eegThetaUv2":                 float(row["eeg_theta_uv2"]),
            "eegDeltaUv2":                 float(row["eeg_delta_uv2"]),
            "eegGammaUv2":                 float(row["eeg_gamma_uv2"]),

            # EEG derived
            "frontalAlphaAsymmetry":       float(row["frontal_alpha_asymmetry"]),
            "attentionIndex":              float(row["attention_index"]),
            "meditationIndex":             float(row["meditation_index"]),
            "cognitiveLoad":               float(row["cognitive_load"]),
            "mentalFatigue":               float(row["mental_fatigue"]),

            # Heart rate & HRV
            "heartRateBpm":                float(row["heart_rate_bpm"]),
            "hrvRmssdMs":                  float(row["hrv_rmssd_ms"]),
            "hrvSdnnMs":                   float(row["hrv_sdnn_ms"]),
            "hrvPnn50Pct":                 float(row["hrv_pnn50_pct"]),
            "hrvLfHfRatio":                float(row["hrv_lf_hf_ratio"]),

            # Respiratory
            "respiratoryRateRpm":          float(row["respiratory_rate_rpm"]),
            "tidalVolumeMl":               float(row["tidal_volume_ml"]),
            "respiratoryVariability":      float(row["respiratory_variability"]),

            # Blood pressure & oxygen
            "systolicBpMmhg":              float(row["systolic_bp_mmhg"]),
            "diastolicBpMmhg":             float(row["diastolic_bp_mmhg"]),
            "spo2Pct":                     float(row["spo2_pct"]),

            # Skin
            "gsrUs":                       float(row["gsr_us"]),
            "skinConductanceLevelUs":      float(row["skin_conductance_level_us"]),
            "skinConductanceResponseUs":   float(row["skin_conductance_response_us"]),
            "skinTempC":                   float(row["skin_temp_c"]),

            # Voice
            "speechRateWpm":               float(row["speech_rate_wpm"]),
            "voiceEnergyNorm":             float(row["voice_energy_norm"]),
            "pauseDurationS":              float(row["pause_duration_s"]),
            "voiceF0Hz":                   float(row["voice_f0_hz"]),
            "speechSentiment":             float(row["speech_sentiment"]),
            "voiceStressScore":            float(row["voice_stress_score"]),

            # Scores
            "anxietyScore":                float(row["anxiety_score"]),
            "stressScore":                 float(row["stress_score"]),
            "depressionIndex":             float(row["depression_index"]),

            # Safety flags
            "adverseEventFlag":            bool(row["adverse_event_flag"]),
            "clinicianInterventionFlag":   bool(row["clinician_intervention_flag"]),
            "aiEscalationFlag":            bool(row["ai_escalation_flag"]),
        }

        message = json.dumps(payload)
        result = client.publish(TOPIC, message)

        print(f"Row {index + 1}/{len(df)} | "
              f"Session: {payload['sessionId']} | "
              f"Phase: {payload['sessionPhase']} | "
              f"HR: {payload['heartRateBpm']} bpm | "
              f"Anxiety: {payload['anxietyScore']}")

        time.sleep(1)  # 1 second between rows — adjust as needed

    print("All rows published")

except KeyboardInterrupt:
    print("\nStopping publisher cleanly...")
except Exception as e:
    print(f"ERROR: {e}")
    input("Press Enter to close...")
finally:
    client.loop_stop()
    client.disconnect()
    print("Disconnected from broker")