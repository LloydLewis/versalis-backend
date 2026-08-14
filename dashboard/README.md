# AVRTE Clinician Dashboard

Integrated Streamlit therapist dashboard for VR exposure therapy: live voice analysis, simulated/real-time physiological signals, descriptive biofeedback ranges, optional AI clinical summaries, and session intensity recommendations.

---

## Quick start

```powershell
streamlit run integrated_avrte_dashboard.py
```

On the setup screen, configure options, then click **Start session**.

---

## Project files

| File | Role |
|------|------|
| `integrated_avrte_dashboard.py` | Main Streamlit app (run this) |
| `clinical_interpretation_snippets.py` | Descriptive ranges + Ollama clinical summary logic |
| `analysis_engine.py` | Microphone capture, pitch/VAD, optional Vosk STT |
| `intensity_feedback_loop.py` | Train/load ML model for physiological distress (optional) |
| `vosk-model-small-en-us-0.15/` | Local speech-to-text model (already included) |
| `model.pkl` | Trained distress model (created after first training — optional) |

---

## Setup options explained

### 1. Voice / microphone — **no dataset needed**

Voice analysis is **live from your microphone**. You do **not** upload a dataset for voice.

| Checkbox | What it does | Required? |
|----------|--------------|-----------|
| **Use real microphone voice analysis** | Captures mic audio, computes pitch, prosody index, speech rate | Optional (falls back to simulated speech if off or mic unavailable) |
| **Use speech-to-text (Vosk)** | Counts filler words, repetitions, words/minute via local STT | Optional (on by default) |

**Vosk — is it working by default?**  
Yes, if:

- The checkbox **Use speech-to-text (Vosk)** is ticked (default: on), and  
- The folder `vosk-model-small-en-us-0.15` sits next to your scripts (it is already in this project).

Your event log confirms this when you see:

```
[voice] Loading Vosk STT model from 'vosk-model-small-en-us-0.15'...
[voice] Vosk STT model loaded.
[voice] Calibrating baseline for 8s - please talk normally...
[voice] Calibration complete. Live analysis started.
```

If Vosk fails, the log will say the model folder was not found or vosk is not installed (`pip install vosk`).

**Speech quality — filler & repeated words**

| Metric | How it is detected |
|--------|-------------------|
| **Filler words** | Vosk must transcribe tokens like `um`, `uh`, `erm`, `ah`, etc. in the live partial + final transcript |
| **Repeated words** | Same word twice **in a row** in the transcript (e.g. “I I think”) |
| **Words recognized** | Total words Vosk has transcribed this session — if this stays **0**, the mic/STT path is not hearing you |

**Why counts can look low (even when Vosk loaded):**

1. **`vosk-model-small-en-us-0.15` is tiny** — it often **drops** quiet fillers or writes them as real words. A larger model (e.g. `vosk-model-en-us-0.22`) improves accuracy; set `VOSK_MODEL_PATH` to its folder.
2. **Old logic only counted finalized utterances** — now fixed: counts update from **live partial** text every second.
3. **Fillers must appear in the transcript** — try saying “um” and “uh” clearly after the 8s calibration; normal fluent speech without fillers will correctly show **0**.
4. **First 8 seconds are calibration** — speech during calibration still runs but baseline mode is active; speak steadily after “Calibration complete”.

To test: after calibration, say aloud: *“Um, I I think, uh, this is a test.”* — you should see fillers ≥ 2, repeated ≥ 1, words recognized increasing.

---

### 2. AI clinical summary (Ollama) — **must run Ollama separately**

| Checkbox | What it does |
|----------|--------------|
| **Use AI clinical summary (Ollama)** | Sends heart rate, GSR, breathing, speech, and voice patterns to a **local** LLM for a short clinician paragraph |

**You need Ollama running before/during the session** if this box is checked:

```powershell
# Terminal 1 — keep this running
ollama serve

# One-time — pull the default model (llama3.2)
ollama pull llama3.2
```

Then start Streamlit in another terminal.

**If Ollama is not running**, you may see in the event log:

```
[clinical AI] synthesis unavailable (... timed out ...); using rule-based summary.
```

That is **not a crash**. The dashboard still works:

- **Biofeedback range** (likely calm / likely anxious / likely very anxious) — always on  
- **Rule-based word summary** — used when Ollama is off or times out  
- **Raw signal values** — always shown  

To avoid timeouts: start `ollama serve` first, or **uncheck** “Use AI clinical summary” on the setup screen.

Ollama calls use a 45-second timeout and run at most once every ~8 seconds so they never block the live loop.

---

### 3. Physiological dataset + `model.pkl` — **optional (not for voice)**

This is what the log line refers to:

```
No existing model and no dataset folder given - running on rules only.
```

**This is NOT about voice.** It means:

| Missing | Effect |
|---------|--------|
| No `model.pkl` in the project folder | No ML distress classifier loaded |
| No dataset folder in setup | Cannot train a new model or simulate full multi-signal physio (breathing, HRV, etc.) |

**What still works in “rules only” mode:**

- Rule-based **composite arousal index** (0–100)  
- **Biofeedback range** labels  
- **Recommendations** (hold / encourage / reduce intensity / stop)  
- Simulated **heart rate + GSR** (simple ramp — no breathing from dataset)  
- **Voice** from mic (if enabled)  

**What you gain with dataset + model:**

| With dataset folder | With `model.pkl` (train or pre-place file) |
|---------------------|------------------------------------------|
| Realistic simulated HR, GSR, **breathing rate**, HRV, etc. from your synthetic Excel/CSV data | ML blends 40% into the composite arousal index (60% rules + 40% ML) |

**Dataset requirements:**

- Folder of `.xlsx`, `.xls`, or `.csv` files  
- Must include a `ground_truth_label` column (for training)  
- Example path: `C:\Users\Fatima\Downloads\Synthetic Data`

**First run with dataset (no `model.pkl` yet):**  
The app trains automatically, saves `model.pkl`, then uses it for the session (can take a minute).

**Later runs:**  
Place `model.pkl` in this folder **and** provide the same dataset folder for best simulation (especially breathing rate in the UI).

---

## Dashboard layout (psych team design)

1. **Inferred emotional state** — descriptive range (*likely calm / anxious / very anxious*), not an objective anxiety score  
2. **AI clinical summary** — Ollama paragraph + ethical disclaimer (or rule-based fallback)  
3. **Raw signal values** — always visible (not a dropdown): HR, GSR, breathing, speech rate, voice prosody, pitch, pauses  
4. **Speech quality** — filler words and repeated words (when mic + Vosk are on)  
5. **Session arousal index** — internal 0–100 fusion score + therapist recommendations  
6. **Live signal trends** + **Session event log**

Head movement and eye tracking are **not shown** to clinicians (still used internally for scoring only). Live transcript text was removed; STT still powers speech-quality metrics.

---

## Typical event log (healthy session)

```
[voice] Loading Vosk STT model from 'vosk-model-small-en-us-0.15'...
[voice] Vosk STT model loaded.
[voice] Calibrating baseline for 8s - please talk normally...
[voice] Calibration complete. Live analysis started.
Loaded existing distress model from model.pkl.          ← only if model.pkl exists
Loaded reference dataset for realistic multi-signal...  ← only if dataset folder set
```

Or without ML:

```
No existing model and no dataset folder given - running on rules only.
```

Both are normal depending on what you configured.

---

## Python dependencies

Install as needed (not all are required — each feature degrades gracefully):

```powershell
pip install streamlit numpy pandas scikit-learn joblib sounddevice requests vosk
```

Optional: `mne` (EEG — not wired in this prototype), `openpyxl` (reading `.xlsx` datasets).

---

## Environment variables (optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model for clinical summaries |
| `VOSK_MODEL_PATH` | `vosk-model-small-en-us-0.15` | Path to unzipped Vosk model |

---

## FAQ

**Do I need Ollama?**  
Only if you want AI-generated clinical summary paragraphs. Otherwise uncheck it on setup; rule-based summaries and all raw numbers still work.

**Is Vosk working?**  
If your log shows `Vosk STT model loaded`, yes. Your screenshot indicates Vosk is working correctly.

**What does “running on rules only” mean?**  
No ML model (`model.pkl`) is loaded. Scoring uses clinician-threshold rules only, and physio simulation uses a simple HR/GSR ramp unless you provide a dataset folder.

**Why is breathing rate “—”?**  
Breathing comes from the dataset-driven simulator. Provide the dataset folder (and ideally `model.pkl`) on setup to populate it.

**Did something “time out”?**  
Most likely **Ollama** if AI summary was enabled but `ollama serve` was not running. Check the log for `[clinical AI] synthesis unavailable`. Vosk loading in your screenshot completed successfully — that was not a timeout.

---

## Privacy note

Only an encrypted session summary is intended for records afterward. Live signals and voice processing are designed to stay on the clinic workstation.
