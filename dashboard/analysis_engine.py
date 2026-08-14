"""
analysis_engine.py

Real-time voice analysis engine: captures microphone audio, extracts
prosodic features (pitch, energy, speech rate / pauses), computes a
heuristic "anxiety" score, and optionally runs a pretrained speech-emotion
model and streaming speech-to-text for a secondary signal.

`AnalysisEngine` has no GUI code in it - it just calls `on_result(dict)` and
`on_status(str)` callbacks, which main_gui.py wires up to a Tkinter window.
"""

import collections
import json
import os
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    HAVE_VOSK = True
except ImportError:
    HAVE_VOSK = False


SAMPLE_RATE = 16000          # matches Vosk models' expected input rate
BLOCK_SECONDS = 1.0          # how often we compute a feature update
VAD_FRAME_MS = 30            # sub-frame size for the energy-based pause detector
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
CALIBRATION_SECONDS = 8.0    # "just talk normally" baseline window
HISTORY_LEN = 300            # rolling history for plotting / smoothing (~5 min at 1s blocks)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
EMOTION_MIN_INTERVAL = 6.0    # seconds between Ollama classification calls
EMOTION_LOOKBACK_SECONDS = 20.0  # how much recent transcript to classify
ANXIETY_EMA_ALPHA = 0.3       # smoothing factor for anxiety_score (lower = smoother/slower)

# Path to an unzipped Vosk model directory. Download one from
# https://alphacephei.com/vosk/models (e.g. vosk-model-small-en-us-0.15 for a
# fast/light model, or a larger one for better accuracy) and unzip it next to
# this file, or set the VOSK_MODEL_PATH environment variable.
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")

# Fillers ASR may emit (small models often skip or garble these — see _recount_speech_quality)
FILLER_WORDS = {
    "um", "uh", "erm", "er", "ah", "eh", "hmm", "hm", "mhm", "huh",
    "uhh", "umm", "uhm", "ummm", "mm", "em", "ermm", "aah", "ehh",
}
# Catch elongated / noisy filler tokens in raw transcript text
FILLER_PATTERN = re.compile(
    r"\b(u+h+m*|u+m+h*|e+r+m*|a+h+|e+h+|h+m+|m+h+m*)\b",
    re.IGNORECASE,
)


class RollingBaseline:
    """Tracks mean/std of a scalar signal during a calibration phase,
    then exposes a z-score for values afterward."""

    def __init__(self):
        self.samples = []
        self.mean = 0.0
        self.std = 1.0
        self.locked = False

    def add(self, value):
        if not self.locked:
            self.samples.append(value)

    def lock(self):
        if self.samples:
            self.mean = float(np.mean(self.samples))
            self.std = float(np.std(self.samples)) or 1.0
        self.locked = True

    def zscore(self, value):
        return (value - self.mean) / self.std


class AnalysisEngine:
    def __init__(self, on_result, on_status=None, use_emotion_model=True, use_stt=True):
        """
        on_result(dict): called from a background thread roughly once per
            BLOCK_SECONDS with the latest analysis result. GUI code must
            marshal this back to the main thread itself (see main_gui.py).
        on_status(str): optional callback for human-readable status/log lines.
        use_emotion_model: if True and Ollama is reachable, classifies the
            emotional tone of recent transcript text via a local Ollama
            model (default 'llama3.2') as a secondary signal.
        use_stt: if True and vosk is installed (with a model available),
            runs streaming speech-to-text for transcript, filler-word,
            word-repetition, and words-per-minute tracking.
        """
        self.on_result = on_result
        self.on_status = on_status or (lambda msg: None)
        self.use_emotion_model = use_emotion_model and HAVE_REQUESTS
        self.use_stt = use_stt and HAVE_VOSK

        self._audio_q = queue.Queue()
        self._stop_event = threading.Event()
        self._stream = None
        self._proc_thread = None

        self._ring = collections.deque(maxlen=SAMPLE_RATE * 6)  # last 6s of audio
        self._block_counter = 0

        self._pitch_baseline = RollingBaseline()
        self._rate_baseline = RollingBaseline()
        self._pause_baseline = RollingBaseline()
        self._smoothed_anxiety = 0.0  # exponential moving average of anxiety_score

        self._calibrating = True
        self._calibration_start = None

        self._transcript_log = collections.deque(maxlen=200)  # (timestamp, text)
        self._latest_emotion = (None, None)  # (label, confidence)
        self._last_emotion_call_time = 0.0
        self._emotion_busy = False

        self._vosk_model = None
        self._vosk_rec = None
        self.session_filler_count = 0
        self.session_repetition_count = 0
        self.session_word_count = 0
        self._speech_start_wall = None  # wall-clock time.time() at first recognized word
        self._last_word_seen = None
        self._accumulated_final_text = ""

        self.history = collections.deque(maxlen=HISTORY_LEN)

    # ---------------------------------------------------------------
    # lifecycle
    # ---------------------------------------------------------------
    def start(self):
        if self.use_emotion_model:
            try:
                tags_url = OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
                requests.get(tags_url, timeout=3).raise_for_status()
                self.on_status(f"Connected to Ollama - using model "
                                f"'{OLLAMA_MODEL}' for emotion classification.")
            except Exception as exc:  # noqa: BLE001
                self.on_status(
                    f"Could not reach Ollama at {OLLAMA_URL} ({exc}). "
                    f"Is 'ollama serve' running and have you run "
                    f"'ollama pull {OLLAMA_MODEL}'? Continuing without "
                    f"emotion classification."
                )
                self.use_emotion_model = False
        elif not HAVE_REQUESTS:
            self.on_status("requests not installed - skipping emotion "
                            "classification (pip install requests)")

        if self.use_stt:
            if not os.path.isdir(VOSK_MODEL_PATH):
                self.on_status(
                    f"STT requested but model folder '{VOSK_MODEL_PATH}' not found. "
                    f"Download a model from https://alphacephei.com/vosk/models, "
                    f"unzip it next to this script (or set VOSK_MODEL_PATH), "
                    f"and restart. Continuing without transcription."
                )
                self.use_stt = False
            else:
                try:
                    self.on_status(f"Loading Vosk STT model from '{VOSK_MODEL_PATH}'...")
                    self._vosk_model = VoskModel(VOSK_MODEL_PATH)
                    self._vosk_rec = KaldiRecognizer(self._vosk_model, SAMPLE_RATE)
                    self._vosk_rec.SetWords(True)  # request word-level timestamps
                    self.on_status("Vosk STT model loaded.")
                except Exception as exc:  # noqa: BLE001
                    self.on_status(f"Could not load Vosk model ({exc}); "
                                    f"continuing without transcription.")
                    self.use_stt = False
        elif not HAVE_VOSK:
            self.on_status("vosk not installed - skipping speech-to-text "
                            "(pip install vosk)")

        self._stop_event.clear()
        self.session_filler_count = 0
        self.session_repetition_count = 0
        self.session_word_count = 0
        self._speech_start_wall = None
        self._last_word_seen = None
        self._accumulated_final_text = ""
        self._calibrating = True
        self._calibration_start = time.time()
        self.on_status(f"Calibrating baseline for {CALIBRATION_SECONDS:.0f}s - "
                        f"please talk normally...")

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
            callback=self._audio_callback,
        )
        self._stream.start()

        self._proc_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._proc_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._vosk_rec = None
        self._vosk_model = None

    # ---------------------------------------------------------------
    # audio capture (runs on sounddevice's internal thread)
    # ---------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.on_status(f"audio status: {status}")
        self._audio_q.put(indata[:, 0].copy())

    # ---------------------------------------------------------------
    # processing (runs on our own worker thread)
    # ---------------------------------------------------------------
    def _process_loop(self):
        while not self._stop_event.is_set():
            try:
                block = self._audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self._ring.extend(block.tolist())

            if self._calibrating and time.time() - self._calibration_start >= CALIBRATION_SECONDS:
                self._pitch_baseline.lock()
                self._rate_baseline.lock()
                self._pause_baseline.lock()
                self._calibrating = False
                self.on_status("Calibration complete. Live analysis started.")

            try:
                result = self._analyze_block(block)
            except Exception as exc:  # noqa: BLE001
                self.on_status(f"analysis error: {exc}")
                continue

            self._block_counter += 1
            self.history.append(result)
            self.on_result(result)

    def _analyze_block(self, block: np.ndarray) -> dict:
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
        energy_db = 20 * np.log10(rms)

        pitch_hz = self._estimate_pitch_autocorr(block)

        speech_ratio, pause_count = self._vad_stats(block)

        if self._calibrating:
            self._pitch_baseline.add(pitch_hz)
            self._rate_baseline.add(speech_ratio)
            self._pause_baseline.add(pause_count)

        raw_anxiety = self._compute_anxiety_score(pitch_hz, speech_ratio, pause_count)
        if raw_anxiety is not None:
            self._smoothed_anxiety = (
                ANXIETY_EMA_ALPHA * raw_anxiety
                + (1 - ANXIETY_EMA_ALPHA) * self._smoothed_anxiety
            )
        # if raw_anxiety is None (pitch detection failed this block), just
        # carry the previous smoothed value forward instead of snapping to 0
        anxiety_score = self._smoothed_anxiety

        stt_info = self._feed_stt(block)

        if stt_info["final_text"]:
            self._transcript_log.append((time.time(), stt_info["final_text"]))
            self._maybe_trigger_emotion_classification()

        emotion_label, emotion_conf = self._latest_emotion

        return {
            "timestamp": time.time(),
            "calibrating": self._calibrating,
            "pitch_hz": pitch_hz,
            "energy_db": energy_db,
            "speech_ratio": speech_ratio,   # fraction of block that is voiced speech
            "pause_count": pause_count,     # number of distinct silence gaps in block
            "anxiety_score": anxiety_score, # 0..1 heuristic
            "emotion_label": emotion_label,
            "emotion_conf": emotion_conf,
            "transcript_partial": stt_info["partial"],
            "transcript_final_delta": stt_info["final_text"],
            "filler_count_session": self.session_filler_count,
            "repetition_count_session": self.session_repetition_count,
            "words_per_minute": stt_info["wpm"],
            "words_recognized_session": self.session_word_count,
            "stt_active": self.use_stt,
        }

    @staticmethod
    def _estimate_pitch_autocorr(block: np.ndarray, fmin=75, fmax=400) -> float:
        """
        Simple autocorrelation-based F0 estimate. Deliberately dependency-free
        (no librosa/numba) since numba's JIT DLL can be blocked by Windows
        Application Control / AppLocker policies on managed machines. Good
        enough for a relative, per-speaker prosody signal - not lab-grade
        pitch tracking.
        """
        y = block.astype(np.float64)
        y = y - np.mean(y)

        if np.max(np.abs(y)) < 1e-4:  # near-silence, nothing to track
            return 0.0

        sr = SAMPLE_RATE
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag >= len(y):
            return 0.0

        corr = np.correlate(y, y, mode="full")
        corr = corr[len(corr) // 2:]  # keep zero-lag onward

        window = corr[min_lag:max_lag]
        if len(window) == 0 or np.max(window) <= 0:
            return 0.0

        peak_lag = min_lag + int(np.argmax(window))
        if peak_lag == 0:
            return 0.0

        # confidence check: peak should be a reasonably strong fraction of
        # zero-lag energy, otherwise this is probably unvoiced/noise
        if corr[peak_lag] < 0.3 * corr[0]:
            return 0.0

        return float(sr / peak_lag)

    def _maybe_trigger_emotion_classification(self):
        """
        Kicks off a background Ollama call at most once every
        EMOTION_MIN_INTERVAL seconds, using recent transcript text. Runs
        async so a slow local-LLM response never blocks the audio loop -
        self._latest_emotion just lags a few seconds behind live speech,
        which is fine for this kind of signal.
        """
        if not self.use_emotion_model or self._emotion_busy:
            return

        now = time.time()
        if now - self._last_emotion_call_time < EMOTION_MIN_INTERVAL:
            return

        recent_text = " ".join(
            text for ts, text in self._transcript_log
            if now - ts <= EMOTION_LOOKBACK_SECONDS
        ).strip()
        if not recent_text:
            return

        self._last_emotion_call_time = now
        self._emotion_busy = True
        threading.Thread(
            target=self._classify_emotion_ollama, args=(recent_text,), daemon=True
        ).start()

    def _classify_emotion_ollama(self, text: str):
        prompt = (
            "You are classifying the emotional tone of a short speech "
            "transcript from someone practicing public speaking in VR. "
            "Respond with ONLY a JSON object, no other text, in exactly "
            "this form: "
            '{"label": "<one of: calm, neutral, anxious, frustrated, '
            'confident, sad>", "confidence": <number between 0 and 1>}. '
            f'Transcript: "{text}"'
        )
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
                timeout=45,
            )
            resp.raise_for_status()
            parsed = json.loads(resp.json().get("response", "{}"))
            label = parsed.get("label")
            conf = float(parsed.get("confidence", 0.0))
            if label:
                self._latest_emotion = (label, conf)
        except Exception as exc:  # noqa: BLE001
            self.on_status(f"Ollama emotion classification error: {exc}")
        finally:
            self._emotion_busy = False

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.strip(".,!?;:'\"").lower()

    def _tokens_from_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [t for t in (self._normalize_token(w) for w in text.split()) if t]

    def _count_fillers_in_text(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        tokens = self._tokens_from_text(text)
        count = sum(1 for t in tokens if t in FILLER_WORDS)
        for match in FILLER_PATTERN.finditer(text.lower()):
            norm = self._normalize_token(match.group())
            if norm not in FILLER_WORDS:
                count += 1
        return count

    def _recount_speech_quality(self, partial: str = "") -> None:
        """
        Re-scan full session transcript (finalized + live partial) so filler/
        repetition counts update while the user is still speaking — not only
        when Vosk finalizes an utterance (which the small model does rarely).
        """
        combined = f"{self._accumulated_final_text} {partial}".strip()
        tokens = self._tokens_from_text(combined)

        if tokens and self._speech_start_wall is None:
            self._speech_start_wall = time.time()

        self.session_word_count = len(tokens)
        self.session_filler_count = self._count_fillers_in_text(combined)
        self.session_repetition_count = sum(
            1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1]
        )

    def _feed_stt(self, block: np.ndarray) -> dict:
        """
        Feeds one audio block into the persistent Vosk recognizer and
        extracts: any newly-finalized transcript text, the current partial
        (in-progress) hypothesis, running filler-word / repeated-word
        counts, and a words-per-minute estimate from word timestamps.
        Returns a dict with defaults even when STT is disabled, so callers
        don't need to branch on self.use_stt.
        """
        empty = {"partial": "", "final_text": "", "wpm": 0.0}
        if not self.use_stt or self._vosk_rec is None:
            return empty

        pcm16 = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        try:
            if self._vosk_rec.AcceptWaveform(pcm16):
                result = json.loads(self._vosk_rec.Result())
                final_text = (result.get("text") or "").strip()

                if final_text:
                    self._accumulated_final_text = (
                        f"{self._accumulated_final_text} {final_text}".strip()
                    )

                self._recount_speech_quality(partial="")
                return {"partial": "", "final_text": final_text, "wpm": self._current_wpm()}

            partial = json.loads(self._vosk_rec.PartialResult()).get("partial", "") or ""
            self._recount_speech_quality(partial=partial)
            return {"partial": partial, "final_text": "", "wpm": self._current_wpm()}
        except Exception as exc:  # noqa: BLE001
            self.on_status(f"STT error: {exc}")
            return empty

    def _current_wpm(self):
        """
        Words per minute using wall-clock elapsed time since the first
        recognized word. Deliberately NOT using Vosk's word-level "start"/
        "end" timestamps for the elapsed-time denominator - those reset
        relative to each finalized utterance rather than running
        continuously for the whole session, which was previously causing
        near-zero elapsed times and wildly inflated (millions of) wpm.
        """
        if self._speech_start_wall is None or self.session_word_count == 0:
            return 0.0
        elapsed_min = max(1e-6, (time.time() - self._speech_start_wall) / 60.0)
        return float(self.session_word_count / elapsed_min)

    def _vad_stats(self, block: np.ndarray):
        """
        Returns (speech_ratio, pause_count) for a block using a simple
        per-frame energy threshold. Deliberately dependency-free (no
        webrtcvad) - splits the block into ~30ms sub-frames, marks each
        "speech" if its RMS is above a fraction of the loudest frame in
        the block, and counts speech->silence transitions as pauses.
        Cruder than a real VAD (no spectral analysis), but good enough for
        a relative, per-speaker signal, and it never needs a compiler.
        """
        frame_len = VAD_FRAME_SAMPLES
        n_frames = len(block) // frame_len
        if n_frames == 0:
            return 1.0, 0

        frame_rms = np.array([
            np.sqrt(np.mean(block[i * frame_len:(i + 1) * frame_len] ** 2)) + 1e-9
            for i in range(n_frames)
        ])

        loudest = np.max(frame_rms)
        if loudest < 1e-4:  # whole block is near-silent
            return 0.0, 0

        threshold = loudest * 0.2  # frames under 20% of the block's peak = silence
        flags = frame_rms > threshold

        speech_ratio = float(np.mean(flags))

        pause_count = 0
        prev = True
        for f in flags:
            if prev and not f:
                pause_count += 1
            prev = bool(f)

        return speech_ratio, pause_count

    def _compute_anxiety_score(self, pitch_hz, speech_ratio, pause_count):
        """
        Heuristic, not clinically validated. Combines:
          - pitch deviation from the user's calibrated baseline (higher = more tense)
          - drop in speech_ratio relative to baseline (more silence than normal)
          - pause_count above baseline (choppier speech)
        Each term is a z-score against the user's own calibration, so the
        score is relative to THIS speaker's normal voice, not an absolute
        threshold. Squashed into 0..1 with a logistic function.

        Returns None (rather than 0.0) when pitch detection failed for this
        block - that's a missing reading, not evidence of calm speech, so
        the caller should just carry the previous smoothed score forward
        instead of snapping to 0.
        """
        if self._calibrating:
            return 0.0
        if pitch_hz == 0.0:
            return None

        pitch_z = abs(self._pitch_baseline.zscore(pitch_hz))
        rate_z = max(0.0, self._rate_baseline.zscore(speech_ratio) * -1)  # less speech = more anxious
        pause_z = max(0.0, self._pause_baseline.zscore(pause_count))

        raw = 0.5 * pitch_z + 0.3 * rate_z + 0.2 * pause_z
        return float(1 / (1 + np.exp(-1.2 * (raw - 1.0))))  # logistic squash centered ~1 std dev