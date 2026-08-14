"""
integrated_avrte_dashboard.py

Combines all four prototype pieces into one running therapist dashboard:

  - avrte_dashboard.py            -> the Streamlit UI + AdaptiveEngine (Section 5/6)
  - analysis_engine.py            -> real-time voice capture, prosody, STT,
                                      Ollama emotion classification
  - intensity_feedback_loop.py    -> the trained physiological distress
                                      classifier (data -> model -> predict)
  - main_gui.py                   -> NOT used directly; it was only a Tk
                                      reference for what analysis_engine's
                                      output dict looks like. That role is
                                      now played by the Streamlit panels below.

What changed vs. the original avrte_dashboard.py:
  1. The "voice stress" / "speech speed" panel is no longer randomly
     simulated - it is fed by a real, running `AnalysisEngine` (mic capture,
     pitch/VAD, optional Vosk STT, optional Ollama emotion tag). See
     `VoiceBridge`.
  2. `AdaptiveEngine`'s optional `ml_adapter` slot, previously always None,
     can now be filled with a real model trained/loaded via
     intensity_feedback_loop.py's pipeline (`MLBundleAdapter`), so the
     composite distress score gets a genuine ML supplement instead of only
     ever running on rules.
  3. HR / GSR / EEG are still simulated (no physical sensors are wired up
     in this prototype) - but if a training dataset folder is provided, the
     simulation bootstraps from `simulate_real_time_stream()`'s own dataset
     distribution instead of a hand-tuned synthetic ramp.

Run with:
    streamlit run integrated_avrte_dashboard.py

Requires analysis_engine.py and intensity_feedback_loop.py to be importable
(i.e. sitting next to this file, or on PYTHONPATH). Everything each of them
needs (sounddevice, vosk, requests, scikit-learn, joblib, pandas, ...) is
optional at import time - each piece degrades to "off" individually if a
dependency or resource (mic, Ollama, a trained model) isn't available,
exactly like the original two prototypes did on their own.
"""

from __future__ import annotations

import glob
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import mne  # EEG processing (optional, same as original dashboard)
except Exception:
    mne = None

# -- the voice / emotion engine ------------------------------------------
try:
    from analysis_engine import AnalysisEngine
    HAVE_VOICE_ENGINE = True
except Exception as exc:  # missing sounddevice, vosk, etc.
    AnalysisEngine = None
    HAVE_VOICE_ENGINE = False
    _VOICE_IMPORT_ERROR = str(exc)

# -- the physiological intensity-feedback ML pipeline ---------------------
try:
    from intensity_feedback_loop import (
        Config as IntensityConfig,
        load_dataset,
        preprocess,
        train_model,
        save_model,
        load_model,
        simulate_real_time_stream,
        predict_distress,
    )
    HAVE_INTENSITY_MODULE = True
except Exception as exc:
    HAVE_INTENSITY_MODULE = False
    _INTENSITY_IMPORT_ERROR = str(exc)


# =============================================================================
# SECTION 1 — Data model (same shape as avrte_dashboard.py's Section 5)
# =============================================================================

class Recommendation(Enum):
    HOLD = "hold"
    ENCOURAGE = "encourage"  # mild elevation - supportive prompt, no intensity change
    REDUCE_INTENSITY = "reduce_intensity"
    STOP_SESSION = "stop_session"  # only ever suggested, therapist still confirms


@dataclass
class BiosignalSample:
    """One tick of signal data. heart_rate_bpm/gsr/head_movement_deg_s are
    simulated (no sensor wired up); speech_rate_wpm/speech_stress come from
    the real voice engine whenever it has produced a reading."""
    timestamp: float
    heart_rate_bpm: float
    gsr_microsiemens: float
    eeg_uv: Optional[np.ndarray] = None
    speech_rate_wpm: Optional[float] = None
    speech_stress: Optional[float] = None  # 0..1, from the voice anxiety score
    head_movement_deg_s: Optional[float] = None


@dataclass
class DistressReading:
    timestamp: float
    score: float
    recommendation: Recommendation
    contributing_signals: dict = field(default_factory=dict)
    explanation: str = ""


@dataclass
class ClinicianThresholds:
    """
    Threshold values below are drawn from Reference_Values_from_PsychTeam.pdf
    where the PDF gives a usable cutoff. Per that document's own caveat, most
    physiological measures are baseline-relative (z-score or % change from a
    personal resting baseline), not universal cutoffs - so AdaptiveEngine
    runs a short calibration window per session (see
    `baseline_calibration_seconds`) before scoring starts, and most rules
    below compare against that captured personal baseline rather than a
    fixed population number.

    Signals the PDF covers but that aren't wired in (no reference value
    found in the PDF, and/or no signal currently produced by this
    dashboard): eye tracking / pupillometry, blood pressure, respiratory
    rate / tidal volume, HRV (RMSSD/SDNN/LF-HF), voice f0 in Hz, EEG
    alpha/beta ratio (EEG is modeled in BiosignalSample but never actually
    populated by the feeder loop today - simulated in shape only).
    """

    # -- baseline calibration --
    baseline_calibration_seconds: float = 20.0

    # -- heart rate (PDF: HRz >= +1.0 sustained >=10s; also HR >=120bpm cue) --
    hr_z_threshold: float = 1.0
    hr_z_sustained_seconds: float = 10.0
    hr_absolute_high_bpm: float = 120.0

    # -- GSR / skin conductance level (PDF: sustained SCL ~2x baseline) --
    gsr_scl_baseline_multiplier: float = 2.0

    # -- speech rate (PDF: +8% vs baseline = stress cue) --
    speech_rate_rise_pct: float = 8.0
    speech_stress_high: float = 0.7  # voice engine's own 0-1 anxiety score

    # -- head movement (PDF: 5s-window z >= +1.0 sustained >=5s) --
    head_movement_z_threshold: float = 1.0
    head_movement_z_sustained_seconds: float = 5.0

    # -- EEG (kept for when a real EEG source is wired up; currently unused
    #    since sample.eeg_uv is never populated) --
    eeg_alpha_suppression_pct: float = 30.0

    # -- composite score bands --
    score_encourage_threshold: float = 35.0
    score_reduce_threshold: float = 60.0
    score_stop_threshold: float = 90.0


# =============================================================================
# SECTION 2 — ML adapter around intensity_feedback_loop.py's trained model
# =============================================================================

class MLBundleAdapter:
    """
    Wraps a SavedModelBundle (produced by intensity_feedback_loop.py's
    save_model/load_model) so AdaptiveEngine can call one uniform method,
    predict_distress_probability(signals: dict) -> float in [0, 1],
    regardless of exactly which physiological columns the trained model
    used. Missing columns are simply left out of the dict - the bundle's
    own SimpleImputer (fit during training) fills them with the training
    mean, the same way intensity_feedback_loop.predict_distress() already
    handles a partial sample.
    """

    def __init__(self, bundle):
        self.bundle = bundle

    def predict_distress_probability(self, signals: dict) -> float:
        prediction = predict_distress(self.bundle, signals)
        prob = prediction.get("distress_probability")
        return float(prob) if prob is not None else 0.0


# =============================================================================
# SECTION 3 — Adaptive engine (rules + optional ML supplement)
# =============================================================================

class AdaptiveEngine:
    """
    Same fusion logic as avrte_dashboard.py's AdaptiveEngine, except the ML
    supplement now goes through an MLBundleAdapter driven by a dict of
    named physiological/voice signals (`ml_signals`) rather than a fixed
    5-column numpy array, so it lines up with the much larger, named
    feature set intensity_feedback_loop.py's model was actually trained on.
    """

    def __init__(
        self,
        thresholds: ClinicianThresholds | None = None,
        resting_hr_baseline: float = 70.0,  # unused now - superseded by the per-session calibrated baseline below
        eeg_alpha_baseline: float = 1.0,
        ml_adapter: Optional[MLBundleAdapter] = None,
        on_reading: Optional[Callable[[DistressReading], None]] = None,
        queue_maxsize: int = 500,
    ):
        self.thresholds = thresholds or ClinicianThresholds()
        self.resting_hr_baseline = resting_hr_baseline
        self.eeg_alpha_baseline = eeg_alpha_baseline
        self.ml_adapter = ml_adapter
        self.on_reading = on_reading

        self._queue: "queue.Queue[tuple[BiosignalSample, dict]]" = queue.Queue(maxsize=queue_maxsize)
        self._hr_window: list[tuple[float, float]] = []
        self._head_movement_window: list[tuple[float, float]] = []
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._last_reading: Optional[DistressReading] = None

        # -- personal baseline calibration (see ClinicianThresholds docstring) --
        self._calibration_start_ts: Optional[float] = None
        self._calibration_buffers: dict[str, list[float]] = {
            "hr": [], "gsr": [], "speech_rate": [], "head_movement": [],
        }
        self._baseline: Optional[dict[str, tuple[float, float]]] = None  # name -> (mean, sd)

    # -- ingestion -----------------------------------------------------
    def ingest(self, sample: BiosignalSample, ml_signals: Optional[dict] = None) -> None:
        item = (sample, ml_signals or {})
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            _ = self._queue.get_nowait()
            self._queue.put_nowait(item)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        if self._worker:
            self._worker.join(timeout=2.0)

    def _run_loop(self) -> None:
        while self._running:
            try:
                sample, ml_signals = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            reading = self._process(sample, ml_signals)
            self._last_reading = reading
            if self.on_reading:
                self.on_reading(reading)

    # -- baseline calibration ------------------------------------------
    def _calibrating_reading(self, sample: BiosignalSample, elapsed: float) -> DistressReading:
        remaining = max(0.0, self.thresholds.baseline_calibration_seconds - elapsed)
        return DistressReading(
            timestamp=sample.timestamp,
            score=0.0,
            recommendation=Recommendation.HOLD,
            contributing_signals={"calibrating": True, "calibration_seconds_remaining": round(remaining, 1)},
            explanation=(
                f"Calibrating personal baseline (HR/GSR/speech rate/head movement) - "
                f"{remaining:.0f}s remaining. Ask the patient to sit normally; scoring "
                f"starts once this completes."
            ),
        )

    def _update_calibration(self, sample: BiosignalSample) -> Optional[DistressReading]:
        """Returns a 'still calibrating' reading while baseline is being
        captured, or None once calibration is complete and normal scoring
        should proceed. Per the reference-values PDF, most of these signals
        are meaningfully interpreted only as a change from a personal
        baseline, not against a fixed population number."""
        if self._baseline is not None:
            return None

        if self._calibration_start_ts is None:
            self._calibration_start_ts = sample.timestamp

        self._calibration_buffers["hr"].append(sample.heart_rate_bpm)
        self._calibration_buffers["gsr"].append(sample.gsr_microsiemens)
        if sample.speech_rate_wpm is not None:
            self._calibration_buffers["speech_rate"].append(sample.speech_rate_wpm)
        if sample.head_movement_deg_s is not None:
            self._calibration_buffers["head_movement"].append(sample.head_movement_deg_s)

        elapsed = sample.timestamp - self._calibration_start_ts
        if elapsed < self.thresholds.baseline_calibration_seconds:
            return self._calibrating_reading(sample, elapsed)

        self._baseline = {}
        for name, values in self._calibration_buffers.items():
            if not values:
                continue
            mean = float(np.mean(values))
            sd = float(np.std(values)) if len(values) > 1 else 0.0
            self._baseline[name] = (mean, sd if sd > 1e-6 else max(mean * 0.05, 1e-6))
        return None

    @staticmethod
    def _z_score(value: float, baseline: Optional[tuple[float, float]]) -> Optional[float]:
        if baseline is None:
            return None
        mean, sd = baseline
        if sd <= 0:
            return None
        return (value - mean) / sd

    # -- fusion logic ------------------------------------------------------
    def _process(self, sample: BiosignalSample, ml_signals: dict) -> DistressReading:
        calibrating = self._update_calibration(sample)
        if calibrating is not None:
            return calibrating

        signals: dict = {}
        baseline = self._baseline or {}

        # ---- heart rate: personal-baseline z-score sustained, plus the
        # near-universal >=120bpm cue from the reference PDF ----
        hr_z = self._z_score(sample.heart_rate_bpm, baseline.get("hr"))
        self._hr_window.append((sample.timestamp, hr_z))
        cutoff = sample.timestamp - self.thresholds.hr_z_sustained_seconds
        self._hr_window = [(t, z) for t, z in self._hr_window if t >= cutoff]
        hr_window_spans_full_duration = (
            len(self._hr_window) > 0
            and (sample.timestamp - self._hr_window[0][0]) >= self.thresholds.hr_z_sustained_seconds
        )
        hr_z_sustained_high = (
            hr_window_spans_full_duration
            and all(z is not None and z >= self.thresholds.hr_z_threshold for _, z in self._hr_window)
        )
        hr_absolute_high = sample.heart_rate_bpm >= self.thresholds.hr_absolute_high_bpm
        signals["hr_bpm"] = sample.heart_rate_bpm
        signals["hr_z"] = round(hr_z, 2) if hr_z is not None else None
        signals["hr_z_sustained_high"] = hr_z_sustained_high
        signals["hr_absolute_high"] = hr_absolute_high

        # ---- GSR: sustained SCL >= 2x personal baseline (PDF) ----
        gsr_mean, _ = baseline.get("gsr", (None, None))
        gsr_scl_elevated = (
            gsr_mean is not None
            and gsr_mean > 0
            and sample.gsr_microsiemens >= gsr_mean * self.thresholds.gsr_scl_baseline_multiplier
        )
        signals["gsr_uS"] = sample.gsr_microsiemens
        signals["gsr_scl_elevated"] = gsr_scl_elevated

        # ---- EEG: kept for when a real EEG source is wired up; sample.eeg_uv
        # is not currently populated by the feeder loop, so this stays inert
        # in today's simulated/voice-only setup ----
        eeg_suppressed = False
        if sample.eeg_uv is not None and mne is not None:
            alpha_power = self._alpha_band_power(sample.eeg_uv)
            drop_pct = 100.0 * (self.eeg_alpha_baseline - alpha_power) / self.eeg_alpha_baseline
            eeg_suppressed = drop_pct >= self.thresholds.eeg_alpha_suppression_pct
            signals["eeg_alpha_drop_pct"] = round(drop_pct, 1)
        signals["eeg_suppressed"] = eeg_suppressed

        # ---- voice: existing 0-1 anxiety score, plus speech rate +8% vs
        # baseline (PDF) - both from the real voice engine when available ----
        speech_high_stress = (
            sample.speech_stress is not None
            and sample.speech_stress >= self.thresholds.speech_stress_high
        )
        signals["speech_stress"] = sample.speech_stress
        signals["speech_high_stress"] = speech_high_stress

        speech_rate_mean, _ = baseline.get("speech_rate", (None, None))
        speech_rate_elevated = (
            speech_rate_mean is not None
            and speech_rate_mean > 0
            and sample.speech_rate_wpm is not None
            and sample.speech_rate_wpm >= speech_rate_mean * (1 + self.thresholds.speech_rate_rise_pct / 100.0)
        )
        signals["speech_rate_wpm"] = sample.speech_rate_wpm
        signals["speech_rate_elevated"] = speech_rate_elevated

        # ---- head movement: baseline z-score sustained >=5s (PDF) ----
        head_z = None
        head_z_sustained_high = False
        if sample.head_movement_deg_s is not None:
            head_z = self._z_score(sample.head_movement_deg_s, baseline.get("head_movement"))
            self._head_movement_window.append((sample.timestamp, head_z))
            hm_cutoff = sample.timestamp - self.thresholds.head_movement_z_sustained_seconds
            self._head_movement_window = [
                (t, z) for t, z in self._head_movement_window if t >= hm_cutoff
            ]
            hm_window_spans_full_duration = (
                len(self._head_movement_window) > 0
                and (sample.timestamp - self._head_movement_window[0][0])
                >= self.thresholds.head_movement_z_sustained_seconds
            )
            head_z_sustained_high = (
                hm_window_spans_full_duration
                and all(
                    z is not None and z >= self.thresholds.head_movement_z_threshold
                    for _, z in self._head_movement_window
                )
            )
        signals["head_movement_deg_s"] = sample.head_movement_deg_s
        signals["head_movement_z"] = round(head_z, 2) if head_z is not None else None
        signals["head_movement_z_sustained_high"] = head_z_sustained_high

        # ---- rule-based score (explainable, always computed) ----
        rule_score = 0.0
        rule_score += 30.0 if hr_z_sustained_high else 0.0
        rule_score += 15.0 if hr_absolute_high else 0.0
        rule_score += 20.0 if gsr_scl_elevated else 0.0
        rule_score += 20.0 if eeg_suppressed else 0.0
        rule_score += 15.0 if speech_high_stress else 0.0
        rule_score += 10.0 if speech_rate_elevated else 0.0
        rule_score += 15.0 if head_z_sustained_high else 0.0
        rule_score = min(rule_score, 100.0)

        # ---- optional ML supplement, from intensity_feedback_loop.py ----
        final_score = rule_score
        if self.ml_adapter is not None:
            try:
                ml_prob = self.ml_adapter.predict_distress_probability(ml_signals) * 100.0
                final_score = 0.6 * rule_score + 0.4 * ml_prob
                signals["ml_score"] = round(ml_prob, 1)
            except Exception as exc:  # noqa: BLE001 - never let ML errors break the loop
                signals["ml_error"] = str(exc)

        recommendation, explanation = self._recommend(final_score, signals)

        return DistressReading(
            timestamp=sample.timestamp,
            score=round(final_score, 1),
            recommendation=recommendation,
            contributing_signals=signals,
            explanation=explanation,
        )

    def _recommend(self, score: float, signals: dict) -> tuple[Recommendation, str]:
        if score >= self.thresholds.score_stop_threshold:
            return (
                Recommendation.STOP_SESSION,
                f"Distress score {score:.0f}/100 exceeds the stop threshold "
                f"({self.thresholds.score_stop_threshold:.0f}). Consider ending the session.",
            )
        if score >= self.thresholds.score_reduce_threshold:
            reasons = [k for k, v in signals.items() if v is True]
            reason_txt = ", ".join(reasons) if reasons else "elevated composite score"
            return (
                Recommendation.REDUCE_INTENSITY,
                f"Distress score {score:.0f}/100 exceeds the reduce threshold "
                f"({self.thresholds.score_reduce_threshold:.0f}). Signals: {reason_txt}.",
            )
        if score >= self.thresholds.score_encourage_threshold:
            reasons = [k for k, v in signals.items() if v is True]
            reason_txt = ", ".join(reasons) if reasons else "mild composite elevation"
            return (
                Recommendation.ENCOURAGE,
                f"Distress score {score:.0f}/100 shows early signs of arousal "
                f"({self.thresholds.score_encourage_threshold:.0f}-{self.thresholds.score_reduce_threshold:.0f} "
                f"band). Signals: {reason_txt}. Consider a brief supportive/grounding "
                f"prompt - no need to change intensity yet.",
            )
        return Recommendation.HOLD, f"Distress score {score:.0f}/100 — within normal range."

    @staticmethod
    def _alpha_band_power(eeg_uv: np.ndarray, sfreq: float = 256.0) -> float:
        if mne is None:
            return 1.0
        freqs, psd = mne.time_frequency.psd_array_welch(
            eeg_uv[np.newaxis, :], sfreq=sfreq, fmin=1, fmax=40, verbose=False
        )
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        return float(psd[0, alpha_mask].mean())

    def last_reading(self) -> Optional[DistressReading]:
        return self._last_reading


# =============================================================================
# SECTION 4 — UnrealBridge (unchanged from avrte_dashboard.py)
# =============================================================================

class UnrealBridge:
    INTENSITY_BANDS = ["very_low", "low", "medium", "high", "very_high"]

    def __init__(self, send_fn: Callable[[dict], None]):
        self._send_fn = send_fn
        self.current_band_index = 2  # start at "medium"

    def apply_intensity_step(self, direction: str) -> str:
        if direction == "down":
            self.current_band_index = max(0, self.current_band_index - 1)
        elif direction == "up":
            self.current_band_index = min(len(self.INTENSITY_BANDS) - 1, self.current_band_index + 1)
        else:
            raise ValueError("direction must be 'down' or 'up'")
        band = self.INTENSITY_BANDS[self.current_band_index]
        self._send_fn({"type": "set_intensity", "band": band})
        return band

    def stop_session(self) -> None:
        self._send_fn({"type": "stop_session"})

    def send_encouragement(self, message: str) -> None:
        """Sends a supportive/grounding prompt cue to the VR side, without
        touching the intensity band. Used for the ENCOURAGE recommendation
        tier (mild elevation - not severe enough to reduce intensity)."""
        self._send_fn({"type": "show_encouragement", "message": message})


# =============================================================================
# SECTION 5 — VoiceBridge: wraps analysis_engine.AnalysisEngine
# =============================================================================

class VoiceBridge:
    """
    Runs the real, headless AnalysisEngine from analysis_engine.py (mic
    capture, pitch/VAD anxiety score, optional Vosk STT, optional Ollama
    emotion tag) and exposes its latest result as a thread-safe snapshot.
    This plays the role main_gui.py played for the standalone Tk prototype,
    except the "GUI" pulling from it is now the Streamlit dashboard below.
    """

    def __init__(self, use_emotion_model: bool, use_stt: bool, on_status: Callable[[str], None]):
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self.engine = AnalysisEngine(
            on_result=self._on_result,
            on_status=on_status,
            use_emotion_model=use_emotion_model,
            use_stt=use_stt,
        )

    def _on_result(self, result: dict) -> None:
        with self._lock:
            self._latest = result

    def start(self) -> None:
        self.engine.start()

    def stop(self) -> None:
        self.engine.stop()

    def latest(self) -> Optional[dict]:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None


# =============================================================================
# SECTION 6 — Shared state between background threads and the Streamlit script
# =============================================================================

class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.history: deque = deque(maxlen=600)
        self.current_reading: DistressReading | None = None
        self.log: list[str] = []
        self.tick: int = 0

    def on_reading(self, sample: BiosignalSample, reading: DistressReading, extra: dict | None = None) -> None:
        with self.lock:
            self.current_reading = reading
            entry = {
                "t": self.tick,
                "hr": sample.heart_rate_bpm,
                "gsr": sample.gsr_microsiemens,
                "stress": sample.speech_stress,
                "score": reading.score,
                "speech_rate": sample.speech_rate_wpm,
            }
            if extra:
                entry.update(extra)
            self.history.append(entry)
            self.tick += 1

    def add_log(self, message: str) -> None:
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')} {message}")

    def snapshot(self):
        with self.lock:
            return (self.current_reading, list(self.history), list(self.log))


# =============================================================================
# SECTION 7 — Physiological simulation (fallback ramp, or dataset-bootstrapped)
# =============================================================================

def _simulate_physio_ramp(tick: int) -> dict:
    """Fallback used when no trained/reference dataset is loaded - a
    hand-tuned synthetic ramp, identical in spirit to the original
    avrte_dashboard.py's _simulate_tick()."""
    rng = np.random.default_rng(tick)
    drift = min(tick / 40.0, 1.0)
    hr = 72 + drift * 30 + rng.normal(0, 2)
    gsr = 0.5 + drift * 2.0 + rng.normal(0, 0.1)
    return {
        "heart_rate_bpm": float(hr),
        "gsr_us": float(max(0.0, gsr)),
    }


def _simulate_extra_signals(tick: int) -> dict:
    rng = np.random.default_rng(tick + 9000)
    drift = min(tick / 40.0, 1.0)
    head_movement = 5 + drift * 25 + rng.normal(0, 3)
    eye_stability = 0.9 - drift * 0.5 + rng.normal(0, 0.05)
    return {
        "head_movement_deg_s": float(max(0.0, head_movement)),
        "eye_tracking_stability": float(np.clip(eye_stability, 0.0, 1.0)),
    }


def _feeder_loop(
    engine: AdaptiveEngine,
    stop_event: threading.Event,
    voice_bridge: Optional[VoiceBridge],
    ml_bundle,
    reference_df,
) -> None:
    """
    Runs in its own thread, once per second:
      - HR/GSR come from the dataset-bootstrapped simulator if a reference
        dataset was loaded, otherwise from the synthetic ramp fallback.
      - speech_stress / speech_rate_wpm come from the REAL voice engine
        whenever it has produced a reading (this is the integration point
        that replaces the old simulated speech numbers).
      - the full physiological signal dict is also handed to AdaptiveEngine
        as `ml_signals`, so the trained intensity_feedback_loop.py model
        (if loaded) can score it.
    """
    tick = 0
    while not stop_event.is_set():
        if reference_df is not None and ml_bundle is not None:
            physio = next(simulate_real_time_stream(
                reference_df, ml_bundle.feature_columns, n_samples=1, random_state=tick,
            ))
        else:
            physio = _simulate_physio_ramp(tick)

        hr = physio.get("heart_rate_bpm")
        if hr is None or pd.isna(hr):
            hr = 72 + min(tick / 40.0, 1.0) * 30
        gsr = physio.get("gsr_us")
        if gsr is None or pd.isna(gsr):
            gsr = physio.get("skin_conductance_level_us")
        if gsr is None or pd.isna(gsr):
            gsr = 0.5 + min(tick / 40.0, 1.0) * 2.0

        voice = voice_bridge.latest() if voice_bridge is not None else None
        if voice is not None:
            speech_stress = float(voice["anxiety_score"])
            speech_rate = float(voice["words_per_minute"])
        else:
            drift = min(tick / 40.0, 1.0)
            speech_stress = float(np.clip(0.15 + drift * 0.7, 0, 1))
            speech_rate = float(max(0.0, 120 - drift * 40))

        extra = _simulate_extra_signals(tick)

        sample = BiosignalSample(
            timestamp=time.time(),
            heart_rate_bpm=float(hr),
            gsr_microsiemens=float(max(0.0, gsr)),
            speech_rate_wpm=speech_rate,
            speech_stress=speech_stress,
            head_movement_deg_s=extra.get("head_movement_deg_s"),
        )

        if voice is not None:
            extra["emotion_label"] = voice.get("emotion_label")
            extra["emotion_conf"] = voice.get("emotion_conf")
            extra["filler_count"] = voice.get("filler_count_session")
            extra["repetition_count"] = voice.get("repetition_count_session")
            extra["transcript_delta"] = voice.get("transcript_final_delta")
            extra["transcript_partial"] = voice.get("transcript_partial")

        ml_signals = dict(physio)
        ml_signals["heart_rate_bpm"] = hr
        ml_signals["gsr_us"] = gsr
        ml_signals["voice_stress_score"] = speech_stress
        ml_signals["speech_rate_wpm"] = speech_rate
        ml_signals["anxiety_score"] = speech_stress

        engine.ingest(sample, ml_signals)
        engine._pending_extra = extra  # picked up by the wrapper in _init_state
        tick += 1
        stop_event.wait(1.0)


# =============================================================================
# SECTION 8 — Streamlit app
# =============================================================================

st.set_page_config(page_title="AVRTE Therapist Dashboard (Integrated)", layout="wide")


def _try_load_or_train_model(dataset_folder: str, model_path: str, status_cb):
    """Loads a model.pkl if present; otherwise, if a dataset folder was
    given, trains one from scratch via intensity_feedback_loop.py's own
    pipeline and saves it. Returns (bundle_or_None, reference_df_or_None)."""
    if not HAVE_INTENSITY_MODULE:
        status_cb(f"intensity_feedback_loop.py not importable: {_INTENSITY_IMPORT_ERROR}")
        return None, None

    cfg = IntensityConfig()
    if model_path:
        cfg.model_path = model_path

    if os.path.isfile(cfg.model_path):
        try:
            bundle = load_model(cfg.model_path)
            status_cb(f"Loaded existing distress model from {cfg.model_path}.")
            ref_df = None
            if dataset_folder and os.path.isdir(dataset_folder):
                ref_df = load_dataset(dataset_folder)
                status_cb("Loaded reference dataset for realistic HR/GSR simulation.")
            return bundle, ref_df
        except Exception as exc:  # noqa: BLE001
            status_cb(f"Could not load {cfg.model_path}: {exc}")

    if dataset_folder and os.path.isdir(dataset_folder):
        try:
            cfg.data_folder = dataset_folder
            status_cb("Training distress model from dataset (this can take a moment)...")
            df = load_dataset(cfg.data_folder)
            prep = preprocess(df, cfg)
            model, X_test, y_test, metrics = train_model(prep, cfg)
            save_model(model, prep, cfg)
            bundle = load_model(cfg.model_path)
            status_cb(
                f"Trained new distress model "
                f"(accuracy={metrics['accuracy']:.2f}, f1={metrics['f1_score']:.2f}) "
                f"-> {cfg.model_path}."
            )
            return bundle, df
        except Exception as exc:  # noqa: BLE001
            status_cb(f"Training failed: {exc}")
            return None, None

    status_cb("No existing model and no dataset folder given - running on rules only.")
    return None, None


def _setup_screen() -> None:
    st.title("AVRTE — Integrated Therapist Dashboard")
    st.caption("Configure the voice engine and distress model, then start the session.")

    with st.form("session_setup"):
        st.subheader("Voice / emotion engine")
        if not HAVE_VOICE_ENGINE:
            st.warning(f"analysis_engine.py not importable ({_VOICE_IMPORT_ERROR}). "
                       f"The session will fall back to simulated speech signals.")
        use_voice = st.checkbox(
            "Use real microphone voice analysis", value=HAVE_VOICE_ENGINE, disabled=not HAVE_VOICE_ENGINE
        )
        use_stt = st.checkbox("Use speech-to-text (Vosk)", value=True)
        use_emotion = st.checkbox("Use emotion classification (Ollama)", value=True)

        st.subheader("Physiological distress model (intensity_feedback_loop.py)")
        dataset_folder = st.text_input(
            "Dataset folder (optional - .xlsx/.xls/.csv files with a "
            "ground_truth_label column)", value="",
        )
        model_path = st.text_input("Model file path", value="model.pkl")

        submitted = st.form_submit_button("Start session", type="primary")

    if submitted:
        st.session_state.cfg_use_voice = use_voice and HAVE_VOICE_ENGINE
        st.session_state.cfg_use_stt = use_stt
        st.session_state.cfg_use_emotion = use_emotion
        st.session_state.cfg_dataset_folder = dataset_folder.strip()
        st.session_state.cfg_model_path = model_path.strip() or "model.pkl"
        st.session_state.session_started = True
        st.rerun()


def _init_state() -> None:
    if "shared" not in st.session_state:
        st.session_state.shared = SharedState()
    # Captured as a plain local so background threads (VoiceBridge's
    # AnalysisEngine thread, AdaptiveEngine's own worker thread) can call
    # into it via closures below. st.session_state itself is only safe to
    # read/write from Streamlit's main script thread - background threads
    # that touch it raise AttributeError/KeyError, which is why status
    # messages and readings weren't showing up before.
    shared: SharedState = st.session_state.shared

    if "bridge" not in st.session_state:
        st.session_state.bridge = UnrealBridge(
            send_fn=lambda msg: shared.add_log(f"-> Unreal: {msg}")
        )

    if "ml_bundle" not in st.session_state:
        bundle, ref_df = _try_load_or_train_model(
            st.session_state.cfg_dataset_folder,
            st.session_state.cfg_model_path,
            status_cb=shared.add_log,
        )
        st.session_state.ml_bundle = bundle
        st.session_state.reference_df = ref_df

    if "voice_bridge" not in st.session_state:
        if st.session_state.cfg_use_voice and HAVE_VOICE_ENGINE:
            vb = VoiceBridge(
                use_emotion_model=st.session_state.cfg_use_emotion,
                use_stt=st.session_state.cfg_use_stt,
                on_status=lambda msg: shared.add_log(f"[voice] {msg}"),
            )
            vb.start()
            st.session_state.voice_bridge = vb
        else:
            st.session_state.voice_bridge = None

    if "engine" not in st.session_state:
        ml_adapter = MLBundleAdapter(st.session_state.ml_bundle) if st.session_state.ml_bundle else None

        _last_extra_box: dict = {"extra": None}

        def _on_reading(reading: DistressReading) -> None:
            # sample fields we display live are already folded into
            # `extra`/history via the feeder loop's ml_signals+extra; we
            # only need the reading itself plus whatever the feeder loop
            # stashed most recently on the engine object.
            # NOTE: this callback runs on AdaptiveEngine's own background
            # thread, so - same reasoning as above - it must not touch
            # st.session_state. `engine` is captured directly from this
            # closure (Python resolves it by name when the callback
            # actually runs, after `engine` below has been assigned).
            extra = getattr(engine, "_pending_extra", None)
            sample = getattr(engine, "_pending_sample", None)
            if sample is not None:
                shared.on_reading(sample, reading, extra)

        engine = AdaptiveEngine(ml_adapter=ml_adapter, on_reading=_on_reading)

        original_ingest = engine.ingest

        def _ingest_and_remember(sample: BiosignalSample, ml_signals: Optional[dict] = None) -> None:
            engine._pending_sample = sample
            original_ingest(sample, ml_signals)

        engine.ingest = _ingest_and_remember  # type: ignore[method-assign]
        engine.start()
        st.session_state.engine = engine

    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if "feeder_thread" not in st.session_state:
        st.session_state.feeder_thread = None

    if "session_active" not in st.session_state:
        st.session_state.session_active = True

    if "dismissed_until_score" not in st.session_state:
        st.session_state.dismissed_until_score = None

    feeder = st.session_state.feeder_thread
    if st.session_state.session_active and (feeder is None or not feeder.is_alive()):
        st.session_state.stop_event.clear()
        thread = threading.Thread(
            target=_feeder_loop,
            args=(
                st.session_state.engine,
                st.session_state.stop_event,
                st.session_state.voice_bridge,
                st.session_state.ml_bundle,
                st.session_state.reference_df,
            ),
            daemon=True,
        )
        thread.start()
        st.session_state.feeder_thread = thread


def _apply_theme(theme: str) -> None:
    if theme == "dark":
        bg, text, panel, border, accent = "#0e1117", "#fafafa", "#1c1f26", "#31333f", "#4da3ff"
    else:
        bg, text, panel, border, accent = "#ffffff", "#111111", "#f5f6f8", "#d8dbe0", "#0068c9"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        [data-testid="stHeader"] {{ background-color: {bg}; }}
        [data-testid="stSidebar"], [data-testid="stMetric"], .stAlert, [data-testid="stExpander"] {{
            background-color: {panel}; border: 1px solid {border}; border-radius: 8px;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {text}; }}
        .stButton > button {{
            background-color: {panel} !important; color: {text} !important;
            border: 1px solid {accent} !important;
        }}
        .stButton > button:hover {{
            background-color: {accent} !important; color: #ffffff !important; border-color: {accent} !important;
        }}
        .stButton > button p {{ color: inherit !important; }}
        [data-stale="true"], [data-stale="true"] * {{ opacity: 1 !important; transition: none !important; }}
        h1, h2, h3, p, span, label, .stCaption, .stMarkdown {{ color: {text} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_setup_or_dashboard() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    _apply_theme(st.session_state.theme)

    if not st.session_state.get("session_started"):
        _setup_screen()
        return

    _init_state()

    title_col, toggle_col = st.columns([5, 1])
    with title_col:
        st.title("AVRTE — Therapist Dashboard (Integrated)")
        voice_status = "live mic" if st.session_state.voice_bridge else "simulated"
        model_status = "trained model + rules" if st.session_state.ml_bundle else "rules only"
        st.caption(f"Speech signal: {voice_status}  •  Distress scoring: {model_status}")
    with toggle_col:
        st.write("")
        toggle_label = "☀️ Light mode" if st.session_state.theme == "dark" else "🌙 Dark mode"
        if st.button(toggle_label, use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    @st.fragment(run_every=1 if st.session_state.session_active else None)
    def render_dashboard() -> None:
        reading, history, log = st.session_state.shared.snapshot()

        col_mirror, col_control = st.columns([2, 1])

        with col_mirror:
            st.subheader("Live patient view (mirror)")
            st.info(
                "Video mirror placeholder — wire this to the local VR streaming layer's "
                "preview output (same feed sent to the Quest, mirrored to this browser)."
            )
            st.image(
                np.zeros((240, 426, 3), dtype=np.uint8),
                caption="Awaiting mirrored stream",
                use_container_width=True,
            )

            st.subheader("Current scenario")
            band = UnrealBridge.INTENSITY_BANDS[st.session_state.bridge.current_band_index]
            st.metric("Intensity band", band.replace("_", " ").title())

        st.subheader("Additional behavioral signals")
        latest = history[-1] if history else {}
        voice_stress = reading.contributing_signals.get("speech_stress") if reading else None
        speech_rate = latest.get("speech_rate")
        head_movement = latest.get("head_movement_deg_s")
        eye_stability = latest.get("eye_tracking_stability")

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Voice stress", f"{voice_stress * 100:.0f}%" if voice_stress is not None else "—")
        with b2:
            st.metric("Speech speed", f"{speech_rate:.0f} wpm" if speech_rate is not None else "—")
        with b3:
            st.metric("Head movement", f"{head_movement:.0f}°/s" if head_movement is not None else "—")
        with b4:
            st.metric(
                "Eye tracking",
                f"{eye_stability * 100:.0f}% stable" if eye_stability is not None else "—",
            )

        if st.session_state.voice_bridge is not None:
            st.subheader("Live transcript & speech quality")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                st.metric("Emotion tag", f"{latest.get('emotion_label') or '—'}")
            with t2:
                conf = latest.get("emotion_conf")
                st.metric("Emotion confidence", f"{conf * 100:.0f}%" if conf else "—")
            with t3:
                st.metric("Filler words", str(latest.get("filler_count") or 0))
            with t4:
                st.metric("Repeated words", str(latest.get("repetition_count") or 0))
            transcript = latest.get("transcript_delta") or latest.get("transcript_partial") or ""
            st.text_area("Recent transcript", transcript, height=70, disabled=True)

        with col_control:
            st.subheader("Distress reading")
            if reading and reading.contributing_signals.get("calibrating"):
                remaining = reading.contributing_signals.get("calibration_seconds_remaining", 0)
                st.info(reading.explanation)
                st.progress(
                    max(0.0, min(1.0, 1 - remaining / max(st.session_state.engine.thresholds.baseline_calibration_seconds, 1)))
                )
            elif reading:
                st.metric("Composite score", f"{reading.score:.0f} / 100")

                if reading.recommendation == Recommendation.HOLD:
                    st.success(reading.explanation)
                elif reading.recommendation == Recommendation.ENCOURAGE:
                    st.info(f"💛 {reading.explanation}")
                    if st.button("Send encouragement prompt to headset", use_container_width=True):
                        msg = "Gentle reminder: breathe, you're safe here — take your time."
                        st.session_state.bridge.send_encouragement(msg)
                        st.session_state.shared.add_log(f"Therapist sent encouragement prompt: \"{msg}\"")
                        st.rerun(scope="fragment")
                elif reading.recommendation == Recommendation.REDUCE_INTENSITY:
                    already_dismissed = (
                        st.session_state.dismissed_until_score is not None
                        and reading.score <= st.session_state.dismissed_until_score
                    )
                    if not already_dismissed:
                        st.warning(f"Recommendation: reduce intensity.\n\n{reading.explanation}")
                        c1, c2 = st.columns(2)
                        if c1.button("Apply — lower intensity", type="primary", use_container_width=True):
                            new_band = st.session_state.bridge.apply_intensity_step("down")
                            st.session_state.shared.add_log(f"Therapist applied: intensity -> {new_band}")
                            st.rerun(scope="fragment")
                        if c2.button("Dismiss for now", use_container_width=True):
                            st.session_state.dismissed_until_score = reading.score - 5
                            st.rerun(scope="fragment")
                    else:
                        st.info("Recommendation dismissed. Will re-alert if distress rises further.")
                elif reading.recommendation == Recommendation.STOP_SESSION:
                    st.error(f"Recommendation: consider stopping the session.\n\n{reading.explanation}")

                with st.expander("Why this recommendation (live signals)"):
                    st.json(reading.contributing_signals)
            else:
                st.write("No reading yet.")

            st.divider()
            st.subheader("Manual controls")
            m1, m2 = st.columns(2)
            if m1.button("⬇ Lower intensity", use_container_width=True):
                new_band = st.session_state.bridge.apply_intensity_step("down")
                st.session_state.shared.add_log(f"Therapist manually lowered -> {new_band}")
                st.rerun(scope="fragment")
            if m2.button("⬆ Raise intensity", use_container_width=True):
                new_band = st.session_state.bridge.apply_intensity_step("up")
                st.session_state.shared.add_log(f"Therapist manually raised -> {new_band}")
                st.rerun(scope="fragment")

            if st.button("⏹ Stop session immediately", type="secondary", use_container_width=True):
                st.session_state.bridge.stop_session()
                st.session_state.session_active = False
                st.session_state.stop_event.set()
                st.session_state.engine.stop()
                if st.session_state.voice_bridge is not None:
                    st.session_state.voice_bridge.stop()
                st.session_state.shared.add_log("Therapist stopped the session.")
                st.rerun()

            st.caption(
                "The patient's on-headset safe word always works independently of this "
                "dashboard, the workstation, and the network."
            )

        st.subheader("Live signal trends")
        if history:
            df = pd.DataFrame(history).set_index("t")
            trend_col1, trend_col2 = st.columns(2)
            with trend_col1:
                st.line_chart(df[["hr"]], height=200)
                st.caption("Heart rate (bpm)")
            with trend_col2:
                st.line_chart(df[["gsr", "stress"]], height=200)
                st.caption("Skin conductance (µS) and voice-derived stress (0–1)")
            st.line_chart(df[["score"]], height=200)
            st.caption("Composite distress/intensity score")
        else:
            st.write("No signal history yet.")

        st.subheader("Session event log")
        if log:
            for entry in reversed(log[-20:]):
                st.text(entry)
        else:
            st.caption("No events yet — applied changes and stops will appear here.")

        st.caption(
            "Only an encrypted summary of this session is synced to records afterward; "
            "live signals and voice never leave the clinic workstation."
        )

    render_dashboard()


render_setup_or_dashboard()
