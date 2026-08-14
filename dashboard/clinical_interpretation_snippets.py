"""
AVRTE — Psychology-team clinical interpretation snippets (Section 1.5)

Paste this block into integrated_avrte_dashboard.py AFTER the intensity_feedback_loop
import block and BEFORE SECTION 2 (ML adapter).

Design goals (psych team 2.2 / 2.3):
  - No objective "anxiety score" shown to clinicians as a diagnosis
  - Descriptive biofeedback ranges: likely calm / likely anxious / likely very anxious
  - Ollama synthesizes HR, GSR, speech rate, voice patterns into a supportive summary
  - Raw numbers remain visible; AI summary is workload-reduction, not clinical judgment
"""

from __future__ import annotations

import json
import os
import threading
import time
from enum import Enum
from typing import Callable, Optional

try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
CLINICAL_SUMMARY_MIN_INTERVAL = 8.0  # seconds between Ollama synthesis calls

ETHICAL_DISCLAIMER = (
    "This AI summary synthesizes biofeedback and speech patterns to reduce cognitive load. "
    "It does not diagnose anxiety, does not replace self-report, and does not replace your "
    "clinical judgment. Always interpret alongside the patient and the raw signal values below."
)


class EmotionalStateLabel(str, Enum):
    """Descriptive ranges — inferred from biofeedback, not objective anxiety measurement."""
    LIKELY_CALM = "likely calm"
    LIKELY_ANXIOUS = "likely anxious"
    LIKELY_VERY_ANXIOUS = "likely very anxious"
    CALIBRATING = "calibrating — establishing personal baseline"
    INSUFFICIENT_DATA = "insufficient data"


# ---------------------------------------------------------------------------
# Snippet A — Rule-based descriptive label (always available, no Ollama needed)
# ---------------------------------------------------------------------------

_BIOFEEDBACK_ELEVATION_KEYS = (
    "hr_z_sustained_high",
    "hr_absolute_high",
    "gsr_scl_elevated",
    "speech_high_stress",
    "speech_rate_elevated",
    "head_movement_z_sustained_high",
)


def infer_emotional_state_label(
    signals: dict,
    composite_arousal_index: float,
    calibrating: bool,
) -> EmotionalStateLabel:
    """
    Map baseline-relative biofeedback + voice-pattern flags to a descriptive range.
    This is inference from physiological/speech cues — not a clinical anxiety score.
    """
    if calibrating:
        return EmotionalStateLabel.CALIBRATING

    elevated_count = sum(1 for key in _BIOFEEDBACK_ELEVATION_KEYS if signals.get(key))

    if elevated_count >= 3 or composite_arousal_index >= 75:
        return EmotionalStateLabel.LIKELY_VERY_ANXIOUS
    if elevated_count >= 1 or composite_arousal_index >= 35:
        return EmotionalStateLabel.LIKELY_ANXIOUS
    return EmotionalStateLabel.LIKELY_CALM


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        return None if f != f else f  # NaN check
    except (TypeError, ValueError):
        return None


def build_clinical_snapshot(
    sample,
    reading,
    extra: Optional[dict],
) -> dict:
    """Structured payload for Ollama — raw numbers + flags, not a single anxiety score."""
    extra = extra or {}
    signals = reading.contributing_signals
    calibrating = bool(signals.get("calibrating"))
    rule_label = infer_emotional_state_label(signals, reading.score, calibrating)

    breathing_rpm = _safe_float(extra.get("respiratory_rate_rpm"))
    breathing_var = _safe_float(extra.get("respiratory_variability"))

    return {
        "calibrating": calibrating,
        "heart_rate_bpm": round(sample.heart_rate_bpm, 1),
        "hr_z_vs_personal_baseline": signals.get("hr_z"),
        "hr_elevated_sustained": signals.get("hr_z_sustained_high"),
        "skin_conductance_us": round(sample.gsr_microsiemens, 3),
        "gsr_elevated_vs_baseline": signals.get("gsr_scl_elevated"),
        "breathing_rate_rpm": round(breathing_rpm, 1) if breathing_rpm is not None else None,
        "respiratory_variability": round(breathing_var, 3) if breathing_var is not None else None,
        "speech_rate_wpm": sample.speech_rate_wpm,
        "speech_rate_elevated_vs_baseline": signals.get("speech_rate_elevated"),
        "voice_pitch_hz": extra.get("pitch_hz"),
        "voice_speech_ratio": extra.get("speech_ratio"),
        "voice_pauses_per_block": extra.get("pause_count"),
        "voice_pattern_elevated": signals.get("speech_high_stress"),
        "filler_words_session": extra.get("filler_count"),
        "repeated_words_session": extra.get("repetition_count"),
        "composite_arousal_index_0_100": reading.score,
        "biofeedback_state_hint": rule_label.value,
    }


def rule_based_clinician_summary(snapshot: dict, label: EmotionalStateLabel) -> str:
    """Fallback when Ollama is off — still gives words, not just numbers."""
    if label == EmotionalStateLabel.CALIBRATING:
        return (
            "Personal baseline is still being established. Biofeedback ranges will be "
            "meaningful once calibration completes."
        )
    parts = [f"Biofeedback pattern suggests the patient is {label.value}."]
    if snapshot.get("hr_elevated_sustained") or snapshot.get("gsr_elevated_vs_baseline"):
        parts.append("Heart rate and/or skin conductance are above the patient's resting baseline.")
    if snapshot.get("breathing_rate_rpm") is not None:
        parts.append(
            f"Breathing rate is {snapshot['breathing_rate_rpm']:.0f} breaths/min "
            f"(from simulated/reference physio stream)."
        )
    if snapshot.get("speech_rate_elevated_vs_baseline") or snapshot.get("voice_pattern_elevated"):
        parts.append("Speech rate or voice prosody show change relative to baseline.")
    parts.append("Use alongside your clinical observation and patient self-report.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Snippet B — Ollama multi-signal clinical summary (replaces transcript-only emotion tag)
# ---------------------------------------------------------------------------

class ClinicalSummarySynthesizer:
    """
    Calls Ollama to synthesize biofeedback + speech into a short clinician overview.
    Runs async so the live loop is never blocked.
    """

    def __init__(self, enabled: bool, on_status: Callable[[str], None]):
        self._enabled = enabled and HAVE_REQUESTS
        self._on_status = on_status
        self._lock = threading.Lock()
        self._latest_summary = ""
        self._latest_ai_label = ""
        self._busy = False
        self._last_call = 0.0

    def maybe_update(self, snapshot: dict, rule_label: EmotionalStateLabel) -> None:
        if not self._enabled or self._busy or snapshot.get("calibrating"):
            return
        now = time.time()
        if now - self._last_call < CLINICAL_SUMMARY_MIN_INTERVAL:
            return
        self._last_call = now
        self._busy = True
        threading.Thread(
            target=self._synthesize,
            args=(snapshot, rule_label),
            daemon=True,
        ).start()

    def _synthesize(self, snapshot: dict, rule_label: EmotionalStateLabel) -> None:
        fallback = rule_based_clinician_summary(snapshot, rule_label)
        prompt = (
            "You assist a VR exposure therapist by synthesizing biofeedback and speech "
            "signals into a brief, supportive overview for the clinician.\n\n"
            "STRICT RULES:\n"
            "- Do NOT diagnose. Do NOT claim objective anxiety measurement.\n"
            "- Use tentative language: 'signals suggest', 'may indicate', 'consistent with'.\n"
            "- Never say the patient IS anxious — only that patterns are 'likely calm', "
            "'likely anxious', or 'likely very anxious' based on biofeedback ranges.\n"
            "- 2-4 sentences max. Supportive tone. Reduce cognitive load, not replace judgment.\n"
            "- Respond with ONLY JSON: "
            '{"state_label": "<likely calm|likely anxious|likely very anxious>", '
            '"summary": "<your paragraph>"}\n\n'
            f"Signal snapshot (JSON):\n{json.dumps(snapshot, indent=2)}"
        )
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
                timeout=45,
            )
            resp.raise_for_status()
            parsed = json.loads(resp.json().get("response", "{}"))
            summary = str(parsed.get("summary") or fallback).strip()
            ai_label = str(parsed.get("state_label") or rule_label.value).strip()
            with self._lock:
                self._latest_summary = summary
                self._latest_ai_label = ai_label
        except Exception as exc:  # noqa: BLE001
            self._on_status(f"[clinical AI] synthesis unavailable ({exc}); using rule-based summary.")
            with self._lock:
                self._latest_summary = fallback
                self._latest_ai_label = rule_label.value
        finally:
            self._busy = False

    def snapshot(self) -> tuple[str, str]:
        with self._lock:
            return self._latest_summary, self._latest_ai_label
