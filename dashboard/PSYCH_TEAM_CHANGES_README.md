# Psychology team changes (sections 2.2 & 2.3)

See **[README.md](README.md)** for full setup, FAQ, and run instructions.

## Summary of psych-team design

1. **No primary “anxiety %”** — clinicians see **likely calm / likely anxious / likely very anxious** from biofeedback ranges.
2. **Raw numbers always visible** under AI clinical summary (HR, GSR, breathing, speech rate, voice prosody).
3. **Ollama** synthesizes biofeedback + speech into a **supportive paragraph** — does not diagnose or replace clinical judgment.
4. **Ethical disclaimer** shown above every AI summary.
5. **Transcript-only emotion tag removed**; multi-signal `ClinicalSummarySynthesizer` replaces it.
6. **Speech quality** section (filler/repeated words) replaces live transcript display.
7. **Head movement / eye tracking** removed from clinician UI.

Implementation: `clinical_interpretation_snippets.py` + patches in `integrated_avrte_dashboard.py`.
