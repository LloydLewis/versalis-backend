package com.example.myapplication.models

import io.realm.kotlin.types.RealmObject
import io.realm.kotlin.types.annotations.PrimaryKey
import org.mongodb.kbson.ObjectId

/*class BioData : RealmObject {
    @PrimaryKey
    var _id: ObjectId = ObjectId()
    var bpm: Int = 0
    var timestamp: Long = 0
}*/

class BiometricReading : RealmObject {
    @PrimaryKey
    var _id: ObjectId = ObjectId()

    // ── Session context ───────────────────────────────────────
    var participantId: String = ""
    var sessionId: String = ""
    var timestampMs: Long = 0
    var sessionPhase: String = ""        // "BASELINE" / "EXPOSURE" / "INTERVENTION" / "RECOVERY"
    var samplingRateHz: Int = 0          // varies per device — needed to interpret signal density
    var dataQualityScore: Float = 0f     // 0-100 — low scores should be excluded from model input

    // ── EEG bands (MNE-Python) ────────────────────────────────
    var eegChannel: String = ""          // Fp1, C3, P4 etc. — brain region matters clinically
    var eegAlphaUv2: Float = 0f          // relaxation / calm
    var eegBetaUv2: Float = 0f           // anxiety / active thinking — primary signal
    var eegThetaUv2: Float = 0f          // drowsiness / meditation
    var eegDeltaUv2: Float = 0f          // deep sleep / low arousal
    var eegGammaUv2: Float = 0f          // high cognitive activity

    // ── EEG derived indices ───────────────────────────────────
    var frontalAlphaAsymmetry: Float = 0f   // negative = withdrawal / anxiety tendency
    var attentionIndex: Float = 0f
    var meditationIndex: Float = 0f
    var cognitiveLoad: Float = 0f
    var mentalFatigue: Float = 0f

    // ── Heart rate & HRV (NeuroKit2) ──────────────────────────
    var heartRateBpm: Float = 0f
    var hrvRmssdMs: Float = 0f           // primary clinical HRV measure — parasympathetic activity
    var hrvSdnnMs: Float = 0f            // overall HRV variability
    var hrvPnn50Pct: Float = 0f          // percentage of successive beats differing >50ms
    var hrvLfHfRatio: Float = 0f         // sympathetic/parasympathetic balance

    // ── Respiratory ───────────────────────────────────────────
    var respiratoryRateRpm: Float = 0f
    var tidalVolumeMl: Float = 0f
    var respiratoryVariability: Float = 0f

    // ── Blood pressure & oxygen ───────────────────────────────
    var systolicBpMmhg: Float = 0f
    var diastolicBpMmhg: Float = 0f
    var spo2Pct: Float = 0f              // blood oxygen — drops under extreme stress

    // ── Skin (NeuroKit2) ──────────────────────────────────────
    var gsrUs: Float = 0f                // raw galvanic skin response
    var skinConductanceLevelUs: Float = 0f
    var skinConductanceResponseUs: Float = 0f
    var skinTempC: Float = 0f            // skin temperature — drops during stress (vasoconstriction)

    // ── Voice derived (ASR pipeline) ─────────────────────────
    var speechRateWpm: Float = 0f
    var voiceEnergyNorm: Float = 0f
    var pauseDurationS: Float = 0f
    var voiceF0Hz: Float = 0f            // fundamental pitch frequency
    var speechSentiment: Float = 0f      // -1.0 to 1.0
    var voiceStressScore: Float = 0f

    // ── Deviation from baseline ───────────────────────────────
    var heartRateDeviation: Float = 0f
    var gsrDeviation: Float = 0f
    var eegBetaDeviation: Float = 0f

    // ── Adaptive engine output ────────────────────────────────
    var anxietyScore: Float = 0f         // 0-100 as per CSV
    var stressScore: Float = 0f
    var depressionIndex: Float = 0f
    var distressScore: Float = 0f        // combined 0.0-1.0 for intensity decision
    var intensityDirection: String = ""  // "INCREASE" / "HOLD" / "DECREASE"
    var upperThresholdBreached: Boolean = false
    var lowerThresholdBreached: Boolean = false
    var triggerSource: String = ""       // "BIOFEEDBACK" / "ASR" / "COMBINED"

    // ── Safety flags ──────────────────────────────────────────
    var adverseEventFlag: Boolean = false       // something went wrong clinically
    var clinicianInterventionFlag: Boolean = false  // MHP manually stepped in
    var aiEscalationFlag: Boolean = false       // AI flagged for escalation
}