package com.example.myapplication.repository

import com.example.myapplication.database.RealmDatabase
import com.example.myapplication.models.BiometricReading
import io.realm.kotlin.ext.query
import io.realm.kotlin.query.Sort
import org.json.JSONObject
import com.example.myapplication.models.LLMInteraction
class BioDataRepository {

    private val realm = RealmDatabase.realm
    suspend fun writeLLMInteraction(
        sessionId: String,
        patientText: String,
        therapistReply: String,
        wasGuardFlagged: Boolean,
        flagCategory: String = "",
        latencyMs: Long = 0,
        modelUsed: String = "llama3.2:1b"
    )
    {
        realm.write {
            copyToRealm(LLMInteraction().apply {
                this.sessionId          = sessionId
                this.timestampMs        = System.currentTimeMillis()
                this.patientText        = patientText
                this.therapistReply     = therapistReply
                this.wasGuardFlagged    = wasGuardFlagged
                this.flagCategory       = flagCategory
                this.latencyMs          = latencyMs
                this.modelUsed          = modelUsed
            })
        }
        println("LLM interaction written to Realm")
    }

    fun getAllLLMInteractions(): List<LLMInteraction> {
        return realm.query<LLMInteraction>().find()
    }
    suspend fun writeBiometricReading(json: JSONObject) {
        realm.write {
            copyToRealm(BiometricReading().apply {
                participantId = json.getString("participantId")
                sessionId = json.getString("sessionId")
                timestampMs = System.currentTimeMillis()
                sessionPhase = json.getString("sessionPhase")
                samplingRateHz = json.getInt("samplingRateHz")
                dataQualityScore = json.getDouble("dataQualityScore").toFloat()

                eegChannel = json.getString("eegChannel")
                eegAlphaUv2 = json.getDouble("eegAlphaUv2").toFloat()
                eegBetaUv2 = json.getDouble("eegBetaUv2").toFloat()
                eegThetaUv2 = json.getDouble("eegThetaUv2").toFloat()
                eegDeltaUv2 = json.getDouble("eegDeltaUv2").toFloat()
                eegGammaUv2 = json.getDouble("eegGammaUv2").toFloat()

                frontalAlphaAsymmetry = json.getDouble("frontalAlphaAsymmetry").toFloat()
                attentionIndex = json.getDouble("attentionIndex").toFloat()
                meditationIndex = json.getDouble("meditationIndex").toFloat()
                cognitiveLoad = json.getDouble("cognitiveLoad").toFloat()
                mentalFatigue = json.getDouble("mentalFatigue").toFloat()

                heartRateBpm = json.getDouble("heartRateBpm").toFloat()
                hrvRmssdMs = json.getDouble("hrvRmssdMs").toFloat()
                hrvSdnnMs = json.getDouble("hrvSdnnMs").toFloat()
                hrvPnn50Pct = json.getDouble("hrvPnn50Pct").toFloat()
                hrvLfHfRatio = json.getDouble("hrvLfHfRatio").toFloat()

                respiratoryRateRpm = json.getDouble("respiratoryRateRpm").toFloat()
                tidalVolumeMl = json.getDouble("tidalVolumeMl").toFloat()
                respiratoryVariability = json.getDouble("respiratoryVariability").toFloat()

                systolicBpMmhg = json.getDouble("systolicBpMmhg").toFloat()
                diastolicBpMmhg = json.getDouble("diastolicBpMmhg").toFloat()
                spo2Pct = json.getDouble("spo2Pct").toFloat()

                gsrUs = json.getDouble("gsrUs").toFloat()
                skinConductanceLevelUs = json.getDouble("skinConductanceLevelUs").toFloat()
                skinConductanceResponseUs = json.getDouble("skinConductanceResponseUs").toFloat()
                skinTempC = json.getDouble("skinTempC").toFloat()

                speechRateWpm = json.getDouble("speechRateWpm").toFloat()
                voiceEnergyNorm = json.getDouble("voiceEnergyNorm").toFloat()
                pauseDurationS = json.getDouble("pauseDurationS").toFloat()
                voiceF0Hz = json.getDouble("voiceF0Hz").toFloat()
                speechSentiment = json.getDouble("speechSentiment").toFloat()
                voiceStressScore = json.getDouble("voiceStressScore").toFloat()

                anxietyScore = json.getDouble("anxietyScore").toFloat()
                stressScore = json.getDouble("stressScore").toFloat()
                depressionIndex = json.getDouble("depressionIndex").toFloat()

                adverseEventFlag = json.getBoolean("adverseEventFlag")
                clinicianInterventionFlag = json.getBoolean("clinicianInterventionFlag")
                aiEscalationFlag = json.getBoolean("aiEscalationFlag")
            })
        }
    }

    fun getLatestReading(): BiometricReading? {
        return realm.query<BiometricReading>()
            .sort("timestampMs", Sort.DESCENDING)
            .first()
            .find()
    }

    fun getAllReadings(): List<BiometricReading> {
        return realm.query<BiometricReading>().find()
    }
}