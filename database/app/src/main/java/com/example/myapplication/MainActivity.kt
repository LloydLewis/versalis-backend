package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.database.RealmDatabase
import com.example.myapplication.models.BiometricReading
import com.example.myapplication.models.LLMInteraction
import com.example.myapplication.mqtt.BioDataSubscriber
import com.example.myapplication.repository.BioDataRepository
import com.example.myapplication.server.LlmInteractionServer
import io.realm.kotlin.ext.query
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.time.Duration.Companion.milliseconds

class MainActivity : ComponentActivity() {

    private lateinit var subscriber: BioDataSubscriber
    private lateinit var bioDataRepository: BioDataRepository
    private lateinit var llmInteractionServer: LlmInteractionServer

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            var isReady by remember { mutableStateOf(false) }
            var queryResult by remember { mutableStateOf("") }
            var llmResult by remember { mutableStateOf("") }
            var latestReading by remember { mutableStateOf("Waiting for MQTT data...") }
            var latestLLM by remember { mutableStateOf("No LLM interactions yet") }

            LaunchedEffect(Unit) {
                kotlinx.coroutines.withContext(Dispatchers.IO) {
                    RealmDatabase.init(this@MainActivity)
                    bioDataRepository = BioDataRepository()
                    subscriber = BioDataSubscriber(bioDataRepository)
                    subscriber.connect()
                    llmInteractionServer = LlmInteractionServer(bioDataRepository)
                    llmInteractionServer.start()
                }
                isReady = true
            }

            if (!isReady) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(16.dp))
                        Text("Initialising secure database...")
                    }
                }
            } else {

                // Live biometric poll — updates every second
                LaunchedEffect(Unit) {
                    while (true) {
                        val latest = bioDataRepository.getLatestReading()
                        latestReading = if (latest != null)
                            "HR: ${latest.heartRateBpm} bpm | " +
                                    "Anxiety: ${latest.anxietyScore} | " +
                                    "Phase: ${latest.sessionPhase}"
                        else
                            "Waiting for MQTT data..."
                        delay(1000.milliseconds)
                    }
                }

                // Live LLM poll — updates every 2 seconds
                LaunchedEffect(Unit) {
                    while (true) {
                        val interactions = RealmDatabase.realm
                            .query<LLMInteraction>()
                            .find()
                        latestLLM = if (interactions.isNotEmpty()) {
                            val last = interactions.last()
                            "Patient: \"${last.patientText}\"\n" +
                                    "Therapist: \"${last.therapistReply}\"\n" +
                                    "Flagged: ${last.wasGuardFlagged} | " +
                                    "Latency: ${last.latencyMs}ms"
                        } else {
                            "No LLM interactions yet"
                        }
                        delay(2000.milliseconds)
                    }
                }

                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(32.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {

                    // ── Biometric section ─────────────────────────────────
                    Text("MQTT -> Realm (Biometrics)", fontSize = 22.sp)

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = latestReading,
                        fontSize = 14.sp,
                        lineHeight = 20.sp,
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    Button(onClick = {
                        CoroutineScope(Dispatchers.IO).launch {
                            val readings = RealmDatabase.realm
                                .query<BiometricReading>()
                                .find()

                            queryResult = if (readings.isEmpty()) {
                                "No biometric readings in Realm yet"
                            } else {
                                val latest = readings.last()
                                buildString {
                                    appendLine("Total records: ${readings.size}")
                                    appendLine("─────────────────────")
                                    appendLine("Participant: ${latest.participantId}")
                                    appendLine("Session: ${latest.sessionId}")
                                    appendLine("Phase: ${latest.sessionPhase}")
                                    appendLine("Channel: ${latest.eegChannel}")
                                    appendLine("Sample rate: ${latest.samplingRateHz} Hz")
                                    appendLine("Data quality: ${latest.dataQualityScore}")
                                    appendLine("─────────────────────")
                                    appendLine("HR: ${latest.heartRateBpm} bpm")
                                    appendLine("HRV RMSSD: ${latest.hrvRmssdMs} ms")
                                    appendLine("HRV SDNN: ${latest.hrvSdnnMs} ms")
                                    appendLine("HRV LF/HF: ${latest.hrvLfHfRatio}")
                                    appendLine("SpO2: ${latest.spo2Pct}%")
                                    appendLine("Resp. rate: ${latest.respiratoryRateRpm} rpm")
                                    appendLine("─────────────────────")
                                    appendLine("BP: ${latest.systolicBpMmhg}/${latest.diastolicBpMmhg} mmHg")
                                    appendLine("─────────────────────")
                                    appendLine("EEG Alpha: ${latest.eegAlphaUv2}")
                                    appendLine("EEG Beta: ${latest.eegBetaUv2}")
                                    appendLine("EEG Theta: ${latest.eegThetaUv2}")
                                    appendLine("EEG Delta: ${latest.eegDeltaUv2}")
                                    appendLine("EEG Gamma: ${latest.eegGammaUv2}")
                                    appendLine("Frontal Asymmetry: ${latest.frontalAlphaAsymmetry}")
                                    appendLine("Attention: ${latest.attentionIndex}")
                                    appendLine("Cognitive Load: ${latest.cognitiveLoad}")
                                    appendLine("─────────────────────")
                                    appendLine("GSR: ${latest.gsrUs}")
                                    appendLine("Skin Temp: ${latest.skinTempC}°C")
                                    appendLine("─────────────────────")
                                    appendLine("Speech Rate: ${latest.speechRateWpm} wpm")
                                    appendLine("Voice Stress: ${latest.voiceStressScore}")
                                    appendLine("Voice F0: ${latest.voiceF0Hz} Hz")
                                    appendLine("Sentiment: ${latest.speechSentiment}")
                                    appendLine("─────────────────────")
                                    appendLine("Anxiety: ${latest.anxietyScore}")
                                    appendLine("Stress: ${latest.stressScore}")
                                    appendLine("Depression: ${latest.depressionIndex}")
                                    appendLine("─────────────────────")
                                    appendLine("Adverse Event: ${latest.adverseEventFlag}")
                                    appendLine("Clinician Intervened: ${latest.clinicianInterventionFlag}")
                                    appendLine("AI Escalation: ${latest.aiEscalationFlag}")
                                }
                            }
                        }
                    }) {
                        Text("Read Latest Biometric Record")
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = queryResult,
                        fontSize = 13.sp,
                        lineHeight = 18.sp,
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(24.dp))
                    HorizontalDivider()
                    Spacer(modifier = Modifier.height(24.dp))

                    // ── LLM Interactions section ──────────────────────────
                    Text("LLM Interactions -> Realm", fontSize = 22.sp)

                    Spacer(modifier = Modifier.height(12.dp))

                    // Live latest interaction — updates every 2 seconds
                    Text(
                        text = latestLLM,
                        fontSize = 13.sp,
                        lineHeight = 18.sp,
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(12.dp))

                    // Read all stored interactions
                    Button(onClick = {
                        CoroutineScope(Dispatchers.IO).launch {
                            val interactions = RealmDatabase.realm
                                .query<LLMInteraction>()
                                .find()

                            llmResult = if (interactions.isEmpty()) {
                                "No LLM interactions stored yet"
                            } else {
                                buildString {
                                    appendLine("Total interactions: ${interactions.size}")
                                    appendLine("─────────────────────")
                                    // Show last 5 interactions
                                    interactions.takeLast(5).forEach { interaction ->
                                        appendLine("Session: ${interaction.sessionId}")
                                        appendLine("Patient:   \"${interaction.patientText}\"")
                                        appendLine("Therapist: \"${interaction.therapistReply}\"")
                                        appendLine("Flagged: ${interaction.wasGuardFlagged}")
                                        appendLine("Latency: ${interaction.latencyMs}ms")
                                        appendLine("Model: ${interaction.modelUsed}")
                                        appendLine("─────────────────────")
                                    }
                                }
                            }
                        }
                    }) {
                        Text("Read LLM Interactions")
                    }

                    Spacer(modifier = Modifier.height(12.dp))

                    Text(
                        text = llmResult,
                        fontSize = 13.sp,
                        lineHeight = 18.sp,
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(24.dp))
                    HorizontalDivider()
                    Spacer(modifier = Modifier.height(24.dp))

                    // ── Clear database ────────────────────────────────────
                    Button(onClick = {
                        CoroutineScope(Dispatchers.IO).launch {
                            RealmDatabase.realm.write {
                                deleteAll()
                            }
                            queryResult = "Database cleared"
                            llmResult = ""
                        }
                    }) {
                        Text("Clear Database")
                    }

                    Spacer(modifier = Modifier.height(32.dp))
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        subscriber.disconnect()
        llmInteractionServer.stop()
        RealmDatabase.realm.close()
    }
}