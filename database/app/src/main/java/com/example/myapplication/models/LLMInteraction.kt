package com.example.myapplication.models

import io.realm.kotlin.types.RealmObject
import io.realm.kotlin.types.annotations.PrimaryKey
import org.mongodb.kbson.ObjectId

class LLMInteraction : RealmObject {
    @PrimaryKey
    var _id: ObjectId = ObjectId()

    // Session context
    var sessionId: String = ""
    var timestampMs: Long = 0

    // What the patient said (from ASR transcript)
    var patientText: String = ""

    // What the therapist avatar replied
    var therapistReply: String = ""

    // Whether the guardrail fired
    var wasGuardFlagged: Boolean = false

    // Which guardrail category triggered (if flagged)
    var flagCategory: String = ""

    // How long the LLM took to respond in milliseconds
    var latencyMs: Long = 0

    // The Ollama model used
    var modelUsed: String = ""
}