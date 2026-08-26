package com.example.myapplication.server

import com.example.myapplication.repository.BioDataRepository
import io.ktor.http.ContentType
import io.ktor.http.HttpStatusCode
import io.ktor.server.application.call
import io.ktor.server.engine.EmbeddedServer
import io.ktor.server.engine.embeddedServer
import io.ktor.server.netty.Netty
import io.ktor.server.request.receiveText
import io.ktor.server.response.respondText
import io.ktor.server.routing.post
import io.ktor.server.routing.routing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONObject

/**
 * Listens on port 8080 for LLM interactions posted by bridge/main.py
 * (POST http://10.0.2.2:8080/realm/llm-interaction) and writes each one
 * to Realm via BioDataRepository.
 */
class LlmInteractionServer(
    private val repository: BioDataRepository
) {
    private var engine: EmbeddedServer<*, *>? = null

    fun start(port: Int = 8080) {
        engine = embeddedServer(Netty, port = port) {
            routing {
                post("/realm/llm-interaction") {
                    val body = call.receiveText()
                    try {
                        val json = JSONObject(body)

                        CoroutineScope(Dispatchers.IO).launch {
                            repository.writeLLMInteraction(
                                sessionId = json.optString("sessionId", ""),
                                patientText = json.optString("patientText", ""),
                                therapistReply = json.optString("therapistReply", ""),
                                wasGuardFlagged = json.optBoolean("wasGuardFlagged", false),
                                latencyMs = json.optLong("latencyMs", 0),
                                modelUsed = json.optString("modelUsed", "llama3.2:1b")
                            )
                        }

                        call.respondText(
                            text = """{"status":"ok"}""",
                            contentType = ContentType.Application.Json,
                            status = HttpStatusCode.OK
                        )
                    } catch (e: Exception) {
                        println("[LlmInteractionServer] Failed to parse payload: ${e.message}")
                        call.respondText(
                            text = """{"status":"error"}""",
                            contentType = ContentType.Application.Json,
                            status = HttpStatusCode.BadRequest
                        )
                    }
                }
            }
        }.start(wait = false)

        println("LlmInteractionServer listening on port $port")
    }

    fun stop() {
        engine?.stop(gracePeriodMillis = 200, timeoutMillis = 500)
        println("LlmInteractionServer stopped")
    }
}
