package com.example.myapplication.mqtt

import com.example.myapplication.repository.BioDataRepository
import com.hivemq.client.mqtt.MqttClient
import com.hivemq.client.mqtt.mqtt3.Mqtt3AsyncClient
import com.hivemq.client.mqtt.mqtt3.message.publish.Mqtt3Publish
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.nio.charset.StandardCharsets

class BioDataSubscriber(
    private val repository: BioDataRepository
) {
    private val topic = "sensor/heartrate"

    // 10.0.2.2 = localhost on your PC from the Android emulator
    private val client: Mqtt3AsyncClient = MqttClient.builder()
        .useMqttVersion3()
        .identifier("versalis-android-${System.currentTimeMillis()}")
        .serverHost("10.0.2.2")
        .serverPort(1883)
        .buildAsync()

    fun connect() {
        client.connectWith()
            .cleanSession(true)
            .keepAlive(30)
            .send()
            .whenComplete { _, throwable ->
                if (throwable != null) {
                    println("MQTT connection failed: ${throwable.message}")
                } else {
                    println("Connected to Mosquitto")
                    subscribe()
                }
            }
    }

    private fun subscribe() {
        client.subscribeWith()
            .topicFilter(topic)
            .qos(com.hivemq.client.mqtt.datatypes.MqttQos.AT_LEAST_ONCE)
            .callback { publish: Mqtt3Publish ->
                val payload = StandardCharsets.UTF_8.decode(publish.payload.get()).toString()
                handleMessage(payload)
            }
            .send()
            .whenComplete { _, throwable ->
                if (throwable != null) {
                    println("Subscription failed: ${throwable.message}")
                } else {
                    println("Subscribed to $topic")
                }
            }
    }

    private fun handleMessage(payload: String) {
        println("Received: $payload bpm")

        try {
            val bpm = payload.trim().toInt()
            val timestamp = System.currentTimeMillis()

            CoroutineScope(Dispatchers.IO).launch {
                repository.writeBioData(bpm, timestamp)
                println("Written to Realm - BPM: $bpm")
            }

        } catch (e: Exception) {
            println("Failed to parse message: ${e.message}")
        }
    }

    fun disconnect() {
        client.disconnect()
        println("MQTT disconnected")
    }
}