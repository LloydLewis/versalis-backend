package com.example.myapplication

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.myapplication.database.RealmDatabase
import com.example.myapplication.models.ButtonPress
import com.example.myapplication.mqtt.BioDataSubscriber
import com.example.myapplication.repository.BioDataRepository
import io.realm.kotlin.ext.query
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var subscriber: BioDataSubscriber
    private lateinit var bioDataRepository: BioDataRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialise Realm with both schemas
        RealmDatabase.init()

        // Initialise MQTT subscriber
        bioDataRepository = BioDataRepository()
        subscriber = BioDataSubscriber(bioDataRepository)
        subscriber.connect()

        setContent {

            // Button press state
            var lastMessage by remember { mutableStateOf("No presses yet") }

            // Live BPM state — updates every second from Realm
            var latestBpm by remember { mutableStateOf("Waiting for MQTT data...") }

            LaunchedEffect(Unit) {
                while (true) {
                    val latest = bioDataRepository.getLatestReading()
                    latestBpm = if (latest != null)
                        "Latest BPM from MQTT: ${latest.bpm}"
                    else
                        "Waiting for MQTT data..."
                    delay(1000)
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(32.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                // MQTT section
                Text("MQTT -> Realm", fontSize = 22.sp)

                Spacer(modifier = Modifier.height(12.dp))

                Text(latestBpm, fontSize = 16.sp)

                Spacer(modifier = Modifier.height(8.dp))

                Button(onClick = {
                    CoroutineScope(Dispatchers.IO).launch {
                        val readings = bioDataRepository.getAllReadings()
                        println("All BPM readings in Realm:")
                        readings.forEach { r ->
                            println("  BPM: ${r.bpm} at ${r.timestamp}")
                        }
                    }
                }) {
                    Text("Print All BPM Readings")
                }

                Spacer(modifier = Modifier.height(40.dp))
                Divider()
                Spacer(modifier = Modifier.height(40.dp))

                //Button press section
                Text("Button Press -> Realm", fontSize = 22.sp)

                Spacer(modifier = Modifier.height(12.dp))

                Text(lastMessage, fontSize = 16.sp)

                Spacer(modifier = Modifier.height(12.dp))

                Button(onClick = {
                    CoroutineScope(Dispatchers.IO).launch {
                        RealmDatabase.realm.write {
                            copyToRealm(ButtonPress().apply {
                                message = "Button pressed"
                                timestamp = System.currentTimeMillis()
                            })
                        }
                        println("Written to Realm")

                        val count = RealmDatabase.realm
                            .query<ButtonPress>()
                            .count()
                            .find()

                        lastMessage = "Total presses in Realm: $count"
                        println("Total button press records: $count")
                    }
                }) {
                    Text("Press Me - Write to Realm")
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = {
                    CoroutineScope(Dispatchers.IO).launch {
                        val presses = RealmDatabase.realm
                            .query<ButtonPress>()
                            .find()

                        println("All button press records in Realm:")
                        presses.forEach { press ->
                            println("  - ${press.message} at ${press.timestamp}")
                        }

                        lastMessage = "Found ${presses.size} records - check Logcat"
                    }
                }) {
                    Text("Read All Records")
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        subscriber.disconnect()
        RealmDatabase.realm.close()
    }
}