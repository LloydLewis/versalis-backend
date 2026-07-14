package com.example.myapplication.repository

import com.example.myapplication.database.RealmDatabase
import com.example.myapplication.models.BioData
import io.realm.kotlin.ext.query
import io.realm.kotlin.query.Sort

class BioDataRepository {

    private val realm = RealmDatabase.realm

    suspend fun writeBioData(bpm: Int, timestamp: Long) {
        realm.write {
            copyToRealm(BioData().apply {
                this.bpm = bpm
                this.timestamp = timestamp
            })
        }
    }

    fun getLatestReading(): BioData? {
        return realm.query<BioData>()
            .sort("timestamp", Sort.DESCENDING)
            .first()
            .find()
    }

    fun getAllReadings(): List<BioData> {
        return realm.query<BioData>().find()
    }
}