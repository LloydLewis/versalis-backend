package com.example.myapplication.database

import com.example.myapplication.models.BioData
import com.example.myapplication.models.ButtonPress
import io.realm.kotlin.Realm
import io.realm.kotlin.RealmConfiguration

object RealmDatabase {
    lateinit var realm: Realm

    fun init() {
        val config = RealmConfiguration.Builder(
            schema = setOf(ButtonPress::class, BioData::class)
        )
            .name("versalis.realm")
            .build()

        realm = Realm.open(config)
        println("Realm opened successfully")
    }
}