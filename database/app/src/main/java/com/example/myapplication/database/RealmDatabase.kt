package com.example.myapplication.database

import android.content.Context
import com.example.myapplication.models.BiometricReading
import io.realm.kotlin.Realm
import io.realm.kotlin.RealmConfiguration

object RealmDatabase {
    lateinit var realm: Realm

    fun init(context: Context) {
        val encryptionKey = KeystoreManager.getRealmEncryptionKey(context)
        val config = RealmConfiguration.Builder(
            schema = setOf(BiometricReading::class)
        )
            .name("versalis.realm")
            .deleteRealmIfMigrationNeeded()  // remove this before production
            .build()

        realm = Realm.open(config)
        println("Realm opened successfully")
        encryptionKey.fill(0)
    }
}