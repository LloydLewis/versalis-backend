package com.example.myapplication.models

import io.realm.kotlin.types.RealmObject
import io.realm.kotlin.types.annotations.PrimaryKey
import org.mongodb.kbson.BsonObjectId.Companion.invoke
import org.mongodb.kbson.ObjectId

class BioData : RealmObject {
    @PrimaryKey
    var _id: ObjectId = ObjectId()
    var bpm: Int = 0
    var timestamp: Long = 0
}