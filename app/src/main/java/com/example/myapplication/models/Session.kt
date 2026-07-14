package com.example.myapplication.models

import io.realm.kotlin.types.RealmObject
import io.realm.kotlin.types.annotations.PrimaryKey
import org.mongodb.kbson.ObjectId

class ButtonPress : RealmObject {
    @PrimaryKey
    var _id: ObjectId = ObjectId()
    var message: String = ""
    var timestamp: Long = 0
}