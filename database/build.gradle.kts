// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlinAndroid) apply false
    alias(libs.plugins.kotlin.compose) apply false
    id("com.infomaniak.realm.kotlin") version "3.2.9" apply false
}

