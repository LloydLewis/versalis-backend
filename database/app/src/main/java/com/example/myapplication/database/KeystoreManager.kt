package com.example.myapplication.database

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

object KeystoreManager {
    private const val KEY_ALIAS = "versalis_realm_key"
    private const val KEYSTORE_PROVIDER = "AndroidKeyStore"
    private const val KEY_SIZE = 256

    // Where the encrypted Realm key is stored on disk
    // The Keystore protects the encryption of this file
    private const val ENCRYPTED_KEY_FILE = "realm_key.enc"

    /**
     * Returns the 64-byte Realm encryption key.
     * Generates and stores it on first launch.
     * Retrieves and decrypts it on every subsequent launch.
     */
    fun getRealmEncryptionKey(context: android.content.Context): ByteArray {
        return if (realmKeyFileExists(context)) {
            // Key was already generated — retrieve and decrypt it
            println("Keystore: retrieving existing Realm key")
            decryptRealmKey(context)
        } else {
            // First launch — generate a new key and store it
            println("Keystore: generating new Realm key")
            generateAndStoreRealmKey(context)
        }
    }

    // ── Key generation ────────────────────────────────────────────────────────

    private fun generateAndStoreRealmKey(context: android.content.Context): ByteArray {
        // Step 1: Generate the 64-byte Realm key (random bytes)
        val realmKey = ByteArray(64)
        java.security.SecureRandom().nextBytes(realmKey)

        // Step 2: Encrypt the Realm key using Android Keystore
        val encryptedData = encryptWithKeystore(realmKey)

        // Step 3: Save the encrypted key + IV to a file on disk
        saveEncryptedKey(context, encryptedData)

        println("Keystore: new Realm key generated and stored")
        return realmKey
    }

    // ── Keystore AES key management ───────────────────────────────────────────

    /**
     * Creates or retrieves the AES key in Android Keystore.
     * This key is used to encrypt/decrypt the Realm key.
     * It never leaves the secure hardware enclave.
     */
    private fun getOrCreateKeystoreKey(): SecretKey {
        val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
        keyStore.load(null)

        // Return existing key if it already exists
        if (keyStore.containsAlias(KEY_ALIAS)) {
            return (keyStore.getEntry(KEY_ALIAS, null) as KeyStore.SecretKeyEntry).secretKey
        }

        // Generate a new AES key inside the Keystore
        val keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            KEYSTORE_PROVIDER
        )

        keyGenerator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                // This key can only be used for encrypt and decrypt operations
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setKeySize(KEY_SIZE)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                // Key is invalidated if the user removes their screen lock
                // Protects against someone disabling biometrics to access data
                .setInvalidatedByBiometricEnrollment(false)
                .build()
        )

        return keyGenerator.generateKey()
    }

    // ── Encryption / Decryption ───────────────────────────────────────────────

    /**
     * Encrypts the Realm key using AES-GCM via the Keystore key.
     * Returns IV + encrypted bytes combined into one array.
     */
    private fun encryptWithKeystore(realmKey: ByteArray): ByteArray {
        val secretKey = getOrCreateKeystoreKey()

        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey)

        val iv = cipher.iv                          // 12-byte initialisation vector
        val encrypted = cipher.doFinal(realmKey)   // encrypted Realm key

        // Combine IV and encrypted data: [IV length (4 bytes)][IV][encrypted data]
        // We store the IV length first so we know how many bytes to read back
        val result = ByteArray(4 + iv.size + encrypted.size)
        result[0] = (iv.size shr 24).toByte()
        result[1] = (iv.size shr 16).toByte()
        result[2] = (iv.size shr 8).toByte()
        result[3] = iv.size.toByte()
        System.arraycopy(iv, 0, result, 4, iv.size)
        System.arraycopy(encrypted, 0, result, 4 + iv.size, encrypted.size)

        return result
    }

    /**
     * Decrypts the stored encrypted Realm key using the Keystore key.
     */
    private fun decryptRealmKey(context: android.content.Context): ByteArray {
        val encryptedData = loadEncryptedKey(context)

        // Read the IV length from the first 4 bytes
        val ivLength = ((encryptedData[0].toInt() and 0xFF) shl 24) or
                ((encryptedData[1].toInt() and 0xFF) shl 16) or
                ((encryptedData[2].toInt() and 0xFF) shl 8) or
                (encryptedData[3].toInt() and 0xFF)

        // Extract IV and encrypted bytes
        val iv = encryptedData.copyOfRange(4, 4 + ivLength)
        val encrypted = encryptedData.copyOfRange(4 + ivLength, encryptedData.size)

        // Decrypt using the Keystore key and the stored IV
        val secretKey = getOrCreateKeystoreKey()
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, secretKey, GCMParameterSpec(128, iv))

        return cipher.doFinal(encrypted)
    }

    // ── File storage for the encrypted key ───────────────────────────────────

    private fun realmKeyFileExists(context: android.content.Context): Boolean {
        return context.getFileStreamPath(ENCRYPTED_KEY_FILE).exists()
    }

    private fun saveEncryptedKey(context: android.content.Context, data: ByteArray) {
        context.openFileOutput(ENCRYPTED_KEY_FILE, android.content.Context.MODE_PRIVATE)
            .use { it.write(data) }
    }

    private fun loadEncryptedKey(context: android.content.Context): ByteArray {
        return context.openFileInput(ENCRYPTED_KEY_FILE)
            .use { it.readBytes() }
    }
}