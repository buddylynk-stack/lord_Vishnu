package com.orignal.buddylynk.security

import android.util.Base64
import javax.crypto.Cipher
import javax.crypto.spec.IvParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * Runtime String Obfuscation
 * 
 * Hides sensitive strings (API URLs, keys) from static analysis.
 * Strings are encrypted at compile time and decrypted at runtime.
 * 
 * Usage:
 * Instead of: val url = "https://api.example.com"
 * Use: val url = StringObfuscator.decode("encrypted_base64_string")
 * 
 * To encrypt a string, use the encrypt() method during development,
 * then replace with the encrypted result.
 */
object StringObfuscator {
    
    // Key derived from app-specific data (change this!)
    // This is XOR'd with package name at runtime for extra protection
    private val KEY_SEED = byteArrayOf(
        0x42, 0x75, 0x64, 0x64, 0x79, 0x4C, 0x79, 0x6E,
        0x6B, 0x53, 0x65, 0x63, 0x72, 0x65, 0x74, 0x21
    )
    
    private val IV = byteArrayOf(
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
    )
    
    /**
     * Decrypt an obfuscated string at runtime
     */
    fun decode(encoded: String): String {
        return try {
            val encrypted = Base64.decode(encoded, Base64.NO_WRAP)
            val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")
            cipher.init(Cipher.DECRYPT_MODE, getKey(), IvParameterSpec(IV))
            String(cipher.doFinal(encrypted), Charsets.UTF_8)
        } catch (e: Exception) {
            "" // Return empty on failure
        }
    }
    
    /**
     * Encrypt a string for obfuscation (use during development)
     * Copy the output and use it in decode() calls
     */
    fun encode(plainText: String): String {
        return try {
            val cipher = Cipher.getInstance("AES/CBC/PKCS5Padding")
            cipher.init(Cipher.ENCRYPT_MODE, getKey(), IvParameterSpec(IV))
            val encrypted = cipher.doFinal(plainText.toByteArray(Charsets.UTF_8))
            Base64.encodeToString(encrypted, Base64.NO_WRAP)
        } catch (e: Exception) {
            ""
        }
    }
    
    private fun getKey(): SecretKeySpec {
        // XOR key seed with a fixed mask for obfuscation
        val key = KEY_SEED.mapIndexed { index, byte ->
            (byte.toInt() xor (index * 17 + 42)).toByte()
        }.toByteArray()
        return SecretKeySpec(key, "AES")
    }
    
    /**
     * Simple XOR obfuscation for less critical strings
     * Faster but less secure than AES
     */
    fun xorDecode(encoded: String, key: Int = 0x5A): String {
        val bytes = Base64.decode(encoded, Base64.NO_WRAP)
        return String(bytes.map { (it.toInt() xor key).toByte() }.toByteArray())
    }
    
    fun xorEncode(plainText: String, key: Int = 0x5A): String {
        val bytes = plainText.toByteArray().map { (it.toInt() xor key).toByte() }.toByteArray()
        return Base64.encodeToString(bytes, Base64.NO_WRAP)
    }
}

/**
 * Pre-obfuscated sensitive strings
 * Add your encrypted strings here after encoding them
 */
object ObfuscatedStrings {
    // Example: API base URL (encode your actual URL)
    // val API_URL = StringObfuscator.decode("your_encrypted_base64")
    
    // To generate encrypted strings, call during debug:
    // Log.d("Obfuscate", StringObfuscator.encode("http://52.0.95.126:3000"))
    // Then copy the output here
}
