package com.orignal.buddylynk.data.security

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Secure storage using EncryptedSharedPreferences
 */
object SecureStorage {
    
    private const val PREFS_NAME = "buddylynk_secure_prefs"
    private var encryptedPrefs: android.content.SharedPreferences? = null
    
    /**
     * Initialize secure storage
     */
    fun init(context: Context) {
        try {
            val masterKey = MasterKey.Builder(context)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build()
            
            encryptedPrefs = EncryptedSharedPreferences.create(
                context,
                PREFS_NAME,
                masterKey,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )
        } catch (e: Exception) {
            // Fallback to regular prefs if encryption fails
            encryptedPrefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        }
    }
    
    /**
     * Store secure string
     */
    fun putString(key: String, value: String) {
        encryptedPrefs?.edit()?.putString(key, value)?.apply()
    }
    
    /**
     * Get secure string
     */
    fun getString(key: String, default: String? = null): String? {
        return encryptedPrefs?.getString(key, default)
    }
    
    /**
     * Remove key
     */
    fun remove(key: String) {
        encryptedPrefs?.edit()?.remove(key)?.apply()
    }
    
    /**
     * Clear all
     */
    fun clear() {
        encryptedPrefs?.edit()?.clear()?.apply()
    }
}

/**
 * Input validation and sanitization
 */
object InputValidator {
    
    // Email regex pattern
    private val EMAIL_PATTERN = Regex(
        "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    )
    
    // Username pattern: alphanumeric, underscore, 3-20 chars
    private val USERNAME_PATTERN = Regex("^[a-zA-Z0-9_]{3,20}$")
    
    // Password: at least 8 chars, 1 upper, 1 lower, 1 digit
    private val PASSWORD_PATTERN = Regex(
        "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).{8,}$"
    )
    
    /**
     * Validate email format
     */
    fun isValidEmail(email: String): Boolean {
        return EMAIL_PATTERN.matches(email.trim())
    }
    
    /**
     * Validate username format
     */
    fun isValidUsername(username: String): Boolean {
        return USERNAME_PATTERN.matches(username.trim())
    }
    
    /**
     * Validate password strength
     */
    fun isValidPassword(password: String): Boolean {
        return PASSWORD_PATTERN.matches(password)
    }
    
    /**
     * Get password strength (0-4)
     */
    fun getPasswordStrength(password: String): Int {
        var score = 0
        if (password.length >= 8) score++
        if (password.length >= 12) score++
        if (password.any { it.isUpperCase() }) score++
        if (password.any { it.isDigit() }) score++
        if (password.any { !it.isLetterOrDigit() }) score++
        return score.coerceAtMost(4)
    }
    
    /**
     * Sanitize text input (remove dangerous chars)
     */
    fun sanitizeText(input: String): String {
        return input
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\"", "&quot;")
            .replace("'", "&#39;")
            .trim()
    }
    
    /**
     * Sanitize for SQL (prevent injection)
     */
    fun sanitizeForSQL(input: String): String {
        return input
            .replace("'", "''")
            .replace("\\", "\\\\")
            .replace("\u0000", "")
            .trim()
    }
    
    /**
     * Validate URL
     */
    fun isValidUrl(url: String): Boolean {
        return try {
            val uri = android.net.Uri.parse(url)
            uri.scheme in listOf("http", "https") && !uri.host.isNullOrBlank()
        } catch (e: Exception) {
            false
        }
    }
}

/**
 * Rate limiter for sensitive operations
 */
object RateLimiter {
    
    private val attempts = mutableMapOf<String, MutableList<Long>>()
    
    /**
     * Check if action is allowed
     * @param key Action identifier (e.g., "login:userId")
     * @param maxAttempts Max attempts in window
     * @param windowMs Time window in milliseconds
     */
    fun isAllowed(key: String, maxAttempts: Int = 5, windowMs: Long = 60_000): Boolean {
        val now = System.currentTimeMillis()
        val windowStart = now - windowMs
        
        // Get or create attempts list
        val keyAttempts = attempts.getOrPut(key) { mutableListOf() }
        
        // Remove old attempts outside window
        keyAttempts.removeAll { it < windowStart }
        
        // Check if under limit
        return if (keyAttempts.size < maxAttempts) {
            keyAttempts.add(now)
            true
        } else {
            false
        }
    }
    
    /**
     * Get remaining cooldown time
     */
    fun getCooldownMs(key: String, windowMs: Long = 60_000): Long {
        val keyAttempts = attempts[key] ?: return 0
        if (keyAttempts.isEmpty()) return 0
        
        val oldestAttempt = keyAttempts.minOrNull() ?: return 0
        val unlockTime = oldestAttempt + windowMs
        
        return (unlockTime - System.currentTimeMillis()).coerceAtLeast(0)
    }
    
    /**
     * Reset attempts for key
     */
    fun reset(key: String) {
        attempts.remove(key)
    }
    
    /**
     * Clear all rate limits
     */
    fun clearAll() {
        attempts.clear()
    }
}

/**
 * Security utilities
 */
object SecurityUtils {
    
    /**
     * Hash a string (for local storage, not for passwords)
     */
    fun hash(input: String): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        val hashBytes = digest.digest(input.toByteArray())
        return hashBytes.joinToString("") { "%02x".format(it) }
    }
    
    /**
     * Generate random token
     */
    fun generateToken(length: Int = 32): String {
        val chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        return (1..length)
            .map { chars.random() }
            .joinToString("")
    }
    
    /**
     * Mask sensitive data (e.g., email)
     */
    fun maskEmail(email: String): String {
        val parts = email.split("@")
        if (parts.size != 2) return email
        
        val name = parts[0]
        val domain = parts[1]
        
        val maskedName = if (name.length > 2) {
            name.first() + "*".repeat(name.length - 2) + name.last()
        } else {
            "*".repeat(name.length)
        }
        
        return "$maskedName@$domain"
    }
    
    /**
     * Mask phone number
     */
    fun maskPhone(phone: String): String {
        return if (phone.length > 4) {
            "*".repeat(phone.length - 4) + phone.takeLast(4)
        } else {
            "*".repeat(phone.length)
        }
    }
}
