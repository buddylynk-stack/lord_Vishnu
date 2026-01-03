package com.orignal.buddylynk.security

import android.content.Context
import android.content.pm.ApplicationInfo
import android.os.Build
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import com.orignal.buddylynk.BuildConfig
import java.io.File
import java.security.KeyStore
import java.security.MessageDigest
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Security Manager for BuddyLynk
 * Handles root detection, tampering detection, and secure encryption
 * 
 * SECURITY: Uses Android Keystore for key storage (hardware-backed on supported devices)
 * SECURITY: Uses AES/GCM for authenticated encryption (NOT ECB which is insecure)
 */
object SecurityManager {
    
    private const val KEYSTORE_ALIAS = "BuddyLynkSecureKey"
    private const val ANDROID_KEYSTORE = "AndroidKeyStore"
    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private const val GCM_IV_LENGTH = 12
    private const val GCM_TAG_LENGTH = 128
    
    /**
     * Check if device is rooted
     */
    fun isDeviceRooted(): Boolean {
        return checkRootMethod1() || checkRootMethod2() || checkRootMethod3()
    }
    
    private fun checkRootMethod1(): Boolean {
        val buildTags = Build.TAGS
        return buildTags != null && buildTags.contains("test-keys")
    }
    
    private fun checkRootMethod2(): Boolean {
        val paths = arrayOf(
            "/system/app/Superuser.apk",
            "/sbin/su",
            "/system/bin/su",
            "/system/xbin/su",
            "/data/local/xbin/su",
            "/data/local/bin/su",
            "/system/sd/xbin/su",
            "/system/bin/failsafe/su",
            "/data/local/su",
            "/su/bin/su"
        )
        
        for (path in paths) {
            if (File(path).exists()) return true
        }
        return false
    }
    
    private fun checkRootMethod3(): Boolean {
        var process: Process? = null
        return try {
            process = Runtime.getRuntime().exec(arrayOf("/system/xbin/which", "su"))
            val bufferedReader = process.inputStream.bufferedReader()
            bufferedReader.readLine() != null
        } catch (t: Throwable) {
            false
        } finally {
            process?.destroy()
        }
    }
    
    /**
     * Check if app is debuggable (tampered)
     */
    fun isAppDebuggable(context: Context): Boolean {
        return (context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
    }
    
    /**
     * Get or create encryption key from Android Keystore
     * Key is hardware-backed on devices with secure hardware
     */
    private fun getOrCreateSecretKey(): SecretKey {
        val keyStore = KeyStore.getInstance(ANDROID_KEYSTORE)
        keyStore.load(null)
        
        // Return existing key if available
        keyStore.getKey(KEYSTORE_ALIAS, null)?.let {
            return it as SecretKey
        }
        
        // Generate new key in Keystore
        val keyGenerator = KeyGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_AES,
            ANDROID_KEYSTORE
        )
        
        val keyGenSpec = KeyGenParameterSpec.Builder(
            KEYSTORE_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        )
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .setKeySize(256)
            .setUserAuthenticationRequired(false) // Set true for biometric protection
            .build()
        
        keyGenerator.init(keyGenSpec)
        return keyGenerator.generateKey()
    }
    
    /**
     * Encrypt sensitive data using AES/GCM with Android Keystore key
     * Output format: IV (12 bytes) + Ciphertext
     */
    fun encrypt(data: String): String {
        return try {
            val secretKey = getOrCreateSecretKey()
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.ENCRYPT_MODE, secretKey)
            
            val iv = cipher.iv
            val encrypted = cipher.doFinal(data.toByteArray(Charsets.UTF_8))
            
            // Prepend IV to ciphertext
            val combined = ByteArray(iv.size + encrypted.size)
            System.arraycopy(iv, 0, combined, 0, iv.size)
            System.arraycopy(encrypted, 0, combined, iv.size, encrypted.size)
            
            Base64.encodeToString(combined, Base64.NO_WRAP)
        } catch (e: Exception) {
            // Log only in debug builds
            if (BuildConfig.DEBUG) {
                android.util.Log.e("SecurityManager", "Encryption failed", e)
            }
            "" // Return empty string on failure - caller should handle
        }
    }
    
    /**
     * Decrypt sensitive data using AES/GCM with Android Keystore key
     * Input format: Base64(IV + Ciphertext)
     */
    fun decrypt(encryptedData: String): String {
        return try {
            if (encryptedData.isEmpty()) return ""
            
            val combined = Base64.decode(encryptedData, Base64.NO_WRAP)
            if (combined.size < GCM_IV_LENGTH + 16) return "" // Minimum: IV + 1 block
            
            // Extract IV and ciphertext
            val iv = combined.copyOfRange(0, GCM_IV_LENGTH)
            val ciphertext = combined.copyOfRange(GCM_IV_LENGTH, combined.size)
            
            val secretKey = getOrCreateSecretKey()
            val cipher = Cipher.getInstance(TRANSFORMATION)
            val gcmSpec = GCMParameterSpec(GCM_TAG_LENGTH, iv)
            cipher.init(Cipher.DECRYPT_MODE, secretKey, gcmSpec)
            
            val decrypted = cipher.doFinal(ciphertext)
            String(decrypted, Charsets.UTF_8)
        } catch (e: Exception) {
            if (BuildConfig.DEBUG) {
                android.util.Log.e("SecurityManager", "Decryption failed", e)
            }
            "" // Return empty string on failure
        }
    }
    
    /**
     * Generate secure hash for integrity checking
     */
    fun generateHash(input: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val hash = digest.digest(input.toByteArray())
        return Base64.encodeToString(hash, Base64.NO_WRAP)
    }
    
    /**
     * Verify app signature to detect tampering
     * Call this at app startup
     */
    fun verifyAppSignature(context: Context, expectedSignatureHash: String): Boolean {
        return try {
            val packageInfo = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                context.packageManager.getPackageInfo(
                    context.packageName,
                    android.content.pm.PackageManager.GET_SIGNING_CERTIFICATES
                )
            } else {
                @Suppress("DEPRECATION")
                context.packageManager.getPackageInfo(
                    context.packageName,
                    android.content.pm.PackageManager.GET_SIGNATURES
                )
            }
            
            val signatures = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                packageInfo.signingInfo?.apkContentsSigners
            } else {
                @Suppress("DEPRECATION")
                packageInfo.signatures
            }
            
            signatures?.any { signature ->
                val signatureHash = generateHash(signature.toByteArray().toString())
                signatureHash == expectedSignatureHash
            } ?: false
        } catch (e: Exception) {
            false
        }
    }
}