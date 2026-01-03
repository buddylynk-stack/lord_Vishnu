package com.orignal.buddylynk.security

import android.content.Context
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.play.core.integrity.IntegrityManagerFactory
import com.google.android.play.core.integrity.IntegrityTokenRequest
import com.orignal.buddylynk.BuildConfig
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Play Integrity API Integration
 * 
 * Verifies:
 * - App is genuine (not modified)
 * - App installed from Play Store
 * - Device is genuine (not rooted/emulated)
 * - Account is licensed
 * 
 * IMPORTANT: You need to:
 * 1. Enable Play Integrity API in Google Cloud Console
 * 2. Link your app in Play Console
 * 3. Set up server-side verification (recommended)
 */
object PlayIntegrityChecker {
    
    private const val TAG = "PlayIntegrityChecker"
    
    // Your cloud project number (get from Google Cloud Console)
    // IMPORTANT: Replace with your actual project number!
    private var cloudProjectNumber: Long = 0L
    
    /**
     * Initialize with your cloud project number
     * Call this in Application.onCreate()
     */
    fun init(projectNumber: Long) {
        cloudProjectNumber = projectNumber
    }
    
    /**
     * Request integrity token for verification
     * 
     * @param context Application context
     * @param nonce Unique nonce for this request (should be generated server-side)
     * @return IntegrityToken or null if failed
     */
    suspend fun requestIntegrityToken(
        context: Context,
        nonce: String
    ): String? {
        if (cloudProjectNumber == 0L) {
            if (BuildConfig.DEBUG) {
                Log.e(TAG, "Cloud project number not set! Call init() first.")
            }
            return null
        }
        
        return try {
            val integrityManager = IntegrityManagerFactory.create(context)
            
            val tokenResponse = integrityManager.requestIntegrityToken(
                IntegrityTokenRequest.builder()
                    .setCloudProjectNumber(cloudProjectNumber)
                    .setNonce(nonce)
                    .build()
            ).await()
            
            tokenResponse.token()
        } catch (e: Exception) {
            if (BuildConfig.DEBUG) {
                Log.e(TAG, "Failed to get integrity token", e)
            }
            null
        }
    }
    
    /**
     * Simple integrity check (client-side only - NOT recommended for production)
     * For production, send token to your server for verification
     */
    suspend fun performBasicIntegrityCheck(context: Context): IntegrityResult {
        val nonce = generateNonce()
        val token = requestIntegrityToken(context, nonce)
        
        return if (token != null) {
            IntegrityResult(
                success = true,
                token = token,
                message = "Integrity token obtained successfully"
            )
        } else {
            IntegrityResult(
                success = false,
                token = null,
                message = "Failed to obtain integrity token"
            )
        }
    }
    
    /**
     * Generate a cryptographic nonce
     * In production, this should come from your server!
     */
    private fun generateNonce(): String {
        val bytes = ByteArray(24)
        java.security.SecureRandom().nextBytes(bytes)
        return android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
    }
    
    /**
     * Extension to await Google Tasks
     */
    private suspend fun <T> Task<T>.await(): T {
        return suspendCancellableCoroutine { continuation ->
            addOnSuccessListener { result ->
                continuation.resume(result)
            }
            addOnFailureListener { exception ->
                continuation.resumeWithException(exception)
            }
            addOnCanceledListener {
                continuation.cancel()
            }
        }
    }
}

data class IntegrityResult(
    val success: Boolean,
    val token: String?,
    val message: String
)
