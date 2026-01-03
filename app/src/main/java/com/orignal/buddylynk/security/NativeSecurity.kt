package com.orignal.buddylynk.security

import android.util.Log

/**
 * Native Security - JNI wrapper for C++ security checks
 * 
 * These checks run in native code which is MUCH harder to bypass than Kotlin:
 * - No Java bytecode to decompile
 * - Xposed cannot hook native functions
 * - Frida hooking is more complex for native
 * - Machine code is harder to reverse engineer
 * 
 * IMPORTANT: This requires the NDK and CMake to be configured in build.gradle
 */
object NativeSecurity {
    
    private const val TAG = "NativeSecurity"
    private var isLibraryLoaded = false
    
    init {
        try {
            System.loadLibrary("buddylynk_security")
            isLibraryLoaded = true
            Log.d(TAG, "Native security library loaded successfully")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native security library: ${e.message}")
            isLibraryLoaded = false
        }
    }
    
    // Native method declarations
    private external fun isRootedNative(): Boolean
    private external fun isFridaDetectedNative(): Boolean
    private external fun isEmulatorNative(): Boolean
    private external fun isDebuggerAttachedNative(): Boolean
    private external fun isMemoryTamperedNative(): Boolean
    private external fun getSecurityStatusNative(): Int
    
    // Renamed native methods to match JNI naming convention
    @JvmStatic
    external fun isRooted(): Boolean
    
    @JvmStatic
    external fun isFridaDetected(): Boolean
    
    @JvmStatic
    external fun isEmulator(): Boolean
    
    @JvmStatic
    external fun isDebuggerAttached(): Boolean
    
    @JvmStatic
    external fun isMemoryTampered(): Boolean
    
    @JvmStatic
    external fun getSecurityStatus(): Int
    
    /**
     * Threat flags from getSecurityStatus()
     */
    object ThreatFlags {
        const val ROOT = 1 shl 0        // 1
        const val FRIDA = 1 shl 1       // 2
        const val EMULATOR = 1 shl 2    // 4
        const val DEBUGGER = 1 shl 3    // 8
        const val MEMORY = 1 shl 4      // 16
    }
    
    /**
     * Check if native library is available
     */
    fun isAvailable(): Boolean = isLibraryLoaded
    
    /**
     * Get comprehensive security check result using native code
     * Falls back to Kotlin checks if native library not loaded
     */
    fun performNativeSecurityCheck(): NativeSecurityResult {
        if (!isLibraryLoaded) {
            Log.w(TAG, "Native library not loaded, using Kotlin fallback")
            return performKotlinFallbackCheck()
        }
        
        return try {
            val status = getSecurityStatus()
            val threats = mutableListOf<String>()
            
            if (status and ThreatFlags.ROOT != 0) threats.add("ROOT")
            if (status and ThreatFlags.FRIDA != 0) threats.add("FRIDA")
            if (status and ThreatFlags.EMULATOR != 0) threats.add("EMULATOR")
            if (status and ThreatFlags.DEBUGGER != 0) threats.add("DEBUGGER")
            if (status and ThreatFlags.MEMORY != 0) threats.add("MEMORY_TAMPERING")
            
            NativeSecurityResult(
                isSecure = threats.isEmpty(),
                threats = threats,
                statusCode = status,
                isNativeCheck = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "Native security check failed", e)
            performKotlinFallbackCheck()
        }
    }
    
    /**
     * Fallback to Kotlin-based checks if native fails
     */
    private fun performKotlinFallbackCheck(): NativeSecurityResult {
        val threats = mutableListOf<String>()
        
        if (AntiTamper.isDebuggerConnected()) threats.add("DEBUGGER")
        if (AntiTamper.isEmulator()) threats.add("EMULATOR")
        
        return NativeSecurityResult(
            isSecure = threats.isEmpty(),
            threats = threats,
            statusCode = 0,
            isNativeCheck = false
        )
    }
}

data class NativeSecurityResult(
    val isSecure: Boolean,
    val threats: List<String>,
    val statusCode: Int,
    val isNativeCheck: Boolean
)
