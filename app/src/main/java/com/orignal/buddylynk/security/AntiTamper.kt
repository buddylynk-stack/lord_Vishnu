package com.orignal.buddylynk.security

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Debug
import java.io.BufferedReader
import java.io.File
import java.io.FileReader

/**
 * Advanced Anti-Tampering and Anti-Reverse Engineering
 * 
 * Detects:
 * - Frida hooking framework
 * - Xposed framework
 * - Debugger attachment
 * - Emulator environments
 * - Memory tampering tools
 */
object AntiTamper {
    
    // Known hooking tool signatures
    private val DANGEROUS_APPS = listOf(
        "de.robv.android.xposed.installer",
        "com.saurik.substrate",
        "com.topjohnwu.magisk",
        "eu.chainfire.supersu",
        "com.koushikdutta.superuser",
        "com.thirdparty.superuser",
        "com.noshufou.android.su",
        "com.yellowes.su",
        "com.kingroot.kinguser",
        "com.kingo.root",
        "com.smedialink.oneclean",
        "com.zhiqupk.root.global",
        "com.alephzain.framaroot"
    )
    
    private val DANGEROUS_PACKAGES = listOf(
        "com.chelpus.lackypatch",
        "com.ramdroid.appquarantine",
        "com.android.vending.billing.InAppBillingService.COIN",
        "org.lsposed.manager"
    )
    
    /**
     * Main security check - call at app startup
     */
    fun performSecurityCheck(context: Context): SecurityCheckResult {
        val threats = mutableListOf<String>()
        
        if (isDebuggerConnected()) {
            threats.add("DEBUGGER")
        }
        
        if (isFridaDetected()) {
            threats.add("FRIDA")
        }
        
        if (isXposedPresent(context)) {
            threats.add("XPOSED")
        }
        
        if (isEmulator()) {
            threats.add("EMULATOR")
        }
        
        if (hasDangerousApps(context)) {
            threats.add("DANGEROUS_APPS")
        }
        
        if (isMemoryTampered()) {
            threats.add("MEMORY_TAMPERING")
        }
        
        return SecurityCheckResult(
            isSecure = threats.isEmpty(),
            threats = threats
        )
    }
    
    /**
     * Check if debugger is attached
     */
    fun isDebuggerConnected(): Boolean {
        return Debug.isDebuggerConnected() || Debug.waitingForDebugger()
    }
    
    /**
     * Detect Frida hooking framework
     * Frida typically opens ports 27042-27047 and injects libraries
     */
    fun isFridaDetected(): Boolean {
        // Check for Frida server ports
        if (isFridaPortOpen()) return true
        
        // Check for Frida libraries in memory
        if (hasFridaLibraries()) return true
        
        // Check for Frida named threads
        if (hasFridaThreads()) return true
        
        return false
    }
    
    private fun isFridaPortOpen(): Boolean {
        val fridaPorts = listOf(27042, 27043, 27044, 27045, 27046, 27047)
        
        for (port in fridaPorts) {
            try {
                val socket = java.net.Socket()
                socket.connect(java.net.InetSocketAddress("127.0.0.1", port), 100)
                socket.close()
                return true // Port is open - Frida might be running
            } catch (e: Exception) {
                // Port not open, continue
            }
        }
        return false
    }
    
    private fun hasFridaLibraries(): Boolean {
        try {
            val mapsFile = File("/proc/self/maps")
            if (mapsFile.exists()) {
                BufferedReader(FileReader(mapsFile)).use { reader ->
                    var line: String?
                    while (reader.readLine().also { line = it } != null) {
                        line?.let {
                            if (it.contains("frida") || 
                                it.contains("gadget") ||
                                it.contains("linjector")) {
                                return true
                            }
                        }
                    }
                }
            }
        } catch (e: Exception) {
            // Ignore
        }
        return false
    }
    
    private fun hasFridaThreads(): Boolean {
        try {
            val taskDir = File("/proc/self/task")
            if (taskDir.exists() && taskDir.isDirectory) {
                taskDir.listFiles()?.forEach { task ->
                    val commFile = File(task, "comm")
                    if (commFile.exists()) {
                        val comm = commFile.readText().trim()
                        if (comm.contains("frida") || 
                            comm.contains("gum-js-loop") ||
                            comm.contains("gmain")) {
                            return true
                        }
                    }
                }
            }
        } catch (e: Exception) {
            // Ignore
        }
        return false
    }
    
    /**
     * Detect Xposed Framework
     */
    fun isXposedPresent(context: Context): Boolean {
        // Check for Xposed installer
        val xposedPackages = listOf(
            "de.robv.android.xposed.installer",
            "io.va.exposed",
            "org.meowcat.edxposed.manager",
            "org.lsposed.manager"
        )
        
        for (pkg in xposedPackages) {
            try {
                context.packageManager.getPackageInfo(pkg, 0)
                return true
            } catch (e: PackageManager.NameNotFoundException) {
                // Not found
            }
        }
        
        // Check for Xposed in stack trace
        try {
            throw Exception("Xposed check")
        } catch (e: Exception) {
            for (element in e.stackTrace) {
                if (element.className.contains("xposed") ||
                    element.className.contains("Xposed")) {
                    return true
                }
            }
        }
        
        // Check for Xposed hooks in system
        try {
            val xposedBridge = Class.forName("de.robv.android.xposed.XposedBridge")
            return true
        } catch (e: ClassNotFoundException) {
            // Not found
        }
        
        return false
    }
    
    /**
     * Detect if running on emulator
     */
    fun isEmulator(): Boolean {
        return (Build.FINGERPRINT.startsWith("generic") ||
                Build.FINGERPRINT.startsWith("unknown") ||
                Build.MODEL.contains("google_sdk") ||
                Build.MODEL.contains("Emulator") ||
                Build.MODEL.contains("Android SDK built for x86") ||
                Build.MANUFACTURER.contains("Genymotion") ||
                Build.BRAND.startsWith("generic") && Build.DEVICE.startsWith("generic") ||
                "google_sdk" == Build.PRODUCT ||
                Build.HARDWARE.contains("goldfish") ||
                Build.HARDWARE.contains("ranchu") ||
                Build.PRODUCT.contains("sdk") ||
                Build.PRODUCT.contains("emulator") ||
                Build.PRODUCT.contains("simulator"))
    }
    
    /**
     * Check for dangerous/hacking apps
     */
    fun hasDangerousApps(context: Context): Boolean {
        val allApps = DANGEROUS_APPS + DANGEROUS_PACKAGES
        
        for (pkg in allApps) {
            try {
                context.packageManager.getPackageInfo(pkg, 0)
                return true
            } catch (e: PackageManager.NameNotFoundException) {
                // Not found
            }
        }
        return false
    }
    
    /**
     * Basic memory tampering check
     */
    fun isMemoryTampered(): Boolean {
        try {
            // Check for common memory editors
            val suspiciousFiles = listOf(
                "/data/local/tmp/re.frida.server",
                "/data/local/tmp/frida-server",
                "/data/data/com.topjohnwu.magisk"
            )
            
            for (path in suspiciousFiles) {
                if (File(path).exists()) {
                    return true
                }
            }
        } catch (e: Exception) {
            // Ignore
        }
        return false
    }
    
    /**
     * Check if app was repackaged
     */
    fun isAppRepackaged(context: Context, originalSignature: String): Boolean {
        return try {
            val packageInfo = context.packageManager.getPackageInfo(
                context.packageName,
                PackageManager.GET_SIGNATURES
            )
            
            @Suppress("DEPRECATION")
            val signatures = packageInfo.signatures
            
            if (signatures.isNullOrEmpty()) {
                true // No signature = definitely tampered
            } else {
                val currentSig = signatures[0].toCharsString()
                currentSig != originalSignature
            }
        } catch (e: Exception) {
            true // Error getting signature = assume tampered
        }
    }
}

data class SecurityCheckResult(
    val isSecure: Boolean,
    val threats: List<String>
)
