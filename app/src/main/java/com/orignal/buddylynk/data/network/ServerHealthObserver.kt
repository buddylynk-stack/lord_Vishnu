package com.orignal.buddylynk.data.network

import android.util.Log
import com.orignal.buddylynk.data.api.ApiConfig
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.net.HttpURLConnection
import java.net.URL

/**
 * Server Health Observer
 * Monitors server availability and emits status changes
 * Checks if the server is responding to health checks
 */
object ServerHealthObserver {
    
    private const val TAG = "ServerHealthObserver"
    private const val HEALTH_CHECK_INTERVAL = 30_000L // Check every 30 seconds
    private const val CONNECTION_TIMEOUT = 5_000 // 5 second timeout for fast detection
    private const val READ_TIMEOUT = 5_000 // 5 second read timeout
    
    // Health check endpoint - simple ping to verify server is alive
    // Uses /health endpoint which is lightweight and fast
    private val HEALTH_CHECK_URL = "${ApiConfig.BASE_URL}/health"
    
    // Shared flow for server status
    private val _serverStatus: MutableStateFlow<ServerStatus> = MutableStateFlow(ServerStatus.Unknown)
    val serverStatus: StateFlow<ServerStatus> = _serverStatus.asStateFlow()
    
    // Track last known status
    private var lastCheckTime = 0L
    private var isChecking = false
    
    /**
     * Check server health once
     * Returns true if server is responding
     */
    suspend fun checkServerHealth(): Boolean {
        return withContext(Dispatchers.IO) {
            try {
                isChecking = true
                Log.d(TAG, "Checking server health: $HEALTH_CHECK_URL")
                
                val url = URL(HEALTH_CHECK_URL)
                val connection = url.openConnection() as HttpURLConnection
                connection.apply {
                    requestMethod = "GET" // Use GET for reliable detection
                    connectTimeout = 5000 // 5 second timeout for faster detection
                    readTimeout = 5000
                    setRequestProperty("Accept", "application/json")
                    instanceFollowRedirects = true
                }
                
                val responseCode = connection.responseCode
                Log.d(TAG, "Server response code: $responseCode")
                connection.disconnect()
                
                val isHealthy = responseCode in 200..399
                
                _serverStatus.value = if (isHealthy) {
                    Log.d(TAG, "Server is healthy (response: $responseCode)")
                    ServerStatus.Online
                } else {
                    Log.w(TAG, "Server returned error: $responseCode")
                    ServerStatus.Error(responseCode)
                }
                
                lastCheckTime = System.currentTimeMillis()
                isChecking = false
                isHealthy
                
            } catch (e: java.net.SocketTimeoutException) {
                Log.e(TAG, "Server timeout: ${e.message}")
                _serverStatus.value = ServerStatus.Offline("Connection timed out")
                lastCheckTime = System.currentTimeMillis()
                isChecking = false
                false
            } catch (e: java.net.ConnectException) {
                Log.e(TAG, "Server connection refused: ${e.message}")
                _serverStatus.value = ServerStatus.Offline("Connection refused")
                lastCheckTime = System.currentTimeMillis()
                isChecking = false
                false
            } catch (e: Exception) {
                Log.e(TAG, "Server health check failed: ${e.message}")
                _serverStatus.value = ServerStatus.Offline(e.message ?: "Connection failed")
                lastCheckTime = System.currentTimeMillis()
                isChecking = false
                false
            }
        }
    }
    
    /**
     * Quick check - returns cached status if recent, otherwise checks
     */
    suspend fun isServerOnline(): Boolean {
        val timeSinceLastCheck = System.currentTimeMillis() - lastCheckTime
        
        // Use cached result if checked within last 10 seconds
        if (timeSinceLastCheck < 10_000 && _serverStatus.value != ServerStatus.Unknown) {
            return _serverStatus.value == ServerStatus.Online
        }
        
        return checkServerHealth()
    }
    
    /**
     * Observe server status with periodic health checks
     */
    fun observeServerHealth(): Flow<ServerStatus> = flow {
        // Emit initial check
        checkServerHealth()
        emit(_serverStatus.value)
        
        // Periodic checks
        while (currentCoroutineContext().isActive) {
            delay(HEALTH_CHECK_INTERVAL)
            checkServerHealth()
            emit(_serverStatus.value)
        }
    }.distinctUntilChanged()
    
    /**
     * Force a health check now (called after retry button)
     */
    suspend fun forceCheck(): Boolean {
        return checkServerHealth()
    }
    
    /**
     * Reset status to unknown (useful when app goes to background)
     */
    fun reset() {
        _serverStatus.value = ServerStatus.Unknown
        lastCheckTime = 0L
    }
}

/**
 * Sealed class for server status
 */
sealed class ServerStatus {
    object Unknown : ServerStatus()
    object Online : ServerStatus()
    data class Offline(val reason: String) : ServerStatus()
    data class Error(val code: Int) : ServerStatus()
    
    fun isOnline(): Boolean = this == Online
    fun isOffline(): Boolean = this is Offline || this is Error
}
