package com.orignal.buddylynk.data.redis

import com.orignal.buddylynk.data.auth.AuthManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * OnlineStatusManager - Tracks user presence in real-time
 * Uses Redis for fast status updates with auto-expire
 */
object OnlineStatusManager {
    
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    private val _isOnline = MutableStateFlow(false)
    val isOnline: StateFlow<Boolean> = _isOnline.asStateFlow()
    
    private val _onlineFriends = MutableStateFlow<Set<String>>(emptySet())
    val onlineFriends: StateFlow<Set<String>> = _onlineFriends.asStateFlow()
    
    private var heartbeatJob: Job? = null
    
    /**
     * Start tracking online status
     * Call when user logs in or app comes to foreground
     */
    fun goOnline() {
        val userId = AuthManager.currentUser.value?.userId ?: return
        
        scope.launch {
            // Set user as online in Redis
            RedisService.setUserOnline(userId)
            _isOnline.value = true
            
            // Start heartbeat to keep online status
            startHeartbeat(userId)
            
            // Get online friends
            refreshOnlineFriends()
        }
    }
    
    /**
     * Stop tracking online status
     * Call when user logs out or app goes to background
     */
    fun goOffline() {
        val userId = AuthManager.currentUser.value?.userId ?: return
        
        heartbeatJob?.cancel()
        
        scope.launch {
            RedisService.setUserOffline(userId)
            _isOnline.value = false
        }
    }
    
    /**
     * Heartbeat to keep online status active
     * Redis TTL will auto-expire if heartbeat stops
     */
    private fun startHeartbeat(userId: String) {
        heartbeatJob?.cancel()
        heartbeatJob = scope.launch {
            while (isActive) {
                // Refresh online status every 4 minutes (TTL is 5 min)
                delay(4 * 60 * 1000L)
                
                if (AuthManager.currentUser.value != null) {
                    RedisService.setUserOnline(userId)
                    refreshOnlineFriends()
                }
            }
        }
    }
    
    /**
     * Refresh list of online friends
     */
    fun refreshOnlineFriends() {
        scope.launch {
            val online = RedisService.getOnlineUsers()
            _onlineFriends.value = online
        }
    }
    
    /**
     * Check if specific user is online
     */
    suspend fun isUserOnline(userId: String): Boolean {
        return RedisService.isUserOnline(userId)
    }
    
    /**
     * Cleanup when app is destroyed
     */
    fun cleanup() {
        goOffline()
        scope.cancel()
    }
}
