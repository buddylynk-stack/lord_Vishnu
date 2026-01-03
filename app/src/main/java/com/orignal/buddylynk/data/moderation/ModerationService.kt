package com.orignal.buddylynk.data.moderation

import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.model.User
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

/**
 * Moderation Service - Block/Report users and content
 * Uses BackendRepository for API-based operations
 */
object ModerationService {
    
    private val _blockedUsers = MutableStateFlow<Set<String>>(emptySet())
    val blockedUsers: StateFlow<Set<String>> = _blockedUsers.asStateFlow()
    
    private val mutedUsers = mutableSetOf<String>()
    
    /**
     * Initialize with current user's blocked list from API
     */
    suspend fun init() = withContext(Dispatchers.IO) {
        val userId = AuthManager.getCurrentUserId() ?: return@withContext
        try {
            // Load blocked users from Backend API
            val blockedList = BackendRepository.getBlockedUsers()
            _blockedUsers.value = blockedList.toSet()
            android.util.Log.d("ModerationService", "Loaded ${blockedList.size} blocked users from API")
        } catch (e: Exception) {
            android.util.Log.e("ModerationService", "Error loading blocked users: ${e.message}")
        }
    }
    
    /**
     * Get list of blocked users with full user info
     */
    suspend fun getBlockedUsers(): List<User> = withContext(Dispatchers.IO) {
        val userId = AuthManager.getCurrentUserId() ?: return@withContext emptyList()
        
        // Always reload blocked IDs from API to ensure we have latest data
        try {
            val blockedList = BackendRepository.getBlockedUsers()
            android.util.Log.d("ModerationService", "Loaded ${blockedList.size} blocked user IDs from API: $blockedList")
            _blockedUsers.value = blockedList.toSet()
        } catch (e: Exception) {
            android.util.Log.e("ModerationService", "Failed to load blocked users from API: ${e.message}")
        }
        
        // Get full user info for each blocked ID
        val users = _blockedUsers.value.mapNotNull { blockedId ->
            try {
                val user = BackendRepository.getUser(blockedId)
                android.util.Log.d("ModerationService", "Got user info for $blockedId: ${user?.username}")
                user ?: User(userId = blockedId, username = "User $blockedId", email = "")
            } catch (e: Exception) {
                android.util.Log.e("ModerationService", "Failed to get user $blockedId: ${e.message}")
                // Return placeholder if user not found
                User(userId = blockedId, username = "User $blockedId", email = "")
            }
        }
        android.util.Log.d("ModerationService", "Returning ${users.size} blocked users for display")
        users
    }
    
    /**
     * Block a user via API
     */
    suspend fun blockUser(targetUserId: String): Result<Unit> = withContext(Dispatchers.IO) {
        val userId = AuthManager.getCurrentUserId() 
            ?: return@withContext Result.failure(Exception("Not logged in"))
        
        try {
            // Update local state first for instant UI feedback
            _blockedUsers.value = _blockedUsers.value + targetUserId
            
            // Persist via API
            val success = BackendRepository.blockUser(targetUserId)
            
            if (success) {
                android.util.Log.d("ModerationService", "Blocked user $targetUserId via API")
                Result.success(Unit)
            } else {
                // Rollback local state
                _blockedUsers.value = _blockedUsers.value - targetUserId
                Result.failure(Exception("API block failed"))
            }
        } catch (e: Exception) {
            // Rollback local state
            _blockedUsers.value = _blockedUsers.value - targetUserId
            Result.failure(e)
        }
    }
    
    /**
     * Unblock a user via API
     */
    suspend fun unblockUser(targetUserId: String): Result<Unit> = withContext(Dispatchers.IO) {
        val userId = AuthManager.getCurrentUserId() 
            ?: return@withContext Result.failure(Exception("Not logged in"))
        
        try {
            // Update local state first
            _blockedUsers.value = _blockedUsers.value - targetUserId
            
            // Persist via API
            val success = BackendRepository.unblockUser(targetUserId)
            
            if (success) {
                android.util.Log.d("ModerationService", "Unblocked user $targetUserId via API")
                Result.success(Unit)
            } else {
                // Rollback
                _blockedUsers.value = _blockedUsers.value + targetUserId
                Result.failure(Exception("API unblock failed"))
            }
        } catch (e: Exception) {
            _blockedUsers.value = _blockedUsers.value + targetUserId
            Result.failure(e)
        }
    }
    
    /**
     * Check if a user is blocked locally
     */
    fun isBlocked(userId: String): Boolean = userId in _blockedUsers.value
    
    /**
     * Check if current user can interact with target user
     */
    fun canInteract(targetUserId: String): Boolean = !isBlocked(targetUserId)
    
    /**
     * Refresh blocked users from API
     */
    suspend fun refresh() {
        init()
    }
}
