package com.orignal.buddylynk.data.auth

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import com.orignal.buddylynk.data.api.ApiService
import com.orignal.buddylynk.data.model.User
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Authentication Manager - Handles login state and session persistence
 * SECURITY: Uses EncryptedSharedPreferences for JWT token storage
 */
object AuthManager {
    
    private const val PREFS_NAME = "buddylynk_auth"
    private const val SECURE_PREFS_NAME = "buddylynk_secure_auth"
    private const val KEY_USER_ID = "user_id"
    private const val KEY_USERNAME = "username"
    private const val KEY_EMAIL = "email"
    private const val KEY_AVATAR = "avatar"
    private const val KEY_BANNER = "banner"
    private const val KEY_BIO = "bio"
    private const val KEY_IS_LOGGED_IN = "is_logged_in"
    private const val KEY_JWT_TOKEN = "jwt_token"
    
    private var prefs: SharedPreferences? = null
    private var securePrefs: SharedPreferences? = null  // For sensitive data like JWT
    
    private val _currentUser = MutableStateFlow<User?>(null)
    val currentUser: StateFlow<User?> = _currentUser.asStateFlow()
    
    private val _isLoggedIn = MutableStateFlow(false)
    val isLoggedIn: StateFlow<Boolean> = _isLoggedIn.asStateFlow()
    
    /**
     * Initialize AuthManager with context - call from Application or MainActivity
     */
    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        
        // SECURITY: Use EncryptedSharedPreferences for JWT token
        try {
            val masterKey = MasterKey.Builder(context)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build()
            
            securePrefs = EncryptedSharedPreferences.create(
                context,
                SECURE_PREFS_NAME,
                masterKey,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
            )
        } catch (e: Exception) {
            // Fallback to regular prefs if encryption fails (older devices)
            securePrefs = context.getSharedPreferences(SECURE_PREFS_NAME, Context.MODE_PRIVATE)
        }
        
        loadSavedSession()
    }
    
    /**
     * Load saved session on app start
     */
    private fun loadSavedSession() {
        prefs?.let { p ->
            val isLoggedIn = p.getBoolean(KEY_IS_LOGGED_IN, false)
            if (isLoggedIn) {
                val userId = p.getString(KEY_USER_ID, null)
                val username = p.getString(KEY_USERNAME, null)
                val email = p.getString(KEY_EMAIL, null)
                val avatar = p.getString(KEY_AVATAR, null)
                val banner = p.getString(KEY_BANNER, null)
                val bio = p.getString(KEY_BIO, null)
                val jwtToken = securePrefs?.getString(KEY_JWT_TOKEN, null)
                
                if (userId != null && username != null && email != null) {
                    _currentUser.value = User(
                        userId = userId,
                        username = username,
                        email = email,
                        avatar = avatar,
                        banner = banner,
                        bio = bio
                    )
                    _isLoggedIn.value = true
                    
                    // Set JWT token in ApiService for authenticated calls
                    jwtToken?.let { ApiService.setAuthToken(it) }
                }
            }
        }
    }
    
    /**
     * Save login session (for API login with JWT token)
     */
    fun login(user: User, jwtToken: String? = null) {
        prefs?.edit()?.apply {
            putBoolean(KEY_IS_LOGGED_IN, true)
            putString(KEY_USER_ID, user.userId)
            putString(KEY_USERNAME, user.username)
            putString(KEY_EMAIL, user.email)
            putString(KEY_AVATAR, user.avatar)
            putString(KEY_BANNER, user.banner)
            putString(KEY_BIO, user.bio)
            apply()
        }
        
        // SECURITY: Store JWT token in encrypted storage
        jwtToken?.let { token ->
            securePrefs?.edit()?.putString(KEY_JWT_TOKEN, token)?.apply()
            ApiService.setAuthToken(token)
        }
        
        _currentUser.value = user
        _isLoggedIn.value = true
    }
    
    /**
     * Logout and clear session
     */
    fun logout() {
        prefs?.edit()?.clear()?.apply()
        securePrefs?.edit()?.clear()?.apply()  // Clear encrypted JWT token
        _currentUser.value = null
        _isLoggedIn.value = false
        ApiService.setAuthToken(null)
    }
    
    /**
     * Check if user is logged in
     */
    fun isUserLoggedIn(): Boolean = _isLoggedIn.value
    /**
     * Get current user ID
     */
    fun getCurrentUserId(): String? = _currentUser.value?.userId
    
    /**
     * Update current user (for profile edits)
     */
    fun updateCurrentUser(user: User) {
        prefs?.edit()?.apply {
            putString(KEY_USER_ID, user.userId)
            putString(KEY_USERNAME, user.username)
            putString(KEY_EMAIL, user.email)
            putString(KEY_AVATAR, user.avatar)
            putString(KEY_BANNER, user.banner)
            putString(KEY_BIO, user.bio)
            apply()
        }
        _currentUser.value = user
    }
    
    /**
     * Refresh current user data from backend API (to get latest followers/following counts)
     */
    suspend fun refreshCurrentUser() {
        val userId = _currentUser.value?.userId ?: return
        try {
            val freshUser = com.orignal.buddylynk.data.repository.BackendRepository.getUser(userId)
            if (freshUser != null) {
                // Keep existing data but update counts
                _currentUser.value = freshUser
            }
        } catch (e: Exception) {
            android.util.Log.e("AuthManager", "Failed to refresh user: ${e.message}")
        }
    }
}
