package com.orignal.buddylynk.data.api

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * BehaviorTracker - Sends user behavior data to MindFlow Algorithm
 * Tracks: likes, shares, comments, saves, views for personalized recommendations
 */
object BehaviorTracker {
    
    private const val TAG = "BehaviorTracker"
    private val BEHAVIOR_API_URL = "${ApiConfig.API_URL}/behavior/track"
    
    // Action types matching backend
    enum class ActionType(val value: Int) {
        VIEW(0),
        LIKE(1),
        SHARE(2),
        COMMENT(3),
        SAVE(4),
        SKIP(5),
        UNLIKE(6),
        UNSAVE(7)
    }
    
    /**
     * Track user behavior asynchronously (fire and forget)
     */
    suspend fun track(
        userId: String,
        contentId: String,
        contentOwnerId: String,
        action: ActionType,
        watchTime: Int = 0,
        isNSFW: Boolean = false
    ) = withContext(Dispatchers.IO) {
        try {
            val url = URL(BEHAVIOR_API_URL)
            val conn = url.openConnection() as HttpURLConnection
            
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true
            conn.connectTimeout = 5000
            conn.readTimeout = 5000
            
            val metadata = JSONObject().apply {
                put("isNSFW", isNSFW)
            }
            
            val body = JSONObject().apply {
                put("userId", userId)
                put("contentId", contentId)
                put("contentOwnerId", contentOwnerId)
                put("actionType", action.value)
                put("watchTime", watchTime)
                put("metadata", metadata)
            }
            
            conn.outputStream.use { os ->
                os.write(body.toString().toByteArray())
            }
            
            val responseCode = conn.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                Log.d(TAG, "Tracked: ${action.name} on $contentId by $userId")
            } else {
                Log.w(TAG, "Track failed: $responseCode")
            }
            
            conn.disconnect()
            
        } catch (e: Exception) {
            // Don't crash the app if tracking fails
            Log.e(TAG, "Track error: ${e.message}")
        }
    }
    
    /**
     * Convenience methods for common actions
     */
    suspend fun trackView(userId: String, contentId: String, contentOwnerId: String, watchTime: Int = 0, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.VIEW, watchTime, isNSFW)
    }
    
    suspend fun trackLike(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.LIKE, 0, isNSFW)
    }
    
    suspend fun trackUnlike(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.UNLIKE, 0, isNSFW)
    }
    
    suspend fun trackShare(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.SHARE, 0, isNSFW)
    }
    
    suspend fun trackComment(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.COMMENT, 0, isNSFW)
    }
    
    suspend fun trackSave(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.SAVE, 0, isNSFW)
    }
    
    suspend fun trackUnsave(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.UNSAVE, 0, isNSFW)
    }
    
    suspend fun trackSkip(userId: String, contentId: String, contentOwnerId: String, isNSFW: Boolean = false) {
        track(userId, contentId, contentOwnerId, ActionType.SKIP, 0, isNSFW)
    }
}
