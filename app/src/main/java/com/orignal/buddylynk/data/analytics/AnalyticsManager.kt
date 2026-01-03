package com.orignal.buddylynk.data.analytics

import android.content.Context
import android.os.Bundle
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID

/**
 * Analytics Manager - Track user events and engagement
 * 
 * Integrates with Firebase Analytics when available,
 * falls back to local storage for offline tracking
 */
object AnalyticsManager {
    
    private var isInitialized = false
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val pendingEvents = mutableListOf<AnalyticsEvent>()
    
    // Session tracking
    private var sessionId: String = ""
    private var sessionStartTime: Long = 0L
    private var currentUserId: String? = null
    
    /**
     * Initialize analytics
     */
    fun init(context: Context, userId: String? = null) {
        sessionId = UUID.randomUUID().toString()
        sessionStartTime = System.currentTimeMillis()
        currentUserId = userId
        isInitialized = true
        
        // Log session start
        logEvent(Event.SESSION_START)
    }
    
    /**
     * Set user ID for analytics
     */
    fun setUserId(userId: String) {
        currentUserId = userId
        // Would set Firebase user ID here
    }
    
    /**
     * Log an analytics event
     */
    fun logEvent(event: String, params: Map<String, Any> = emptyMap()) {
        scope.launch {
            val analyticsEvent = AnalyticsEvent(
                name = event,
                params = params.toMutableMap().apply {
                    put("session_id", sessionId)
                    put("timestamp", System.currentTimeMillis())
                    currentUserId?.let { put("user_id", it) }
                },
                timestamp = System.currentTimeMillis()
            )
            
            pendingEvents.add(analyticsEvent)
            
            // Process events (would send to Firebase/backend)
            processEvents()
        }
    }
    
    /**
     * Log screen view
     */
    fun logScreenView(screenName: String, screenClass: String? = null) {
        logEvent(Event.SCREEN_VIEW, mapOf(
            "screen_name" to screenName,
            "screen_class" to (screenClass ?: screenName)
        ))
    }
    
    /**
     * Log user action
     */
    fun logAction(action: String, targetId: String? = null, targetType: String? = null) {
        val params = mutableMapOf<String, Any>("action" to action)
        targetId?.let { params["target_id"] = it }
        targetType?.let { params["target_type"] = it }
        logEvent(Event.USER_ACTION, params)
    }
    
    /**
     * Log engagement metrics
     */
    fun logEngagement(type: String, duration: Long? = null) {
        val params = mutableMapOf<String, Any>("engagement_type" to type)
        duration?.let { params["duration_ms"] = it }
        logEvent(Event.ENGAGEMENT, params)
    }
    
    /**
     * Log error
     */
    fun logError(errorType: String, errorMessage: String, screen: String? = null) {
        logEvent(Event.ERROR, mapOf(
            "error_type" to errorType,
            "error_message" to errorMessage,
            "screen" to (screen ?: "unknown")
        ))
    }
    
    /**
     * Log content interaction
     */
    fun logContentInteraction(
        contentId: String,
        contentType: String,
        interactionType: String
    ) {
        logEvent(Event.CONTENT_INTERACTION, mapOf(
            "content_id" to contentId,
            "content_type" to contentType,
            "interaction_type" to interactionType
        ))
    }
    
    /**
     * Log social action
     */
    fun logSocialAction(action: String, targetUserId: String) {
        logEvent(Event.SOCIAL_ACTION, mapOf(
            "social_action" to action,
            "target_user_id" to targetUserId
        ))
    }
    
    /**
     * Log search
     */
    fun logSearch(query: String, resultsCount: Int) {
        logEvent(Event.SEARCH, mapOf(
            "search_query" to query,
            "results_count" to resultsCount
        ))
    }
    
    /**
     * Log content creation
     */
    fun logContentCreation(contentType: String, hasMedia: Boolean) {
        logEvent(Event.CONTENT_CREATED, mapOf(
            "content_type" to contentType,
            "has_media" to hasMedia
        ))
    }
    
    /**
     * End session
     */
    fun endSession() {
        val sessionDuration = System.currentTimeMillis() - sessionStartTime
        logEvent(Event.SESSION_END, mapOf(
            "session_duration_ms" to sessionDuration
        ))
    }
    
    /**
     * Process pending events (would send to backend)
     */
    private fun processEvents() {
        // In production, this would send to Firebase Analytics
        // For now, just log locally
        if (pendingEvents.size > 100) {
            pendingEvents.removeAll(pendingEvents.take(50))
        }
    }
    
    /**
     * Get session info
     */
    fun getSessionInfo(): Map<String, Any> = mapOf(
        "session_id" to sessionId,
        "start_time" to sessionStartTime,
        "duration_ms" to (System.currentTimeMillis() - sessionStartTime),
        "events_count" to pendingEvents.size
    )
    
    /**
     * Event names
     */
    object Event {
        const val SESSION_START = "session_start"
        const val SESSION_END = "session_end"
        const val SCREEN_VIEW = "screen_view"
        const val USER_ACTION = "user_action"
        const val ENGAGEMENT = "engagement"
        const val ERROR = "error"
        const val CONTENT_INTERACTION = "content_interaction"
        const val SOCIAL_ACTION = "social_action"
        const val SEARCH = "search"
        const val CONTENT_CREATED = "content_created"
        
        // Specific events
        const val POST_LIKED = "post_liked"
        const val POST_SHARED = "post_shared"
        const val POST_SAVED = "post_saved"
        const val POST_VIEWED = "post_viewed"
        const val PROFILE_VIEWED = "profile_viewed"
        const val USER_FOLLOWED = "user_followed"
        const val USER_UNFOLLOWED = "user_unfollowed"
        const val MESSAGE_SENT = "message_sent"
        const val STORY_VIEWED = "story_viewed"
        const val STORY_CREATED = "story_created"
        const val COMMENT_POSTED = "comment_posted"
    }
}

/**
 * Analytics event data class
 */
data class AnalyticsEvent(
    val name: String,
    val params: Map<String, Any>,
    val timestamp: Long
)

/**
 * User properties for analytics
 */
object UserProperties {
    const val USER_TYPE = "user_type"
    const val ACCOUNT_AGE_DAYS = "account_age_days"
    const val POSTS_COUNT = "posts_count"
    const val FOLLOWERS_COUNT = "followers_count"
    const val FOLLOWING_COUNT = "following_count"
    const val IS_VERIFIED = "is_verified"
    const val LAST_ACTIVE = "last_active"
}
