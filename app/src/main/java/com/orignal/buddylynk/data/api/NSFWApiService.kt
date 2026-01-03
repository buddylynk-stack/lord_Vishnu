package com.orignal.buddylynk.data.api

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * NSFW API Service - Connects to DynamoDB NSFW table
 * 
 * Manages admin-flagged 18+/sensitive content in real-time.
 * When admin marks a post as NSFW, app reflects it immediately.
 */
object NSFWApiService {
    
    private const val TAG = "NSFWApiService"
    private const val BASE_URL = "https://app.buddylynk.com/api" // Same as ApiConfig.BASE_URL/api
    
    // Cache of NSFW post IDs for quick lookup
    private val _nsfwPostIds = MutableStateFlow<Set<String>>(emptySet())
    val nsfwPostIds: StateFlow<Set<String>> = _nsfwPostIds.asStateFlow()
    
    // Last refresh timestamp
    private var lastRefreshTime: Long = 0
    private const val REFRESH_INTERVAL_MS = 30_000L // Refresh every 30 seconds
    
    /**
     * Fetch all NSFW post IDs from DynamoDB table
     * Called on app start and periodically refreshed
     */
    suspend fun fetchNSFWPosts(): Result<Set<String>> = withContext(Dispatchers.IO) {
        try {
            val url = URL("$BASE_URL/nsfw/posts")
            val connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "GET"
            connection.setRequestProperty("Content-Type", "application/json")
            connection.connectTimeout = 10000
            connection.readTimeout = 10000
            
            val responseCode = connection.responseCode
            if (responseCode == HttpURLConnection.HTTP_OK) {
                val response = connection.inputStream.bufferedReader().readText()
                val jsonObject = JSONObject(response)
                val postIds = mutableSetOf<String>()
                
                if (jsonObject.has("nsfwPosts")) {
                    val postsArray = jsonObject.getJSONArray("nsfwPosts")
                    for (i in 0 until postsArray.length()) {
                        val post = postsArray.getJSONObject(i)
                        if (post.has("postId")) {
                            postIds.add(post.getString("postId"))
                        }
                    }
                }
                
                // Update cache
                _nsfwPostIds.value = postIds
                lastRefreshTime = System.currentTimeMillis()
                
                Log.d(TAG, "Loaded ${postIds.size} NSFW posts from server")
                Result.success(postIds)
            } else {
                Log.e(TAG, "Failed to fetch NSFW posts: HTTP $responseCode")
                Result.failure(Exception("HTTP $responseCode"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching NSFW posts: ${e.message}")
            Result.failure(e)
        }
    }
    
    /**
     * Check if a specific post is flagged as NSFW
     */
    fun isPostNSFW(postId: String): Boolean {
        return _nsfwPostIds.value.contains(postId)
    }
    
    /**
     * Get NSFW status for multiple posts
     * Returns a map of postId -> isNSFW
     */
    fun getNSFWStatusForPosts(postIds: List<String>): Map<String, Boolean> {
        val nsfwSet = _nsfwPostIds.value
        return postIds.associateWith { nsfwSet.contains(it) }
    }
    
    /**
     * Refresh NSFW posts if cache is stale
     */
    suspend fun refreshIfNeeded() {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastRefreshTime > REFRESH_INTERVAL_MS) {
            fetchNSFWPosts()
        }
    }
    
    /**
     * Force refresh NSFW posts (called on pull-to-refresh)
     */
    suspend fun forceRefresh() {
        fetchNSFWPosts()
    }
    
    /**
     * Apply NSFW flags to a list of posts based on cached NSFW data
     * Used when loading posts to mark them as NSFW in real-time
     * IMPORTANT: Fetches fresh data if cache is empty (e.g., after app restart)
     */
    suspend fun applyNSFWFlags(posts: List<com.orignal.buddylynk.data.model.Post>): List<com.orignal.buddylynk.data.model.Post> {
        // CRITICAL: If cache is empty or stale, fetch fresh data first
        var nsfwSet = _nsfwPostIds.value
        if (nsfwSet.isEmpty() || System.currentTimeMillis() - lastRefreshTime > REFRESH_INTERVAL_MS) {
            Log.d(TAG, "NSFW cache empty or stale, fetching fresh data...")
            val result = fetchNSFWPosts()
            if (result.isSuccess) {
                nsfwSet = result.getOrNull() ?: emptySet()
            }
        }
        
        Log.d(TAG, "Applying NSFW flags: ${nsfwSet.size} flagged posts, checking ${posts.size} posts")
        
        return posts.map { post ->
            val isMarkedNSFW = nsfwSet.contains(post.postId)
            if (isMarkedNSFW) {
                Log.d(TAG, "Post ${post.postId.take(8)} is NSFW")
            }
            post.copy(
                isNSFW = isMarkedNSFW,
                isSensitive = isMarkedNSFW // Same as NSFW for now
            )
        }
    }
    
    /**
     * Subscribe to real-time NSFW updates via WebSocket (optional enhancement)
     * This would allow instant updates when admin marks content as NSFW
     */
    fun subscribeToRealTimeUpdates(onUpdate: (Set<String>) -> Unit) {
        // TODO: Implement WebSocket connection for real-time updates
        // For now, use polling with refreshIfNeeded()
    }
}
