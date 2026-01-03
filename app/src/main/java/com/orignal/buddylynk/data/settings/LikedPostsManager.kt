package com.orignal.buddylynk.data.settings

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * LikedPostsManager - Stores liked post IDs locally to persist glow state
 * 
 * This ensures liked posts show red heart even after app restart,
 * since the backend might not have a liked posts endpoint.
 */
object LikedPostsManager {
    
    private const val PREFS_NAME = "liked_posts_prefs"
    private const val KEY_LIKED_POST_IDS = "liked_post_ids"
    
    private var prefs: SharedPreferences? = null
    
    private val _likedPostIds = MutableStateFlow<Set<String>>(emptySet())
    val likedPostIds: StateFlow<Set<String>> = _likedPostIds.asStateFlow()
    
    /**
     * Initialize with context (call from MainActivity)
     */
    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        loadLikedPosts()
    }
    
    /**
     * Load liked post IDs from local storage
     */
    private fun loadLikedPosts() {
        val savedIds = prefs?.getStringSet(KEY_LIKED_POST_IDS, emptySet()) ?: emptySet()
        _likedPostIds.value = savedIds
        android.util.Log.d("LikedPostsManager", "Loaded ${savedIds.size} liked posts from local storage")
    }
    
    /**
     * Add a liked post ID and save to local storage
     */
    fun likePost(postId: String) {
        val newSet = _likedPostIds.value + postId
        _likedPostIds.value = newSet
        saveLikedPosts(newSet)
    }
    
    /**
     * Remove a liked post ID and save to local storage
     */
    fun unlikePost(postId: String) {
        val newSet = _likedPostIds.value - postId
        _likedPostIds.value = newSet
        saveLikedPosts(newSet)
    }
    
    /**
     * Toggle like state for a post
     */
    fun toggleLike(postId: String): Boolean {
        val isCurrentlyLiked = _likedPostIds.value.contains(postId)
        if (isCurrentlyLiked) {
            unlikePost(postId)
        } else {
            likePost(postId)
        }
        return !isCurrentlyLiked // Return new state
    }
    
    /**
     * Check if a post is liked
     */
    fun isLiked(postId: String): Boolean = _likedPostIds.value.contains(postId)
    
    /**
     * Get all liked post IDs
     */
    fun getLikedPostIds(): Set<String> = _likedPostIds.value
    
    /**
     * Save liked posts to local SharedPreferences
     */
    private fun saveLikedPosts(likedIds: Set<String>) {
        prefs?.edit()?.putStringSet(KEY_LIKED_POST_IDS, likedIds)?.apply()
        android.util.Log.d("LikedPostsManager", "Saved ${likedIds.size} liked posts to local storage")
    }
}
