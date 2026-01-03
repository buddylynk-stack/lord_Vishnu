package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.moderation.ModerationService
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.redis.RedisService
import com.orignal.buddylynk.data.api.ApiService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * HomeViewModel - Real-time posts with Redis caching
 */
class HomeViewModel : ViewModel() {
    
    // Current user ID for ownership checks
    val currentUserId: String = AuthManager.currentUser.value?.userId ?: ""
    
    private val _posts = MutableStateFlow<List<Post>>(emptyList())
    val posts: StateFlow<List<Post>> = _posts.asStateFlow()
    
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _isRefreshing = MutableStateFlow(false)
    val isRefreshing: StateFlow<Boolean> = _isRefreshing.asStateFlow()
    
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    // Pagination state for infinite scroll
    private val _hasMorePosts = MutableStateFlow(true)
    val hasMorePosts: StateFlow<Boolean> = _hasMorePosts.asStateFlow()
    
    private val _isLoadingMore = MutableStateFlow(false)
    val isLoadingMore: StateFlow<Boolean> = _isLoadingMore.asStateFlow()
    
    private var currentPage = 0
    private val pageSize = 30
    
    // Trending posts from Redis
    private val _trendingPosts = MutableStateFlow<List<String>>(emptyList())
    val trendingPosts: StateFlow<List<String>> = _trendingPosts.asStateFlow()
    
    // Blocked users list
    private val _blockedUsers = MutableStateFlow<Set<String>>(emptySet())
    val blockedUsers: StateFlow<Set<String>> = _blockedUsers.asStateFlow()
    
    // Saved post IDs
    private val _savedPostIds = MutableStateFlow<Set<String>>(emptySet())
    val savedPostIds: StateFlow<Set<String>> = _savedPostIds.asStateFlow()
    
    // Liked post IDs - persists across app sessions
    private val _likedPostIds = MutableStateFlow<Set<String>>(emptySet())
    val likedPostIds: StateFlow<Set<String>> = _likedPostIds.asStateFlow()
    
    init {
        loadBlockedUsers()
        loadSavedPosts()
        loadLikedPosts() // Load liked posts to show filled hearts
        loadNSFWPosts() // Load NSFW flags from admin database
        loadPosts()
        loadTrending()
        startAutoRefresh()
    }
    
    /**
     * Load blocked users from ModerationService (local + synced state)
     */
    private fun loadBlockedUsers() {
        if (currentUserId.isBlank()) return
        
        // First, load blocked users from API
        viewModelScope.launch(Dispatchers.IO) {
            try {
                android.util.Log.d("HomeViewModel", "Initializing ModerationService...")
                ModerationService.init()
                
                // Get initial blocked users
                val blocked = ModerationService.blockedUsers.value
                _blockedUsers.value = blocked
                android.util.Log.d("HomeViewModel", "Loaded ${blocked.size} blocked users from API")
                
                // Re-filter posts if we have any
                if (blocked.isNotEmpty() && _posts.value.isNotEmpty()) {
                    _posts.value = _posts.value.filter { post -> post.userId !in blocked }
                    android.util.Log.d("HomeViewModel", "Filtered posts, remaining: ${_posts.value.size}")
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error loading blocked users: ${e.message}")
            }
        }
        
        // Observe blocked users changes for real-time updates
        viewModelScope.launch {
            ModerationService.blockedUsers.collect { blocked ->
                if (blocked != _blockedUsers.value) {
                    _blockedUsers.value = blocked
                    android.util.Log.d("HomeViewModel", "Blocked users changed: ${blocked.size} users")
                    
                    // Re-filter posts when blocked list changes
                    if (blocked.isNotEmpty() && _posts.value.isNotEmpty()) {
                        _posts.value = _posts.value.filter { post -> post.userId !in blocked }
                    }
                }
            }
        }
    }
    
    fun loadPosts() {
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null
            try {
                android.util.Log.d("HomeViewModel", "Starting to load posts from BackendRepository...")
                
                // Reset pagination on fresh load
                currentPage = 0
                _hasMorePosts.value = true
                
                // ALWAYS fetch fresh blocked users from API BEFORE loading posts
                try {
                    val freshBlockedIds = BackendRepository.getBlockedUsers()
                    _blockedUsers.value = freshBlockedIds.toSet()
                    android.util.Log.d("HomeViewModel", "Loaded ${freshBlockedIds.size} blocked users for filtering")
                } catch (e: Exception) {
                    android.util.Log.e("HomeViewModel", "Failed to load blocked users: ${e.message}")
                }
                
                // Fetch posts via BackendRepository with pagination
                val feedResult = BackendRepository.getFeedPosts(page = 0, limit = pageSize)
                android.util.Log.e("FEED_DEBUG", "======= FEED LOADED =======")
                android.util.Log.e("FEED_DEBUG", "Posts: ${feedResult.posts.size}, hasMore: ${feedResult.hasMore}, total: ${feedResult.totalPosts}")
                android.util.Log.e("FEED_DEBUG", "============================")
                
                _hasMorePosts.value = feedResult.hasMore
                val fetchedPosts = feedResult.posts
                
                if (fetchedPosts.isEmpty()) {
                    android.util.Log.d("HomeViewModel", "No posts found.")
                    _posts.value = emptyList()
                } else {
                    // Filter out blocked users using fresh list
                    val blockedSet = _blockedUsers.value
                    val filteredPosts = fetchedPosts.filter { post -> 
                        post.userId !in blockedSet 
                    }
                    android.util.Log.d("HomeViewModel", "Filtered ${fetchedPosts.size} -> ${filteredPosts.size} posts (blocked ${blockedSet.size} users)")
                    
                    // Priority boost: User's posts from last 20 minutes appear at TOP
                    val now = System.currentTimeMillis()
                    val twentyMinutesAgo = now - (20 * 60 * 1000) // 20 minutes in ms

                    val (boostedPosts, otherPosts) = filteredPosts.partition { post ->
                        // Check if post is from current user AND within last 20 minutes
                        post.userId == currentUserId && isWithinTimeWindow(post.createdAt, twentyMinutesAgo)
                    }
                    
                    // Boosted posts first (sorted by newest), then others (sorted by newest)
                    val sortedPosts = boostedPosts.sortedByDescending { it.createdAt } + 
                                     otherPosts.sortedByDescending { it.createdAt }
                    
                    // Apply liked AND saved state before showing (persists glow after app close)
                    val likedSet = _likedPostIds.value
                    val savedSet = _savedPostIds.value
                    val postsWithState = sortedPosts.map { post ->
                        post.copy(
                            isLiked = likedSet.contains(post.postId),
                            isBookmarked = savedSet.contains(post.postId)
                        )
                    }
                    
                    // PROGRESSIVE LOADING: Show posts one by one for smooth animation
                    // First show first 3 posts immediately for fast first paint
                    val initialPosts = postsWithState.take(3)
                    _posts.value = initialPosts
                    _isLoading.value = false
                    
                    // Then progressively add remaining posts with staggered delay
                    launch {
                        val remainingPosts = postsWithState.drop(3)
                        for ((index, post) in remainingPosts.withIndex()) {
                            kotlinx.coroutines.delay(30L) // 30ms delay between each post
                            _posts.value = _posts.value + post
                        }
                        android.util.Log.d("HomeViewModel", "Progressive loading complete: ${_posts.value.size} posts")
                    }
                    
                    // Enhance with Redis views and NSFW flags in BACKGROUND (non-blocking)
                    launch {
                        try {
                            // Wait for progressive loading to finish
                            kotlinx.coroutines.delay(50L * postsWithState.size)
                            
                            val enhancedPosts = _posts.value.map { post ->
                                val redisViews = RedisService.getViews(post.postId)
                                post.copy(viewsCount = maxOf(post.viewsCount, redisViews.toInt()))
                            }
                            // Apply NSFW flags after all enhancements
                            val nsfwEnhancedPosts = com.orignal.buddylynk.data.api.NSFWApiService.applyNSFWFlags(enhancedPosts)
                            _posts.value = nsfwEnhancedPosts
                            android.util.Log.d("HomeViewModel", "Applied NSFW flags to ${nsfwEnhancedPosts.count { it.isNSFW }} posts")
                        } catch (e: Exception) {
                            // Redis enhancement failed, keep original posts but still apply NSFW flags
                            _posts.value = com.orignal.buddylynk.data.api.NSFWApiService.applyNSFWFlags(_posts.value)
                            android.util.Log.w("HomeViewModel", "Redis enhancement failed: ${e.message}")
                        }
                    }
                    return@launch
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error loading posts", e)
                _error.value = e.message ?: "Unknown error"
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    /**
     * Load more posts for infinite scroll - called when user reaches bottom of feed
     */
    fun loadMorePosts() {
        if (_isLoadingMore.value || !_hasMorePosts.value) return
        
        viewModelScope.launch(Dispatchers.IO) {
            _isLoadingMore.value = true
            try {
                currentPage++
                android.util.Log.d("HomeViewModel", "Loading more posts - page: $currentPage")
                
                val feedResult = BackendRepository.getFeedPosts(page = currentPage, limit = pageSize)
                android.util.Log.d("HomeViewModel", "Loaded ${feedResult.posts.size} more posts, hasMore: ${feedResult.hasMore}")
                
                _hasMorePosts.value = feedResult.hasMore
                
                if (feedResult.posts.isNotEmpty()) {
                    // Filter blocked users and append to existing posts
                    val blockedSet = _blockedUsers.value
                    val filteredPosts = feedResult.posts.filter { post -> post.userId !in blockedSet }
                    
                    // Apply liked and saved state
                    val likedSet = _likedPostIds.value
                    val savedSet = _savedPostIds.value
                    val postsWithState = filteredPosts.map { post ->
                        post.copy(
                            isLiked = likedSet.contains(post.postId),
                            isBookmarked = savedSet.contains(post.postId)
                        )
                    }
                    
                    // Append to existing posts (avoid duplicates)
                    val existingIds = _posts.value.map { it.postId }.toSet()
                    val newPosts = postsWithState.filter { it.postId !in existingIds }
                    
                    _posts.value = _posts.value + newPosts
                    android.util.Log.d("HomeViewModel", "Total posts now: ${_posts.value.size}")
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error loading more posts", e)
                currentPage-- // Revert page on error
            } finally {
                _isLoadingMore.value = false
            }
        }
    }
    
    // Helper: Check if post createdAt timestamp is within the time window
    private fun isWithinTimeWindow(createdAt: String, thresholdMs: Long): Boolean {
        return try {
            val formatter = java.text.SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", java.util.Locale.US)
            formatter.timeZone = java.util.TimeZone.getTimeZone("UTC")
            val postTime = formatter.parse(createdAt.take(19))?.time ?: 0L
            postTime >= thresholdMs
        } catch (e: Exception) {
            false
        }
    }

    // Add a newly created post to the TOP of the feed immediately
    fun addNewPost(post: Post) {
        val currentPosts = _posts.value.toMutableList()
        // Remove any existing post with same ID (in case of duplicate)
        currentPosts.removeAll { it.postId == post.postId }
        // Add new post to the TOP
        currentPosts.add(0, post)
        _posts.value = currentPosts
        android.util.Log.d("HomeViewModel", "Added new post ${post.postId} to top of feed")
    }
    
    // Refresh and force user's posts to show at top
    fun refreshWithUserPostsFirst() {
        viewModelScope.launch {
            _isRefreshing.value = true
            try {
                val fetchedPosts = BackendRepository.getFeedPosts()
                // Sort: user's posts first (by createdAt desc), then others
                val sortedPosts = fetchedPosts.sortedWith(
                    compareByDescending<Post> { it.userId == currentUserId }
                        .thenByDescending { it.createdAt }
                )
                _posts.value = sortedPosts
            } catch (e: Exception) {
                _error.value = e.message
            } finally {
                _isRefreshing.value = false
            }
        }
    }

    // Seeding removed for production

    
    fun refresh() {
        viewModelScope.launch {
            _isRefreshing.value = true
            _error.value = null
            try {
                // Fetch fresh blocked users
                val freshBlockedIds = BackendRepository.getBlockedUsers()
                _blockedUsers.value = freshBlockedIds.toSet()
                
                val fetchedPosts = BackendRepository.getFeedPosts()
                
                // Filter blocked users
                val blockedSet = _blockedUsers.value
                val filteredPosts = fetchedPosts.filter { post -> post.userId !in blockedSet }
                
                // Enhance with Redis counters
                val enhancedPosts = filteredPosts.map { post ->
                    val redisViews = RedisService.getViews(post.postId)
                    post.copy(viewsCount = maxOf(post.viewsCount, redisViews.toInt()))
                }
                
                _posts.value = enhancedPosts
                _error.value = null // Clear error on success
                loadTrending()
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Refresh failed: ${e.message}")
                _error.value = e.message ?: "Server connection failed"
                // Clear posts if this is initial load (no posts yet)
                if (_posts.value.isEmpty()) {
                    _error.value = "Server is not responding"
                }
            } finally {
                _isRefreshing.value = false
            }
        }
    }
    
    private fun loadTrending() {
        viewModelScope.launch {
            try {
                val trending = RedisService.getTrendingPosts(limit = 20)
                _trendingPosts.value = trending
            } catch (e: Exception) {
                // Silent fail for trending
            }
        }
    }
    
    // Auto-refresh every 30 seconds for real-time feel
    private fun startAutoRefresh() {
        viewModelScope.launch {
            try {
                while (true) {
                    delay(30_000) // 30 seconds
                    try {
                        // Fetch fresh blocked users
                        val freshBlockedIds = BackendRepository.getBlockedUsers()
                        _blockedUsers.value = freshBlockedIds.toSet()
                        
                        val fetchedPosts = BackendRepository.getFeedPosts()
                        
                        // Filter blocked users
                        val blockedSet = _blockedUsers.value
                        val filteredPosts = fetchedPosts.filter { post -> post.userId !in blockedSet }
                        
                        val enhancedPosts = filteredPosts.map { post ->
                            val redisViews = RedisService.getViews(post.postId)
                            post.copy(viewsCount = maxOf(post.viewsCount, redisViews.toInt()))
                        }
                        _posts.value = enhancedPosts
                        _error.value = null // Clear error on successful auto-refresh
                    } catch (e: Exception) {
                        android.util.Log.w("HomeViewModel", "Auto-refresh failed: ${e.message}")
                        // Set error if posts are empty (server was never reachable)
                        if (_posts.value.isEmpty()) {
                            _error.value = "Server is not responding"
                        }
                    }
                }
            } catch (e: kotlinx.coroutines.CancellationException) {
                // Normal cancellation when ViewModel is cleared - do nothing
            }
        }
    }
    
    /**
     * Like post - INSTANT UI update, then async DB
     */
    fun likePost(postId: String) {
        // Get current post state
        val currentPost = _posts.value.find { it.postId == postId } ?: return
        val newIsLiked = !currentPost.isLiked
        val newCount = if (newIsLiked) currentPost.likesCount + 1 else (currentPost.likesCount - 1).coerceAtLeast(0)
        
        // UPDATE UI IMMEDIATELY (no coroutine delay!)
        _posts.value = _posts.value.map { post ->
            if (post.postId == postId) {
                post.copy(isLiked = newIsLiked, likesCount = newCount)
            } else post
        }
        
        // Also update the likedPostIds set for persistence
        if (newIsLiked) {
            _likedPostIds.value = _likedPostIds.value + postId
            com.orignal.buddylynk.data.settings.LikedPostsManager.likePost(postId)
        } else {
            _likedPostIds.value = _likedPostIds.value - postId
            com.orignal.buddylynk.data.settings.LikedPostsManager.unlikePost(postId)
        }
        
        // Update Redis + DynamoDB in background
        viewModelScope.launch(Dispatchers.IO) {
            try {
                if (newIsLiked) {
                    RedisService.incrementLikes(postId)
                    RedisService.incrementPostScore(postId, 2.0)
                } else {
                    RedisService.decrementLikes(postId)
                }
                // Save to API permanently (via BackendRepository)
                BackendRepository.likePost(postId)
                
                // Track for MindFlow Algorithm
                val contentOwnerId = currentPost.userId ?: ""
                val isNSFW = currentPost.isNSFW
                if (newIsLiked) {
                    com.orignal.buddylynk.data.api.BehaviorTracker.trackLike(currentUserId, postId, contentOwnerId, isNSFW)
                } else {
                    com.orignal.buddylynk.data.api.BehaviorTracker.trackUnlike(currentUserId, postId, contentOwnerId, isNSFW)
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error updating likes: ${e.message}")
            }
        }
    }
    
    /**
     * Load saved posts from API
     */
    private fun loadSavedPosts() {
        if (currentUserId.isBlank()) return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val savedIds = BackendRepository.getSavedPostIds()
                _savedPostIds.value = savedIds.toSet()
                android.util.Log.d("HomeViewModel", "Loaded ${savedIds.size} saved posts from API")
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error loading saved posts: ${e.message}")
            }
        }
    }
    
    /**
     * Load liked posts - tries API first, then merges with local storage
     * This ensures likes persist both across reinstalls (API) and when offline (local)
     */
    private fun loadLikedPosts() {
        if (currentUserId.isBlank()) return
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Start with local storage (always available, works offline)
                val localLikedIds = com.orignal.buddylynk.data.settings.LikedPostsManager.getLikedPostIds()
                
                // Also try to load from API (for cross-device sync)
                val apiLikedIds = try {
                    BackendRepository.getLikedPostIds().toSet()
                } catch (e: Exception) {
                    android.util.Log.w("HomeViewModel", "API liked posts failed, using local only: ${e.message}")
                    emptySet()
                }
                
                // Merge both - union of local and API
                val mergedLikedIds = localLikedIds + apiLikedIds
                _likedPostIds.value = mergedLikedIds
                
                // Sync any new likes from API back to local storage
                apiLikedIds.forEach { postId ->
                    if (postId !in localLikedIds) {
                        com.orignal.buddylynk.data.settings.LikedPostsManager.likePost(postId)
                    }
                }
                
                android.util.Log.d("HomeViewModel", "Loaded liked posts: ${localLikedIds.size} local + ${apiLikedIds.size} API = ${mergedLikedIds.size} total")
                
                // Update posts with isLiked state
                if (mergedLikedIds.isNotEmpty() && _posts.value.isNotEmpty()) {
                    _posts.value = _posts.value.map { post ->
                        post.copy(isLiked = mergedLikedIds.contains(post.postId))
                    }
                    android.util.Log.d("HomeViewModel", "Applied isLiked state to posts")
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error loading liked posts: ${e.message}")
            }
        }
    }
    
    /**
     * Load NSFW posts from admin database - shows blur/hide based on user settings
     * Connects to DynamoDB NSFW table in real-time
     */
    private fun loadNSFWPosts() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Fetch NSFW flags from server
                val result = com.orignal.buddylynk.data.api.NSFWApiService.fetchNSFWPosts()
                result.onSuccess { nsfwIds ->
                    android.util.Log.d("HomeViewModel", "Loaded ${nsfwIds.size} NSFW posts from admin database")
                    
                    // Apply NSFW flags to existing posts
                    if (nsfwIds.isNotEmpty() && _posts.value.isNotEmpty()) {
                        _posts.value = com.orignal.buddylynk.data.api.NSFWApiService.applyNSFWFlags(_posts.value)
                        android.util.Log.d("HomeViewModel", "Applied NSFW flags to posts")
                    }
                }
                result.onFailure { e ->
                    android.util.Log.w("HomeViewModel", "NSFW API not available: ${e.message}")
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error loading NSFW posts: ${e.message}")
            }
        }
    }
    
    /**
     * Refresh NSFW flags (called on pull-to-refresh)
     */
    fun refreshNSFWFlags() {
        viewModelScope.launch(Dispatchers.IO) {
            com.orignal.buddylynk.data.api.NSFWApiService.forceRefresh()
            // Re-apply to posts
            _posts.value = com.orignal.buddylynk.data.api.NSFWApiService.applyNSFWFlags(_posts.value)
        }
    }
    
    /**
     * Toggle save/bookmark post - INSTANT UI update, then save to DB
     */
    fun bookmarkPost(postId: String) {
        val isSaved = _savedPostIds.value.contains(postId)
        
        // Update UI immediately
        if (isSaved) {
            _savedPostIds.value = _savedPostIds.value - postId
        } else {
            _savedPostIds.value = _savedPostIds.value + postId
        }
        
        // Update post isBookmarked state
        _posts.value = _posts.value.map { post ->
            if (post.postId == postId) {
                post.copy(isBookmarked = !isSaved)
            } else post
        }
        
        // Persist to API in background
        viewModelScope.launch(Dispatchers.IO) {
            try {
                if (isSaved) {
                    BackendRepository.unsavePost(postId)
                    android.util.Log.d("HomeViewModel", "Unsaved post $postId via API")
                } else {
                    BackendRepository.savePost(postId)
                    android.util.Log.d("HomeViewModel", "Saved post $postId via API")
                }
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error saving/unsaving post: ${e.message}")
            }
        }
    }
    
    /**
     * Share post - INSTANT UI update, then save to DB
     */
    fun sharePost(postId: String) {
        // Get current post
        val currentPost = _posts.value.find { it.postId == postId } ?: return
        val newCount = currentPost.sharesCount + 1
        
        android.util.Log.d("HomeViewModel", "sharePost: old=${currentPost.sharesCount}, new=$newCount")
        
        // UPDATE UI IMMEDIATELY (main thread, no coroutine)
        _posts.value = _posts.value.map { post ->
            if (post.postId == postId) {
                post.copy(sharesCount = newCount)
            } else post
        }
        
        android.util.Log.d("HomeViewModel", "sharePost: state updated, list size=${_posts.value.size}")
        
        // Update Redis + DynamoDB in background
        viewModelScope.launch(Dispatchers.IO) {
            try {
                // Persist to DynamoDB via API
                ApiService.sharePost(postId)
                
                // Also update Redis cache
                RedisService.incrementShares(postId)
                RedisService.incrementPostScore(postId, 3.0)
                android.util.Log.d("HomeViewModel", "Share count updated to $newCount via API + Redis")
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Share update error: ${e.message}")
            }
        }
    }
    
    /**
     * Track view - update Redis, DynamoDB, and local state
     */
    fun incrementViews(postId: String) {
        // Get current count first
        val currentPost = _posts.value.find { it.postId == postId }
        val newViewCount = (currentPost?.viewsCount ?: 0) + 1
        
        // Update local state immediately
        _posts.value = _posts.value.map { post ->
            if (post.postId == postId) {
                post.copy(viewsCount = newViewCount)
            } else post
        }
        
        // Update Redis + DynamoDB in background
        viewModelScope.launch(Dispatchers.IO) {
            try {
                RedisService.incrementViews(postId)
                RedisService.incrementPostScore(postId, 0.1)
                // Views tracked via Redis for now (no API endpoint yet)
                android.util.Log.d("HomeViewModel", "View count updated to $newViewCount via Redis")
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Views update error: ${e.message}")
            }
        }
    }
    
    /**
     * Delete post (own posts only) via API
     */
    fun deletePost(postId: String) {
        viewModelScope.launch {
            try {
                val success = BackendRepository.deletePost(postId)
                if (success) {
                    // Remove from local state
                    _posts.value = _posts.value.filter { it.postId != postId }
                    android.util.Log.d("HomeViewModel", "Post $postId deleted via API")
                } else {
                    _error.value = "Failed to delete post"
                }
            } catch (e: Exception) {
                _error.value = "Failed to delete post"
            }
        }
    }
    
    /**
     * Block user - instantly hide their posts and save to DB
     */
    fun blockUser(userId: String) {
        // Immediately add to blocked set and hide posts
        _blockedUsers.value = _blockedUsers.value + userId
        _posts.value = _posts.value.filter { it.userId != userId }
        
        // Save to API in background
        viewModelScope.launch(Dispatchers.IO) {
            try {
                BackendRepository.blockUser(userId)
                android.util.Log.d("HomeViewModel", "User $userId blocked via API")
            } catch (e: Exception) {
                android.util.Log.e("HomeViewModel", "Error blocking user: ${e.message}")
            }
        }
    }
    
    /**
     * Report post via API (TODO: implement report API)
     */
    fun reportPost(postId: String) {
        viewModelScope.launch {
            try {
                // TODO: Implement report API endpoint
                android.util.Log.d("HomeViewModel", "Report submitted for post $postId")
            } catch (e: Exception) {
                _error.value = "Failed to report post"
            }
        }
    }
}
