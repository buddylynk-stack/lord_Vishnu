package com.orignal.buddylynk.data.local

import android.content.Context
import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.data.util.Result
import com.orignal.buddylynk.data.util.safeCall
import com.orignal.buddylynk.data.util.withRetry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Offline-first repository with caching and sync
 */
class OfflineRepository private constructor(context: Context) {
    
    private val database = AppDatabase.getInstance(context)
    private val postDao = database.postDao()
    private val userDao = database.userDao()
    private val pendingDao = database.pendingActionDao()
    
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    
    companion object {
        @Volatile
        private var INSTANCE: OfflineRepository? = null
        
        fun getInstance(context: Context): OfflineRepository {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: OfflineRepository(context).also { INSTANCE = it }
            }
        }
    }
    
    // =========================================================================
    // POSTS
    // =========================================================================
    
    /**
     * Get posts - cache-first, then network
     */
    fun getPosts(): Flow<List<Post>> {
        // Start background refresh
        scope.launch { refreshPosts() }
        
        // Return cached posts as Flow
        return postDao.getRecentPosts().map { cached ->
            cached.map { it.toPost() }
        }
    }
    
    /**
     * Refresh posts from network
     */
    suspend fun refreshPosts(): Result<List<Post>> = withContext(Dispatchers.IO) {
        if (!NetworkMonitor.isNetworkAvailable()) {
            return@withContext Result.Error(
                Exception("No network"),
                "No internet connection"
            )
        }
        
        withRetry {
            val posts = DynamoDbService.getPosts()
            // Cache posts
            postDao.insertPosts(posts.map { CachedPost.fromPost(it) })
            posts
        }
    }
    
    /**
     * Get user posts
     */
    fun getUserPosts(userId: String): Flow<List<Post>> {
        // Start background refresh
        scope.launch { refreshUserPosts(userId) }
        
        return postDao.getUserPosts(userId).map { cached ->
            cached.map { it.toPost() }
        }
    }
    
    /**
     * Refresh user posts from network
     */
    private suspend fun refreshUserPosts(userId: String) {
        if (!NetworkMonitor.isNetworkAvailable()) return
        
        safeCall {
            val posts = DynamoDbService.getUserPosts(userId)
            postDao.insertPosts(posts.map { CachedPost.fromPost(it) })
        }
    }
    
    /**
     * Like post - optimistic update with offline queue
     */
    suspend fun likePost(postId: String, isLiked: Boolean): Result<Unit> = withContext(Dispatchers.IO) {
        // Optimistic local update
        val delta = if (isLiked) 1 else -1
        postDao.updateLike(postId, isLiked, delta)
        
        if (NetworkMonitor.isNetworkAvailable()) {
            // Sync immediately via Redis counter
            try {
                if (isLiked) {
                    com.orignal.buddylynk.data.redis.RedisService.incrementLikes(postId)
                } else {
                    com.orignal.buddylynk.data.redis.RedisService.decrementLikes(postId)
                }
            } catch (e: Exception) {
                // Queue for later
                queueAction(if (isLiked) "like" else "unlike", postId)
            }
        } else {
            // Queue for later sync
            queueAction(if (isLiked) "like" else "unlike", postId)
        }
        
        Result.Success(Unit)
    }
    
    /**
     * Bookmark post - optimistic update
     */
    suspend fun bookmarkPost(postId: String, isBookmarked: Boolean): Result<Unit> = withContext(Dispatchers.IO) {
        postDao.updateBookmark(postId, isBookmarked)
        
        if (!NetworkMonitor.isNetworkAvailable()) {
            queueAction(if (isBookmarked) "bookmark" else "unbookmark", postId)
        }
        
        Result.Success(Unit)
    }
    
    /**
     * Queue an action for later sync
     */
    private suspend fun queueAction(actionType: String, targetId: String, payload: String? = null) {
        pendingDao.insert(
            PendingAction(
                actionType = actionType,
                targetId = targetId,
                payload = payload
            )
        )
    }
    
    // =========================================================================
    // USERS
    // =========================================================================
    
    /**
     * Get user - cache first, then network
     */
    suspend fun getUser(userId: String): Result<User?> = withContext(Dispatchers.IO) {
        // Try cache first
        val cached = userDao.getUser(userId)
        
        if (NetworkMonitor.isNetworkAvailable()) {
            // Refresh from network
            val result = safeCall { DynamoDbService.getUser(userId) }
            if (result is Result.Success && result.data != null) {
                userDao.insertUser(CachedUser.fromUser(result.data))
                return@withContext result
            }
        }
        
        // Return cached
        Result.Success(cached?.toUser())
    }
    
    /**
     * Cache a user
     */
    suspend fun cacheUser(user: User) = withContext(Dispatchers.IO) {
        userDao.insertUser(CachedUser.fromUser(user))
    }
    
    // =========================================================================
    // SYNC
    // =========================================================================
    
    /**
     * Sync pending actions when online
     */
    suspend fun syncPendingActions(): Result<Int> = withContext(Dispatchers.IO) {
        if (!NetworkMonitor.isNetworkAvailable()) {
            return@withContext Result.Error(Exception("No network"), "Offline")
        }
        
        val pending = pendingDao.getAllPending()
        var syncedCount = 0
        
        pending.forEach { action ->
            val success = try {
                when (action.actionType) {
                    "like" -> {
                        com.orignal.buddylynk.data.redis.RedisService.incrementLikes(action.targetId)
                        true
                    }
                    "unlike" -> {
                        com.orignal.buddylynk.data.redis.RedisService.decrementLikes(action.targetId)
                        true
                    }
                    else -> false
                }
            } catch (e: Exception) {
                false
            }
            
            if (success) {
                pendingDao.delete(action)
                syncedCount++
            } else {
                pendingDao.incrementRetryCount(action.id)
            }
        }
        
        // Clean up failed actions
        pendingDao.deleteFailedActions(5)
        
        Result.Success(syncedCount)
    }
    
    /**
     * Clear old cached data
     */
    suspend fun cleanupOldCache() = withContext(Dispatchers.IO) {
        val oneWeekAgo = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000L)
        postDao.deleteOldPosts(oneWeekAgo)
        userDao.deleteOldUsers(oneWeekAgo)
    }
    
    /**
     * Clear all cache
     */
    suspend fun clearCache() = withContext(Dispatchers.IO) {
        postDao.clearAll()
        userDao.clearAll()
    }
}
