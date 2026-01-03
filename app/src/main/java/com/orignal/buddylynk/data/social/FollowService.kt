package com.orignal.buddylynk.data.social

import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.aws.DynamoDbService  // Kept for fallback operations not yet in API
import com.orignal.buddylynk.data.model.Follow
import com.orignal.buddylynk.data.model.User
import com.orignal.buddylynk.data.model.Activity
import com.orignal.buddylynk.data.redis.RedisService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * FollowService - Handles all follow/follower operations
 * Production-ready with Redis caching for fast counts
 */
object FollowService {
    
    // Cache for follow status
    private val _followingCache = MutableStateFlow<Set<String>>(emptySet())
    val followingCache: StateFlow<Set<String>> = _followingCache.asStateFlow()
    
    private val _followersCache = MutableStateFlow<Set<String>>(emptySet())
    val followersCache: StateFlow<Set<String>> = _followersCache.asStateFlow()
    
    // Redis keys
    private const val KEY_FOLLOWING = "following:"
    private const val KEY_FOLLOWERS = "followers:"
    private const val KEY_FOLLOW_COUNT = "follow_count:"
    
    /**
     * Follow a user
     */
    suspend fun followUser(targetUserId: String): Boolean = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext false
        
        // Don't follow yourself
        if (currentUser.userId == targetUserId) return@withContext false
        
        // Check if already following
        if (isFollowing(targetUserId)) return@withContext false
        
        try {
            // Create follow record
            val follow = Follow(
                followId = UUID.randomUUID().toString(),
                followerId = currentUser.userId,
                followingId = targetUserId,
                followerUsername = currentUser.username,
                followerAvatar = currentUser.avatar,
                createdAt = System.currentTimeMillis().toString()
            )
            
            // Save to DynamoDB
            val success = DynamoDbService.createFollow(follow)
            
            if (success) {
                // Update local cache
                _followingCache.value = _followingCache.value + targetUserId
                
                // Increment follower count in Redis
                RedisService.incrementLikes("${KEY_FOLLOWERS}$targetUserId")
                RedisService.incrementLikes("${KEY_FOLLOWING}${currentUser.userId}")
                
                // Create activity for the followed user
                createFollowActivity(currentUser, targetUserId)
                
                // Update user's following count
                updateUserCounts(currentUser.userId, followingDelta = 1)
                updateUserCounts(targetUserId, followersDelta = 1)
            }
            
            success
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Unfollow a user
     */
    suspend fun unfollowUser(targetUserId: String): Boolean = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext false
        
        try {
            // Delete follow record from DynamoDB
            val success = DynamoDbService.deleteFollow(currentUser.userId, targetUserId)
            
            if (success) {
                // Update local cache
                _followingCache.value = _followingCache.value - targetUserId
                
                // Decrement counts in Redis
                RedisService.decrementLikes("${KEY_FOLLOWERS}$targetUserId")
                RedisService.decrementLikes("${KEY_FOLLOWING}${currentUser.userId}")
                
                // Update user counts
                updateUserCounts(currentUser.userId, followingDelta = -1)
                updateUserCounts(targetUserId, followersDelta = -1)
            }
            
            success
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Check if current user is following target user
     */
    suspend fun isFollowing(targetUserId: String): Boolean = withContext(Dispatchers.IO) {
        // Check cache first
        if (_followingCache.value.contains(targetUserId)) return@withContext true
        
        val currentUser = AuthManager.currentUser.value ?: return@withContext false
        
        // Check DynamoDB
        DynamoDbService.checkFollowing(currentUser.userId, targetUserId)
    }
    
    /**
     * Get list of users the current user is following
     */
    suspend fun getFollowing(userId: String? = null): List<User> = withContext(Dispatchers.IO) {
        val targetId = userId ?: AuthManager.currentUser.value?.userId ?: return@withContext emptyList()
        
        try {
            val follows = DynamoDbService.getFollowing(targetId)
            
            // Update cache if it's current user
            if (userId == null || userId == AuthManager.currentUser.value?.userId) {
                _followingCache.value = follows.map { it.followingId }.toSet()
            }
            
            // Get user details for each following
            follows.mapNotNull { follow ->
                DynamoDbService.getUser(follow.followingId)
            }
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Get list of followers for a user
     */
    suspend fun getFollowers(userId: String? = null): List<User> = withContext(Dispatchers.IO) {
        val targetId = userId ?: AuthManager.currentUser.value?.userId ?: return@withContext emptyList()
        
        try {
            val follows = DynamoDbService.getFollowers(targetId)
            
            // Update cache if it's current user
            if (userId == null || userId == AuthManager.currentUser.value?.userId) {
                _followersCache.value = follows.map { it.followerId }.toSet()
            }
            
            // Get user details for each follower
            follows.mapNotNull { follow ->
                DynamoDbService.getUser(follow.followerId)
            }
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Get follower count (from Redis for speed)
     */
    suspend fun getFollowerCount(userId: String): Long = withContext(Dispatchers.IO) {
        RedisService.getViews("${KEY_FOLLOWERS}$userId")
    }
    
    /**
     * Get following count (from Redis for speed)
     */
    suspend fun getFollowingCount(userId: String): Long = withContext(Dispatchers.IO) {
        RedisService.getViews("${KEY_FOLLOWING}$userId")
    }
    
    /**
     * Get mutual followers (friends)
     */
    suspend fun getMutualFollowers(userId: String): List<User> = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext emptyList()
        
        val myFollowing = getFollowing().map { it.userId }.toSet()
        val theirFollowing = getFollowing(userId).map { it.userId }.toSet()
        
        val mutualIds = myFollowing.intersect(theirFollowing)
        
        mutualIds.mapNotNull { DynamoDbService.getUser(it) }
    }
    
    /**
     * Check if users are mutual followers
     */
    suspend fun areMutual(userId: String): Boolean = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext false
        
        val iFollow = isFollowing(userId)
        val theyFollow = DynamoDbService.checkFollowing(userId, currentUser.userId)
        
        iFollow && theyFollow
    }
    
    /**
     * Get suggested users to follow
     */
    suspend fun getSuggestions(limit: Int = 10): List<User> = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext emptyList()
        
        try {
            // Get users not already followed
            val allUsers = DynamoDbService.getUsers(limit = limit * 2)
            val following = _followingCache.value
            
            allUsers
                .filter { it.userId != currentUser.userId && !following.contains(it.userId) }
                .take(limit)
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Load current user's following list to cache
     */
    suspend fun loadFollowingCache() = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext
        
        try {
            val follows = DynamoDbService.getFollowing(currentUser.userId)
            _followingCache.value = follows.map { it.followingId }.toSet()
            
            val followers = DynamoDbService.getFollowers(currentUser.userId)
            _followersCache.value = followers.map { it.followerId }.toSet()
        } catch (e: Exception) {
            // Silent fail
        }
    }
    
    /**
     * Create follow activity notification
     */
    private suspend fun createFollowActivity(follower: User, followedUserId: String) {
        try {
            val activity = Activity(
                activityId = UUID.randomUUID().toString(),
                type = "follow",
                actorId = follower.userId,
                actorUsername = follower.username,
                actorAvatar = follower.avatar,
                targetId = followedUserId,
                isRead = false,
                createdAt = System.currentTimeMillis().toString()
            )
            
            DynamoDbService.createActivity(activity, followedUserId)
        } catch (e: Exception) {
            // Silent fail - not critical
        }
    }
    
    /**
     * Update user follower/following counts
     */
    private suspend fun updateUserCounts(
        userId: String,
        followersDelta: Int = 0,
        followingDelta: Int = 0
    ) {
        try {
            DynamoDbService.updateUserCounts(userId, followersDelta, followingDelta)
        } catch (e: Exception) {
            // Silent fail
        }
    }
}
