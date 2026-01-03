package com.orignal.buddylynk.data.repository

import android.util.Log
import com.orignal.buddylynk.data.api.ApiService
import com.orignal.buddylynk.data.model.Group
import com.orignal.buddylynk.data.model.Message
import com.orignal.buddylynk.data.model.Post
import com.orignal.buddylynk.data.model.User
import org.json.JSONObject
import java.util.UUID

/**
 * Backend Repository - All API calls go through secure backend
 * 
 * This is the PRODUCTION configuration.
 * All calls go through the secure backend API at http://52.0.95.126:3000
 */
object BackendRepository {
    
    private const val TAG = "BackendRepository"
    
    // ========== AUTH OPERATIONS ==========
    
    suspend fun login(email: String, password: String): Pair<User, String>? {
        return try {
            val result = ApiService.login(email, password)
            result.getOrNull()?.let { json ->
                val userJson = json.optJSONObject("user")
                val token = json.optString("token", "")
                userJson?.let { 
                    val user = parseUserFromJson(it)
                    if (user != null && token.isNotEmpty()) {
                        Pair(user, token)
                    } else null
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "API login failed", e)
            null
        }
    }
    
    suspend fun register(email: String, username: String, password: String): Pair<User, String>? {
        return try {
            val result = ApiService.register(email, username, password, username)
            result.getOrNull()?.let { json ->
                val userJson = json.optJSONObject("user")
                val token = json.optString("token", "")
                userJson?.let { 
                    val user = parseUserFromJson(it)
                    if (user != null && token.isNotEmpty()) {
                        Pair(user, token)
                    } else null
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "API register failed", e)
            null
        }
    }
    
    // ========== USER OPERATIONS ==========
    
    suspend fun getUser(userId: String): User? {
        return try {
            val result = ApiService.getUser(userId)
            result.getOrNull()?.let { parseUserFromJson(it) }
        } catch (e: Exception) {
            Log.e(TAG, "API getUser failed", e)
            null
        }
    }
    
    suspend fun searchUsers(query: String): List<User> {
        return try {
            val result = ApiService.searchUsers(query)
            result.getOrNull()?.mapNotNull { parseUserFromJson(it) } ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "API searchUsers failed", e)
            emptyList()
        }
    }
    
    // ========== POST OPERATIONS ==========
    
    // Data class for feed result with pagination
    data class FeedResult(
        val posts: List<Post>,
        val hasMore: Boolean,
        val nextPage: Int?,
        val totalPosts: Int
    )
    
    suspend fun getFeedPosts(page: Int = 0, limit: Int = 30): FeedResult {
        return try {
            val result = ApiService.getFeed(page, limit)
            val response = result.getOrNull()
            if (response != null) {
                val posts = response.posts.mapNotNull { parsePostFromJson(it) }
                FeedResult(posts, response.hasMore, response.nextPage, response.totalPosts)
            } else {
                FeedResult(emptyList(), false, null, 0)
            }
        } catch (e: Exception) {
            Log.e(TAG, "API getFeed failed", e)
            FeedResult(emptyList(), false, null, 0)
        }
    }
    
    suspend fun getUserPosts(userId: String): List<Post> {
        return try {
            // Filter feed posts by userId
            val allPosts = getFeedPosts()
            allPosts.filter { it.userId == userId }
        } catch (e: Exception) {
            Log.e(TAG, "API getUserPosts failed", e)
            emptyList()
        }
    }
    
    suspend fun getUsers(limit: Int = 20): List<User> {
        return try {
            val result = ApiService.getRecommendedUsers(limit)
            result.getOrNull()?.mapNotNull { parseUserFromJson(it) } ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "API getUsers failed", e)
            emptyList()
        }
    }
    
    suspend fun createPost(post: Post): Boolean {
        return try {
            val result = ApiService.createPost(post.content, post.mediaUrls, post.mediaType ?: "text")
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API createPost failed", e)
            false
        }
    }
    
    suspend fun likePost(postId: String, userId: String = "", currentLikes: Int = 0): Boolean {
        return try {
            val result = ApiService.likePost(postId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API likePost failed", e)
            false
        }
    }
    
    // ========== GROUP OPERATIONS ==========
    
    suspend fun getUserGroups(userId: String): List<Group> {
        Log.d(TAG, "getUserGroups called for userId: $userId")
        return try {
            val result = ApiService.getGroups()
            if (result.isSuccess) {
                val groups = result.getOrNull()?.mapNotNull { parseGroupFromJson(it) } ?: emptyList()
                Log.d(TAG, "API returned ${groups.size} groups")
                groups
            } else {
                Log.e(TAG, "API getGroups failed")
                emptyList()
            }
        } catch (e: Exception) {
            Log.e(TAG, "API getUserGroups exception: ${e.message}", e)
            emptyList()
        }
    }
    
    suspend fun createGroup(group: Group): Boolean {
        return try {
            val result = ApiService.createGroup(group.name, group.description, group.isPublic)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API createGroup failed", e)
            false
        }
    }
    
    // ========== FOLLOW OPERATIONS ==========
    
    suspend fun followUser(targetUserId: String, currentUserId: String): Boolean {
        return try {
            val result = ApiService.follow(targetUserId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API followUser failed", e)
            false
        }
    }
    
    suspend fun unfollowUser(targetUserId: String, currentUserId: String): Boolean {
        return try {
            val result = ApiService.unfollow(targetUserId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API unfollowUser failed", e)
            false
        }
    }
    
    suspend fun isFollowing(targetUserId: String): Boolean {
        return try {
            val result = ApiService.isFollowing(targetUserId)
            result.getOrNull() ?: false
        } catch (e: Exception) {
            Log.e(TAG, "API isFollowing failed", e)
            false
        }
    }
    
    // ========== BLOCK OPERATIONS ==========
    
    suspend fun getBlockedUsers(): List<String> {
        return try {
            val result = ApiService.getBlockedUsers()
            result.getOrNull() ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "API getBlockedUsers failed", e)
            emptyList()
        }
    }
    
    suspend fun blockUser(targetUserId: String): Boolean {
        return try {
            val result = ApiService.blockUser(targetUserId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API blockUser failed", e)
            false
        }
    }
    
    suspend fun unblockUser(targetUserId: String): Boolean {
        return try {
            val result = ApiService.unblockUser(targetUserId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API unblockUser failed", e)
            false
        }
    }
    
    // ========== LIKED POSTS OPERATIONS ==========
    
    suspend fun getLikedPostIds(): List<String> {
        return try {
            val result = ApiService.getLikedPostIds()
            result.getOrNull() ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "API getLikedPostIds failed", e)
            emptyList()
        }
    }
    
    suspend fun toggleLikePost(postId: String): Boolean {
        return try {
            val result = ApiService.likePost(postId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API likePost failed", e)
            false
        }
    }
    
    // ========== SAVED POSTS OPERATIONS ==========
    
    suspend fun getSavedPostIds(): List<String> {
        return try {
            val result = ApiService.getSavedPostIds()
            result.getOrNull() ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "API getSavedPostIds failed", e)
            emptyList()
        }
    }
    
    suspend fun savePost(postId: String): Boolean {
        return try {
            val result = ApiService.savePost(postId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API savePost failed", e)
            false
        }
    }
    
    suspend fun unsavePost(postId: String): Boolean {
        return try {
            val result = ApiService.unsavePost(postId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API unsavePost failed", e)
            false
        }
    }
    
    suspend fun deletePost(postId: String): Boolean {
        return try {
            val result = ApiService.deletePost(postId)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API deletePost failed", e)
            false
        }
    }
    
    // ========== MESSAGING OPERATIONS ==========
    
    suspend fun getConversationPartners(userId: String): List<User> {
        return try {
            val result = ApiService.getConversations()
            // getConversations returns list of partner user IDs
            val partnerIds = result.getOrNull() ?: return emptyList()
            
            val users = mutableListOf<User>()
            for (partnerId in partnerIds) {
                try {
                    val user = getUser(partnerId)
                    if (user != null) {
                        users.add(user)
                    }
                } catch (e: Exception) {
                    // Skip this entry
                }
            }
            users
        } catch (e: Exception) {
            Log.e(TAG, "API getConversationPartners failed", e)
            emptyList()
        }
    }
    
    suspend fun getMessages(conversationId: String): List<Message> {
        return try {
            // conversationId format is "conv_userId1_userId2"
            // Extract partner ID (the one that's not the current user)
            val parts = conversationId.removePrefix("conv_").split("_")
            val currentUserId = com.orignal.buddylynk.data.auth.AuthManager.getCurrentUserId() ?: ""
            val partnerId = parts.firstOrNull { it != currentUserId && it.isNotEmpty() } ?: parts.lastOrNull() ?: return emptyList()
            
            Log.d(TAG, "getMessages: conversationId=$conversationId, extractedPartnerId=$partnerId")
            val result = ApiService.getMessages(partnerId)
            result.getOrNull()?.mapNotNull { parseMessageFromJson(it) } ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "API getMessages failed", e)
            emptyList()
        }
    }
    
    suspend fun sendMessage(partnerId: String, content: String): Boolean {
        return try {
            val result = ApiService.sendMessage(partnerId, content)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API sendMessage failed", e)
            false
        }
    }
    
    // ========== UPLOAD OPERATIONS ==========
    
    suspend fun getPresignedUploadUrl(filename: String, contentType: String, folder: String): Pair<String, String>? {
        return try {
            val result = ApiService.getPresignedUrl(filename, contentType, folder)
            result.getOrNull()?.let { json ->
                val uploadUrl = json.optString("uploadUrl", "")
                val fileUrl = json.optString("fileUrl", "")
                if (uploadUrl.isNotEmpty() && fileUrl.isNotEmpty()) {
                    Pair(uploadUrl, fileUrl)
                } else null
            }
        } catch (e: Exception) {
            Log.e(TAG, "API getPresignedUploadUrl failed", e)
            null
        }
    }
    
    // ========== GROUP MESSAGES ==========
    
    suspend fun getGroupMessages(groupId: String): List<Message> {
        Log.d(TAG, "getGroupMessages called for groupId: $groupId")
        return try {
            Log.d(TAG, "Calling API getGroupMessages...")
            val result = ApiService.getGroupMessages(groupId)
            
            if (result.isSuccess) {
                val messages = result.getOrNull()?.mapNotNull { parseMessageFromJson(it) } ?: emptyList()
                Log.d(TAG, "API returned ${messages.size} messages")
                messages
            } else {
                Log.e(TAG, "API getGroupMessages failed")
                emptyList()
            }
        } catch (e: Exception) {
            Log.e(TAG, "API getGroupMessages exception: ${e.message}", e)
            emptyList()
        }
    }
    
    suspend fun sendGroupMessage(groupId: String, content: String): Boolean {
        return try {
            val result = ApiService.sendGroupMessage(groupId, content)
            result.isSuccess
        } catch (e: Exception) {
            Log.e(TAG, "API sendGroupMessage failed", e)
            false
        }
    }
    
    // ========== HELPER FUNCTIONS ==========
    
    private fun parseUserFromJson(json: JSONObject): User? {
        return try {
            User(
                userId = json.optString("userId", json.optString("_id", "")),
                email = json.optString("email", ""),
                username = json.optString("username", ""),
                avatar = json.optString("avatar", null),
                bio = json.optString("bio", null),
                // Check both field names for compatibility (backend uses followerCount, model uses followersCount)
                followersCount = json.optInt("followerCount", json.optInt("followersCount", 0)),
                followingCount = json.optInt("followingCount", 0),
                postsCount = json.optInt("postsCount", 0),
                createdAt = json.optString("createdAt", "")
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing user: ${e.message}")
            null
        }
    }
    
    private fun parsePostFromJson(json: JSONObject): Post? {
        return try {
            val mediaUrls = mutableListOf<String>()
            var detectedMediaType = "text"
            
            // Check for mediaUrls array (new format)
            val mediaUrlsArray = json.optJSONArray("mediaUrls")
            if (mediaUrlsArray != null) {
                for (i in 0 until mediaUrlsArray.length()) {
                    val url = mediaUrlsArray.optString(i, "")
                    if (url.isNotEmpty()) {
                        mediaUrls.add(url)
                    }
                }
            }
            
            // Check for 'media' array (DynamoDB format with objects containing url and type)
            val mediaArray = json.optJSONArray("media")
            if (mediaArray != null && mediaArray.length() > 0) {
                for (i in 0 until mediaArray.length()) {
                    val mediaObj = mediaArray.optJSONObject(i)
                    if (mediaObj != null) {
                        val url = mediaObj.optString("url", "")
                        val type = mediaObj.optString("type", "image")
                        if (url.isNotEmpty()) {
                            mediaUrls.add(url)
                            // Use the first media item's type
                            if (i == 0) {
                                detectedMediaType = type
                            }
                        }
                    }
                }
            }
            
            // Also check for single mediaUrl field
            val singleMediaUrl = json.optString("mediaUrl", "")
            if (singleMediaUrl.isNotEmpty() && mediaUrls.isEmpty()) {
                mediaUrls.add(singleMediaUrl)
            }
            
            // Get mediaType from JSON or use detected type
            val mediaType = json.optString("mediaType", "").ifEmpty { 
                if (mediaUrls.isNotEmpty()) detectedMediaType else "text" 
            }
            
            // Log for debugging
            Log.d(TAG, "Post ${json.optString("postId", "?")} - mediaUrls: ${mediaUrls.size}, type: $mediaType, first: ${mediaUrls.firstOrNull() ?: "none"}")
            
            Post(
                postId = json.optString("postId", json.optString("_id", UUID.randomUUID().toString())),
                userId = json.optString("userId", ""),
                username = json.optString("username", "User"),
                userAvatar = json.optString("userAvatar", null),
                content = json.optString("content", ""),
                mediaUrl = mediaUrls.firstOrNull(),
                mediaUrls = mediaUrls,
                mediaType = mediaType,
                likesCount = json.optInt("likesCount", json.optInt("likes", json.optInt("likeCount", 0))),
                commentsCount = json.optInt("commentsCount", json.optInt("commentCount", json.optJSONArray("comments")?.length() ?: 0)),
                sharesCount = json.optInt("sharesCount", json.optInt("shares", json.optInt("shareCount", 0))),
                viewsCount = json.optInt("viewsCount", json.optInt("views", json.optInt("viewCount", 0))),
                createdAt = json.optString("createdAt", ""),
                isLiked = json.optBoolean("isLiked", json.optBoolean("isLikedByMe", false)),
                isBookmarked = json.optBoolean("isBookmarked", json.optBoolean("isSaved", false)),
                // NSFW flags from backend - persists from DynamoDB
                isNSFW = json.optBoolean("isNSFW", json.optBoolean("isNsfw", false)),
                isSensitive = json.optBoolean("isSensitive", json.optBoolean("isNsfw", false))
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing post: ${e.message}")
            null
        }
    }
    
    private fun parseGroupFromJson(json: JSONObject): Group? {
        return try {
            Group(
                groupId = json.optString("groupId", json.optString("_id", "")),
                name = json.optString("name", ""),
                description = json.optString("description", null),
                imageUrl = json.optString("imageUrl", null),
                creatorId = json.optString("creatorId", ""),
                isPublic = json.optBoolean("isPublic", false),
                memberCount = json.optInt("memberCount", 0),
                createdAt = json.optString("createdAt", "")
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing group: ${e.message}")
            null
        }
    }
    
    private fun parseMessageFromJson(json: JSONObject): Message? {
        return try {
            val createdAt = json.optString("createdAt", "")
            Log.d(TAG, "parseMessageFromJson: createdAt=$createdAt, content=${json.optString("content", "")}")
            
            Message(
                messageId = json.optString("messageId", json.optString("_id", "")),
                senderId = json.optString("senderId", ""),
                receiverId = json.optString("receiverId", ""),
                conversationId = json.optString("conversationId", ""),
                content = json.optString("content", ""),
                mediaUrl = json.optString("mediaUrl", null),
                mediaType = json.optString("mediaType", null),
                createdAt = createdAt,
                isRead = json.optBoolean("isRead", false)
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error parsing message: ${e.message}")
            null
        }
    }
}
