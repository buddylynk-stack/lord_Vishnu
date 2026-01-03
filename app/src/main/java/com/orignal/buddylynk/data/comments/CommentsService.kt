package com.orignal.buddylynk.data.comments

import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.model.Comment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * Comments Service - Manage post comments
 */
object CommentsService {
    
    // Local cache for demo
    private val commentsCache = mutableMapOf<String, MutableList<Comment>>()
    
    /**
     * Create a comment
     */
    suspend fun createComment(
        postId: String,
        content: String,
        parentCommentId: String? = null
    ): Comment? = withContext(Dispatchers.IO) {
        val user = AuthManager.currentUser.value ?: return@withContext null
        
        val comment = Comment(
            commentId = UUID.randomUUID().toString(),
            postId = postId,
            userId = user.userId,
            username = user.username,
            userAvatar = user.avatar,
            content = content,
            parentCommentId = parentCommentId,
            likesCount = 0,
            repliesCount = 0,
            isLiked = false,
            createdAt = System.currentTimeMillis().toString()
        )
        
        // Cache locally
        val postComments = commentsCache.getOrPut(postId) { mutableListOf() }
        postComments.add(0, comment)
        
        // If reply, increment parent reply count
        if (parentCommentId != null) {
            postComments.find { it.commentId == parentCommentId }?.let { parent ->
                val index = postComments.indexOf(parent)
                if (index >= 0) {
                    postComments[index] = parent.copy(repliesCount = parent.repliesCount + 1)
                }
            }
        }
        
        // TODO: Save to DynamoDB
        comment
    }
    
    /**
     * Get comments for a post
     */
    suspend fun getComments(
        postId: String,
        limit: Int = 50
    ): List<Comment> = withContext(Dispatchers.IO) {
        // Return cached or empty
        commentsCache[postId]?.take(limit) ?: emptyList()
    }
    
    /**
     * Get replies to a comment
     */
    suspend fun getReplies(
        commentId: String,
        limit: Int = 20
    ): List<Comment> = withContext(Dispatchers.IO) {
        commentsCache.values.flatten()
            .filter { it.parentCommentId == commentId }
            .take(limit)
    }
    
    /**
     * Like a comment
     */
    suspend fun likeComment(commentId: String): Boolean = withContext(Dispatchers.IO) {
        commentsCache.values.forEach { comments ->
            val index = comments.indexOfFirst { it.commentId == commentId }
            if (index >= 0) {
                val c = comments[index]
                comments[index] = c.copy(isLiked = true, likesCount = c.likesCount + 1)
                return@withContext true
            }
        }
        false
    }
    
    /**
     * Unlike a comment
     */
    suspend fun unlikeComment(commentId: String): Boolean = withContext(Dispatchers.IO) {
        commentsCache.values.forEach { comments ->
            val index = comments.indexOfFirst { it.commentId == commentId }
            if (index >= 0) {
                val c = comments[index]
                comments[index] = c.copy(isLiked = false, likesCount = maxOf(0, c.likesCount - 1))
                return@withContext true
            }
        }
        false
    }
    
    /**
     * Delete a comment
     */
    suspend fun deleteComment(commentId: String, postId: String): Boolean = withContext(Dispatchers.IO) {
        val userId = AuthManager.getCurrentUserId() ?: return@withContext false
        
        val comments = commentsCache[postId] ?: return@withContext false
        val comment = comments.find { it.commentId == commentId } ?: return@withContext false
        
        // Verify ownership
        if (comment.userId != userId) return@withContext false
        
        comments.remove(comment)
        true
    }
    
    /**
     * Edit a comment
     */
    suspend fun editComment(commentId: String, newContent: String): Boolean = withContext(Dispatchers.IO) {
        val userId = AuthManager.getCurrentUserId() ?: return@withContext false
        
        commentsCache.values.forEach { comments ->
            val index = comments.indexOfFirst { it.commentId == commentId }
            if (index >= 0) {
                val c = comments[index]
                if (c.userId != userId) return@withContext false
                comments[index] = c.copy(content = newContent, isEdited = true)
                return@withContext true
            }
        }
        false
    }
    
    /**
     * Add demo comments for a post
     */
    fun addDemoComments(postId: String) {
        if (commentsCache.containsKey(postId)) return
        
        commentsCache[postId] = mutableListOf(
            Comment(
                commentId = "c1",
                postId = postId,
                userId = "demo1",
                username = "alex_smith",
                content = "This is amazing! 🔥",
                likesCount = 12,
                createdAt = (System.currentTimeMillis() - 3600000).toString()
            ),
            Comment(
                commentId = "c2",
                postId = postId,
                userId = "demo2",
                username = "sarah_j",
                content = "Love this content!",
                likesCount = 5,
                createdAt = (System.currentTimeMillis() - 7200000).toString()
            ),
            Comment(
                commentId = "c3",
                postId = postId,
                userId = "demo3",
                username = "mike_tech",
                content = "Great post, keep it up! 👏",
                parentCommentId = "c1",
                likesCount = 2,
                createdAt = (System.currentTimeMillis() - 1800000).toString()
            )
        )
    }
}
