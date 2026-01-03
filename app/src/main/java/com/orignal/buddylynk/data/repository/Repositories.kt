package com.orignal.buddylynk.data.repository

import android.content.Context
import android.net.Uri
import com.orignal.buddylynk.data.model.*

/**
 * User Repository - Uses BackendRepository API calls
 */
class UserRepository {
    
    suspend fun getUser(userId: String): User? {
        return BackendRepository.getUser(userId)
    }
    
    suspend fun createUser(user: User): Boolean {
        // Use API - registration creates user
        return false // Handled by BackendRepository.register
    }
    
    suspend fun updateUser(user: User): Boolean {
        // TODO: Add updateUser API endpoint
        return false
    }
    
    suspend fun uploadProfileImage(context: Context, userId: String, imageUri: Uri): String? {
        // Use API pre-signed URL for upload
        val result = BackendRepository.getPresignedUploadUrl(
            "${System.currentTimeMillis()}.jpg",
            "image/jpeg",
            "avatars"
        )
        return result?.second // Return the file URL
    }
}

/**
 * Post Repository - Uses BackendRepository API calls
 */
class PostRepository {
    
    suspend fun getFeedPosts(limit: Int = 20): List<Post> {
        return BackendRepository.getFeedPosts()
    }
    
    suspend fun createPost(post: Post): Boolean {
        return BackendRepository.createPost(post)
    }
    
    suspend fun uploadPostMedia(context: Context, postId: String, mediaUri: Uri, isVideo: Boolean = false): String? {
        // Use API pre-signed URL for upload
        val contentType = if (isVideo) "video/mp4" else "image/jpeg"
        val ext = if (isVideo) "mp4" else "jpg"
        val result = BackendRepository.getPresignedUploadUrl(
            "${System.currentTimeMillis()}.$ext",
            contentType,
            "posts"
        )
        return result?.second // Return the file URL
    }
}

/**
 * Message Repository - Uses BackendRepository API calls
 */
class MessageRepository {
    
    suspend fun getMessages(conversationId: String): List<Message> {
        return BackendRepository.getMessages(conversationId)
    }
    
    suspend fun sendMessage(message: Message): Boolean {
        return BackendRepository.sendMessage(message.receiverId, message.content)
    }
    
    suspend fun uploadMessageMedia(context: Context, conversationId: String, mediaUri: Uri): String? {
        // Use API pre-signed URL for upload
        val result = BackendRepository.getPresignedUploadUrl(
            "${System.currentTimeMillis()}.jpg",
            "image/jpeg",
            "messages"
        )
        return result?.second // Return the file URL
    }
}

/**
 * Notification Repository
 */
class NotificationRepository {
    
    suspend fun getNotifications(userId: String): List<Notification> {
        // TODO: Add notifications API endpoint
        return emptyList()
    }
}
