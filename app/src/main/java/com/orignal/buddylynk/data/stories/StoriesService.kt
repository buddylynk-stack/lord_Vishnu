package com.orignal.buddylynk.data.stories

import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.aws.AwsConfig
import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.aws.S3Service
import com.orignal.buddylynk.data.model.Story
import com.orignal.buddylynk.data.model.User
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.util.UUID

/**
 * Grouped stories by user
 */
data class UserStories(
    val user: User,
    val stories: List<Story>,
    val hasUnviewed: Boolean
)

/**
 * StoriesService - Manages 24h disappearing stories
 */
object StoriesService {
    
    private val _myStories = MutableStateFlow<List<Story>>(emptyList())
    val myStories: StateFlow<List<Story>> = _myStories.asStateFlow()
    
    private val _feedStories = MutableStateFlow<List<UserStories>>(emptyList())
    val feedStories: StateFlow<List<UserStories>> = _feedStories.asStateFlow()
    
    /**
     * Create a new story
     */
    suspend fun createStory(
        context: Context,
        mediaUri: Uri,
        isVideo: Boolean,
        caption: String? = null
    ): Story? = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext null
        
        try {
            val storyId = UUID.randomUUID().toString()
            
            // Upload media to S3
            val mediaUrl = S3Service.uploadStoryMedia(context, storyId, mediaUri, isVideo)
                ?: return@withContext null
            
            val story = Story(
                storyId = storyId,
                userId = currentUser.userId,
                username = currentUser.username,
                userAvatar = currentUser.avatar,
                mediaUrl = mediaUrl,
                mediaType = if (isVideo) "video" else "image",
                caption = caption,
                createdAt = System.currentTimeMillis(),
                expiresAt = System.currentTimeMillis() + 24 * 60 * 60 * 1000
            )
            
            // Save to DynamoDB
            val success = DynamoDbService.createStory(story)
            
            if (success) {
                _myStories.value = listOf(story) + _myStories.value
                story
            } else null
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Get stories from users I follow
     */
    suspend fun loadStoriesFeed(): List<UserStories> = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext emptyList()
        
        try {
            // Get all recent stories
            val allStories = DynamoDbService.getStories()
            
            // Filter expired stories
            val activeStories = allStories.filter { !it.isExpired() }
            
            // Group by user
            val groupedStories = activeStories.groupBy { it.userId }
            
            // Create UserStories for each group
            val userStoriesList = groupedStories.map { (userId, stories) ->
                val user = DynamoDbService.getUser(userId) ?: User(
                    userId = userId,
                    username = stories.firstOrNull()?.username ?: "User",
                    email = "",
                    avatar = stories.firstOrNull()?.userAvatar
                )
                
                val hasUnviewed = stories.any { story ->
                    !story.viewers.contains(currentUser.userId)
                }
                
                UserStories(
                    user = user,
                    stories = stories.sortedBy { it.createdAt },
                    hasUnviewed = hasUnviewed
                )
            }
            
            // Sort: unviewed first, then by most recent
            val sorted = userStoriesList.sortedWith(
                compareByDescending<UserStories> { it.hasUnviewed }
                    .thenByDescending { it.stories.maxOfOrNull { story -> story.createdAt } }
            )
            
            _feedStories.value = sorted
            sorted
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Load my own stories
     */
    suspend fun loadMyStories(): List<Story> = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext emptyList()
        
        try {
            val stories = DynamoDbService.getUserStories(currentUser.userId)
                .filter { !it.isExpired() }
                .sortedBy { it.createdAt }
            
            _myStories.value = stories
            stories
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Mark story as viewed
     */
    suspend fun markViewed(storyId: String): Boolean = withContext(Dispatchers.IO) {
        val currentUser = AuthManager.currentUser.value ?: return@withContext false
        
        try {
            DynamoDbService.addStoryViewer(storyId, currentUser.userId)
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Delete a story
     */
    suspend fun deleteStory(storyId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val success = DynamoDbService.deleteStory(storyId)
            if (success) {
                _myStories.value = _myStories.value.filter { it.storyId != storyId }
            }
            success
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * Get story viewers
     */
    suspend fun getViewers(storyId: String): List<User> = withContext(Dispatchers.IO) {
        try {
            val story = DynamoDbService.getStory(storyId) ?: return@withContext emptyList()
            story.viewers.mapNotNull { userId ->
                DynamoDbService.getUser(userId)
            }
        } catch (e: Exception) {
            emptyList()
        }
    }
}
