package com.orignal.buddylynk.data.aws

import com.orignal.buddylynk.data.model.*

/**
 * STUB: DynamoDbService - This is a placeholder for migration to API-only
 * 
 * All methods return empty/null results. The actual data should come from the API.
 * This file exists only to satisfy imports during the migration period.
 */
object DynamoDbService {
    
    // User operations - return null/empty
    suspend fun getUser(userId: String): User? = null
    suspend fun createUser(user: User): Boolean = false
    suspend fun updateUser(user: User): Boolean = false
    suspend fun loginByEmail(email: String): User? = null
    suspend fun registerUser(email: String, username: String, password: String): User? = null
    suspend fun searchUsers(query: String): List<User> = emptyList()
    suspend fun getUsers(limit: Int = 20): List<User> = emptyList()
    suspend fun updateUserCounts(userId: String, followersDelta: Int, followingDelta: Int): Boolean = false
    suspend fun updateUserPostsProfile(userId: String, newUsername: String, newAvatar: String?): Boolean = false
    
    // Post operations - return null/empty
    suspend fun getPosts(limit: Int = 50): List<Post> = emptyList()
    suspend fun getUserPosts(userId: String): List<Post> = emptyList()
    suspend fun createPost(post: Post): Boolean = false
    suspend fun updatePostLikes(postId: String, userId: String): Boolean = false
    suspend fun deletePost(postId: String): Boolean = false
    
    // Message operations - return empty
    suspend fun getMessages(conversationId: String): List<Message> = emptyList()
    suspend fun sendMessage(message: Message): Boolean = false
    
    // Follow operations - return empty/false
    suspend fun createFollow(follow: Follow): Boolean = false
    suspend fun deleteFollow(followerId: String, followingId: String): Boolean = false
    suspend fun checkFollowing(userId: String, targetId: String): Boolean = false
    suspend fun getFollowing(userId: String): List<Follow> = emptyList()
    suspend fun getFollowers(userId: String): List<Follow> = emptyList()
    
    // Activity operations - return empty
    suspend fun getActivities(userId: String): List<Activity> = emptyList()
    suspend fun createActivity(activity: Activity, targetUserId: String): Boolean = false
    
    // Notification operations - return empty
    suspend fun getNotifications(userId: String): List<Notification> = emptyList()
    
    // Story operations - return empty/false
    suspend fun getStories(): List<Story> = emptyList()
    suspend fun getUserStories(userId: String): List<Story> = emptyList()
    suspend fun getStory(storyId: String): Story? = null
    suspend fun createStory(story: Story): Boolean = false
    suspend fun deleteStory(storyId: String): Boolean = false
    suspend fun addStoryViewer(storyId: String, viewerId: String): Boolean = false
    
    // Event operations - return empty/false
    suspend fun getEvents(): List<Event> = emptyList()
    suspend fun createEvent(event: Event): Boolean = false
    
    // Group operations - return empty
    suspend fun getUserGroups(userId: String): List<Group> = emptyList()
    suspend fun createGroup(group: Group): Boolean = false
    suspend fun getGroup(groupId: String): Group? = null
    suspend fun getGroupMessages(groupId: String): List<Message> = emptyList()
}

// Extension functions that were in the aws package
suspend fun DynamoDbService.getUserGroups(userId: String): List<Group> = emptyList()
suspend fun DynamoDbService.createGroup(group: Group): Boolean = false
suspend fun DynamoDbService.getUserConversationPartners(userId: String): List<User> = emptyList()

// GroupPost for TeamUpViewModel
data class GroupPost(
    val postId: String = "",
    val userId: String = "",
    val content: String = "",
    val createdAt: String = "",
    val likes: Int = 0,
    val media: List<GroupPostMedia> = emptyList()
)

data class GroupPostMedia(
    val type: String = "image",
    val url: String = ""
)

suspend fun DynamoDbService.getGroupPosts(groupId: String): List<GroupPost> = emptyList()
