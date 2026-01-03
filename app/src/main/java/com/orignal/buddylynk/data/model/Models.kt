package com.orignal.buddylynk.data.model

/**
 * User data model - Matches Buddylynk_Users DynamoDB table
 */
data class User(
    val userId: String,
    val username: String,
    val email: String,
    val password: String? = null, // Only for local use, never expose
    val avatar: String? = null,
    val avatarColor: String? = null, // Hex color for default avatar (e.g. "#E91E63")
    val banner: String? = null,
    val bio: String? = null,
    val location: String? = null,
    val website: String? = null,
    val followersCount: Int = 0,
    val followingCount: Int = 0,
    val postsCount: Int = 0,
    val isVerified: Boolean = false,
    val isOnline: Boolean = false,
    val createdAt: String = ""
)

/**
 * Post data model - Matches Buddylynk_Posts DynamoDB table
 */
data class Post(
    val postId: String,
    val userId: String,
    val username: String? = null,
    val userAvatar: String? = null,
    val content: String,
    val mediaUrl: String? = null,
    val mediaUrls: List<String> = emptyList(), // Multiple media support
    val mediaType: String? = null, // "image" or "video"
    val likesCount: Int = 0,
    val commentsCount: Int = 0,
    val sharesCount: Int = 0,
    val viewsCount: Int = 0,
    val isLiked: Boolean = false,
    val isBookmarked: Boolean = false,
    val isNSFW: Boolean = false, // Admin-flagged 18+ content (from NSFW DynamoDB table)
    val isSensitive: Boolean = false, // General sensitive content flag
    val createdAt: String = ""
)

/**
 * Message data model - Matches Buddylynk_Messages DynamoDB table
 */
data class Message(
    val messageId: String,
    val conversationId: String,
    val senderId: String,
    val receiverId: String,
    val senderName: String? = null,
    val senderAvatar: String? = null,
    val content: String,
    val mediaUrl: String? = null,
    val mediaType: String? = null,
    val isRead: Boolean = false,
    val createdAt: String = ""
)

/**
 * Group data model - Matches Buddylynk_Groups DynamoDB table
 */
data class Group(
    val groupId: String,
    val name: String,
    val description: String? = null,
    val imageUrl: String? = null,
    val creatorId: String,
    val memberIds: List<String> = emptyList(),
    val memberCount: Int = 0,
    val isPublic: Boolean = true,
    val createdAt: String = ""
)

/**
 * Notification data model - Matches Buddylynk_Notifications
 */
data class Notification(
    val notificationId: String,
    val userId: String,
    val type: String, // "like", "comment", "follow", "message"
    val title: String,
    val body: String,
    val fromUserId: String? = null,
    val fromUsername: String? = null,
    val fromAvatar: String? = null,
    val postId: String? = null,
    val isRead: Boolean = false,
    val createdAt: String = ""
)

/**
 * PostView data model - Matches Buddylynk_PostViews
 */
data class PostView(
    val viewId: String,
    val postId: String,
    val userId: String,
    val viewedAt: String = ""
)

/**
 * Follow relationship - Matches Buddylynk_Follows table
 */
data class Follow(
    val followId: String,
    val followerId: String, // Who is following
    val followingId: String, // Who is being followed
    val followerUsername: String? = null,
    val followerAvatar: String? = null,
    val createdAt: String = ""
)

/**
 * Story data model - 24h disappearing content
 */
data class Story(
    val storyId: String,
    val userId: String,
    val username: String? = null,
    val userAvatar: String? = null,
    val mediaUrl: String,
    val mediaType: String = "image", // "image" or "video"
    val caption: String? = null,
    val viewsCount: Int = 0,
    val viewers: List<String> = emptyList(),
    val createdAt: Long = System.currentTimeMillis(),
    val expiresAt: Long = System.currentTimeMillis() + 24 * 60 * 60 * 1000 // 24 hours
) {
    fun isExpired(): Boolean = System.currentTimeMillis() > expiresAt
}

/**
 * Conversation for chat
 */
data class Conversation(
    val conversationId: String,
    val participantIds: List<String>,
    val participantNames: List<String> = emptyList(),
    val participantAvatars: List<String> = emptyList(),
    val lastMessage: String = "",
    val lastMessageTime: String = "",
    val lastSenderId: String? = null,
    val unreadCount: Int = 0,
    val isOnline: Boolean = false,
    val createdAt: String = ""
) {
    // Convenience properties for single-participant chats
    val participantName: String get() = participantNames.firstOrNull() ?: "Unknown"
    val participantAvatar: String? get() = participantAvatars.firstOrNull()
}

/**
 * Activity for activity feed
 */
data class Activity(
    val activityId: String,
    val type: String, // "like", "comment", "follow", "mention"
    val actorId: String,
    val actorUsername: String,
    val actorAvatar: String? = null,
    val targetId: String? = null, // postId or userId
    val targetPreview: String? = null, // post preview or null
    val isRead: Boolean = false,
    val createdAt: String = ""
)

/**
 * Hashtag for trending and search
 */
data class Hashtag(
    val tag: String,
    val postsCount: Int = 0,
    val isFollowing: Boolean = false
)

/**
 * User profile with follow status
 */
data class UserProfile(
    val user: User,
    val isFollowing: Boolean = false,
    val isFollower: Boolean = false,
    val isMutual: Boolean = false,
    val posts: List<Post> = emptyList()
)

/**
 * Comment data model
 */
data class Comment(
    val commentId: String,
    val postId: String,
    val userId: String,
    val username: String,
    val userAvatar: String? = null,
    val content: String,
    val parentCommentId: String? = null,
    val likesCount: Int = 0,
    val repliesCount: Int = 0,
    val isLiked: Boolean = false,
    val isEdited: Boolean = false,
    val isPinned: Boolean = false,
    val createdAt: String = ""
)

