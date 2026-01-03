package com.orignal.buddylynk.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.orignal.buddylynk.data.auth.AuthManager
import com.orignal.buddylynk.data.repository.BackendRepository
import com.orignal.buddylynk.data.aws.DynamoDbService
import com.orignal.buddylynk.data.aws.getGroupPosts
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * TeamUpViewModel - Fetches real teams/groups via BackendRepository
 * Uses API or DynamoDB based on BackendRepository.USE_API setting
 */
class TeamUpViewModel : ViewModel() {
    
    data class TeamItem(
        val id: String,
        val name: String,
        val type: String, // "group" or "channel"
        val members: Int,
        val active: Int,
        val avatar: String?,
        val lastMsg: String,
        val time: String,
        val unread: Int
    )
    
    private val _teams = MutableStateFlow<List<TeamItem>>(emptyList())
    val teams: StateFlow<List<TeamItem>> = _teams.asStateFlow()
    
    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    init {
        loadTeams()
    }
    
    fun loadTeams() {
        viewModelScope.launch {
            _isLoading.value = true
            _error.value = null
            
            try {
                val currentUser = AuthManager.currentUser.value
                if (currentUser == null) {
                    _teams.value = emptyList()
                    _isLoading.value = false
                    return@launch
                }
                
                // Fetch groups via BackendRepository (API or DynamoDB)
                val groups = BackendRepository.getUserGroups(currentUser.userId)
                
                // Convert Group model to TeamItem
                val teamItems = groups.map { group ->
                    TeamItem(
                        id = group.groupId,
                        name = group.name,
                        type = if (group.isPublic) "channel" else "group",
                        members = group.memberCount,
                        active = 0, // Could track online members later
                        avatar = group.imageUrl,
                        lastMsg = group.description ?: "Tap to start chatting",
                        time = formatTime(group.createdAt),
                        unread = 0 // Could track unread messages later
                    )
                }
                
                _teams.value = teamItems
                
            } catch (e: Exception) {
                android.util.Log.e("TeamUpViewModel", "Error loading teams: ${e.message}", e)
                _error.value = "Failed to load teams"
                _teams.value = emptyList()
            } finally {
                _isLoading.value = false
            }
        }
    }
    
    private fun formatTime(timestamp: String): String {
        return try {
            val millis = timestamp.toLongOrNull() ?: return "Just now"
            val diff = System.currentTimeMillis() - millis
            val minutes = diff / 60000
            val hours = minutes / 60
            val days = hours / 24
            
            when {
                minutes < 1 -> "Just now"
                minutes < 60 -> "${minutes}m ago"
                hours < 24 -> "${hours}h ago"
                days < 7 -> "${days}d ago"
                else -> "A while ago"
            }
        } catch (e: Exception) {
            "Just now"
        }
    }
    
    fun refreshTeams() {
        loadTeams()
    }
    
    // Group Messages with media support
    data class MediaItem(
        val type: String,  // "image" or "video"
        val url: String
    )
    
    data class GroupMessage(
        val id: String,
        val senderId: String,
        val senderName: String,
        val senderAvatar: String?,
        val content: String,
        val createdAt: String,
        val isMe: Boolean,
        val media: List<MediaItem> = emptyList()  // S3 URLs for images/videos
    )
    
    private val _groupMessages = MutableStateFlow<List<GroupMessage>>(emptyList())
    val groupMessages: StateFlow<List<GroupMessage>> = _groupMessages.asStateFlow()
    
    private val _messagesLoading = MutableStateFlow(false)
    val messagesLoading: StateFlow<Boolean> = _messagesLoading.asStateFlow()
    
    fun loadGroupMessages(groupId: String) {
        viewModelScope.launch {
            _messagesLoading.value = true
            try {
                android.util.Log.d("TeamUpViewModel", "Loading posts for group via API: $groupId")
                
                val currentUserId = AuthManager.currentUser.value?.userId ?: ""
                var posts = mutableListOf<GroupMessage>()
                
                // Try API first
                val apiResult = com.orignal.buddylynk.data.api.ApiService.getGroup(groupId)
                if (apiResult.isSuccess) {
                    val groupJson = apiResult.getOrNull()
                    android.util.Log.d("TeamUpViewModel", "API returned group: ${groupJson?.optString("name", "")}")
                    
                    // Parse posts array from group JSON
                    val postsArray = groupJson?.optJSONArray("posts")
                    if (postsArray != null && postsArray.length() > 0) {
                        android.util.Log.d("TeamUpViewModel", "Found ${postsArray.length()} posts in API response")
                        
                        for (i in 0 until postsArray.length()) {
                            val postJson = postsArray.optJSONObject(i) ?: continue
                            
                            // Parse media array
                            val mediaItems = mutableListOf<MediaItem>()
                            val mediaArray = postJson.optJSONArray("media")
                            if (mediaArray != null) {
                                for (j in 0 until mediaArray.length()) {
                                    val mediaJson = mediaArray.optJSONObject(j) ?: continue
                                    mediaItems.add(MediaItem(
                                        type = mediaJson.optString("type", "image"),
                                        url = mediaJson.optString("url", "")
                                    ))
                                }
                            }
                            
                            val postUserId = postJson.optString("userId", "")
                            posts.add(GroupMessage(
                                id = postJson.optString("postId", ""),
                                senderId = postUserId,
                                senderName = "Group Member",
                                senderAvatar = null,
                                content = postJson.optString("content", ""),
                                createdAt = formatTime(postJson.optString("createdAt", "")),
                                isMe = postUserId == currentUserId,
                                media = mediaItems
                            ))
                        }
                    }
                } else {
                    android.util.Log.e("TeamUpViewModel", "API failed, falling back to DynamoDB")
                    // Fallback to DynamoDB
                    val dynamoPosts = DynamoDbService.getGroupPosts(groupId)
                    posts = dynamoPosts.map { post ->
                        GroupMessage(
                            id = post.postId,
                            senderId = post.userId,
                            senderName = "Group Member",
                            senderAvatar = null,
                            content = post.content,
                            createdAt = formatTime(post.createdAt),
                            isMe = post.userId == currentUserId,
                            media = post.media.map { m -> MediaItem(type = m.type, url = m.url) }
                        )
                    }.toMutableList()
                }
                
                android.util.Log.d("TeamUpViewModel", "Total posts loaded: ${posts.size}")
                _groupMessages.value = posts.sortedByDescending { it.createdAt }
                
            } catch (e: Exception) {
                android.util.Log.e("TeamUpViewModel", "Error loading posts: ${e.message}", e)
                _groupMessages.value = emptyList()
            } finally {
                _messagesLoading.value = false
            }
        }
    }
    
    fun sendGroupMessage(groupId: String, content: String) {
        viewModelScope.launch {
            try {
                val currentUser = AuthManager.currentUser.value ?: return@launch
                
                val newMessage = BackendRepository.sendGroupMessage(groupId, content)
                
                // Add to local list immediately if successful
                if (newMessage) {
                    _groupMessages.value = _groupMessages.value + GroupMessage(
                        id = java.util.UUID.randomUUID().toString(),
                        senderId = currentUser.userId,
                        senderName = currentUser.username,
                        senderAvatar = currentUser.avatar,
                        content = content,
                        createdAt = "Just now",
                        isMe = true
                    )
                }
            } catch (e: Exception) {
                android.util.Log.e("TeamUpViewModel", "Error sending message: ${e.message}", e)
            }
        }
    }
}

